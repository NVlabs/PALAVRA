# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file 
# located at the root directory.

import torch
import clip
import numpy as np
import sys
sys.path.append("../")
sys.path.append("clip_language_set")
import torch.optim as optim
import torch.nn.functional as F
import re
from utils.nv import l2_norm_loss, contrastive_loss


emb_dim: int = 512
num_tokens = 77
device = "cuda" if torch.cuda.is_available() else "cpu"


def optimize_token(args, object_tokens, model, train_dataloader, gt_text_label, text_vis_model):
    # Second stage token optimzation

    prompt = args.evalparams.text_prompt #set ptompt to be used throughout optimization
    prompt = prompt + "* "
    asterix_token = clip.tokenize(["*"]).to(device)[0][1]

    trainable_estimated_tokens = torch.nn.Embedding.from_pretrained(object_tokens, freeze=False) #create learnble tokens
    optimizer = optim.Adam(trainable_estimated_tokens.parameters(), lr=args.evalparams.latent_lr)

    #For positive samples, or neative+positive samples, train as usual
    if (args.evalparams.token_optimize_mode == 0) or (args.evalparams.token_optimize_mode == 2):
        trainable_estimated_tokens = optimize_trainable_token(args, object_tokens, asterix_token, prompt, model,
                                                                  optimizer, train_dataloader, trainable_estimated_tokens)

    #Training with coarse grained as negative sample needs: (a) the coarse grained phrase embeddings (which might be class dependant)
    if (args.evalparams.token_optimize_mode == 1):
        with torch.no_grad():
            if args.evalparams.is_coarse_grained_negative_per_class:
                print("gt_text_label",gt_text_label)
            else:
                gt_text_label = [args.evalparams.per_dataset_coarse_grained_phrase for i in range(len(gt_text_label))]

            coarse_grained_embeddings = get_clip_embedding_from_tokens(args, model, gt_text_label = gt_text_label)
            coarse_grained_embeddings = F.normalize(coarse_grained_embeddings, dim=-1)
            coarse_grained_embeddings = text_vis_model(coarse_grained_embeddings.float())

        trainable_estimated_tokens = optimize_trainable_token(args, object_tokens, asterix_token, prompt, model,
                                                                  optimizer, train_dataloader, trainable_estimated_tokens,
                                                                    coarse_grained_embeddings = coarse_grained_embeddings)

    res_object_tokens = trainable_estimated_tokens.weight

    return res_object_tokens


def parse_coarse_grained_strings(args, gt_text_label):
    gt_text_label = [re.sub(r"(\w)([A-Z])", r"\1 \2", gt_text_label_i) for gt_text_label_i in gt_text_label]

    return gt_text_label



def infer_tokens_from_f_theta(args, train_args, model, set_model, train_dataloader, text_vis_model, gt_text_coarse_func=None):

    label_acc, object_tokens_acc, image_features_list_acc = [], [], []

    for batch_num, sample in enumerate(train_dataloader):
        #get image features
        images, label = sample

        image_features_list = []
        object_tokens = torch.tensor([])

        for i in range(images.shape[1]):
            image_features = model.encode_image(images[:,i].cuda())
            image_features = F.normalize(image_features, dim=-1)
            image_features_list.append(image_features)

        image_features_list = torch.stack(image_features_list, 1)

        if train_args.is_learn_token:
            object_tokens = set_model(image_features_list.float())


        label_acc.append(np.asarray(label))
        object_tokens_acc.append(object_tokens)
        image_features_list_acc.append(image_features_list)

    label_acc = np.concatenate(label_acc)
    object_tokens_acc = torch.cat(object_tokens_acc)
    image_features_list_acc = torch.cat(image_features_list_acc)

    return label_acc, object_tokens_acc, image_features_list_acc, image_features_list




def calc_contrastive_loss_with_coarse_grained_token(image_features_means, clip_embeddings, coarse_grained_embeddings, contrastive_temp):
    #calculte 2-entries contrastive for each learnt token and its coarse grained concept
    coarse_grained_embeddings.requires_grad = False
    image_features_and_coarse = [ torch.stack((image_feature, coarse_grained_embeddings[i])) for i,image_feature in enumerate(image_features_means)]
    clip_embeddings_and_coarse = [ torch.stack((clip_embedding, coarse_grained_embeddings[i])) for i,clip_embedding in enumerate(clip_embeddings)]

    loss_i = [contrastive_loss(image_features_and_coarse[i], clip_embeddings, contrastive_temp)
                                                        for i,clip_embeddings in enumerate(clip_embeddings_and_coarse)]

    return torch.mean(torch.stack(loss_i))



def optimize_trainable_token(args, object_tokens, asterix_token, prompt, model,
                                 optimizer, train_dataloader, trainable_estimated_tokens,
                                    coarse_grained_embeddings = None):

    for ep in range(args.evalparams.latent_ep):
        sample_ind = 0

        for batch_num, sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            images, label = sample

            image_features_list = []

            #get image features
            with torch.no_grad():
                for i in range(images.shape[1]):
                    image_features = model.encode_image(images[:,i].cuda())
                    image_features = F.normalize(image_features, dim=-1)
                    image_features_list.append(image_features.cpu())
                image_features_list = torch.stack(image_features_list)
                image_features_means = torch.mean(image_features_list, axis = 0).cuda()
            torch.cuda.empty_cache()

            #Get token to optimize
            prompt_asterix = [prompt for object_token in (object_tokens)]
            text = clip.tokenize(prompt_asterix).to(device)
            clip_embeddings = model.encode_text_with_learnt_tokens(text, asterix_token, trainable_estimated_tokens, is_emb = True)
            batch_clip_embeddings = clip_embeddings[sample_ind:sample_ind + len(images)]
            batch_coarse_grained_embeddings = coarse_grained_embeddings[sample_ind:sample_ind + len(images)]
            sample_ind = sample_ind + len(images)

            if (args.evalparams.token_optimize_mode == 0):
                loss = l2_norm_loss(image_features_means, batch_clip_embeddings)
            elif (args.evalparams.token_optimize_mode == 1):
                loss = calc_contrastive_loss_with_coarse_grained_token(image_features_means, batch_clip_embeddings,
                                                         batch_coarse_grained_embeddings, args.evalparams.contrastive_temp)
            elif (args.evalparams.token_optimize_mode == 2):
                loss = contrastive_loss(image_features_means, batch_clip_embeddings, args.evalparams.contrastive_temp)

            loss.backward()
            optimizer.step()

    return trainable_estimated_tokens



def get_clip_embedding_from_tokens(args, model, is_estimated_tokens = False, estimated_tokens = None, gt_text_label = None):

    #embed tokens with evalparams.text_prompt for second stage optimization
    prompt = args.evalparams.text_prompt

    if is_estimated_tokens:
        prompt = prompt + "* "
        asterix_token = clip.tokenize(["*"]).to(device)[0][1]
        prompt_asterix = [prompt for object_token in (estimated_tokens)]
        text = clip.tokenize(prompt_asterix).to(device)
        if args.evalparams.is_token_mean:
            prompt_w_gt_text = [prompt + label for label in (gt_text_label)]
            gt_text = clip.tokenize(prompt_w_gt_text).to(device)
            clip_embeddings = model.encode_text_avg(text, asterix_token, estimated_tokens, gt_text)
        else:
            clip_embeddings = model.encode_text_with_learnt_tokens(text, asterix_token, estimated_tokens)
    else:
        prompt_w_gt_text = [prompt + label for label in (gt_text_label)]
        text = clip.tokenize(prompt_w_gt_text).to(device)
        clip_embeddings = model.encode_text(text)

    clip_embeddings = F.normalize(clip_embeddings, dim=-1)

    return clip_embeddings





