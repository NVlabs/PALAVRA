# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file 
# located at the root directory.

import torch
import clip
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from torch.utils.data import DataLoader
sys.path.append("../")
from utils.non_nv import encode_text_with_learnt_tokens
from utils.deep_set_clf import D as deep_set
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from dataclasses import dataclass
from simple_parsing import ArgumentParser
import wandb
import time
import os
from utils.nv import TextVisualMap, TextVisualMapAbl, natural_prompt_multi, CustomTextDataset, l2_norm_loss, cosine_loss, MLP, contrastive_loss
import random

emb_dim: int = 512
natural_prompt_multi = natural_prompt_multi
num_tokens = 77


@dataclass
class HParams:
    """Set of options for the training of a Model."""
    lr: float = 1e-4 #learning rate
    lr_text_vis: float = 1e-4 #learning rate text_vis
    epochs: int = 1000
    batch_size: int = 200 #input batch size for training
    no_of_new_tokens: int = 1
    is_learn_token: bool = True
    sentence_wise_split: bool = False #split according to setence or to pretraining dataset object
    sentence_wise_ratio: float = 0.05
    is_image_input_train: bool = False
    is_image_input_test: bool = False
    is_text_visual_map: bool = False # Do we want to train the "A" to align image-visual domain shift
    is_text_vis_map_abl: bool = False # Do we want to use the "A" to align image-visual domain shift
    is_prompt_multi: bool = False #Use prompt augmentations:    {"This is a photo of a *", "This photo contains a *",..,} etc.
    is_augment_object: bool = False #use object textual augmentation
    is_save_models: bool = False
    is_learn_prefix: bool = False  #Use also coarse grained textual prompt
    is_learn_prefix_also_image: bool = False #dependant on is_learn_prefix
    is_gt_object_loss: bool = False  #Use gt (l2) loss
    coeff_gt_object_loss: float = 1 #Coefficent for the gt (l2) loss
    coeff_cycle_loss: float = 1 #Coefficent for the cycle loss
    contrastive_temp: float = 0.25
    set_size: int = 5
    deep_set_d_dim: int = 2048
    dropout: float = 0.5
    project_name: str = "FSL_clip"
    loss_str: str = "contrastive" #cycle loss type
    save_model_name: str = "first_save"
    natural_prompt_default: str = "This is a photo of a *"

    pooling_type: str = "mean" #deepset pooling

    #Set transfomer parameters below, currently deprecated
    is_set_transformer: bool = False
    st_num_outputs: int = 1
    st_num_inds: int = 32
    st_dim_hidden: int = 128
    st_num_heads: int = 4
    st_ln: bool = False

    is_multi_set_size: bool = False #Variable set input size, currently deprecated

    object_dict_path: str = "../pervl_data_prep/data/inversion_model_train_data/txt_for_training/commom_obj_dict.npy"
    visual_features_path: str = "../pervl_data_prep/data/inversion_model_train_data/visual_features/visual_features_dict_center_crop_300_224.npz"
    text_aug_map_path: str = "../pervl_data_prep/data/inversion_model_train_data/open_images/open_images_to_mscoco_map.npz"
    text_obj_path: str = "../pervl_data_prep/data/inversion_model_train_data/open_images/open_images_obj_names.npz"

def save_trained_model(save_path, model_name, args, trained_model, timestr):
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path , "%s_%s"%(model_name,timestr) )
    args_path = os.path.join(save_path, "%s_%s"%("args",timestr) )
    np.save(args_path,args.hparams)
    torch.save(trained_model.state_dict(), model_path)



def get_clip_text_acc(natural_prompt_embeddings, natural_prompt_asterix_embeddings):
    #Calculate accuracy
    pred = torch.argmax(torch.mm(natural_prompt_embeddings.float(), natural_prompt_asterix_embeddings.t().contiguous().float()), dim = 1)
    gt = np.arange(len(pred))
    correct_num = (len(np.where(pred.cpu().numpy() == gt)[0]))
    print("acc %d / %d"%(correct_num, len(pred)))
    return correct_num, len(pred)

def run_epoch(args, epoch, dataloader,  optimizer, optimizer_text_vis, scheduler, model, mlp_model, set_model, text_vis_model,
              device, is_train = True, is_image_input = False, img_features_dict = None, is_augment_object = False):
    #run a single training epoch

    if args.hparams.is_prompt_multi and is_train:
        chosen_prompt = np.random.randint(len(natural_prompt_multi))
        natural_prompt = natural_prompt_multi[chosen_prompt]
    else:
        natural_prompt = args.hparams.natural_prompt_default

    asterix_token = clip.tokenize(["*"]).to(device)[0][1]
    no_of_new_tokens = args.hparams.no_of_new_tokens
    criterion = contrastive_loss

    for batch_num, sample in enumerate(dataloader):
        ####### Image input
        if is_image_input:
            visual_features_list = []
            local_batch, local_batch_asterix, local_labels, chosen_img_id = sample

            for i, chosen_img_id_inst in enumerate(chosen_img_id):
                img_features_inst = [img_features_dict[chosen_img_id_inst_i][0] for chosen_img_id_inst_i in chosen_img_id_inst]
                img_features_inst = torch.from_numpy(np.asarray(img_features_inst)).cuda()
                img_features_inst = F.normalize(img_features_inst, dim=-1)
                visual_features_list.append(img_features_inst[:,0,:])
            visual_features = torch.stack(visual_features_list, 1)

        total_loss_text_vis = 0.0
        optimizer_text_vis.zero_grad()

        ####### Text input
        if (is_image_input==False) or (args.hparams.is_text_visual_map):
            text_features_list = []
            est_text_features_list = []
            if args.hparams.is_text_visual_map and is_augment_object==False:
                local_batch, local_batch_asterix, local_labels, chosen_img_id = sample
            else:
                local_batch, local_batch_asterix, local_labels, coarse_labels = sample

            for i, local_batch_inst in enumerate(local_batch):
                    text = clip.tokenize(local_batch_inst).to(device)
                    with torch.no_grad():
                        text_features_inst = model.encode_text(text)
                        text_features_list.append(text_features_inst)
                    if is_train:
                        est_text_features_inst = text_vis_model(text_features_inst.float())
                    else:
                        with torch.no_grad():
                            est_text_features_inst = text_vis_model(text_features_inst.float())
                    est_text_features_list.append(est_text_features_inst)
            text_features = torch.stack(text_features_list, 1)
        else:
            text_features = visual_features

        if args.hparams.is_text_visual_map:
            est_text_features_list = torch.stack(est_text_features_list, 1)
            if is_train and is_augment_object == False:
                if not args.hparams.is_text_vis_map_abl:
                    vis_text_loss = l2_norm_loss(est_text_features_list, visual_features.float())
                    total_loss_text_vis += vis_text_loss.item()
                    vis_text_loss.backward()
                    optimizer_text_vis.step()
                text_features = visual_features
            else:
                text_features = est_text_features_list

        total_loss, total_num = 0.0, 0
        optimizer.zero_grad()

        #Use text_features to get estimated object tokens
        text_features = text_features.float()
        text_features = F.normalize(text_features, dim=-1)
        out_features = set_model(text_features)

        estimated_tokens = out_features.reshape((out_features.shape[0],no_of_new_tokens,emb_dim))

        #Find target text_features
        with torch.no_grad():
            natural_prompt_labels = [natural_prompt[:-1] + local_label for local_label in (local_labels)]
            text = clip.tokenize(natural_prompt_labels).to(device)
            natural_prompt_embeddings = model.encode_text(text)
            natural_prompt_embeddings /= natural_prompt_embeddings.norm(dim=-1, keepdim=True)
            natural_prompt_embeddings = F.normalize(natural_prompt_embeddings, dim=-1)

        #Use estimated object tokens to get embeddings with text_asterix
        if args.hparams.is_learn_token:
            natural_prompt_asterix = [natural_prompt for local_label in (local_labels)]
            if (args.hparams.is_learn_prefix and is_image_input==False):
                for i, sentence in enumerate(natural_prompt_asterix):
                    coarse_label = coarse_labels[i]
                    sentence = sentence.replace("*", coarse_label + " *")
                    natural_prompt_asterix[i] = sentence

            elif (args.hparams.is_learn_prefix and args.hparams.is_learn_prefix_also_image):
                for i, sentence in enumerate(natural_prompt_asterix):
                    local_label = local_labels[i]
                    #sentence = sentence.replace("*", "* " + local_label)
                    sentence = sentence.replace("*", local_label + " *")
                    natural_prompt_asterix[i] = sentence
            text = clip.tokenize(natural_prompt_asterix).to(device)
            base_token = None
            natural_prompt_asterix_embeddings = model.encode_text_with_learnt_tokens(text, asterix_token, estimated_tokens, base_token)
        else:
            natural_prompt_asterix_embeddings = torch.mean(text_features,dim=1).float()
            natural_prompt_asterix_embeddings = mlp_model(natural_prompt_asterix_embeddings)
        natural_prompt_asterix_embeddings = F.normalize(natural_prompt_asterix_embeddings, dim=-1)

        #Optimize for embeddings with text_asterix to reconstruct text_features
        if args.hparams.loss_str == "contrastive":
            loss = criterion(natural_prompt_embeddings, natural_prompt_asterix_embeddings, args.hparams.contrastive_temp)
        if args.hparams.loss_str == "cosine":
            loss = cosine_loss(natural_prompt_embeddings, natural_prompt_asterix_embeddings)
        total_num += natural_prompt_embeddings.size(0)
        total_loss += loss.item() * natural_prompt_embeddings.size(0)

        if args.hparams.is_gt_object_loss:
            text = clip.tokenize(local_labels).to(device)
            gt_tokens = model.token_embedding(text)
            gt_object_loss = l2_norm_loss(gt_tokens[:,[1],:], estimated_tokens)

        if is_train:
            loss = args.hparams.coeff_cycle_loss*loss + args.hparams.coeff_gt_object_loss*gt_object_loss
            loss.backward()
            optimizer.step()
            if batch_num == 0:

                if is_augment_object:
                    print("augment object")
                else:
                    print("")
                print("ep %d train loss %.5f"%(epoch, total_loss / (total_num)))

                correct_num, pred = get_clip_text_acc(natural_prompt_embeddings, natural_prompt_asterix_embeddings)

                if is_augment_object:
                    wandb.log({'aug train/loss': total_loss / (total_num),
                       'aug train/accuracy': correct_num / pred,
                               'train gt_object_loss': gt_object_loss / (total_num)}, step=epoch, sync=False)
                    args.hparams.train_aug_accuracy = correct_num / pred

                else:
                    wandb.log({'train/loss': total_loss / (total_num),
                               'tsmap/loss': total_loss_text_vis / (total_num),
                       'train/accuracy': correct_num / pred,
                               'train gt_object_loss': gt_object_loss / (total_num)}, step=epoch, sync=False)

                    args.hparams.train_accuracy = correct_num / pred

        else:
            if batch_num == 0:
                print("")
                print("ep %d test  loss %.5f"%(epoch, total_loss / (total_num)))
                correct_num, pred = get_clip_text_acc(natural_prompt_embeddings, natural_prompt_asterix_embeddings)

                wandb.log({'test/loss': total_loss / (total_num),
                   'test/accuracy': correct_num / pred,
                           'train gt_object_loss': gt_object_loss / (total_num)}, step=epoch, sync=False)
                args.hparams.test_accuracy = correct_num / pred
    scheduler.step()

def main():

    parser = ArgumentParser()
    parser.add_arguments(HParams, dest="hparams")
    args, unknown = parser.parse_known_args()
    print("")
    print("args",args)

    wandb.init(project=args.hparams.project_name, config=args.hparams) #I can name the run here

    batch_size = args.hparams.batch_size
    deep_set_d_dim = args.hparams.deep_set_d_dim
    dropout = args.hparams.dropout
    set_size = args.hparams.set_size
    learning_rate = args.hparams.lr
    learning_rate_text_vis = args.hparams.lr_text_vis
    epochs = args.hparams.epochs
    is_image_input_train = args.hparams.is_image_input_train
    is_image_input_test = args.hparams.is_image_input_test
    is_augment_object = args.hparams.is_augment_object
    is_text_visual_map = args.hparams.is_text_visual_map

    #Deep set model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    #Add personalized text encoder method to CLIP
    funcType = type(model.encode_text)
    model.encode_text_with_learnt_tokens = funcType(encode_text_with_learnt_tokens, model)

    model.eval()

    #Deep set model
    set_model = deep_set(deep_set_d_dim, x_dim=emb_dim, out_dim = args.hparams.no_of_new_tokens*emb_dim, pool = args.hparams.pooling_type, dropout=dropout)
    set_model = set_model.to(device)

    mlp_model = MLP(dropout = dropout)
    mlp_model = mlp_model.to(device)

    #Text visual map model
    if args.hparams.is_text_vis_map_abl:
        text_vis_model = TextVisualMapAbl()
    else:
        text_vis_model = TextVisualMap()
    text_vis_model.to(device)

    #Load pretraining text object descriptors
    object_dict = np.load(args.hparams.object_dict_path, allow_pickle=True)
    object_dict = object_dict.tolist()
    key_vals = list(object_dict.keys())
    print("key_vals",key_vals[0])
    print("object_dict",object_dict[key_vals[0]])

    if args.hparams.sentence_wise_split:
        train_dataset = CustomTextDataset(key_vals, object_dict, set_size, sentence_wise_split=True, is_train=True, is_image_input=is_image_input_train)
        test_dataset = CustomTextDataset(key_vals, object_dict, set_size, sentence_wise_split=True, is_train=False, is_image_input=is_image_input_test)
        print("L_train",len(train_dataset))
        print("L_test",len(test_dataset))
    else:
        key_vals_train, key_vals_test = train_test_split(key_vals, test_size=args.hparams.sentence_wise_ratio)
        L_train = len(key_vals_train)
        L_test = len(key_vals_test)
        print("L_train",L_train)
        print("L_test",L_test)
        train_dataset = CustomTextDataset(key_vals_train, object_dict, set_size, is_image_input=is_image_input_train)
        test_dataset = CustomTextDataset(key_vals_test, object_dict, set_size, is_image_input=is_image_input_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    if is_augment_object:
        #Load object augmentation names and map
        aug_dataset_objects = np.load(args.hparams.text_obj_path, allow_pickle=True)
        aug_dataset_objects = aug_dataset_objects["arr_0"]
        text_aug_map = np.load(args.hparams.text_aug_map_path, allow_pickle=True)
        text_aug_map = text_aug_map["arr_0"]
        train_dataset_augment = CustomTextDataset(key_vals, object_dict, set_size, sentence_wise_split=True, is_train=True, is_text_aug = True,
                                                  aug_dataset_objects = aug_dataset_objects, text_aug_map = text_aug_map)
        train_dataloader_augment = DataLoader(train_dataset_augment, batch_size=batch_size, shuffle=True)

    if is_image_input_train or is_image_input_test:
        #Load image feautres
        data_img_features = np.load(args.hparams.visual_features_path, allow_pickle=True)
        data_img_features_dict = data_img_features["arr_0"].tolist()
    else:
        data_img_features_dict = None

    #Optimization
    optimizer = optim.Adam(set_model.parameters(), lr=learning_rate)
    optimizer_text_vis = optim.Adam(text_vis_model.parameters(), lr=learning_rate_text_vis)
    scheduler = StepLR(optimizer, step_size=200 , gamma=0.2)

    #Training
    for epoch in range(epochs):
        mlp_model.train()
        text_vis_model.train()
        model.eval()
        print("train_dataloader")

        # Regular epoch
        run_epoch(args, epoch, train_dataloader, optimizer, optimizer_text_vis, scheduler, model, mlp_model, set_model, text_vis_model,
                  device, is_train = True, is_image_input = is_image_input_train, img_features_dict = data_img_features_dict)

        # Text augmentation epoch
        if is_augment_object and is_text_visual_map:
            print("train_dataloader_augment")
            run_epoch(args, epoch, train_dataloader_augment, optimizer, optimizer_text_vis, scheduler, model, mlp_model, set_model, text_vis_model,
                      device, is_train = True, is_image_input = False, img_features_dict = data_img_features_dict,
                      is_augment_object = True)

        #Test epoch
        print("test_dataloader")
        with torch.no_grad():
            mlp_model.eval()
            model.eval()
            text_vis_model.eval()
            run_epoch(args, epoch, test_dataloader, optimizer, optimizer_text_vis, scheduler, model, mlp_model, set_model, text_vis_model,
                      device, is_train = False, is_image_input = is_image_input_test,  img_features_dict = data_img_features_dict)

    ### Save models
    torch.save(set_model.state_dict(), f"{wandb.run.dir}/set_model.pt")
    torch.save(text_vis_model.state_dict(), f"{wandb.run.dir}/text_vis_model.pt")

    if args.hparams.is_save_models:
        save_path = "../sandbox/checkpoints/"
        os.makedirs(save_path, exist_ok=True)
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")

        model_name = "deep_set_model_%s"%args.hparams.save_model_name
        save_trained_model(save_path, model_name, args, set_model, timestr)

        txt_vis_model_name = "txt_vis_model_%s"%args.hparams.save_model_name
        save_trained_model(save_path, txt_vis_model_name, args, text_vis_model, timestr)
        return timestr



if __name__ == '__main__':
    main()


