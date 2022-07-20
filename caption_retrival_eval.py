# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file 
# located at the root directory.

import torch
import clip
import numpy as np
import sys
from torch.utils.data import DataLoader
sys.path.append("../")
sys.path.append("clip_language_set")
from utils.nv import ClipEvalutionEncodeData, TextVisualMap
from utils.deep_set_clf import D as deep_set
import torch.nn.functional as F
from dataclasses import dataclass
from simple_parsing import ArgumentParser
import time
import os
from get_f_theta import HParams
import faiss
from fsl_eval import infer_tokens_from_f_theta, optimize_token, parse_coarse_grained_strings
import pandas as pd
import wandb
from utils.non_nv import encode_text_with_learnt_tokens

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

emb_dim: int = 512
num_tokens = 77
device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class EvalParams:
    """Set of options for the evaluation of a Model."""
    model_name: str =  "2022_07_18-04_00_07"
    model_path: str = "../sandbox/checkpoints/"
    set_captions_path: str = "captions.csv"
    data_name: str = ""
    gt_text_path: str = ""
    text_prompt: str = ""  #general retrival caption; no "*"
    set_size_override: int = -1 #override deepset input size
    batch_size: int = 128
    random_seed: int = 1
    no_fsl: bool = False #no few shot learning, use coarse grain object only
    is_class_name_folders: bool = False

    #Second stage token optimization parameters
    is_optimize_token: bool = False #use second stage token optimization
    latent_lr: float = 0.01
    latent_ep: int = 10
    token_optimize_mode: int = 0  # 0 - positive only, 1 - coarse negative, 2 - positive negative
    contrastive_temp: float = 0.25
    is_coarse_grained_negative_per_class: bool = False #use per class coarse grained concept
    per_dataset_coarse_grained_phrase: str = "item"

    #Token file parameters
    out_tokens_path: str = "data/out_tokens/"
    is_save: bool = False
    is_load: bool = False
    tokens_path: str = "data/out_tokens/2022_02_06-21_43_39"

    #evaluation data
    data_path: str = ""
    coarse_gained_class_names_path: str = "" #path to coarse grained class names
    captions_path: str = ""
    is_short_captions: bool = True #when there are two caption lengths
    project_name: str = "RetrivalEval"
    is_token_as_suffix: bool = False #use learnt token togather with coarse grained descriptor
    is_train_loader_no_reps: bool = True #randomized train images w.o. repetitions

    use_base_code: bool = False # Use token from coarse grained (for stage 2 optimization)
    modified_object_desc: str = "" #personalize object descriptor for


    is_eval_transform: bool = False # use augmentation for evaluation dataset, deprecated

    is_constant_caption_abl: bool = False #use constant caption
    constant_caption_abl_test: str = "A photo of a *" #which constant caption
    is_score_means: bool = False #Score according to mean embeddings (comparison)
    is_caption_visual_mean: bool = False #mean embeddings for images+text, or images only
    is_random_token_init: bool = False #random token as stage 1 ablation

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_coarse_gained_class_names(args, gt_text_label):
    gt_text_label_str = [""]*len(gt_text_label)


    if args.evalparams.is_class_name_folders:
        for i, label in enumerate(gt_text_label):
            gt_text_label_str[i] = label.split("_")[0]
    else:
        df_all_set = pd.read_csv(args.evalparams.coarse_gained_class_names_path)
        unique_pair_ids = df_all_set["unique_pair_ids"].to_numpy()
        pair_id_categories = df_all_set["pair_id_categories"].to_numpy()
        for i, label in enumerate(gt_text_label):
            label_ind_in_df = np.where(int(label) == unique_pair_ids)[0][0]
            gt_text_label_str[i] = pair_id_categories[int(label_ind_in_df)]

    return gt_text_label_str

def get_caption_image_score(args, model, index, captions_i, object_token, index_len, gt_text_label):
    captions_i_modified = prepare_caption_for_eval(args, captions_i, gt_text_label)
    if args.evalparams.no_fsl: #use coarse grain string, no learning
        text = clip.tokenize(captions_i_modified.replace("*", gt_text_label)).to(device)
        emb_for_retrive_phrase = model.encode_text(text)
    else: #use token
        text = clip.tokenize(captions_i_modified).to(device)
        asterix_token = clip.tokenize(["*"]).to(device)[0][1]
        emb_for_retrive_phrase = model.encode_text_with_learnt_tokens(text, asterix_token, object_token.unsqueeze(0).unsqueeze(0), is_emb=False)

    emb_for_retrive_phrase = F.normalize(emb_for_retrive_phrase, dim=-1)
    emb_for_retrive_phrase = emb_for_retrive_phrase.detach().cpu().numpy()

    #Use input index of image embeddings to retrive images
    D, I = index.search(np.ascontiguousarray(emb_for_retrive_phrase.astype('float32')), index_len)
    return D, I

def get_avg_image_score(args, model, index, captions_i, index_len, gt_text_label, object_features_i):
    #Score with mean of fsl (eval) train set
    object_features_i = F.normalize(object_features_i, dim=-1)
    object_features_i = torch.mean(object_features_i, axis=0, keepdim = True)
    object_features_i = F.normalize(object_features_i, dim=-1)
    #Find embedding using coarse grain string, no learning
    captions_i_modified = prepare_caption_for_eval(args, captions_i, gt_text_label)
    text = clip.tokenize(captions_i_modified.replace("*", gt_text_label)).to(device)
    caption_emb = model.encode_text(text)
    caption_emb = F.normalize(caption_emb, dim=-1)
    if args.evalparams.is_caption_visual_mean:
        emb_for_retrive_phrase = caption_emb + object_features_i
    else:
        emb_for_retrive_phrase = object_features_i

    emb_for_retrive_phrase = F.normalize(emb_for_retrive_phrase, dim=-1)
    emb_for_retrive_phrase = emb_for_retrive_phrase.detach().cpu().numpy()
    D, I = index.search(np.ascontiguousarray(emb_for_retrive_phrase.astype('float32')), index_len)

    return D, I

def prepare_caption_for_eval(args, captions_i, gt_text_label):
    #Modify retrival captions context (coarse grained next to learnable token, prompts)
    if args.evalparams.is_constant_caption_abl: #constant caption
        if args.evalparams.is_token_as_suffix: #with coarse grained
            return args.evalparams.text_prompt + args.evalparams.constant_caption_abl_test.replace("*", "%s * "%(gt_text_label))
        else: #only constant caption with prompt
            return args.evalparams.text_prompt + args.evalparams.constant_caption_abl_test

    if args.evalparams.is_class_name_folders: #in is_class_name_folders, "*" is labelled in the amt captions
        if args.evalparams.is_token_as_suffix: #with coarse grained
            captions_i_modified = captions_i.replace("*", " %s * "%gt_text_label)
        else:
            captions_i_modified = captions_i.replace("*", " * ")
    else:
        captions_i = args.evalparams.text_prompt + captions_i
        if args.evalparams.is_token_as_suffix: #with coarse grained
                captions_i_modified = captions_i.replace("person", "%s %s * " % (args.evalparams.modified_object_desc, gt_text_label))
        else:
            captions_i_modified = captions_i.replace("person", "%s * " % (args.evalparams.modified_object_desc))

    captions_i_modified = args.evalparams.text_prompt + captions_i_modified
    return captions_i_modified


def calc_metrics_from_image_scores(args, I_inds, gt_inds):
    #Produce results based on retrival image ranking and ground truth

    places = [] # list of "n"s where each image was retrival in the n'th place
    for i, I in enumerate(I_inds):
        place = np.where(gt_inds[i] == I)[1][0] + 1
        places.append(place)

    print("places",np.unique(places))
    print("places",places)
    print("places", len(places))
    print("means",np.mean(np.array(places)))
    mrr = np.mean(1 / np.array(places))
    print("mrr",mrr)

    for k in range(1, len(places) + 1):
        correct_recall = [place <= k for place in places]
        recall_at_k = np.mean(correct_recall)
        wandb.log({'%s/PrecisionAtK/%d' % (args.evalparams.test_data_path, k): recall_at_k/k,
                   '%s/RecallAtK/%d' % (args.evalparams.test_data_path, k): recall_at_k}, sync=False)

    return places, mrr


def eval_tokens(args, model, test_dataloader, test_data_path, gt_text_label, object_tokens, object_features = None):
    image_features_list = []
    label_list = []

    #Extract fsl (eval) test set features
    for batch_num, sample in enumerate(test_dataloader):
        images, label = sample
        image_features = model.encode_image(images.cuda())
        image_features = F.normalize(image_features, dim=-1)
        image_features_list.append(image_features.detach().cpu().numpy())
        label_list.append(label)
    image_features_list = np.concatenate(image_features_list)

    #built kNN index for image retrival
    index = faiss.IndexFlatL2(image_features_list.shape[1])
    index.add(np.ascontiguousarray(image_features_list.astype('float32')))

    #get dir names
    test_file_names = os.listdir(os.path.join(test_data_path, "test"))
    test_file_names.sort()
    print("test_file_names",test_file_names)

    #get class ids
    #captions_df = pd.read_csv(os.path.join(test_data_path, "captions.csv"))
    captions_df = pd.read_csv(args.evalparams.set_captions_path)

    image_name_list = captions_df["image_name"].to_numpy()
    if args.evalparams.is_class_name_folders:
        pair_id_list = captions_df["class_id"].to_numpy()
    else:
        pair_id_list = captions_df["pair_id"].to_numpy()

    #get retrival captions
    if args.evalparams.is_short_captions:
        short_captions_df = pd.read_csv(args.evalparams.captions_path)
        captions_list = [""]*len(image_name_list)
        for i,image_name in enumerate(image_name_list):
            row = short_captions_df.loc[short_captions_df['image_name'] == image_name]
            captions_list[i] = row["caption"].to_numpy()[0]
        captions_list = np.asarray(captions_list)
    else:
        captions_list = captions_df["caption"].to_numpy()

    gt_inds = []
    I_inds = []
    gt_text_corase = get_coarse_gained_class_names(args, gt_text_label)

    for i, image_name_i in enumerate(image_name_list): #For each test image
        pair_id_i = pair_id_list[i]
        captions_i = captions_list[i]

        #Find image index
        if args.evalparams.is_class_name_folders:
            image_name_i = "_".join(image_name_i.split("_")[:-1]) + ".jpg"
            ind_gt = (np.where(np.asarray(test_file_names) == image_name_i.split("/")[-1]))
        else:
            ind_gt = (np.where(np.asarray(test_file_names) == image_name_i.split("/")[-1]))

        #Find class index
        gt_text_label = gt_text_label.astype('str')
        pair_id_ind = np.where(gt_text_label == str(pair_id_i))[0][0]

        #Eval method
        if args.evalparams.is_score_means:
            D, I = get_avg_image_score(args, model, index, captions_i, len(image_features_list),
                                           gt_text_corase[pair_id_ind], object_features[pair_id_ind])
        else:
            object_token = object_tokens[pair_id_ind]
            D, I = get_caption_image_score(args, model, index, captions_i, object_token, len(image_features_list),
                                            gt_text_corase[pair_id_ind])
        gt_inds.append(ind_gt)
        I_inds.append(I)
    gt_inds = np.asarray(gt_inds)
    I_inds = np.asarray(I_inds)[:,0,:]

    #Calculate scores
    places, mrr = calc_metrics_from_image_scores(args, I_inds, gt_inds)

    wandb.log({'%s/places'%(args.evalparams.test_data_path): places,
               '%s/mrr'%(args.evalparams.test_data_path): mrr}, sync=False)


def tokenize_coarse_grained_strings(args, model, gt_text_label): #get tokens corresponding to coarse grained strings
    gt_text_label_coarse = np.asarray(gt_text_label)
    gt_text_coarse = get_coarse_gained_class_names(args, gt_text_label_coarse)
    gt_text_coarse = [gt_text.split(" ")[-1] for gt_text in gt_text_coarse]
    text = clip.tokenize(gt_text_coarse).to(device)
    x = model.token_embedding(text).type(model.dtype)
    x = x[:,1,:]
    return x

def main(args_path = None, is_wandb_init = True):
    parser = ArgumentParser()
    parser.add_arguments(EvalParams, dest="evalparams")
    args, unknown = parser.parse_known_args()
    print("args",args)

    if args_path is not None:
        args.evalparams.model_name = args_path

    if is_wandb_init:
        wandb.init(project=args.evalparams.project_name, config=args.evalparams) #run name goes here

    args.evalparams.train_data_path = os.path.join(args.evalparams.data_path, "codes_infer")
    args.evalparams.test_data_path = os.path.join(args.evalparams.data_path, "eval")

    train_args = np.load(os.path.join(args.evalparams.model_path, "args_" + args.evalparams.model_name + ".npy"), allow_pickle = True)
    train_args = train_args.tolist()

    if args.evalparams.set_size_override !=-1:
        train_args.set_size = args.evalparams.set_size_override

    ######### Load tokenization model
    model, preprocess = clip.load("ViT-B/32", device=device)
    data_augment_preprocess = Compose([
        Resize(300, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    #Inser CLIP text encoding with learnt token methods
    funcType = type(model.encode_text)
    model.encode_text_with_learnt_tokens = funcType(encode_text_with_learnt_tokens, model)

    if train_args.is_learn_token:
        #Load deep set network (f_theta)
        set_model = deep_set(train_args.deep_set_d_dim, x_dim=emb_dim, out_dim = train_args.no_of_new_tokens*emb_dim, pool = 'mean', dropout=train_args.dropout)
        model_name = "deep_set_model_%s_%s"%(train_args.save_model_name, args.evalparams.model_name)

        set_model = set_model.to(device)
        set_model.load_state_dict(torch.load(os.path.join(args.evalparams.model_path,model_name)))
        set_model.eval()
        mlp_model = None

        txt_vis_model_name = "txt_vis_model_%s_%s"%(train_args.save_model_name, args.evalparams.model_name)
        text_vis_model = TextVisualMap()
        text_vis_model.to(device)
        text_vis_model.load_state_dict(torch.load(os.path.join(args.evalparams.model_path,txt_vis_model_name)))



    ######### Get tokens
    if args.evalparams.is_load:
        #load already inferred tokens
        object_tokens = torch.load(args.evalparams.tokens_path)
        print("object_tokens",object_tokens.shape)
        print("args.evalparams.tokens_path",args.evalparams.tokens_path)
        gt_text_label = np.load("/".join(args.evalparams.tokens_path.split("/")[:-1]) + "/gt_text_" + args.evalparams.tokens_path.split("/")[-1] + ".npy")
        gt_text_label = np.asarray(gt_text_label)
    else:
        #Get train data loader
        train_dataset = ClipEvalutionEncodeData(args.evalparams.train_data_path, transform = preprocess, is_train = True, k = train_args.set_size, is_fold_prefix = False,
                                                   is_train_no_reps = args.evalparams.is_train_loader_no_reps, seed = args.evalparams.random_seed)
        train_dataloader = DataLoader(train_dataset, batch_size=args.evalparams.batch_size, shuffle=False)

        with torch.no_grad():
            gt_text_label, object_tokens, image_features_list, _ = infer_tokens_from_f_theta(args, train_args, model, set_model, train_dataloader,
                                                       text_vis_model, gt_text_coarse_func = get_coarse_gained_class_names)


            if args.evalparams.use_base_code:  # Use token from coarse grained (for stage 2 optimization)
                print("object_tokens",object_tokens)
                object_tokens = tokenize_coarse_grained_strings(args, model, gt_text_label)
                object_tokens = object_tokens.double()
                print("object_tokens",object_tokens)

            object_tokens = object_tokens.reshape((object_tokens.shape[0],train_args.no_of_new_tokens,emb_dim))[:,0]

        gt_text_label = np.asarray(gt_text_label)
        gt_text_coarse = get_coarse_gained_class_names(args, gt_text_label)

        if args.evalparams.is_random_token_init:
            object_tokens = torch.rand(object_tokens.shape)
            object_tokens = F.normalize(object_tokens, dim=-1).cuda()

        print("object_tokens1",object_tokens)

        if args.evalparams.is_optimize_token:

            if args.evalparams.is_eval_transform: # use augmentation for evaluation dataset, deprecated
                train_dataset = ClipEvalutionEncodeData(args.evalparams.train_data_path, transform=data_augment_preprocess,
                                                           is_train=True, k=train_args.set_size, is_fold_prefix=False,
                                                           is_train_no_reps=args.evalparams.is_train_loader_no_reps,
                                                           seed=args.evalparams.random_seed)
                train_dataloader = DataLoader(train_dataset, batch_size=args.evalparams.batch_size, shuffle=False)
            gt_text_label_parsed = parse_coarse_grained_strings(args, gt_text_coarse)
            object_tokens = optimize_token(args, object_tokens, model, train_dataloader, gt_text_label_parsed, text_vis_model)
            print("object_tokens2", object_tokens)


        if args.evalparams.is_save:
            timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
            args_out_path = os.path.join(args.evalparams.out_tokens_path, "args_" + timestr)
            gt_text_out_path = os.path.join(args.evalparams.out_tokens_path, "gt_text_" + timestr)
            tokens_out_path = os.path.join(args.evalparams.out_tokens_path, "" + timestr)
            os.makedirs(args.evalparams.out_tokens_path, exist_ok=True)
            np.save(args_out_path,args)
            np.save(gt_text_out_path,gt_text_label)
            torch.save(object_tokens, tokens_out_path)


    ############ Evaluation

    test_dataset = ClipEvalutionEncodeData(args.evalparams.test_data_path, transform = preprocess, is_train = False, is_fold_prefix = False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.evalparams.batch_size, shuffle=False)
    print("test_dataloader",len(test_dataloader))
    torch.save(object_tokens, f"{wandb.run.dir}/object_tokens.pt")
    eval_tokens(args, model, test_dataloader, args.evalparams.test_data_path, gt_text_label, object_tokens,
                      object_features = image_features_list)

if __name__ == '__main__':
    main()




