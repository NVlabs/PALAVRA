# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file 
# located at the root directory.

import os
import numpy as np
from torch.utils.data import Dataset
from utils.non_nv import temporary_random_numpy_seed
import torchvision
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CosineSimilarity, MSELoss


natural_prompt_multi = [
    "This is a photo of a *",
    "This photo contains a *",
    "A photo of a *",
    "This is an illustrations of a *",
    "This illustrations contains a *",
    "An illustrations of a *",
    "This is a sketch of a *",
    "This sketch contains a *",
    "A sketch of a *",
    "This is a diagram of a *",
    "This diagram contains a *",
    "A diagram of a *",
    "A *",
    "We see a *",
    "*",
    "We see an * in this photo ",
    "We see an * in this image ",
    "We see an * in this illustration ",
    "We see a * photo ",
    "We see a * image ",
    "We see a * illustration ",
    " * photo ",
    " * image ",
    " * illustration ",
    ]



class ClipEvalutionEncodeData(Dataset):

    def __init__(self, image_folder_path, transform, is_train, k=5, ds_factor = 1, is_train_no_reps = False, is_fold_prefix = False, seed = 0):
        self.k = k
        self.transform = transform
        self.is_train = is_train
        if is_train:
            if is_fold_prefix:
                image_folder_path = os.path.join(image_folder_path, "train")
        else:
            if is_fold_prefix:
                image_folder_path = os.path.join(image_folder_path, "test")
        imagefoler_data = torchvision.datasets.ImageFolder(image_folder_path, transform = transform)
        self.labels = imagefoler_data.classes
        self.imagefoler_data = imagefoler_data

        label_idx_list = []
        for k, label in enumerate(self.labels):
            label_idx = [i for i in range(len(imagefoler_data)) if imagefoler_data.imgs[i][1] == imagefoler_data.class_to_idx[label]]
            temp_seed = seed + k
            with temporary_random_numpy_seed(temp_seed):
                if is_train:
                    if is_train_no_reps:
                        label_idx = np.random.choice(label_idx, self.k, replace=False)
                    else:
                        label_idx = np.random.choice(label_idx, self.k, replace=True)
            label_idx_list.append(label_idx)


        self.label_idx_list = label_idx_list

    def __len__(self):
        if self.is_train:
            return len(self.label_idx_list)
        else:
            return len(self.imagefoler_data)

    def __getitem__(self, idx):
        if self.is_train:
            label = self.labels[idx]
            images_inds = self.label_idx_list[idx]

            cls_images_list = []
            for images_ind in images_inds:
                cls_image = self.imagefoler_data[images_ind]
                cls_images_list.append(cls_image[0])
            cls_images_list = torch.stack(cls_images_list)
            return cls_images_list, label

        else:

            return self.imagefoler_data.__getitem__(idx)

class TextVisualMap(nn.Module):

  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(512, 512),
    )
  def forward(self, x):
    return self.layers(x) + x

class TextVisualMapAbl(nn.Module):

  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(512, 512),
    )
  def forward(self, x):
    return x


##


class LinearEmb(nn.Module):
  #No license needed, can be moved to nv.py?
  def __init__(self, no_of_tokens, emb_dim):
    super().__init__()
    self.layers = nn.Sequential(
        torch.nn.Linear(no_of_tokens,emb_dim)
    )

  def forward(self, x):
    return self.layers(x)

class MLP(nn.Module):
  #No license needed, can be moved to nv.py?
  def __init__(self, dropout = 0.5):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(p=dropout),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Dropout(p=dropout),
      nn.Linear(512, 512),
    )

  def forward(self, x):
    return self.layers(x) + x

def cosine_loss(out_1, out_2):
    # No license needed, can be moved to nv.py?
    cos = CosineSimilarity(dim=1, eps=1e-6)
    loss = -cos(out_1, out_2).mean()

    return loss

def l2_norm_loss(out_1, out_2):
    # No license needed, can be moved to nv.py?
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    loss = MSELoss()
    output = loss(out_1,out_2)
    return output


class CustomTextDataset(Dataset):
    # No license needed, can be moved to nv.py?
    # Is there usage here at all?
    def __init__(self, labels, sentences_dict, k=50, sentence_wise_split = False, is_train = False, test_freq = 5, is_image_input = False, is_text_aug = False,
                 aug_dataset_objects = None, text_aug_map = None):
        self.labels = labels
        self.sentences_dict = sentences_dict
        self.k = k
        self.is_image_input = is_image_input
        self.is_text_aug = is_text_aug
        self.aug_dataset_objects = aug_dataset_objects
        self.text_aug_map = text_aug_map

        if sentence_wise_split: #if sentence wise split
            for label in self.labels:
                img_id = self.sentences_dict[label][0]
                img_id_unique = np.unique(img_id)
                img_id_unique_test = img_id_unique[::test_freq] #in sentence wise split, use every test_freq'th sample for test
                img_id_unique_train = [i for i in img_id_unique if i not in img_id_unique_test]
                if self.is_image_input:
                    #split images by img id
                    label_sentences_ind_test = [np.where(i == img_id)[0][0] for i in img_id_unique_test]
                    label_sentences_ind_train = [np.where(i == img_id)[0][0] for i in img_id_unique_train]
                else:
                    label_sentences_ind_test = [np.where(i == img_id) for i in img_id_unique_test]
                    label_sentences_ind_train = [np.where(i == img_id) for i in img_id_unique_train]
                    label_sentences_ind_test = np.hstack(label_sentences_ind_test)[0]
                    label_sentences_ind_train = np.hstack(label_sentences_ind_train)[0]

                if is_train:
                    label_sentences_ind_chosen = label_sentences_ind_train
                else:
                    label_sentences_ind_chosen = label_sentences_ind_test

                self.sentences_dict[label][0] = self.sentences_dict[label][0][label_sentences_ind_chosen]
                self.sentences_dict[label][1] = self.sentences_dict[label][1][label_sentences_ind_chosen]
                self.sentences_dict[label][2] = self.sentences_dict[label][2][label_sentences_ind_chosen]

    def __len__(self):
        return len(self.labels)

    def augment_text(self, chosen_sentences_list, chosen_sentences_asterix_list, label, idx):
        #Note that we always return both chosen_sentences_list and chosen_sentences_asterix_list
        inds_for_aug = np.where(self.text_aug_map == idx)[0]
        if len(inds_for_aug) == 0: #If no augmentation avilable, leave as is
            return chosen_sentences_list, chosen_sentences_asterix_list, label, label
        else: #Replace the sentence with a random object according to text_aug_map
            chosen_aug_inds = np.random.choice(inds_for_aug, 1)[0]
            chosen_aug_obj = self.aug_dataset_objects[chosen_aug_inds].lower()
            chosen_sentences_list = [chosen_sentence_asterix.replace("*", chosen_aug_obj) for chosen_sentence_asterix in chosen_sentences_asterix_list]
            return chosen_sentences_list, chosen_sentences_asterix_list, chosen_aug_obj, label

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_id = self.sentences_dict[label][0]
        sentences = self.sentences_dict[label][1]
        sentences_asterix = self.sentences_dict[label][2]
        chosen_inds = np.random.choice(np.arange(len(sentences)), self.k)
        chosen_img_id = img_id[chosen_inds]
        chosen_sentences = sentences[chosen_inds]
        chosen_sentences_asterix = sentences_asterix[chosen_inds]

        if self.is_image_input:
            return list(chosen_sentences), list(chosen_sentences_asterix), label, list(chosen_img_id)
        elif self.is_text_aug:
            return self.augment_text(list(chosen_sentences), list(chosen_sentences_asterix), label, idx)
        else:
            return list(chosen_sentences), list(chosen_sentences_asterix), label


def contrastive_loss(v1, v2, temperature = 0.25):
    v1 = F.normalize(v1, dim=1)
    v2 = F.normalize(v2, dim=1)

    numerator = torch.exp(torch.diag(torch.inner(v1,v2))/temperature)
    numerator = torch.cat((numerator,numerator), 0)
    joint_vector = torch.cat((v1,v2), 0)
    pairs_product = torch.exp(torch.mm(joint_vector,joint_vector.t()) / temperature)
    denominator = torch.sum(pairs_product - pairs_product*torch.eye(joint_vector.shape[0]).cuda(), 0)

    loss = -torch.mean(torch.log(numerator/denominator))

    return loss

