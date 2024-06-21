#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import torch

import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset


def c2chw(x):
    return x.unsqueeze(1).unsqueeze(2)


def inverse_list(list):
    """
    List to dict: index -> element
    """
    dict = {}

    for idx, x in enumerate(list):
        dict[x] = idx

    return dict


class KolektorSDD2(Dataset):
    """"
    Kolektor Surface-Defect 2 dataset

        Args:
            dataroot (string): path to the root directory of the dataset
            split    (string): data split ['train', 'test']
            scale    (string): input image scale
            debug    (bool)  : debug mode
    """

    labels = ['ok', 'defect']
    # scales = {'1408x512': 1., '704x256': .5, 'half': .5}
    # output_sizes = {'1408x512': (1408, 512),
    #                 '704x256': (704, 256),
    #                 'half': (704, 256)}

    # ImageNet.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self,
                 dataroot='/path/to/dataset/'
                          'KolektorSDD2',
                 split='train', negative_only=False,
                 add_augmented=False, num_augmented=0, zero_shot=False):
        super(KolektorSDD2, self).__init__()

        self.fold = None
        self.dataroot = dataroot

        self.split_path = None
        self.split = 'train' if 'val' == split else split

        # self.scale = scale
        # self.fxy = self.scales[scale]
        # self.output_size = self.output_sizes[scale]
        self.output_size = (704, 256)
        self.negative_only = negative_only
        self.add_augmented = add_augmented
        if self.add_augmented:
            assert self.split == 'train', 'Augmented images are only for the training set!'
        self.num_augmented = num_augmented
        self.zero_shot = zero_shot
        if self.zero_shot:
            assert self.add_augmented, 'Zero-shot learning requires augmented images!'
            assert self.split == 'train', 'Zero-shot learning is only for the training set!'

        self.class_to_idx = inverse_list(self.labels)
        self.classes = self.labels
        self.transform = KolektorSDD2.get_transform(output_size=self.output_size)
        self.normalize = T.Normalize(KolektorSDD2.mean, KolektorSDD2.std)

        self.load_imgs()
            # torch.save((self.samples, self.masks, self.product_ids), image_cache_path)
        if negative_only:
            m = self.masks.sum(-1).sum(-1) == 0
            self.samples = self.samples[m]
            self.masks = self.masks[m]
            self.product_ids = [pid for flag, pid in zip(m, self.product_ids)
                                    if flag]


    def load_imgs(self):
        # Please remove this duplicated files in the official dataset:
        #   -- 10301_GT (copy).png
        #   -- 10301 (copy).png
        if self.num_augmented > 0:
            augmented_imgs_path = os.path.join(self.dataroot, f'augmented_{self.num_augmented}', 'imgs')
            augmented_masks_path = os.path.join(self.dataroot, f'augmented_{self.num_augmented}', 'masks')
        else:
            augmented_imgs_path = os.path.join(self.dataroot, f'augmented', 'imgs')
            augmented_masks_path = os.path.join(self.dataroot, f'augmented', 'masks')

        if self.split == 'test':
            N = 1004
        elif self.split == 'train' and self.zero_shot:
            # only augmented positives and original negatives
            N = 2085 # number of original negatives
        else:
            # all original data + augmented
            N = 2331 # number of original negatives and positives
        
        if self.add_augmented:
            N += len(os.listdir(augmented_imgs_path))
            if self.num_augmented > 0:
                assert len(os.listdir(augmented_imgs_path)) == self.num_augmented, f'Number of augmented images requested ({self.num_augmented}) does not match with number found ({len(os.listdir(augmented_imgs_path))})!'

        self.samples = torch.Tensor(N, 3, *self.output_size).zero_()
        self.masks = torch.LongTensor(N, *self.output_size).zero_()
        self.product_ids = []

        cnt = 0
        path = "%s/%s/" % (self.dataroot, self.split)
        image_list = [f for f in os.listdir(path)
                      if re.search(r'[0-9]+\.png$', f)]
        assert 0 < len(image_list), self.dataroot

        for img_name in image_list:
            product_id = img_name[:-4]
            img = self.transform(Image.open(path + img_name))
            lab = self.transform(
                Image.open(path + product_id + '_GT.png').convert('L'))
            if self.zero_shot:
                # check that the mask is negative
                if lab.sum() == 0:
                    self.samples[cnt] = img
                    self.masks[cnt] = lab
                    self.product_ids.append(product_id)
                    cnt += 1
            else:
                # default
                self.samples[cnt] = img
                self.masks[cnt] = lab
                self.product_ids.append(product_id)
                cnt += 1

        # Add the augmented images.
        if self.add_augmented:
            if 'train' == self.split:
                image_list = os.listdir(augmented_imgs_path)
                
                for img_name in image_list:
                    product_id = img_name[:-4]
                    img = self.transform(Image.open(os.path.join(augmented_imgs_path, img_name)))
                    lab = self.transform(
                        Image.open(os.path.join(augmented_masks_path, img_name)).convert('L'))
                    self.samples[cnt] = img
                    self.masks[cnt] = lab
                    self.product_ids.append(product_id)
                    cnt += 1

        assert N == cnt, '{} should be {}!'.format(cnt, N)


    def __getitem__(self, index):
        x = self.samples[index]
        a = self.masks[index] > 0
        if self.normalize is not None:
            x = self.normalize(x)

        if 0 == a.sum():
            y = self.class_to_idx['ok']
        else:
            y = self.class_to_idx['defect']

        return x, y, a, 0


    def __len__(self):
        return self.samples.size(0)


    @staticmethod
    def get_transform(output_size=(704, 256)):
        transform = [
            T.Resize(output_size),
            T.ToTensor()
        ]
        transform = T.Compose(transform)
        return transform


    @staticmethod
    def denorm(x):
        return x * c2chw(torch.Tensor(KolektorSDD2.std)) + c2chw(torch.Tensor(KolektorSDD2.mean))
