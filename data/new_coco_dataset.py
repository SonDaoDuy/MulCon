import os
import xml.etree.ElementTree as ET
import glob
import pickle
import json

import numpy as np
import torch
import scipy.io as sio
from PIL import Image
import torchvision.transforms as transforms

from data.base_dataset import BaseDataset, get_transform
from util.coco_preprocess import MultiScaleCrop, Warp, Cutout
from util.autoaugmentation import ImageNetPolicy
from randaugment import RandAugment

class NewCOCODataset(BaseDataset):
    """docstring for DyceDataset"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--image_normalization_mean', type=tuple, default=(0.485, 0.456, 0.406), help='coco image_normalization_mean')
        parser.add_argument('--image_normalization_std', type=tuple, default=(0.229, 0.224, 0.225), help='coco image_normalization_std')
        parser.add_argument('--randaug', type=bool, default=True, help='using randaug')
        parser.add_argument('--cutout', type=bool, default=False, help='using cutout')
        parser.add_argument('--n_holes', type=int, default=1, help='no of holes')
        parser.add_argument('--length', type=int, default=244, help='length of holes')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self._root = self.opt.dataroot
        self.phase = self.opt.data_type
        #load data 
        img_infos, cat2idx = self.load_annotations()
        if self.opt.data_type =='trainsplit':
            self.phase = 'train'
        self.cat2idx = cat2idx
        self.img_infos = img_infos[:min(opt.max_dataset_size, len(img_infos))]

        #transform
        if self.opt.isTrain:
            if opt.randaug:
                normalize = transforms.Normalize(mean=self.opt.image_normalization_mean,
                                                std=self.opt.image_normalization_std)
                self._img_transform  = transforms.Compose([
                    transforms.RandomResizedCrop(size=opt.load_size, scale=(0.875, 1.)),
                    transforms.RandomHorizontalFlip(),
                    RandAugment(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                normalize = transforms.Normalize(mean=self.opt.image_normalization_mean,
                                                std=self.opt.image_normalization_std)
                self._img_transform  = transforms.Compose([
                    MultiScaleCrop(opt.load_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

            if opt.cutout:
                self._img_transform.transforms.append(Cutout(opt.n_holes, opt.length))

        else:
            normalize = transforms.Normalize(mean=self.opt.image_normalization_mean,
                                             std=self.opt.image_normalization_std)
            self._img_transform  = transforms.Compose([
                Warp(self.opt.load_size),
                transforms.ToTensor(),
                normalize,
            ])

    def __getitem__(self, index):
        #feature processing
        item = self.img_infos[index]
        filename = item['file_name']
        
        #image processing
        img = Image.open(os.path.join(self._root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')

        if self._img_transform is not None:
            img_tensor = self._img_transform(img)

        #label processing
        labels = sorted(item['labels'])
        # print(item['labels'])
        gt_label = np.zeros(self.opt.num_classes, np.float32)
        gt_label[labels] = 1

        ann = dict(
            img=img_tensor,
            gt_label=gt_label,
            img_id=filename
            )
        return ann

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self):
        list_path = os.path.join(self._root, 'data', '{}_anno.json'.format(self.phase))
        img_list = json.load(open(list_path, 'r'))
        cat2idx = json.load(open(os.path.join(self._root, 'data', 'category.json'), 'r'))
        return img_list, cat2idx
