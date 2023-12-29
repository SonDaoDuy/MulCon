"""This module contains simple helper functions """
from __future__ import print_function

import os
import pickle
import random

import numpy as np
import torch
from PIL import Image


def clean_mask(mask, clean_mask_thr):
    # mask = torch.Tensor(np.array(mask))
    if torch.sum(mask) == 0:
        return mask.float()
    return torch.ge(mask, clean_mask_thr).float()


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        image_numpy = (np.transpose(image_numpy,
                                    (1, 2, 0)) * 0.5 + 0.5) * 255.0  # post-processing: tranpose and scaling
        # round((img + 1) * 255 / 2)
        # image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def convert_to_img(tensor, img_mean):
    if not isinstance(tensor, np.ndarray):
        """do sth"""
        if isinstance(tensor, torch.Tensor):
            # print(tensor.size())
            tensor = tensor.data
        else:
            return tensor

        # use random index to generate visual output
        image_numpy = tensor[0].cpu().float().numpy()
        # if img_name == 'fake_obj':
        #     img_mean = list(img_mean)
        #     img_mean[-1] = 0.5
        # print(image_numpy.shape)
        if len(image_numpy.shape) == 4:
            image_numpy = image_numpy[0]
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
            # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) / 2.0 * 255.0
        else:
            # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + np.array(img_mean))
            # image_numpy = np.transpose(image_numpy, (1, 2, 0))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 0.5 + 0.5) * 255.0
            # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) / 2.0 * 255.0
            # image_numpy[image_numpy > 1] = 1.0
            # image_numpy = image_numpy*255.0
    else:
        image_numpy = tensor
    return image_numpy.astype(np.uint8)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_image_v2(image, image_path):
    image.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
            interpolation=self.interpolation)

class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


    def __str__(self):
        return self.__class__.__name__

def gen_A(num_classes, t=0.4, adj_file=None):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    # _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int32)
    return _adj

def gen_A_no_threshold(num_classes, t=0.4, adj_file=None):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj = _adj + np.identity(num_classes, np.int32)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

def gen_correlated(num_classes, adj_file=None):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj = _adj + np.identity(num_classes, np.int32)
    return _adj

def make_adj_file(dataroot, label_file):
    label_file = os.path.join(dataroot, label_file)
    with open(label_file, 'rb') as f:
        gt = pickle.load(f)
    count = np.zeros((gt.shape[1], gt.shape[1]))
    nums = np.zeros(gt.shape[1])
    for i in range(gt.shape[1]):
        for item in gt:
            if item[i] == 1:
                for k in range(gt.shape[1]):
                    count[i,k] += item[k]
        nums[i] = count[i,i]
        count[i,i] = 0
    save_dict = dict(
            nums=nums,
            adj=count
            )
    save_file = os.path.join(dataroot, 'adj.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(save_dict, f)

def make_correlated(root, label_file):
    label_file = os.path.join(root, label_file)
    with open(label_file, 'rb') as f:
        label = pickle.load(f)

    print("go correlated calculation")
    Y = np.matmul(label.T, label)
    max_value = Y.max()
    C1 = (Y/max_value).astype(np.float)
    save_file = os.path.join(root, 'correlated.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(C1,f)

def normalize_sigmoid(C1):
    num_classes = C1.shape[1]
    new_C1 = np.zeros(C1.shape)
    for i in range(num_classes):
        row = np.delete(C1[i],i)
        mean = np.mean(row)
        std = np.std(row)
        temp_row = (C1[i] - mean)/std
        new_C1[i] = 1/(1 + np.exp(-temp_row))
        new_C1[i][i] = 0
    return new_C1
