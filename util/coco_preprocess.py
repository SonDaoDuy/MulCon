import math
import os
from urllib.request import urlretrieve
import torch
from PIL import Image, ImageFilter, ImageDraw
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from util.autoaugmentation import ImageNetPolicy
from randaugment import RandAugment
from models import create_model


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, p=0.5):
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        if torch.rand(1) < self.p:
            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class BboxCutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, p=0.5):
        self.max_area = 224*224
        self.p = p

    def __call__(self, img, bbox_list):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        
        if torch.rand(1) < self.p:
            bbox_idx = torch.randperm(len(bbox_list))
            picked_area = 0
            picked_bbox = []
            for idx in bbox_idx:
                picked_area += bbox_list[idx]['area']
                if picked_area > self.max_area and len(picked_bbox) > 0:
                    break
                picked_bbox.append(bbox_list[idx])

            for bbox_item in picked_bbox:
                x1,y1,w1,h1 = bbox_item['bbox']
                if bbox_item['area'] <= 0 or w1 < 1 or h1 < 1:
                    continue
                x2 = x1 + w1 - 1
                y2 = y1 + h1 - 1

                mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class Maskout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, opt, p=0.5, epoch=10):
        # pretrained model for cut out
        self.opt = opt
        self.opt.gpu_ids = []
        # self.opt.isTrain = False
        self.model = create_model(self.opt)
        self.model.save_dir = os.path.join(opt.checkpoints_dir, opt.mask_model)  # save all the checkpoints to save_dir
        # self.model.device=torch.device('cpu')
        self.model.isTrain=False
        load_suffix = str(epoch)
        self.model.load_networks(load_suffix)
        self.model.eval()
        # print(opt.gpu_ids)
        self.opt.gpu_ids = [0, 1]
        # print(self.model.netE.device)
        self.p = p
        self.up_scale = nn.Upsample(size=opt.load_size)

    def __call__(self, img, method='max'):
        # print(img.device)
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        final_mask = torch.ones(self.opt.load_size, self.opt.load_size)
        h = 14
        w = 14

        if torch.rand(1) < self.p:
            gt_label = torch.zeros(self.opt.num_classes)
            mask_ann = dict(img=img.unsqueeze(0), gt_label=gt_label.unsqueeze(0))
            # do forward pass
            self.model.set_input(mask_ann)
            self.model.test()  # run inference and calculate metrics
            mask = self.model.get_current_attention()
            mask = mask.squeeze()
            pred, _ = self.model.get_current_results()
            pred = pred.squeeze()
            pred = torch.sigmoid(pred)
            # make binary mask
            mask = mask.reshape(mask.size(0), mask.size(1), h,w) # N , C, 14, 14
            if method=='max':
                picked_idx = pred.argmax()
            else:
                picked_idx = pred.argmax()
            mask_head_idx = torch.randint(0,3,(1,))
            # print(mask.size())
            mask_vis = mask[mask_head_idx, picked_idx,:,:] # 14, 14
            mask_vis = mask_vis.squeeze()
            mask_vis = mask_vis - torch.min(mask_vis)
            cam_img = mask_vis / torch.max(mask_vis)
            mean_heat = cam_img.mean()
            mean_map = (cam_img < mean_heat).float() # 14,14
            mean_map = mean_map.view(1,1,h,w)
            final_mask = self.up_scale(mean_map)
            final_mask = final_mask.squeeze()
            
        mask = final_mask.expand_as(img)
        img = img * mask

        return img

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
        self.scales = scales if scales is not None else [1, .875, .75, .66]
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


def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.

    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.

    Returns
    -------
    filename : str
        The location of the downloaded file.

    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= (pos_count +1e-6)
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        # scores = np.zeros((n, c)) - 1
        scores = np.zeros((n, c))
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                # scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
                scores[i, ind] = 1 if tmp[i, ind] >= 0.5 else 0
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0.5)
            Nc[k] = np.sum(targets * (scores >= 0.5))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1

class NewAveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False, n_class=157):
        super(NewAveragePrecisionMeter, self).__init__()
        self.n_class = n_class
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.sample_num = 0.
        self.ap = 0
        self.Nc = np.zeros(self.n_class)
        self.Np = np.zeros(self.n_class)
        self.Ng = np.zeros(self.n_class)
        
        self.ap_topk = 0
        self.Nc_topk = np.zeros(self.n_class)
        self.Np_topk = np.zeros(self.n_class)
        self.Ng_topk = np.zeros(self.n_class)

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if output.numel() > 0:
            assert target.size(1) == output.size(1), \
                'dimensions for output should match target.'

        for k in range(output.size(0)):
            # sort scores
            outputk = output[k, :]
            targetk = target[k, :]
            # compute average precision
            self.ap += AveragePrecisionMeter.average_precision(outputk, targetk, self.difficult_examples)
            self.sample_num += 1

        Nc = np.zeros(self.n_class)
        Np = np.zeros(self.n_class)
        Ng = np.zeros(self.n_class)
        for i in range(self.n_class):
            outputk = output[:, i]
            targetk = target[:, i]
            Ng[i] = sum(targetk == 1)
            Np[i] = sum(outputk >= 0)
            print(outputk.device, targetk.device)
            Nc[i] = sum(targetk * (outputk >= 0))
        
        self.Nc += Nc
        self.Np += Np
        self.Ng += Ng

        n = output.shape[0]
        output_topk = torch.zeros((n, self.n_class)) - 1
        index = output.topk(3, 1, True, True)[1].cpu().numpy()
        tmp = output.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                output_topk[i, ind] = 1 if tmp[i, ind] >= 0 else - 1
        for k_topk in range(output_topk.shape[0]):
            # sort scores
            outputk_topk = output_topk[k_topk, :]
            targetk_topk = target[k_topk, :]
            # compute average precision
            self.ap_topk += AveragePrecisionMeter.average_precision(outputk_topk, targetk_topk, self.difficult_examples)

        Nc_topk = np.zeros(self.n_class)
        Np_topk = np.zeros(self.n_class)
        Ng_topk = np.zeros(self.n_class)
        for i in range(self.n_class):
            outputk_topk = output_topk[:, i]
            targetk_topk = target[:, i]
            Ng_topk[i] = sum(targetk_topk == 1)
            Np_topk[i] = sum(outputk_topk >= 0)
            Nc_topk[i] = sum(targetk_topk * (outputk_topk >= 0))
        
        self.Nc_topk += Nc_topk
        self.Np_topk += Np_topk
        self.Ng_topk += Ng_topk

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each sample k
        """
        ap = self.ap / self.sample_num
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=False):
        
        # sort examples
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)
            
        _, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        precision_at_i_list = torch.zeros(int(target.sum()))
        for i in indices:
            label = target[i]
            total_count += 1
            if label == 1:
                pos_count += 1
                precision_at_i_list[int(pos_count) - 1] = pos_count / total_count

        for i in range(len(precision_at_i_list) - 1, 0, -1):
            if precision_at_i_list[i - 1] < precision_at_i_list[i]:
                precision_at_i_list[i - 1] = precision_at_i_list[i]

        precision_at_i = precision_at_i_list.mean()
        return precision_at_i

    def overall(self):
        if self.sample_num == 0:
            return 0
        
        OP = sum(self.Nc) / max(sum(self.Np), 1e-8)
        OR = sum(self.Nc) / max(sum(self.Ng), 1e-8)
        OF1 = (2 * OP * OR) / max(OP + OR, 1e-8)

        clip_Np = self.Np.copy()
        for i in range(len(self.Np)):
            if self.Np[i] < 1e-8:
                clip_Np[i] = 1e-8

        clip_Ng = self.Ng.copy()
        for i in range(len(self.Ng)):
            if self.Ng[i] < 1e-8:
                clip_Ng[i] = 1e-8

        CP = sum(self.Nc / clip_Np) / self.n_class
        CR = sum(self.Nc / clip_Ng) / self.n_class
        CF1 = (2 * CP * CR) / max(CP + CR, 1e-8)
        
        return OP, OR, OF1, CP, CR, CF1

    def overall_topk(self, k):
        '''
        k is useless
        '''
        if self.sample_num == 0:
            return 0
        
        OP = sum(self.Nc_topk) / max(sum(self.Np_topk), 1e-8)
        OR = sum(self.Nc_topk) / max(sum(self.Ng_topk), 1e-8)
        OF1 = (2 * OP * OR) / max(OP + OR, 1e-8)

        clip_Np_topk = self.Np_topk.copy()
        for i in range(len(self.Np_topk)):
            if self.Np_topk[i] < 1e-8:
                clip_Np_topk[i] = 1e-8

        clip_Ng_topk = self.Ng_topk.copy()
        for i in range(len(self.Ng_topk)):
            if self.Ng_topk[i] < 1e-8:
                clip_Ng_topk[i] = 1e-8

        CP = sum(self.Nc_topk / clip_Np_topk) / self.n_class
        CR = sum(self.Nc_topk / clip_Ng_topk) / self.n_class
        CF1 = (2 * CP * CR) / max(CP + CR, 1e-8)
        
        return OP, OR, OF1, CP, CR, CF1

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, opt):
        normalize = transforms.Normalize(mean=opt.image_normalization_mean, std=opt.image_normalization_std)
        transform = transforms.Compose([
            transforms.Resize((opt.load_size, opt.load_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            normalize,
        ])
        if opt.cutout:
            transform.transforms.append(Cutout(opt.n_holes, opt.length))
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TwoCropTransformOri:
    """Create two crops of the same image"""
    def __init__(self, opt):
        normalize = transforms.Normalize(mean=opt.image_normalization_mean, std=opt.image_normalization_std)
        transform = transforms.Compose([
            transforms.Resize((opt.load_size, opt.load_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_ori = transforms.Compose([
            MultiScaleCrop(opt.load_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        if opt.cutout:
            transform.transforms.append(Cutout(opt.n_holes, opt.length))
        self.transform = transform
        self.transform_ori = transform_ori

    def __call__(self, x):
        return [self.transform_ori(x), self.transform(x)]

class TwoCropTransformCutout:
    """Create two crops of the same image"""
    def __init__(self, opt):
        normalize = transforms.Normalize(mean=opt.image_normalization_mean, std=opt.image_normalization_std)
        transform_1 = transforms.Compose([
            transforms.Resize((opt.load_size, opt.load_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_2 = transforms.Compose([
            transforms.Resize((opt.load_size, opt.load_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            normalize,
            Cutout(opt.n_holes, opt.length),
        ])
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __call__(self, x):
        return [self.transform_1(x), self.transform_2(x)]

class TwoCropTransformDiffViews:
    """Create two crops of the same image"""
    def __init__(self, opt):
        normalize = transforms.Normalize(mean=opt.image_normalization_mean, std=opt.image_normalization_std)
        transform_1 = transforms.Compose([
            transforms.Resize((opt.load_size, opt.load_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_2 = transforms.Compose([
            MultiScaleCrop(opt.load_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __call__(self, x):
        return [self.transform_1(x), self.transform_2(x)]

class TwoCropTransformSIMCLR:
    """Create two crops of the same image"""
    def __init__(self, opt):
        normalize = transforms.Normalize(mean=opt.image_normalization_mean, std=opt.image_normalization_std)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.load_size, scale=(0.875, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


