import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'multi_step':
        milestones = [int(item) for item in opt.milestones] 
        print(milestones)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        T_max = opt.niter_decay - opt.warm_epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0, last_epoch=opt.last_epoch)
    elif opt.lr_policy == 'one_cycle':
        EPOCHS = opt.niter_decay
        STEPS_PER_EPOCH = opt.steps_per_epoch
        TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

        MAX_LRS = [p['lr'] for p in optimizer.param_groups]

        scheduler = lr_scheduler.OneCycleLR(optimizer,
                                            max_lr = MAX_LRS,
                                            steps_per_epoch = STEPS_PER_EPOCH,
                                            epochs=EPOCHS)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        print(classname)
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, pretrained=False, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        # net = torch.nn.parallel.DistributedDataParallel(net, gpu_ids)
    if not pretrained:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def add_weight_decay(model, lr, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'lr': lr, 'weight_decay': 0.},
        {'params': decay, 'lr': lr, 'weight_decay': weight_decay}]

##############################################################################
# Classes
##############################################################################

class WSLoss(nn.Module):
    def __init__(self):
        super(WSLoss, self).__init__()

    def __call__(self, pred, gt_label, C1, sinkhorn_reg, sinkhorn_iter):
        loss = self.sinkhorn(pred, gt_label, C1, sinkhorn_reg, numItermax=sinkhorn_iter)
        return loss

    def sinkhorn(self, a, b, M, lambda_sh, numItermax=1000, stopThr=.5e-2, cuda=True):
        a = a.t() #Cx32
        b = b.t() #Cx32

        if cuda:
            u = (torch.ones_like(a) / a.size()[0]).double().cuda() #Cx32
            v = (torch.ones_like(b)).double().cuda()
        else:
            u = (torch.ones_like(a) / a.size()[0])
            v = (torch.ones_like(b))

        K = torch.exp(-M / lambda_sh) #CxC

        err = 1
        cpt = 0

        while err > stopThr and cpt < numItermax:

            u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t()))) #Cx32

            cpt += 1

            if cpt % 20 == 1:

                v = torch.div(b, torch.matmul(K.t(), u))  # (nb, N)
                u = torch.div(a, torch.matmul(K, v))

                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        sinkhorn_divergence = torch.mean(torch.sum(torch.mul(u, torch.matmul(torch.mul(K, M), v)), dim=0))

        return sinkhorn_divergence

class BCEWSLoss(nn.Module):
    def __init__(self, C1):
        super(BCEWSLoss, self).__init__()
        self.C1 = C1.float()

    def __call__(self, pred, gt_label, reduction='avg'):
        loss = gt_label * torch.log(pred) #+ (1-gt_label) * torch.log(1-pred)
        loss = torch.matmul(loss, self.C1)
        if reduction == 'avg':
            loss = torch.mean(loss)
        return loss

class NCALoss(nn.Module):
    def __init__(self):
        super(NCALoss, self).__init__()

    def __call__(self, pred, gt_label):
        loss = self.nca(pred, gt_label)
        return loss

    def nca(self, feature, gt_label, temperature=100):
        # compute pairwise squared Euclidean distances
        # in transformed space
        distances = self._pairwise_l2_sq(feature)
        # fill diagonal values such that exponentiating them
        # makes them equal to 0
        distances.diagonal().copy_(np.inf*torch.ones(len(distances)))
        weight_mat = self._gen_weight_mat(gt_label)
        #p_ij = self._weighted_softmax(weight_mat, -distances)
        p_ij = self._softmax(-distances, temperature) 

        #find the instance of the same class
        y_mask = torch.matmul(gt_label, gt_label.t()) # batch x batch
        y_mask = (y_mask > 0).float()
        y_mask.diagonal().copy_(torch.zeros(feature.size(0)))
        #actual nca
        average = y_mask.sum(dim=1)
        zero_index = (average == 0)
        average[zero_index] = 1

        p_ij_mask = (p_ij * y_mask.float())/average
        p_ij_mask = p_ij_mask * weight_mat.float()
        p_i = p_ij_mask.sum(dim=1)

        loss = -torch.log(torch.masked_select(p_i, p_i != 0)).sum()
        return loss

    def _pairwise_l2_sq(self, x):
        """Compute pairwise squared Euclidean distances.
        """
        dot = torch.mm(x.double(), torch.t(x.double()))
        norm_sq = torch.diag(dot)
        dist = norm_sq[None, :] - 2*dot + norm_sq[:, None]
        dist = torch.clamp(dist, min=0)  # replace negative values with 0
        return dist.float()

    def _softmax(self,x, temperature):
        """Compute row-wise softmax.

        Notes:
          Since the input to this softmax is the negative of the
          pairwise L2 distances, we don't need to do the classical
          numerical stability trick.
        """
        exp = torch.exp(x/temperature)
        return exp / exp.sum(dim=1)

    def _weighted_softmax(self, weight_mat, x):
        # wieght_mat size : bs x bs 
        # x size: bs x bs
        exp = torch.exp(x) * weight_mat
        return exp / exp.sum(dim=1)

    def _gen_weight_mat(self, label):
        #label size : bs x label_size
        count_occ = torch.matmul(label, label.t()) #bs x bs
        label_per_img = torch.sum(label,1) #bs
        # label_per_img = label_per_img.unsqueeze(1)
        label_per_img_mat = label_per_img.unsqueeze(1).repeat(1,label.size(0)) # bsxbs
        # label_per_img_mat = label_per_img_mat.view(label.size(0), label.size(0))
        count_union = label_per_img_mat + label_per_img - count_occ
        weight_mat = count_occ/count_union
        weight_mat.diagonal().copy_(torch.zeros(label.size(0)))
        return weight_mat.float()

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt, len_pos):
        return self.loss_log(features, gt, len_pos)

    def loss_log(self, features, labels=None, len_pos=0):
        features = features.squeeze()
        batch_size = features.size(0)
        mask = torch.matmul(labels, labels.t()) # batch x batch
        #mask = (mask > 3).float().cuda()
        #mask = torch.eye(8, dtype=torch.float32).cuda()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        zero_index = (average == 0)
        average[zero_index] = 1
        if self.weighted:
            weight_mat = self._gen_weight_mat(labels)
            mean_log_prob_pos = (mask * log_prob * weight_mat).sum(1) / average
        else:
            mean_log_prob_pos = (mask * log_prob ).sum(1) / average

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(batch_size).mean()
        loss = loss.view(batch_size).sum()/len_pos

        return loss
    
    def _gen_weight_mat(self, label):
        #label size : bs x label_size
        count_occ = torch.matmul(label, label.t()) #bs x bs
        label_per_img = torch.sum(label,1) #bs
        # label_per_img = label_per_img.unsqueeze(1)
        label_per_img_mat = label_per_img.unsqueeze(1).repeat(1,label.size(0)) # bsxbs
        # label_per_img_mat = label_per_img_mat.view(label.size(0), label.size(0))
        count_union = label_per_img_mat + label_per_img - count_occ
        weight_mat = count_occ/count_union
        weight_mat.diagonal().copy_(torch.zeros(label.size(0)))
        return weight_mat.detach().float()

class SupConNewMeanLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SupConNewMeanLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt, true_bs):
        return self.loss_log(features, gt, true_bs)

    def loss_log(self, features, labels=None, true_bs=0):
        features = features.squeeze()
        batch_size = features.size(0)
        mask = torch.matmul(labels, labels.t()) # batch x batch
        # make division vector
        pos_item = labels.sum(1) # N
        div_vector = torch.zeros(batch_size)
        start_index = 0
        for item in pos_item:
            end_index = int(start_index + item)
            div_vector[start_index:end_index] = item
            start_index = end_index
        div_vector = div_vector.cuda() # N_pos
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        zero_index = (average == 0)
        average[zero_index] = 1
        if self.weighted:
            weight_mat = self._gen_weight_mat(labels)
            mean_log_prob_pos = (mask * log_prob * weight_mat).sum(1) / average
        else:
            mean_log_prob_pos = (mask * log_prob ).sum(1) / average

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos / div_vector
        # loss = loss.view(batch_size).mean()
        loss = loss.view(batch_size).sum()/true_bs

        return loss
    
    def _gen_weight_mat(self, label):
        #label size : bs x label_size
        count_occ = torch.matmul(label, label.t()) #bs x bs
        label_per_img = torch.sum(label,1) #bs
        # label_per_img = label_per_img.unsqueeze(1)
        label_per_img_mat = label_per_img.unsqueeze(1).repeat(1,label.size(0)) # bsxbs
        # label_per_img_mat = label_per_img_mat.view(label.size(0), label.size(0))
        count_union = label_per_img_mat + label_per_img - count_occ
        weight_mat = count_occ/count_union
        weight_mat.diagonal().copy_(torch.zeros(label.size(0)))
        return weight_mat.detach().float()

class SupConReWLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SupConReWLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt, temp_mat, len_pos):
        return self.loss_log(features, gt, temp_mat, len_pos)

    def loss_log(self, features, labels=None, temp_mat=None, len_pos=0):
        features = features.squeeze()
        batch_size = features.size(0)
        mask = torch.matmul(labels, labels.t()) # batch x batch
        #mask = (mask > 3).float().cuda()
        #mask = torch.eye(8, dtype=torch.float32).cuda()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute logits
        # print(temp_mat.size())
        # print(batch_size)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temp_mat)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        zero_index = (average == 0)
        average[zero_index] = 1
        if self.weighted:
            weight_mat = self._gen_weight_mat(labels)
            mean_log_prob_pos = (mask * log_prob * weight_mat).sum(1) / average
        else:
            mean_log_prob_pos = (mask * log_prob ).sum(1) / average

        # loss
        loss = -mean_log_prob_pos
        # loss = loss.view(batch_size).mean()
        loss = loss.view(batch_size).sum()/len_pos

        return loss
    
    def _gen_weight_mat(self, label):
        #label size : bs x label_size
        count_occ = torch.matmul(label, label.t()) #bs x bs
        label_per_img = torch.sum(label,1) #bs
        # label_per_img = label_per_img.unsqueeze(1)
        label_per_img_mat = label_per_img.unsqueeze(1).repeat(1,label.size(0)) # bsxbs
        # label_per_img_mat = label_per_img_mat.view(label.size(0), label.size(0))
        count_union = label_per_img_mat + label_per_img - count_occ
        weight_mat = count_occ/count_union
        weight_mat.diagonal().copy_(torch.zeros(label.size(0)))
        return weight_mat.detach().float()

class SameImgSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SameImgSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt, label_mask, len_pos):
        return self.loss_log(features, gt, label_mask, len_pos)

    def loss_log(self, features, labels=None, label_mask=None, len_pos=0):
        features = features.squeeze()
        batch_size = features.size(0)
        mask = torch.matmul(labels, labels.t()) # batch x batch
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        label_mask = label_mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        deno_not_same_img = torch.log((label_mask * exp_logits).sum(1, keepdim=True))
        log_prob = logits - deno_not_same_img

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        average_mask = label_mask.sum(dim=1)
        average = average + average_mask
        zero_index = (average == 0)
        average[zero_index] = 1

        mean_log_prob_pos = (mask * log_prob ).sum(1)/ average

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).sum()/len_pos

        return loss

class SameImgNoNegSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SameImgNoNegSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt, len_pos):
        return self.loss_log(features, gt, len_pos)

    def loss_log(self, features, labels=None, len_pos=0):
        features = features.squeeze()
        batch_size = features.size(0)
        mask = torch.matmul(labels, labels.t()) # batch x batch
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        # label_mask = label_mask * logits_mask

        # compute logits
        logits = 1 - torch.matmul(features, features.T)

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        # average_mask = label_mask.sum(dim=1)
        # average = average + average_mask
        zero_index = (average == 0)
        average[zero_index] = 1

        mean_log_prob_pos = (mask * logits ).sum(1)/ average

        # loss
        loss = mean_log_prob_pos
        loss = loss.view(batch_size).sum()/len_pos

        return loss

class SupConRandomWalkLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SupConRandomWalkLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt, len_pos):
        return self.loss_log(features, gt, len_pos)

    def loss_log(self, features, labels=None, len_pos=0):
        features = features.squeeze()
        batch_size = features.size(0)
        mask = labels # batch x batch
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        zero_index = (average == 0)
        average[zero_index] = 1
        mean_log_prob_pos = (mask * log_prob ).sum(1) / average

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).sum()/len_pos

        return loss

class SupConMeanLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SupConMeanLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt):
        return self.loss_log(features, gt)

    def loss_log(self, features, mask=None):
        features = features.squeeze()
        batch_size = features.size(0)
        # mask = torch.matmul(labels, labels.t()) # batch x batch
        # mask = (mask > 0).float().cuda()
        # mask = torch.eye(8, dtype=torch.float32).cuda()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        zero_index = (average == 0)
        average[zero_index] = 1
        if self.weighted:
            weight_mat = self._gen_weight_mat(labels)
            mean_log_prob_pos = (mask * log_prob * weight_mat).sum(1) / average
        else:
            mean_log_prob_pos = (mask * log_prob ).sum(1) / average

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()
        # loss = loss.view(batch_size).sum()/batch_size

        return loss
    
    def _gen_weight_mat(self, label):
        #label size : bs x label_size
        count_occ = torch.matmul(label, label.t()) #bs x bs
        label_per_img = torch.sum(label,1) #bs
        # label_per_img = label_per_img.unsqueeze(1)
        label_per_img_mat = label_per_img.unsqueeze(1).repeat(1,label.size(0)) # bsxbs
        # label_per_img_mat = label_per_img_mat.view(label.size(0), label.size(0))
        count_union = label_per_img_mat + label_per_img - count_occ
        weight_mat = count_occ/count_union
        weight_mat.diagonal().copy_(torch.zeros(label.size(0)))
        return weight_mat.detach().float()

class SupConMeanThresholdLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, threshold=2, weighted=False):
        super(SupConMeanThresholdLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.threshold=threshold
        self.weighted = weighted

    def __call__(self, features, gt):
        return self.loss_log(features, gt)

    def loss_log(self, features, labels=None):
        features = features.squeeze()
        batch_size = features.size(0)
        label_coocc_mat = torch.matmul(labels, labels.t()) # batch x batch
        mask = (label_coocc_mat > self.threshold).float()
        total_pos_mat = labels.sum(1) # batch
        idx_lower_pos = torch.where(total_pos_mat < self.threshold)[0]
        for i in idx_lower_pos:
            item_idx = int(i)
            pos_row = (label_coocc_mat[item_idx] == total_pos_mat[item_idx]).float()
            mask[item_idx] = pos_row
        mask = mask.cuda()
        # mask = torch.eye(8, dtype=torch.float32).cuda()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        zero_index = (average == 0)
        average[zero_index] = 1
        if self.weighted:
            weight_mat = self._gen_weight_mat(labels)
            mean_log_prob_pos = (mask * log_prob * weight_mat).sum(1) / average
        else:
            mean_log_prob_pos = (mask * log_prob ).sum(1) / average

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()
        # loss = loss.view(batch_size).sum()/batch_size

        return loss
    
    def _gen_weight_mat(self, label):
        #label size : bs x label_size
        count_occ = torch.matmul(label, label.t()) #bs x bs
        label_per_img = torch.sum(label,1) #bs
        # label_per_img = label_per_img.unsqueeze(1)
        label_per_img_mat = label_per_img.unsqueeze(1).repeat(1,label.size(0)) # bsxbs
        # label_per_img_mat = label_per_img_mat.view(label.size(0), label.size(0))
        count_union = label_per_img_mat + label_per_img - count_occ
        weight_mat = count_occ/count_union
        weight_mat.diagonal().copy_(torch.zeros(label.size(0)))
        return weight_mat.detach().float()

class SupConLabelLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SupConLabelLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt):
        return self.loss_log(features, gt)

    def loss_log(self, features, labels=None):
        total_loss = 0
        features = features.squeeze()
        batch_size, num_classes = features.size(0), features.size(1)
        count_item = 0
        for i in range(batch_size):
            label = labels[i]
            label = label.unsqueeze(-1)
            len_pos = label.sum()
            if len_pos <= 1:
                continue
            count_item += 1
            feature = features[i]
            mask = torch.matmul(label, label.t()) # num_classes x num_classes
            # print(mask.size())
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(num_classes).view(-1, 1).cuda(),
                0
            )
            mask = mask * logits_mask

            # compute logits
            anchor_dot_contrast = torch.div(torch.matmul(feature, feature.T), self.temperature)

            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive 
            average = mask.sum(dim=1)
            zero_index = (average == 0)
            average[zero_index] = 1
            mean_log_prob_pos = (mask * log_prob ).sum(1) / average

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(num_classes).sum()/len_pos
            total_loss += loss
        total_loss = total_loss / count_item
        return total_loss

class SupConIntraLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SupConIntraLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self,mean, features, gt):
        return self.loss_log(mean, features, gt)

    def loss_log(self, mean, features, labels=None):
        total_loss = 0
        features = features.squeeze()
        mean = mean.squeeze()
        batch_size, num_classes = features.size(0), features.size(1)
        count_item = 0
        for i in range(batch_size):
            mask = labels[i]
            mean_item = mean[i]
            feature = features[i]
            count_item += 1

            # compute logits
            anchor_dot_contrast = torch.div(torch.matmul(feature, mean_item), self.temperature)
            logits_max = torch.matmul(mean_item, mean_item.t())
            # for numerical stability
            logits = anchor_dot_contrast - logits_max.detach()
            # compute log_prob
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum())
            log_prob = log_prob.squeeze()

            # compute mean of log-likelihood over positive 
            average = mask.sum()
            mean_log_prob_pos = (mask * log_prob ).sum() / average

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            total_loss += loss
        total_loss = total_loss / count_item
        return total_loss

class SupConWithMaskLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SupConWithMaskLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt, len_pos):
        return self.loss_log(features, gt, len_pos)

    def loss_log(self, features, labels=None, len_pos=0):
        features = features.squeeze()
        batch_size = features.size(0)
        mask = labels # batch x batch
        #mask = (mask > 3).float().cuda()
        #mask = torch.eye(8, dtype=torch.float32).cuda()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        zero_index = (average == 0)
        average[zero_index] = 1
        if self.weighted:
            weight_mat = self._gen_weight_mat(labels)
            mean_log_prob_pos = (mask * log_prob * weight_mat).sum(1) / average
        else:
            mean_log_prob_pos = (mask * log_prob ).sum(1) / average

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(batch_size).mean()
        loss = loss.view(batch_size).sum()/len_pos

        return loss
    
    def _gen_weight_mat(self, label):
        #label size : bs x label_size
        count_occ = torch.matmul(label, label.t()) #bs x bs
        label_per_img = torch.sum(label,1) #bs
        # label_per_img = label_per_img.unsqueeze(1)
        label_per_img_mat = label_per_img.unsqueeze(1).repeat(1,label.size(0)) # bsxbs
        # label_per_img_mat = label_per_img_mat.view(label.size(0), label.size(0))
        count_union = label_per_img_mat + label_per_img - count_occ
        weight_mat = count_occ/count_union
        weight_mat.diagonal().copy_(torch.zeros(label.size(0)))
        return weight_mat.detach().float()

class SupConMultiLabelLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SupConMultiLabelLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt, label_mask, len_pos):
        return self.loss_log(features, gt, label_mask, len_pos)

    def loss_log(self, features, labels=None, label_mask=None, len_pos=0):
        features = features.squeeze()
        batch_size = features.size(0)
        mask = torch.matmul(labels, labels.t()) # batch x batch
        #mask = (mask > 3).float().cuda()
        #mask = torch.eye(8, dtype=torch.float32).cuda()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        label_mask = label_mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        average_mask = label_mask.sum(dim=1)
        average = average + average_mask
        zero_index = (average == 0)
        average[zero_index] = 1
        if self.weighted:
            weight_mat = self._gen_weight_mat(labels)
            mean_log_prob_pos = (mask * log_prob * weight_mat).sum(1) / average
        else:
            mean_log_prob_pos = ((mask * log_prob ).sum(1) + (label_mask * log_prob).sum(1))/ average

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(batch_size).mean()
        loss = loss.view(batch_size).sum()/len_pos

        return loss
    
    def _gen_weight_mat(self, label):
        #label size : bs x label_size
        count_occ = torch.matmul(label, label.t()) #bs x bs
        label_per_img = torch.sum(label,1) #bs
        # label_per_img = label_per_img.unsqueeze(1)
        label_per_img_mat = label_per_img.unsqueeze(1).repeat(1,label.size(0)) # bsxbs
        # label_per_img_mat = label_per_img_mat.view(label.size(0), label.size(0))
        count_union = label_per_img_mat + label_per_img - count_occ
        weight_mat = count_occ/count_union
        weight_mat.diagonal().copy_(torch.zeros(label.size(0)))
        return weight_mat.detach().float()

class NewSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(NewSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt):
        return self.loss_log(features, gt)

    def loss_log(self, features, labels=None):
        """
        features: batch x L x feat_size
        labels: batch x L
        """
        batch_size = features.size(0)
        num_classes = labels.size(1)
        # make mask
        a = torch.diag(torch.ones(num_classes)).cuda()
        a = a.unsqueeze(-1)
        class_mask = a.repeat(1,1,batch_size) # L x L x batch_size
        labels = labels.t() # L x batch
        class_over_sample = class_mask * labels # L x L x batch
        mask = torch.bmm(class_over_sample.permute(0,2,1), class_over_sample) # L x b x b
        # mask-out self-contrast cases
        logits_mask = 1 - torch.diag(torch.ones(batch_size)).cuda() # batch x batch
        mask = mask * logits_mask

        # compute logits

        features = features.permute(1,0,2) # L x batch x size
        features_T = features.permute(0,2,1) # L x size x batch
        new_features = features.unsqueeze(0) # 1 x L x batch x size
        new_features = new_features.repeat(num_classes, 1, 1, 1) # L x L x batch x size
        # denominator

        new_features = new_features.view(new_features.size(0), -1, new_features.size(3))
        anchor_dot_contrast = torch.div(torch.bmm(new_features, features_T), self.temperature) # L x L x batch x batch
        anchor_dot_contrast = anchor_dot_contrast.view(num_classes, num_classes, batch_size, batch_size)
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        sum_exp_logits = exp_logits.sum(1) # L x b x b
        sum_exp_logits = sum_exp_logits.sum(2, keepdim=True)
        # numerator
        numerator = torch.div(torch.bmm(features, features_T), self.temperature) # L x b x b
        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # sum_exp_logits = exp_logits.sum(3, keepdim=True)
        # sum_exp_logits = exp_logits.sum(1, keepdim=True)
        log_prob = numerator - torch.log(sum_exp_logits)

        # compute mean of log-likelihood over positive 
        mean_log_prob_pos = (mask * log_prob ).sum(0).sum(1)
        average = mask.sum(0).sum(1)
        zero_index = (average == 0)
        average[zero_index] = 1
        mean_log_prob_pos = mean_log_prob_pos/average

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss
    
    def _gen_weight_mat(self, label):
        #label size : bs x label_size
        count_occ = torch.matmul(label, label.t()) #bs x bs
        label_per_img = torch.sum(label,1) #bs
        # label_per_img = label_per_img.unsqueeze(1)
        label_per_img_mat = label_per_img.unsqueeze(1).repeat(1,label.size(0)) # bsxbs
        # label_per_img_mat = label_per_img_mat.view(label.size(0), label.size(0))
        count_union = label_per_img_mat + label_per_img - count_occ
        weight_mat = count_occ/count_union
        weight_mat.diagonal().copy_(torch.zeros(label.size(0)))
        return weight_mat.detach().float()

class SupConV2Loss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.5, contrast_mode='all', base_temperature=0.5):
        super(SupConV2Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
        has the same class as sample i. Can be asymmetric.
        Returns:
        A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                    'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConBaselineLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, weighted=False):
        super(SupConBaselineLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.weighted = weighted

    def __call__(self, features, gt):
        return self.loss_log(features, gt)

    def loss_log(self, features, labels=None):
        features = features.squeeze()
        batch_size = features.size(0)
        mask = torch.matmul(labels, labels.t()) # batch x batch
        mask = (mask > 0).float().cuda()
        #mask = torch.eye(8, dtype=torch.float32).cuda()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive 
        average = mask.sum(dim=1)
        zero_index = (average == 0)
        average[zero_index] = 1
        if self.weighted:
            weight_mat = self._gen_weight_mat(labels)
            mean_log_prob_pos = (mask * log_prob * weight_mat).sum(1) / average
        else:
            mean_log_prob_pos = (mask * log_prob ).sum(1) / average

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()
        # loss = loss.view(batch_size).sum()/len_pos

        return loss
    
    def _gen_weight_mat(self, label):
        #label size : bs x label_size
        count_occ = torch.matmul(label, label.t()) #bs x bs
        label_per_img = torch.sum(label,1) #bs
        # label_per_img = label_per_img.unsqueeze(1)
        label_per_img_mat = label_per_img.unsqueeze(1).repeat(1,label.size(0)) # bsxbs
        # label_per_img_mat = label_per_img_mat.view(label.size(0), label.size(0))
        count_union = label_per_img_mat + label_per_img - count_occ
        weight_mat = count_occ/count_union
        weight_mat.diagonal().copy_(torch.zeros(label.size(0)))
        return weight_mat.detach().float()