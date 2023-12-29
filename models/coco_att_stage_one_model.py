import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import os
import pickle
import numpy as np

from . import networks
from .base_model import BaseModel
from util.util import *
from models.mab import ISABMultiLabel
from models.resnet import resnet101

class COCOAttStageOneModel(BaseModel):
    """docstring for DummyModel"""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--coef_nce', type=float, default=0.5, help='Pre-calculated cost matrix')
        parser.add_argument('--coef_att', type=float, default=1.0, help='weight loss for bce')
        parser.add_argument('--pretrained', type=bool, default=True, help='weight loss for bce')
        parser.add_argument('--bce_type', type=str, default='bce', help='bce| focal| asl')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # constant
        self.num_classes = opt.num_classes
        self.coef_nce = opt.coef_nce
        self.coef_att = opt.coef_att
        # self.batch_size = opt.batch_size
        self.pretrained = opt.pretrained
        self.bce_type = opt.bce_type

        # specify training losses
        self.loss_names = ['total', 'attention']
        # specify visual results
        self.visual_names = []
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['E', 'A', 'C']
        # define networks
        # netE
        model = models.resnet101(pretrained=self.pretrained)
        self.netE= nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.netE = networks.init_net(self.netE, pretrained=self.pretrained, gpu_ids=self.gpu_ids)
        #attention module
        dim_in = 2048
        dim_out = 1024
        num_head = 4
        self.netA = ISABMultiLabel(dim_in, dim_out, num_head, self.num_classes, ln=True)
        self.netA = networks.init_net(self.netA, gpu_ids=self.gpu_ids)
        self.netC = []
        for i in range(self.num_classes):
            item_c = nn.Linear(dim_out, 1)
            item_c = networks.init_net(item_c, gpu_ids=self.gpu_ids)
            self.netC.append(item_c)
        
        if self.isTrain:
            if self.bce_type == 'bce':
                self.criterionBCE = nn.BCEWithLogitsLoss(reduction='sum').to(self.device)

            if not self.pretrained:
                params = list(self.netE.parameters()) + list(self.netA.parameters()) + list(self.netP.parameters()) 
                for i in range(self.num_classes):
                    params = params + list(self.netC[i].parameters())
            else:
                params = self.get_config_optim(opt.lr, opt.lrp)
                print('len param:', len(params))
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=1e-4) 
            self.optimizers.append(self.optimizer)
            print('in model len of param_group: ',len(self.optimizers[0].param_groups))
            self.scaler = GradScaler()

    def get_config_optim(self, lr, lrp):
        # pretrained imagenet
        params_list = [
                {'params': self.netE.parameters(), 'lr': lr * lrp},
                {'params': self.netA.parameters(), 'lr': lr}
                ]
        for i in range(self.num_classes):
            params_list.append({'params': self.netC[i].parameters(), 'lr': lr})
        return params_list

    def set_input(self, input):
        self.img = input['img'].to(self.device)
        self.gt_label = input['gt_label'].to(self.device)
        self.batch_size = self.gt_label.size(0)
        if self.isTrain:
            #make new label for attention
            self.gt_label_T = self.gt_label.t().float() # Lx N
            self.new_gt = self.gt_label_T.contiguous().view(self.num_classes * self.batch_size)

    def forward(self):
        with autocast():
            # feed forward
            self.features = self.netE(self.img)
            self.features = self.features.view(self.features.size(0), self.features.size(1), -1) # B, C, HW
            self.features = self.features.permute(0,2,1) # B, HW, C
            self.new_feat, self.mask = self.netA(self.features)
            #attention
            self.attention_feat = self.new_feat.permute(1, 0, 2) # L, B, C
            self.pred_attention = []
            for i in range(self.num_classes):
                pred = self.netC[i](self.attention_feat[i]) # N x 1
                self.pred_attention.append(pred.squeeze())
            self.pred = torch.stack(self.pred_attention)
            if self.batch_size > 1:
                self.pred = self.pred.permute(1, 0)
            # losses
            if self.isTrain:
                #loss attention
                self.loss_attention = 0
                for i in range(self.num_classes):
                    loss_item = self.criterionBCE(self.pred_attention[i], self.gt_label_T[i])
                    self.loss_attention += loss_item
                self.loss_total = self.loss_attention*self.coef_att/(self.batch_size*self.num_classes)

    def backward_net(self):
        self.optimizer.zero_grad()
        self.scaler.scale(self.loss_total).backward() 
        # self.loss_total.backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def optimize_parameters(self):
        self.forward()
        # update 
        self.set_requires_grad(self.netE, True)
        self.set_requires_grad(self.netA, True)
        self.set_requires_grad(self.netC, True)
        self.backward_net()

    def get_current_attention(self):
        return self.mask

    def get_current_feature(self):
        return self.new_feat, self.pred

    def load_component(self, pretrain_folder, epoch, component=['E', 'A']):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in component:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if not isinstance(net, list):
                    if not self.opt.ema:
                        load_filename = 'epoch_%s_net_%s.pth' % (epoch, name)
                    else:
                        load_filename = 'ema_epoch_%s_net_%s.pth' % (epoch, name)
                    load_path = os.path.join(pretrain_folder, load_filename)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                    if 'I' in state_dict:
                        I = torch.Tensor(1, self.num_classes, 1024)
                        nn.init.xavier_uniform_(I)
                        state_dict['I'] = I
                    net.load_state_dict(state_dict)
                else:
                    for i, net_item in enumerate(net):
                        if not self.opt.ema:
                            load_filename = 'epoch_%s_net_%s_%s.pth' % (epoch, name, i)
                        else:
                            load_filename = 'ema_epoch_%s_net_%s_%s.pth' % (epoch, name, i)
                        
                        load_path = os.path.join(pretrain_folder, load_filename)
                        if isinstance(net_item, torch.nn.DataParallel):
                            net_item = net_item.module
                        print('loading the model from %s' % load_path)
                        # if you are using PyTorch newer than 0.4 (e.g., built from
                        # GitHub source), you can remove str() on self.device
                        state_dict = torch.load(load_path, map_location=str(self.device))
                        if hasattr(state_dict, '_metadata'):
                            del state_dict._metadata
                        net_item.load_state_dict(state_dict)

