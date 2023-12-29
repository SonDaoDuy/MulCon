import os
from copy import deepcopy
import random
import time
from copy import deepcopy

import numpy as np
from torchvision import datasets as datasets
import torch

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class MultiLabelModelEMA(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(MultiLabelModelEMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.module_name = model.model_names

    def _update(self, model, update_fn):
        with torch.no_grad():
            for name in self.module_name:
                net_ema = getattr(self.module, 'net' + name)
                net = getattr(model, 'net' + name)
                for ema_v, model_v in zip(net_ema.state_dict().values(), net.state_dict().values()):
                    ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class MultiLabelModelEMAMultiC(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(MultiLabelModelEMAMultiC, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.module_name = model.model_names

    def _update(self, model, update_fn):
        with torch.no_grad():
            for name in self.module_name:
                net_ema = getattr(self.module, 'net' + name)
                net = getattr(model, 'net' + name)
                if not isinstance(net, list):
                    for ema_v, model_v in zip(net_ema.state_dict().values(), net.state_dict().values()):
                        ema_v.copy_(update_fn(ema_v, model_v))
                else:
                    for i in range(len(net)):
                        for ema_v, model_v in zip(net_ema[i].state_dict().values(), net[i].state_dict().values()):
                            ema_v.copy_(update_fn(ema_v, model_v))
    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)