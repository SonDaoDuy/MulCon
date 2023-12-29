import time
import os

import torch
import numpy as np
import torch.nn.functional as F

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util import metrics
from util import visualizer
from models.model_ema import MultiLabelModelEMAMultiC

import math

def main():
    # nomaly detection from pytorch
    #torch.autograd.set_detect_anomaly(True)
    opt = TrainOptions().parse()

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training  image = %d' % dataset_size)
    print('length of one epoch = %d' % len(dataset.dataloader))
    model = create_model(opt)
    model.setup(opt)
    #warm up learning rate for large batch training
    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True

    if opt.warm:
        rate_warmup = 0.1
        lr_parts = [opt.lr*opt.lrp, opt.lr]
        opt.warmup_from = [opt.lr*opt.lrp*rate_warmup, opt.lr*rate_warmup]
        for i in range(opt.num_classes):
            lr_parts.append(opt.lr)
            opt.warmup_from.append(opt.lr*rate_warmup)
        opt.warm_epochs = opt.niter_decay/5
        if opt.lr_policy == 'cosine':
            opt.warmup_to = []
            for i in range(len(opt.warmup_from)):
                eta_min = lr_parts[i] * (0.1 ** 3)
                opt.warmup_to.append(eta_min + (lr_parts[i] - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.niter_decay)) / 2)
        else:
            opt.warmup_to = [opt.lr*opt.lrp, opt.lr]
    # visualizer
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)
    #EMA model
    ema_model = MultiLabelModelEMAMultiC(model, 0.9997)
    #start training
    total_iters = 0
    best_score = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # training
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            #warm up for large batch size
            model.warmup_learning_rate(opt, epoch, i, len(dataset.dataloader))
            model.optimize_parameters()

            ema_model.update(model)
            
            if total_iters % opt.print_freq == 0:
                #losses
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_results(log_name, epoch, epoch_iter, losses, t_comp, t_data, best_score)


            iter_data_time = time.time()


        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            print('time taken for epoch %d: %.6f' % (epoch, time.time() - epoch_start_time))
            save_suffix = 'epoch_%d' % epoch
            model.save_networks(save_suffix)
            save_suffix_ema = 'ema_epoch_%d' % epoch
            ema_model.module.save_networks(save_suffix_ema)

        print('End of epoch: %d / %d \t iter_no: %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, total_iters, time.time() - epoch_start_time))
        #print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        if opt.warm:
            if epoch > opt.warm_epochs:
                model.update_learning_rate()
            else:
                for pg in model.optimizers[0].param_groups:
                    lr = pg['lr']
                    print('learning rate = %.7f' % lr)
        else:
            model.update_learning_rate()

if __name__ == '__main__':
    main()
