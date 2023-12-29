import time
import os
import numpy as np
import torch

from data import create_dataset
from models import create_model
from options.test_options import TestOptions

from util.coco_preprocess import AveragePrecisionMeter, NewAveragePrecisionMeter

def display_result(ap_meter):
    map = 100 * ap_meter.value().mean()
    # map = 100 * ap_meter.value()
    # map=0
    OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(3)
    print('Test: \t mAP {map:.3f}'.format(map=map))
    print('OP: {OP:.4f}\t'
          'OR: {OR:.4f}\t'
          'OF1: {OF1:.4f}\t'
          'CP: {CP:.4f}\t'
          'CR: {CR:.4f}\t'
          'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
    print('OP_3: {OP:.4f}\t'
          'OR_3: {OR:.4f}\t'
          'OF1_3: {OF1:.4f}\t'
          'CP_3: {CP:.4f}\t'
          'CR_3: {CR:.4f}\t'
          'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

    return map

def main():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    # opt.num_threads = 0  # test code only supports num_threads = 1
    #opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    no_of_test = len(dataset)
    all_mess = ""
    print("test data length: %d" % len(dataset))

    ap_meter = AveragePrecisionMeter()
    # ap_meter = NewAveragePrecisionMeter(n_class=opt.num_classes)
    ap_meter.reset()
    start_time = time.time()
    for i, data in enumerate(dataset):
        #print(data['img'][0].size())
        if i%100 ==0:
            print('image %d'%(i))
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference and calculate metrics
        pred, gt_label = model.get_current_results()
        pred = pred.squeeze()
        pred = pred.view(gt_label.size(0),opt.num_classes)
        pred = torch.sigmoid(pred)
        ap_meter.add(pred.data, gt_label)

    result_2 = display_result(ap_meter)
    file_save = os.path.join('./checkpoints', opt.name + '_ap_class.txt')
    np.savetxt(file_save, ap_meter.value().numpy(), fmt='%.5f')
    print('time taken for testing: %.6f' %  (time.time() - start_time))
if __name__ == '__main__':
    main()
