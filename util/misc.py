'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
    - accuracy: calculates accuracy.
'''
import errno
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'accuracy']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

#initialize the weight and bias
def init_params(net):
    '''Init layer parameters.'''
    #m : network's modules -> if/ elif 's Generalization
    for m in net.modules():
        #if m is a instance of Conv2d
        if isinstance(m, nn.Conv2d):
            #described in Delving deep into rectifiers  - He, K. et al. (2015),
            #fan_out : preserves the magnitudes in the backwards pass.
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias: #if bias exist
                init.constant(m.bias, 0)
        #if m is a instance of BatchNorm2d
        elif isinstance(m, nn.BatchNorm2d):
            #Initialize to weight = 1/ bias =0 
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        #if m is a instance of Linear
        elif isinstance(m, nn.Linear):
            #Use Normal Distribution for initializing
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def accuracy(output, target, topk=(1,)): 
    #top1 accuracy : best 1's percentage if it is correct
    #top5 accuracy : best 5's percentage if the answer is in there
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
