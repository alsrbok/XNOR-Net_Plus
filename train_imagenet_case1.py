import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import gc

#sys.path.append("../")
from utils import *
from torchvision import datasets, transforms
from torch.autograd import Variable
from XNOR_Net_Plus_model import xnornetplus18_case1

from util.progress.progress.bar import Bar as Bar

parser = argparse.ArgumentParser("xnornetplus18_case1")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--save', type=str, default='./models/imagenet', help='path for saving trained models')
parser.add_argument('--data', metavar='DIR', default='/home/data/imagenet/images/', help='path to dataset')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

CLASSES = 1000

if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled=True
    logging.info("args = %s", args)

    # load model
    model = xnornetplus18_case1().cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or pname=='classifier.0.weight' or pname == 'classifier.0.bias':
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate)
   
    start_epoch = 0
    best_top1_acc= 0
    
    #Load pretrained weight from BAN_network
    pre_trained_tar = os.path.join(args.save, 'model_best_imagenet_BAN.pth.tar')
    if os.path.exists(pre_trained_tar):
        logging.info('loading best_model_of_BAN {} ..........'.format(pre_trained_tar))
        pre_train = torch.load(pre_trained_tar)
        model.load_state_dict(pre_train['state_dict'], strict=False)


     #Load the checkpoint_tar
    checkpoint_tar = os.path.join(args.save, 'checkpoint_imagenet.pth.tar')
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch'] + 1
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))


    # load training data
    traindir = os.path.join(args.data, 'train') # You should insert -data args for location of data
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data augmentation
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        adjust_learning_rate(optimizer, epoch, args)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch: [{epoch} | {args.epochs}] LR: {current_lr:.3e}")
        accuracy_file = open('imagenet_accuracy.txt', 'a')
        print(f"\nEpoch: [{epoch} | {args.epochs}] LR: {current_lr:.3e}", file=accuracy_file, flush=True)
        accuracy_file.close()

        train_loss, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, criterion_smooth, optimizer)
        valid_loss, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

        #adjust_learning_rate(optimizer, epoch, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save, type=3)

        epoch += 1

    training_time = (time.time() - start_t) / 36000
    print('total training time = {} hours'.format(training_time))
    accuracy_file = open('imagenet_accuracy.txt', 'a')
    print('total training time = {} hours'.format(training_time), file=accuracy_file, flush=True)
    accuracy_file.close()


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    len_trainloader = len(train_loader)
    bar = Bar('Processing', max=len_trainloader)

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        batch_time.update(time.time() - end)
        end = time.time()

        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            torch.clamp(param.grad, min=-5.0, max=5.0)
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #progress.display(i)
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=i + 1,
            size=len_trainloader,
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    accuracy_file = open('imagenet_accuracy.txt', 'a')
    print(bar.suffix, file=accuracy_file, flush=True)
    accuracy_file.close()
    bar.finish()

    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluation mode
    model.eval()

    end = time.time()

    len_valloader = len(val_loader)
    bar = Bar('Processing', max=len_valloader)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            data_time.update(time.time() - end)

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #progress.display(i)
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=i + 1,
                size=len_valloader,
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        accuracy_file = open('imagenet_accuracy.txt', 'a')
        print(bar.suffix, file=accuracy_file, flush=True)
        
        bar.finish()

        print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5), file=accuracy_file, flush=True)
        accuracy_file.close()

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
