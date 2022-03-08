# simple test for torch.gpu
'''
Author: Shenghao Qiu
Last Edit: 10/12/2021
'''
import time
import os
import sys
import numpy as np
import pandas as pd
import parser
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from utils import progress_bar, data_loader, train_val_dataset

from timm.models import create_model
import os
os.environ['TORCH_HOME']='/home/shenghao/pretrained_model'

# 虞美人
# 文章代码何时了，无奈天知晓
# 近邻夜半又欢歌，身倦思枯平躺望楼奢
# 红颜挚友应苏醒，何故卿难寝
# 问君真理几多思？不若将期情恋半分痴。

datasets_name = ['ImageNet', 'CIFAR10']

parser = argparse.ArgumentParser(description='PyTorch ImageNet/CIFAR10 Training')
parser.add_argument('--dataset', '-d', metavar='DATA', default='CIFAR10',
                    choices=datasets_name,
                    help='datasets chosen: ' + ' | '.join(datasets_name) +
                    ' (default: CIFAR10)')

parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'convnext_small'], help='models choosing')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--use-cuda', default=True, action='store_true', help='use GPU CUDA for training improvement (default: True)')
parser.add_argument('--device', default='cuda:0', type=str, help='If use-cuda is True then choose which device to use (default: cuda:0)')
parser.add_argument('--pretrain', default=False, action='store_true', help='use pre-trained models? (default: False)')
parser.add_argument('--path', default='../dataset/ImageNet', type=str, metavar='PATH', help='path to the dataset, (default: ImageNet)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--use-val', default=False, action='store_true',
                                 help='Since the training set is big, so this is to determine if only validation set '
                                      'is required (default: False)')

args = parser.parse_args()

device = args.device if torch.cuda.is_available() and args.use_cuda else 'cpu'

# ImageNet dataset prepare
if args.dataset == 'ImageNet':
    print('==> Preparing data ImageNet..')
    train_path = os.path.join(args.path, 'train/')
    val_path = os.path.join(args.path, 'val/')
    save_dir = './'
    best_prec = 0.0

    data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    if args.use_val:
        # Optional: use val_dataset as the training dataset for shorter training time
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=data_transform)
        datasets = train_val_dataset(val_dataset, val_split=0.25)
        dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
                       for x in ['train', 'val']}
        trainloader = dataloaders['train']
        testloader = dataloaders['val']
    
    else:
        # Data loading
        trainloader, testloader = data_loader(args.path, args.batch_size, args.workers, True)

# CIFAR10 dataset prepare
elif args.dataset == 'CIFAR10':
    print('==> Preparing data CIFAR10..')
    save_dir = './'
    best_prec = 0.0

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
# net = models.resnet18(pretrained=args.pretrain)
net = create_model('convnext_small', pretrained=True, num_classes=1000, drop_path_rate=0, head_init_scale=1.0)
#net = getattr(models, args.model)(pretrained=args.pretrain)
print(args.model)
net = net.to(device)
best_acc = 0
start_epoch = 0

# GPU Info
if device != "cpu":
    print("Current Device Num.: {}, CUDA Device Name: {}, Devices Count: {}".format(torch.cuda.current_device(), torch.cuda.get_device_name(0) ,torch.cuda.device_count()))

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/{}_ori_ckpt.pth'.format(args.model))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Training[{}/{}] Loss: {:.2f} | Acc: {:.2f}%'.format(batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            print('Testing[{}/{}] Loss: {:.2f} | Acc: {:.2f}%'.format(batch_idx, len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_ori_ckpt.pth'.format(args.model))
        best_acc = acc
        ############# Two New Saving Method ####################
        torch.save(net, './checkpoint/{}_ckpt.pth'.format(args.model))
        torch.save(net.state_dict(), './checkpoint/{}_stat.pth'.format(args.model))

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()



sourceTensor.clone().detach()
sourceTensor.clone().detach().requires_grad_(True)
