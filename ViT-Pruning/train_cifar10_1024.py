# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import wandb

from models import *
from models.vit import ViT, channel_selection, Attention, FeedForward
from models.vit11 import ViT11, channel_selection2
from utils import progress_bar, CosineAnnealingWarmupRestarts, CIFAR10Policy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit11')
parser.add_argument('--bs', default='128')
parser.add_argument('--n_epochs', type=int, default='1000')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--cos', action='store_true', help='Train with cosine annealing scheduling')
args = parser.parse_args()

# wandb.init(project='InitialLearning_movement_cos', config={
#     "batch_size": args.bs,
#     "num_epochs": args.n_epochs,
#     "lr": args.lr
# })

# if args.cos:
#     from warmup_scheduler import GradualWarmupScheduler
# if args.aug:
#     import albumentations
bs = int(args.bs)
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/lxc/ABCPruner/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='/home/lxc/ABCPruner/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = 32,
    patch_size = args.patch,
    num_classes = 10,
    dim = 512,                  # 512
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1)
elif args.net =="vit11":
    net = ViT11(
    in_c=3,
    img_size = 32,
    patch = args.patch,
    num_classes = 10,
    hidden = 384,                  # 512
    num_layers = 7,
    head = 8,
    mlp_hidden = 384*4,
    dropout = 0.1,
    is_cls_token=True
)

net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net) # make parallel
#     cudnn.benchmark = True
# cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-4-ckpt_l1magnitude_cos.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'], strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()
# reduce LR on Plateau
if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)    
elif args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)

scheduler = CosineAnnealingWarmupRestarts(optimizer, 100, 1, 0.001, 0.00001, 0, 0.9, -1)

wandb.init(project='InitialLearning', config={
    "batch_size": args.bs,
    "num_epochs": args.n_epochs
})


def sparse_selection():
    s = 1e-4 # 1e-4
    for m in net.modules():
        if isinstance(m, channel_selection2):
            m.indexes.grad.data.add_(s*torch.sign(m.indexes.data))
            # m.indexes.grad.data.mul_(-1*m.indexes.data)  # L1

def hes_cal(loss):
    for m in net.modules():
        if isinstance(m, channel_selection2):
            grads = torch.autograd.grad(loss, m.indexes, create_graph=True)
            m.grads.data.add_(grads[0])
            # arr = np.zeros((grads[0].shape[0], grads[0].shape[0]))
            for i, grad in enumerate(grads[0]):
                hess = torch.autograd.grad(grad, m.indexes, retain_graph=True)
                m.hessian_diagonal.data[i].add_(hess[0])

##### Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # if(epoch == 999):
    #     for m in net.modules():
    #         if isinstance(m, channel_selection2):
    #             dim_1=m.grads.data.shape[0]
    #             m.grads.data = torch.zeros(dim_1).to(device)
    #             m.hessian_diagonal.data = torch.zeros((dim_1, dim_1)).to(device)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # grad_cal(loss)
        if(epoch==999):
            if(batch_idx % 200 == 0):
                hes_cal(loss)
        loss.backward()
        sparse_selection()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss/(batch_idx+1)

net_param = sum([param.nelement() for param in net.parameters()])
##### Validation
import time
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

            # if batch_idx % 10 == 9:
            #     wandb.log({
            #             "epoch": epoch,
            #             "step": batch_idx,
            #             "Loss": test_loss/(batch_idx+1),
            #             "Acc": 100.*correct/total,
            #             "Parameter": net_param,
            #             "Learning_Rate": optimizer.param_groups[0]["lr"]
            #         })

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    # if not args.cos:
    #     scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    wandb.log({
      "epoch": epoch,
      "Loss": test_loss,
      "Acc": acc,
      "Parameter": net_param,
      "Learning_Rate": optimizer.param_groups[0]["lr"]
    })
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt_l1magnitude_cos.t7'.format(args.patch))
        best_acc = acc
    if epoch ==999:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/'+args.net+'-4-ckpt_l1magnitude_cos_{}.t7'.format(epoch))
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []
for epoch in range(start_epoch, args.n_epochs):
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch)
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # write as csv for analysis
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    # print(list_loss)
    
    
