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
from utils import progress_bar, CosineAnnealingWarmupRestarts

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='64')
parser.add_argument('--n_epochs', type=int, default='100')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--cos', action='store_true', help='Train with cosine annealing scheduling')
args = parser.parse_args()

# wandb.init(project='InitialLearning_l1movement_cos', config={
#     "batch_size": args.bs,
#     "num_epochs": args.n_epochs,
#     "lr": args.lr
# })

# if args.cos:
#     from warmup_scheduler import GradualWarmupScheduler
# if args.aug:
#     import albumentations
bs = int(args.bs)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/lxc/ABCPruner/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='/home/lxc/ABCPruner/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

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
    emb_dropout = 0.1
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
    net.load_state_dict(checkpoint['net'])
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

scheduler = CosineAnnealingWarmupRestarts(optimizer, 10, 1, 0.0001, 0.000001, 0, 0.9, -1)

wandb.init(project='InitialLearning_l1magnitude_cos', config={
    "batch_size": args.bs,
    "num_epochs": args.n_epochs,
    "lr": scheduler.get_lr()
})


def sparse_selection():
    s = 1e-4 # 1e-4
    # s_2 = 1e-7
    # s2 = -1e-5
    for m in net.modules():
        # if isinstance(m, Attention):
            # m.to_q.weight.grad.data.add_(s_2*torch.sign(m.to_q.weight.data))
            # m.to_k.weight.grad.data.add_(s_2*torch.sign(m.to_k.weight.data))
            # m.to_v.weight.grad.data.add_(s_2*torch.sign(m.to_v.weight.data))
            # m.to_out[0].weight.grad.data.add_(s_2*torch.sign(m.to_out[0].weight.data))
            # grad_q_copy = m.to_q.weight.grad.clone().to(device)
            # grad_k_copy = m.to_k.weight.grad.clone().to(device)
            # grad_v_copy = m.to_v.weight.grad.clone().to(device)
            # grad_out_copy = m.to_out[0].weight.grad.clone().to(device)
            # weight_q_copy = m.to_q.weight.clone().to(device)
            # weight_k_copy = m.to_k.weight.clone().to(device)
            # weight_v_copy = m.to_v.weight.clone().to(device)
            # weight_out_copy = m.to_out[0].weight.clone().to(device)
            # m.to_q_score.data = m.to_q_score.data.to(device)
            # m.to_k_score.data = m.to_k_score.data.to(device)
            # m.to_v_score.data = m.to_v_score.data.to(device)
            # m.to_out_score.data = m.to_out_score.data.to(device)
            # m.to_q_score.data.add_(s2 * grad_q_copy.mul_(weight_q_copy))
            # m.to_k_score.data.add_(s2 * grad_k_copy.mul_(weight_k_copy))
            # m.to_v_score.data.add_(s2 * grad_v_copy.mul_(weight_v_copy))
            # m.to_out_score.data.add_(s2 * grad_out_copy.mul_(weight_out_copy))
            # m.to_q_score.data = m.to_q_score.data.to('cpu')
            # m.to_k_score.data = m.to_k_score.data.to('cpu')
            # m.to_v_score.data = m.to_v_score.data.to('cpu')
            # m.to_out_score.data = m.to_out_score.data.to('cpu')
            # grad_q_copy = grad_q_copy.to('cpu')
            # grad_k_copy = grad_k_copy.to('cpu')
            # grad_v_copy = grad_v_copy.to('cpu')
            # grad_out_copy = grad_out_copy.to('cpu')
            # weight_q_copy = weight_q_copy.to('cpu')
            # weight_k_copy = weight_k_copy.to('cpu')
            # weight_v_copy = weight_v_copy.to('cpu')
            # weight_out_copy = weight_out_copy.to('cpu')
            # del grad_q_copy, grad_k_copy, grad_v_copy, grad_out_copy
            # del weight_q_copy, weight_k_copy, weight_v_copy, weight_out_copy

        # if isinstance(m, FeedForward):
            # m.net1[0].weight.grad.data.add_(s_2*torch.sign(m.net1[0].weight.data))
            # m.net2[0].weight.grad.data.add_(s_2*torch.sign(m.net2[0].weight.data))
            # grad1_copy = m.net1[0].weight.grad.clone().to(device)
            # grad2_copy = m.net2[0].weight.grad.clone().to(device)
            # weight1_copy = m.net1[0].weight.clone().to(device)
            # weight2_copy = m.net2[0].weight.clone().to(device)
            # m.net1_score.data = m.net1_score.data.to(device)
            # m.net2_score.data = m.net2_score.data.to(device)
            # m.net1_score.data.add_(s2 * grad1_copy.mul_(weight1_copy))
            # m.net2_score.data.add_(s2 * grad2_copy.mul_(weight2_copy))
            # m.net1_score.data = m.net1_score.data.to('cpu')
            # m.net2_score.data = m.net2_score.data.to('cpu')
            # grad1_copy = grad1_copy.to('cpu')
            # grad2_copy = grad2_copy.to('cpu')
            # weight1_copy = weight1_copy.to('cpu')
            # weight2_copy = weight2_copy.to('cpu')
            # del grad1_copy, grad2_copy
            # del weight1_copy, weight2_copy
        if isinstance(m, channel_selection):
            m.indexes.grad.data.add_(s*torch.sign(m.indexes.data)) # L1
            m.indexes.grad.data.mul_(-1*m.indexes.data)  # movement

# def get_second_order_grad(grads, xs):
#     start = time.time()
#     grads2 = []
#     for j, (grad, x) in enumerate(zip(grads, xs)):
#         print('2nd order on ', j, 'th layer')
#         print(x.size())
#         grad = torch.reshape(grad, [-1])
#         grads2_tmp = []
#         for count, g in enumerate(grad):
#             g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
#             g2 = torch.reshape(g2, [-1])
#             grads2_tmp.append(g2[count].data.cpu().numpy())
#         grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())).to(DEVICE_IDS[0]))
#         print('Time used is ', time.time() - start)
#     for grad in grads2:  # check size
#         print(grad.size())

#     grads2 = torch.Tensor(grads2)

#     return grads2

# def get_second_order_grad(grads, xs):
#     start = time.time()
#     grads2 = []
#     for j, (grad, x) in enumerate(zip(grads, xs)):
#         print('2nd order on ', j, 'th layer')
#         print(x.size())
#         g2 = torch.autograd.grad(grad, x, retain_graph=True)
#         grads.append(g2)
#         print('Time used is ', time.time() - start)
#     grads2 = torch.Tensor(grads2)
#     for grad in grads2:  # check size
#         print(grad.size())
#     return grads2


# def grad_cal(loss):
#     for m in net.modules():
#         if isinstance(m, channel_selection):
#             grads = torch.autograd.grad(loss, m.indexes, create_graph=True)
#             print(grads[0])
#             m.grads.data = grads[0]

# grads = torch.autograd.grad(loss, m.indexes, create_graph=True)
#             m.grads.data = grads[0]
#             arr = np.zeros(())
#             for i, grad in enumerate(grads[0]):
#                 hess = torch.autograd.grad(grad, m.indexes, retain_graph=True)
#                 hess[0].cpu().numpy()
#                 m.hessian_diagonal.data[i] = hess[0]


def hes_cal(loss):
    for m in net.modules():
        if isinstance(m, channel_selection):
            grads = torch.autograd.grad(loss, m.indexes, create_graph=True)
            m.grads.data.add_(grads[0])
            # arr = np.zeros((grads[0].shape[0], grads[0].shape[0]))
            for i, grad in enumerate(grads[0]):
                hess = torch.autograd.grad(grad, m.indexes, retain_graph=True)
                m.hessian_diagonal.data[i].add_(hess[0])

            # hess = torch.autograd.grad(m.indexes.grad)
            # print(hess)
            # m.hessian_diagonal.data = hess[0]
            # grad_clone = m.indexes.grad.clone()
            # m.grads = grad_clone.data

            # print(grads[0].shape)

            # grads2 = get_second_order_grad(grads, m.indexes)

            # m.hessian_diagonal.data = grads2.data

            

# def sparse_selection_2():
#     s2 = -1e-2
#     for m in net.modules():
#         if isinstance(m, Attention):
#             grad_q_copy = m.to_q.weight.grad.clone().to(device)
#             grad_k_copy = m.to_k.weight.grad.clone().to(device)
#             grad_v_copy = m.to_v.weight.grad.clone().to(device)
#             grad_out_copy = m.to_out[0].weight.grad.clone().to(device)
#             weight_q_copy = m.to_q.weight.clone().to(device)
#             weight_k_copy = m.to_k.weight.clone().to(device)
#             weight_v_copy = m.to_v.weight.clone().to(device)
#             weight_out_copy = m.to_out[0].weight.clone().to(device)
#             m.to_q_score.data = m.to_q_score.data.to(device)
#             m.to_k_score.data = m.to_k_score.data.to(device)
#             m.to_v_score.data = m.to_v_score.data.to(device)
#             m.to_out_score.data = m.to_out_score.data.to(device)
#             m.to_q_score.data.add_(s2 * grad_q_copy.mul_(weight_q_copy))
#             m.to_k_score.data.add_(s2 * grad_k_copy.mul_(weight_k_copy))
#             m.to_v_score.data.add_(s2 * grad_v_copy.mul_(weight_v_copy))
#             m.to_out_score.data.add_(s2 * grad_out_copy.mul_(weight_out_copy))
#             m.to_q_score.data = m.to_q_score.data.to('cpu')
#             m.to_k_score.data = m.to_k_score.data.to('cpu')
#             m.to_v_score.data = m.to_v_score.data.to('cpu')
#             m.to_out_score.data = m.to_out_score.data.to('cpu')
#             grad_q_copy = grad_q_copy.to('cpu')
#             grad_k_copy = grad_k_copy.to('cpu')
#             grad_v_copy = grad_v_copy.to('cpu')
#             grad_out_copy = grad_out_copy.to('cpu')
#             weight_q_copy = weight_q_copy.to('cpu')
#             weight_k_copy = weight_k_copy.to('cpu')
#             weight_v_copy = weight_v_copy.to('cpu')
#             weight_out_copy = weight_out_copy.to('cpu')
#             del grad_q_copy, grad_k_copy, grad_v_copy, grad_out_copy
#             del weight_q_copy, weight_k_copy, weight_v_copy, weight_out_copy
#         if isinstance(m, FeedForward):
#             grad1_copy = m.net1[0].weight.grad.clone().to(device)
#             grad2_copy = m.net2[0].weight.grad.clone().to(device)
#             weight1_copy = m.net1[0].weight.clone().to(device)
#             weight2_copy = m.net2[0].weight.clone().to(device)
#             m.net1_score.data = m.net1_score.data.to(device)
#             m.net2_score.data = m.net2_score.data.to(device)
#             m.net1_score.data.add_(s2 * grad1_copy.mul_(weight1_copy))
#             m.net2_score.data.add_(s2 * grad2_copy.mul_(weight2_copy))
#             m.net1_score.data = m.net1_score.data.to('cpu')
#             m.net2_score.data = m.net2_score.data.to('cpu')
#             grad1_copy = grad1_copy.to('cpu')
#             grad2_copy = grad2_copy.to('cpu')
#             weight1_copy = weight1_copy.to('cpu')
#             weight2_copy = weight2_copy.to('cpu')
#             del grad1_copy, grad2_copy
#             del weight1_copy, weight2_copy

##### Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # if(epoch == 30 ):
    #     for m in net.modules():
    #         if isinstance(m, channel_selection):
    #             dim_1=m.grads.data.shape[0]
    #             m.grads.data = torch.zeros(dim_1).to(device)
    #             m.hessian_diagonal.data = torch.zeros((dim_1, dim_1)).to(device)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # grad_cal(loss)
        if(epoch == 30 or epoch==40):
            if(batch_idx % 240 == 0):
                hes_cal(loss)
        loss.backward()
        sparse_selection()
        optimizer.step()
        # sparse_selection_2()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss/(batch_idx+1)

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

            if batch_idx % 10 == 0:
                wandb.log({
                        "epoch": epoch,
                        "step": batch_idx,
                        "Loss": test_loss/(batch_idx+1),
                        "Acc": 100.*correct/total,
                        "Learning_Rate": optimizer.param_groups[0]["lr"]
                    })

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    # if not args.cos:
    #     scheduler.step(test_loss)
    
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
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt_l1magnitude_cos.t7'.format(args.patch))
        best_acc = acc
    if epoch == 30 or epoch ==40:
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
    
    scheduler.step()
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # write as csv for analysis
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    # print(list_loss)
    
    
