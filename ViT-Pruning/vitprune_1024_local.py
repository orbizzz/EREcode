import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from einops import rearrange
import wandb

from models.vit import ViT, channel_selection, Attention, FeedForward
from models.vit11 import ViT11, channel_selection2
from models.vit_slim import ViT_slim
from models.vit11_slim import ViT11_slim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Pruning')
parser.add_argument('-s', '--select', default=0, help='select pruned method type', type=int)
parser.add_argument('-p','--percent', default=0.5, help='select sparsity level', type=float)
args = parser.parse_args()

model = ViT11(
    in_c=3,
    img_size = 32,
    patch = 4,
    num_classes = 10,
    hidden = 384,                  # 512
    num_layers = 7,
    head = 8,
    mlp_hidden = 384*4,
    dropout = 0.1,
    is_cls_token=True
)
model = model.to(device)

model_path = "/content/drive/MyDrive/ViT-cifar10-pruning/checkpoint/vit11-4-ckpt_l1magnitude_cos.t7"
print("=> loading checkpoint '{}'".format(model_path))
checkpoint = torch.load(model_path)
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['acc']
model.load_state_dict(checkpoint['net'])
print("=> loaded checkpoint '{}' (epoch {})\n Prec1: {:f}".format(model_path, checkpoint['epoch'], best_prec1))



select = args.select   # 0: mag, 1: mag&fisher, 2: fisher, 3: taylor 
percent = args.percent
glob = 0

total = 0
# total_2 = 0
for m in model.modules():
    if isinstance(m, channel_selection2):
        total += m.indexes.data.shape[0]

    # if isinstance(m, Attention):
    #     total_2 += m.to_q_score.data.shape[0] * m.to_q_score.data.shape[1]
    #     total_2 += m.to_k_score.data.shape[0] * m.to_k_score.data.shape[1]
    #     total_2 += m.to_q_score.data.shape[0] * m.to_q_score.data.shape[1]
    #     total_2 += m.to_out_score.data.shape[0] * m.to_out_score.data.shape[1]
    # if isinstance(m, FeedForward):
    #     total_2 += m.net1_score.data.shape[0] * m.net1_score.data.shape[1]
    #     total_2 += m.net2_score.data.shape[0] * m.net2_score.data.shape[1]

bn = torch.zeros(total)
# bn_2 = torch.zeros(total_2)
index = 0
# index_2 = 0
with torch.no_grad():
    for m in model.modules():
        #magnitude pruning
        if select == 0:
            if isinstance(m, channel_selection2):
                size = m.indexes.data.shape[0]
                bn[index:(index+size)] = m.indexes.data.abs().clone()
                index += size

        # movement pruning
        if select == 1:
            if isinstance(m, channel_selection2):
                # size = m.indexes.data.shape[0]
                # weight_clone = m.indexes.data.clone()
                # hessian_inverse = torch.linalg.inv(m.hessian_diagonal)
                # hessian_diagonal_inverse = torch.zeros(size)
                # for i in range(size):
                #     hessian_diagonal_inverse[i] = hessian_inverse[i][i]
                # bn[index:(index+size)] = (-0.5 * (weight_clone.mul_(weight_clone)).div_(hessian_diagonal_inverse.cpu()))
                # print(bn[index:(index+size)])
                size = m.indexes.data.shape[0]
                weight_clone = m.indexes.data.clone()
                hessian_inverse = torch.linalg.inv(m.hessian_diagonal.data)
                for i in range(size):
                    bn[index + i] = 1000 * 0.5 * weight_clone[i] * weight_clone[i] / hessian_inverse[i][i]
                bn[index:(index+size)] = bn[index:(index+size)].add_(m.indexes.data.abs().cpu())


                index += size

        # WoodFisher
        if select == 2:
            if isinstance(m, channel_selection2):
                size = m.indexes.data.shape[0]
                weight_clone = m.indexes.data.clone()
                hessian_inverse = torch.linalg.inv(m.hessian_diagonal)
                for i in range(size):
                    bn[index + i] = 0.5 * weight_clone[i] * weight_clone[i] / hessian_inverse[i][i]
                
                index += size

        # WoodTaylor
        if select == 3:
            if isinstance(m, channel_selection2):
                size = m.indexes.data.shape[0]
                weight_clone = m.indexes.data.clone()
                grad_clone = m.grads.clone()
                hessian_inverse = torch.linalg.inv(m.hessian_diagonal)
                inverse_grad = hessian_inverse.matmul(grad_clone)
                for i in range(size):
                    bn[index + i] = -0.5 * weight_clone[i] * weight_clone[i] / hessian_inverse[i][i] - 0.5 * inverse_grad[i] * inverse_grad[i] / hessian_inverse[i][i] + grad_clone[i] * inverse_grad[i] / hessian_inverse[i][i]
                index += size

        # if isinstance(m, Attention):
        #     size_q = m.to_q_score.data.shape[0] * m.to_q_score.data.shape[1]
        #     size_k = m.to_k_score.data.shape[0] * m.to_k_score.data.shape[1]
        #     size_v = m.to_v_score.data.shape[0] * m.to_v_score.data.shape[1]
        #     size_out = m.to_out_score.data.shape[0] * m.to_out_score.data.shape[1]
        #     bn_2[index_2:(index_2+size_q)] = m.to_q_score.data.flatten(0)
        #     index_2 += size_q
        #     bn_2[index_2:(index_2+size_k)] = m.to_k_score.data.flatten(0)
        #     index_2 += size_k
        #     bn_2[index_2:(index_2+size_v)] = m.to_v_score.data.flatten(0)
        #     index_2 += size_v
        #     bn_2[index_2:(index_2+size_out)] = m.to_out_score.data.flatten(0)
        #     index_2 += size_out
        # if isinstance(m, FeedForward):
        #     size_1 = m.net1_score.data.shape[0] * m.net1_score.data.shape[1]
        #     size_2 = m.net2_score.data.shape[0] * m.net2_score.data.shape[1]
        #     bn_2[index_2:(index_2+size_1)] = m.net1_score.data.flatten(0)
        #     index_2 += size_1
        #     bn_2[index_2:(index_2+size_2)] = m.net2_score.data.flatten(0)
        #     index_2 += size_2

# percent_2 = 0.5
y, i = torch.sort(bn)
# y_2, i_2 = torch.sort(bn_2)
thre_index = int(total * percent)
# thre_index_2 = int(total_2 * percent_2)
thre = y[thre_index]
# thre_2 = y_2[thre_index_2]

print(thre)
# print(thre_2)

pruned = 0
# pruned_2 = 0
cfg = []
cfg_mask = []
with torch.no_grad():
    for k, m in enumerate(model.modules()):
        # if isinstance(m, Attention):
        #     weight_q_copy = m.to_q_score.data.clone()
        #     mask_q = weight_q_copy.gt(thre_2).float().cuda()
        #     m.to_q.weight.mul_(mask_q)
        #     pruned_2 += mask_q.shape[0] - torch.sum(mask_q)
        #     weight_k_copy = m.to_k_score.data.clone()
        #     mask_k = weight_k_copy.gt(thre_2).float().cuda()
        #     m.to_k.weight.mul_(mask_k)
        #     pruned_2 += mask_k.shape[0] - torch.sum(mask_k)
        #     weight_v_copy = m.to_v_score.data.clone()
        #     mask_v = weight_v_copy.gt(thre_2).float().cuda()
        #     m.to_v.weight.mul_(mask_v)
        #     pruned_2 += mask_v.shape[0] - torch.sum(mask_v)
        #     weight_out_copy = m.to_out_score.data.clone()
        #     mask_out = weight_out_copy.gt(thre_2).float().cuda()
        #     m.to_out[0].weight.mul_(mask_out)
        #     pruned_2 = pruned_2 + mask_out.shape[0] - torch.sum(mask_out)

        # if isinstance(m, FeedForward):
        #     weight_1_copy = m.net1_score.data.clone()
        #     mask_1 = weight_1_copy.gt(thre_2).float().cuda()
        #     m.net1[0].weight.mul_(mask_1)
        #     pruned_2 += mask_1.shape[0] - torch.sum(mask_1)
        #     weight_2_copy = m.net2_score.data.clone()
        #     mask_2 = weight_2_copy.gt(thre_2).float().cuda()
        #     m.net2[0].weight.mul_(mask_2)
        #     pruned_2 = pruned_2 + mask_2.shape[0] - torch.sum(mask_2)


        if isinstance(m, channel_selection2):
            #print(k)
            #print(m)
            if k in [12, 31, 50, 69, 88, 107, 126]:
                if select ==  0:
                    weight_copy = m.indexes.data.abs().clone() # magnitude
                
                if select == 1:
                    # size = m.indexes.data.shape[0]
                    # weight_clone = m.indexes.data.clone()
                    # hessian_inverse = torch.linalg.inv(m.hessian_diagonal).cpu()
                    # hessian_diagonal_inverse = torch.zeros(size)
                    # for i in range(size):
                    #     hessian_diagonal_inverse[i] = hessian_inverse[i][i]
                    # weight_copy = m.indexes.data.abs().clone().add_(-0.5 * (weight_clone.mul_(weight_clone)).div_(hessian_diagonal_inverse)) 
                    size = m.indexes.data.shape[0]
                    weight_clone = m.indexes.data.clone()
                    hessian_inverse = torch.linalg.inv(m.hessian_diagonal.data)
                    weight_copy = torch.zeros(size)
                    for i in range(size):
                        weight_copy[i] = 1000* 0.5 * weight_clone[i] * weight_clone[i] / hessian_inverse[i][i]
                    weight_copy = weight_copy.add_(m.indexes.data.abs().cpu())
                    

                    # movement
                
                if select == 2:
                    size = m.indexes.data.shape[0]
                    weight_clone = m.indexes.data.clone()
                    hessian_inverse = torch.linalg.inv(m.hessian_diagonal)
                    weight_copy = torch.zeros(size)
                    for i in range(size):
                        weight_copy[i] = 0.5 * weight_clone[i] * weight_clone[i] / hessian_inverse[i][i] # 피셔
                
                if select == 3:
                    size = m.indexes.data.shape[0]
                    weight_clone = m.indexes.data.clone()
                    grad_clone = m.grads.clone()
                    hessian_inverse = torch.linalg.inv(m.hessian_diagonal)
                    inverse_grad = hessian_inverse.matmul(grad_clone)
                    weight_copy = torch.zeros(size)
                    for i in range(size):
                        weight_copy[i] = -0.5 * weight_clone[i] * weight_clone[i] / hessian_inverse[i][i] - 0.5 * inverse_grad[i] * inverse_grad[i] / hessian_inverse[i][i] + grad_clone[i] * inverse_grad[i] / hessian_inverse[i][i]
                
                mask = weight_copy.gt(thre).float().cuda()
                thre_ = thre.clone()
                while (torch.sum(mask)%8 !=0 or torch.sum(mask) < 8):                       # heads
                    thre_ = thre_ - 0.00001
                    mask = weight_copy.gt(thre_).float().cuda()
            else:
                if select ==  0:
                    weight_copy = m.indexes.data.abs().clone() # magnitude
                
                if select == 1:
                    # size = m.indexes.data.shape[0]
                    # weight_clone = m.indexes.data.clone()
                    # hessian_inverse = torch.linalg.inv(m.hessian_diagonal).cpu()
                    # hessian_diagonal_inverse = torch.zeros(size)
                    # for i in range(size):
                    #     hessian_diagonal_inverse[i] = hessian_inverse[i][i]
                    # weight_copy = m.indexes.data.abs().clone().add_(-0.5 * (weight_clone.mul_(weight_clone)).div_(hessian_diagonal_inverse)) 
                    # movement
                    size = m.indexes.data.shape[0]
                    weight_clone = m.indexes.data.clone()
                    hessian_inverse = torch.linalg.inv(m.hessian_diagonal.data)
                    weight_copy = torch.zeros(size)
                    for i in range(size):
                        weight_copy[i] = 1000 * 0.5 * weight_clone[i] * weight_clone[i] / hessian_inverse[i][i]
                    weight_copy = weight_copy.add_(m.indexes.data.abs().cpu())
                
                if select == 2:
                    size = m.indexes.data.shape[0]
                    weight_clone = m.indexes.data.clone()
                    hessian_inverse = torch.linalg.inv(m.hessian_diagonal)
                    weight_copy = torch.zeros(size)
                    for i in range(size):
                        weight_copy[i] = 0.5 * weight_clone[i] * weight_clone[i] / hessian_inverse[i][i] # 피셔
                
                if select == 3:
                    size = m.indexes.data.shape[0]
                    weight_clone = m.indexes.data.clone()
                    grad_clone = m.grads.clone()
                    hessian_inverse = torch.linalg.inv(m.hessian_diagonal)
                    inverse_grad = hessian_inverse.matmul(grad_clone)
                    weight_copy = torch.zeros(size)
                    for i in range(size):
                        weight_copy[i] = -0.5 * weight_clone[i] * weight_clone[i] / hessian_inverse[i][i] - 0.5 * inverse_grad[i] * inverse_grad[i] / hessian_inverse[i][i] + grad_clone[i] * inverse_grad[i] / hessian_inverse[i][i]
                
                mask = weight_copy.gt(thre).float().cuda()
                thre_ = thre.clone()
                while(torch.sum(mask) < 8):
                    thre_ = thre_ - 0.00001
                    mask = weight_copy.gt(thre_).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.indexes.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total
print(pruned_ratio)
# pruned_ratio_2 = pruned_2/total_2
# print(pruned_ratio_2)
print('Pre-processing Successful!')
for num in cfg:
    print("{}".format(num), end=" ")


def test(model):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='/home/lxc/ABCPruner/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

test(model)
cfg_prune = []
for i in range(len(cfg)):
    if i%2!=0:
        cfg_prune.append([cfg[i-1],cfg[i]])
newmodel = ViT11_slim(in_c=3,
    img_size = 32,
    patch = 4,
    num_classes = 10,
    hidden = 384,                  # 512
    num_layers = 7,
    head = 8,
    mlp_hidden = 384*4,
    dropout = 0.1,
    is_cls_token=True,
    cfg=cfg_prune)

newmodel.to(device)
# num_parameters = sum([param.nelement() for param in newmodel.parameters()])
# print(num_parameters)
newmodel_dict = newmodel.state_dict().copy()

i = 0
newdict = {}
# 실제 사이즈 축소화(사이즈 크기 축소화)
for k,v in model.state_dict().items():
    if 'mlp1.0.weight' in k:
        #print(k)
        #print(v)
        #print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'mlp1.0.bias' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    # elif 'net1_score' in k:
    #     continue
    # elif 'net2_score' in k:
    #     continue
    # elif 'to_q_score' in k:
    #     continue
    # elif 'to_k_score' in k:
    #     continue
    # elif 'to_v_score' in k:
    #     continue
    # elif 'to_out_score' in k:
    #     continue
    elif 'grads' in k:
        continue
    elif 'hessian_diagonal' in k:
        continue

    elif 'msa.q.weight' in k or 'msa.k.weight' in k or 'msa.v.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'mlp2.0.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:,idx.tolist()].clone()
        i = i + 1
    elif 'msa.o.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:,idx.tolist()].clone()
        i = i + 1

    elif k in newmodel.state_dict():
        newdict[k] = v

model_num_parameters = sum([param.nelement() for param in model.parameters()])
newmodel_num_parameters = sum([param.nelement() for param in newmodel.parameters()])
print(model_num_parameters, newmodel_num_parameters)

newmodel_dict.update(newdict)
newmodel.load_state_dict(newmodel_dict)

if select == 0:
    model_name = "pruned_new_movement_cos_mag_{}.pth".format(percent)
    torch.save(newmodel.state_dict(), model_name)
if select == 1:
    model_name = "pruned_new_movement_cos_magfisher_{}.pth".format(percent)
    torch.save(newmodel.state_dict(), model_name)
if select == 2:
    model_name = "pruned_new_movement_cos_fisher_{}.pth".format(percent)
    torch.save(newmodel.state_dict(), model_name)
if select == 3:
    model_name = "pruned_new_movement_cos_taylor_{}.pth".format(percent)
    torch.save(newmodel.state_dict(), model_name)
print('after pruning: ', end=' ')
test(newmodel)
