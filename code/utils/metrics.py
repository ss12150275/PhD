# -*- coding:utf-8 -*-
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    # log_p = log_p.view(target.numel())
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)


    target = target.view(target.numel())
    print('log_p',log_p.size(),target.size())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

def weight_cross_entropy_3D(input, targets):
    num_sum = torch.sum(targets, dim=(0, 2, 3, 4))
    w = torch.Tensor([0, 0])#.cuda()
    w[0 ] = 1
    w[1] = (torch.sum(num_sum) - num_sum[1] + 1)/(num_sum[1] + 1)
    print(w)
    class_weight = targets
    for i in range(targets.size(1)):
        class_weight[:,i,:,:,:] = class_weight[:,i,:,:,:] + w[i]
    loss = nn.BCELoss(weight=class_weight)(input, targets)
    return loss

def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    print('输入的大小',n, c, h, w, s)
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class Dice_coff_loss(nn.Module):
    def __init__(self):
        super(Dice_coff_loss, self).__init__()
    def forward(self, logits, targets):
        dice_loss = 1 - dice_coeff(logits, targets)
        return dice_loss

class DiceMean(nn.Module):
    def __init__(self):
        super(DiceMean, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return dice_sum / class_num





class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)
        # print('class_number',class_num)
        # print('target,logits',logits.shape,targets.shape)
        g = targets.cpu().detach().numpy()
        p = logits.cpu().detach().numpy()
        # print('2222',p.shape,g.shape)


        # cv2.imshow('one-label-0', np.array(g[0, 0, :, :, 16]))
        # cv2.waitKey(3)
        # cv2.imshow('pred-0', np.array(p[0, 0, :, :, 16]))
        # cv2.waitKey(3)
        # cv2.imshow('one-label-1', np.array(g[0, 1, :, :, 16]))
        # cv2.waitKey(3)
        # cv2.imshow('pred-1', np.array(p[0, 1, :, :, 16]))
        # cv2.waitKey(3)
        # cv2.imshow('one-label-2', np.array(g[0, 1, :, :, 16]))
        # cv2.waitKey(3)
        # cv2.imshow('pred-2', np.array(p[0, 1, :, :, 16]))
        # cv2.waitKey(3)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            # print('inter--union',inter,union)
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - dice_sum / class_num

def GUI(pred, target):
    class_num = pred.size(1)
    print('class_number', class_num)
    dice_sum = 0
    for i in range(class_num):
        # 计算距离图
        print('F.mse_loss',F.mse_loss(pred[:, i, :, :, :], target[:, i, :, :, :], reduction='none').sum(dim=1, keepdim=True))
        dist = torch.sqrt(F.mse_loss(pred[:, i, :, :, :], target[:, i, :, :, :], reduction='none').sum(dim=1, keepdim=True)+0.000001)
        print('------dist:',dist)
        weight = dist*target[:, i, :, :, :] + (1-target[:, i, :, :, :])
        #when weight = 1, this loss becomes Root Tversky loss
        smooth = 1.0
        alpha = 0.1 #alpha=0.1 in stage1 and 0.2 in stage2
        beta = 1 - alpha
        sigma1 = 0.0001
        sigma2 = 0.0001
        weight_i = target[:, i, :, :, :]*sigma1 + (1-target[:, i, :, :, :])*sigma2
        intersection = (weight*((pred[:, i, :, :, :]+weight_i)**0.7)*target[:, i, :, :, :]).sum()
        intersection2 = (weight*(alpha*pred[:, i, :, :, :] + beta*target[:, i, :, :, :])).sum()
        return 1-(intersection + smooth)/(intersection2 + smooth)

class DiceMeanLoss2d(nn.Module):
    def __init__(self):
        super(DiceMeanLoss2d, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)
        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :] * targets[:, i, :, :])
            union = torch.sum(logits[:, i, :, :]) + torch.sum(targets[:, i, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - dice_sum / class_num

class WeightDiceLoss(nn.Module):
    def __init__(self):
        super(WeightDiceLoss, self).__init__()

    def forward(self, logits, targets):



        num_sum = torch.sum(targets, dim=(0, 2, 3, 4))
        w = torch.Tensor([0, 0]).cuda()
        for i in range(targets.size(1)):
            if (num_sum[i] < 1):
                w[i] = 0
            else:
                w[i] = (0.1 * num_sum[i] + num_sum[i - 1] + 1) / (torch.sum(num_sum) + 1)
        print(w)
        inter = w * torch.sum(targets * logits, dim=(0, 2, 3, 4))
        inter = torch.sum(inter)

        union = w * torch.sum(targets + logits, dim=(0, 2, 3, 4))
        union = torch.sum(union)

        return 1 - 2. * inter / union


def dice(logits, targets, class_index):

    if (len(targets.size()) == 5):
        inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
        union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    else:
        # print('dice 4 channel')
        inter = torch.sum(logits[:, class_index, :, :] * targets[:, class_index, :, :])
        union = torch.sum(logits[:, class_index, :, :]) + torch.sum(targets[:, class_index, :, :])
    smooth = 10E-20
    dice = (2. * inter + smooth) / (union + smooth)
    return dice

def dice_coeff(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output)#.view(-1).data.cpu().numpy()
    target = target#.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

def dice_coff_loss(output, target):
    1 - dice_coeff(output, target)


def T(logits, targets):
    if (len(targets.size()) == 5):
        return torch.sum(targets[:, 1, :, :, :])
    else:
        return torch.sum(targets[:, 1, :, :])



def P(logits, targets):
    if (len(logits.size()) == 5):
        return torch.sum(logits[:, 1, :, :, :])
    else:
        return torch.sum(logits[:, 1, :, :])


def TP(logits, targets):
    if (len(logits.size()) == 5):
        return torch.sum(targets[:, 1, :, :, :] * logits[:, 1, :, :, :])
    else:
        return torch.sum(targets[:, 1, :, :] * logits[:, 1, :, :])

