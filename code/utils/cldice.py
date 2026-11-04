import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import soft_skel
# from utils.metrics import *

def Dicemeanloss(logits,targets):
    class_num = logits.size(1)
    dice_sum = 0
    for i in range(class_num):
        inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
        union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
        dice = (2. * inter + 1) / (union + 1)
        dice_sum += dice
    return 1 - dice_sum / class_num


class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self,y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.mul(skel_pred, y_true)[:,1:,:,:,:])+self.smooth)/(torch.sum(skel_pred[:,1:,:,:,:])+self.smooth)
        tsens = (torch.sum(torch.mul(skel_true, y_pred)[:,1:,:,:,:])+self.smooth)/(torch.sum(skel_true[:,1:,:,:,:])+self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred)[:,1:,:,:,:])
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true[:,1:,:,:,:]) + torch.sum(y_pred[:,1:,:,:,:]) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.2, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self,y_true, y_pred):
        dice = Dicemeanloss(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.mul(skel_pred, y_true)[:,1:,:,:,:])+self.smooth)/(torch.sum(skel_pred[:,1:,:,:,:])+self.smooth)
        tsens = (torch.sum(torch.mul(skel_true, y_pred)[:,1:,:,:,:])+self.smooth)/(torch.sum(skel_true[:,1:,:,:,:])+self.smooth)
        cl_dice = 1.- 2.0 * (tprec * tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice