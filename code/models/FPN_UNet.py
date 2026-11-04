#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/10/7 11:09
#@Author: dada
#@File  : FPN_UNet.py

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.non_local import *
from utils import init_util
import argparse
# from mmcv.cnn import xavier_init


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self, block, layers):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y
    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        #print(f'c1:{c1.shape}')
        c2 = self.layer1(c1)
        #print(f'c2:{c2.shape}')
        c3 = self.layer2(c2)
        #print(f'c3:{c3.shape}')
        c4 = self.layer3(c3)
        #print(f'c4:{c4.shape}')
        c5 = self.layer4(c4)
        #print(f'c5:{c5.shape}')
        # Top-down
        p5 = self.toplayer(c5)
        #print(f'p5:{p5.shape}')
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        #print(f'latlayer1(c4):{self.latlayer1(c4).shape}, p4:{p4.shape}')
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        #print(f'latlayer1(c3):{self.latlayer2(c3).shape}, p3:{p3.shape}')
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        #print(f'latlayer1(c2):{self.latlayer3(c2).shape}, p2:{p2.shape}')
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5

class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = in_channels//2 + in_channels if in_channels > out_channels else out_channels//2
        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm3d(inter_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)

class unet3dDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dDown, self).__init__()
        self.pub =pub(in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub =pub(in_channels//2 + in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        print('x',x.shape)
        print('x1',x1.shape)
        # x = x + x1
        x = self.sample(x)
        # c1 = (x1.size(2) - x.size(2)) // 2
        # c2 = (x1.size(3) - x.size(3)) // 2
        # print(x1.size(2), x.size(2))
        # x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        # print('相加后大小',x.shape)
        x = torch.cat((x, x1), dim=1)
        # print("卷积前通道数",x.shape)
        x = self.pub(x)
        return x

class _upsample_add(nn.Module):
    def __init__(self):
        super(_upsample_add, self).__init__()
        self.upample = nn.Upsample(scale_factor=2,mode='nearest')

    def forward(self, x,y):
        x= self.upample(x)
        return x + y

class BFP(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=1,
                 conv_cfg=None,
                 norm_cfg=None):
        super(BFP, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        assert 0 <= self.refine_level < self.num_levels


        self.refine = NONLocalBlock3D(
            self.in_channels,
            sub_sample= True, bn_layer= True)

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        print('缩放尺寸',gather_size)
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool3d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        # if self.refine_type is not None:
        bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool3d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])

        return tuple(outs)

class FPNUNet(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(FPNUNet, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1])
        self.down2 = unet3dDown(filter_num_list[1], filter_num_list[2])
        self.down3 = unet3dDown(filter_num_list[2], filter_num_list[3])

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up
        self.up1 = unet3dUp(filter_num_list[3], filter_num_list[2])
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1])
        self.up3 = unet3dUp(filter_num_list[1], filter_num_list[0])

        self.up_sample_add = _upsample_add()

        # Lateral layers
        self.latlayer1 = nn.Conv3d(filter_num_list[3], filter_num_list[2], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv3d(filter_num_list[2], filter_num_list[2], kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv3d(filter_num_list[1], filter_num_list[2], kernel_size=1, stride=1, padding=0)

        self.bfp = BFP(filter_num_list[2], 3)

        #rescalebfp
        self.rescalayer1 = nn.Conv3d(filter_num_list[2], filter_num_list[3], kernel_size=1, stride=1, padding=0)
        self.rescalayer2 = nn.Conv3d(filter_num_list[2], filter_num_list[2], kernel_size=1, stride=1, padding=0)
        self.rescalayer3 = nn.Conv3d(filter_num_list[2], filter_num_list[1], kernel_size=1, stride=1, padding=0)

        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def forward(self, input):

        x = input

        x1 = self.inc(x)

        dowx1 = self.down1(x1)
        # print('dowx1', dowx1.shape)

        dowx2 = self.down2(dowx1)
        # print('dowx2',dowx2.shape)

        dowx3 = self.down3(dowx2)
        # print('dowx3', dowx3.shape)

        p3 = self.latlayer1(dowx3)
        print('p3', p3.shape)
        p2 = self.up_sample_add(p3, self.latlayer2(dowx2))
        print('p2', p2.shape)
        p1 = self.up_sample_add(p2, self.latlayer3(dowx1))
        print('p1', p1.shape)

        bfp_input = [p1,p2,p3]
        bfp_out = self.bfp(bfp_input)

        r1,r2,r3 = bfp_out[0], bfp_out[1], bfp_out[2]
        r3 = self.rescalayer1(r3)
        print('r3', r3.shape)
        r2 = self.rescalayer2(r2)
        print('r2', r2.shape)
        r1 = self.rescalayer3(r1)
        print('r1', r1.shape)



        # bridgex = self.bridge(dowx4)

        upx1 = self.up1(dowx3,r2)

        upx2 = self.up2(upx1,r1)

        upx3 = self.up3(upx2,x1)


        final_x = self.class_conv(upx3)

        out_x = nn.Softmax(1)(final_x)

        return out_x


# def FPN101():
#     # return FPN(Bottleneck, [2,4,23,3])
#     return FPN(Bottleneck, [3,4,6,3])
# def test():
#     net = FPN101()
#     fms = net(Variable(torch.randn(1,3,224,224)))
#     for fm in fms:
#         print(fm.size())
if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(4, 1, 32, 32, 32))
    print(img.shape)
    # inputs = [img,img2,img3]
    # # inputs = tuple(inputs)
    net = FPNUNet(1, [32, 64, 128, 256], 2, net_mode='3d')
    init_util.print_network(net)
    out = net(img)
    print(out.size())