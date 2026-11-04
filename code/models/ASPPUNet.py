#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/9/13 15:35
#@Author: dada
#@File  : ASPPUNet.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

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
    def __init__(self, in_channels, out_channels, scale,batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub =pub(in_channels//2  + in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=scale, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        # print(x.shape)
        # print(x1.shape)
        x = self.sample(x)
        # c1 = (x1.size(2) - x.size(2)) // 2
        # c2 = (x1.size(3) - x.size(3)) // 2
        # # print(x1.size(2), x.size(2))
        # x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        # print(x1.shape,x.shape)
        x = torch.cat((x, x1), dim=1)
        # print("卷积前通道数",x.shape)
        x = self.pub(x)
        return x

class conv_single(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch,kernel,stride,pad,rate,net_mode='3d'):
        super(conv_single, self).__init__()
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None


        self.conv = nn.Sequential(
            conv(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, dilation=rate,bias=True),
            bn(out_ch),
            nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel,net_mode = '3d'):
        super(ASPP, self).__init__()
        rate = [4,8,12]
        if net_mode == '2d':
            conv = nn.Conv2d
            avpool = nn.AdaptiveAvgPool2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            avpool = nn.AdaptiveAvgPool3d
        else:
            conv = None
            avpool = None

        self.pool = nn.MaxPool3d(2, stride=2)

        self.mean = avpool((1, 1, 1))  # (1,1)means ouput_dim
        self.conv_1 = conv_single(in_channel, out_channel, 1, 1,1,1)
        self.atrous_block1 = conv_single(in_channel, out_channel, 1, 1,0,1)
        self.atrous_block6 = conv_single(in_channel, out_channel, 3, 1, rate[0],rate[0])
        self.atrous_block12 = conv_single(in_channel, out_channel, 3, 1, rate[1],rate[1])
        self.atrous_block18 = conv_single(in_channel, out_channel, 3, 1, rate[2],rate[2])
        self.conv_1x1_output = conv_single(out_channel * 5, out_channel, 1, 1,0,1)

    def forward(self, x):
        x = self.pool(x)
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv_1(image_features)
        image_features = F.upsample(image_features, size=size, mode='trilinear')
        # print('size 0', image_features.shape)

        atrous_block1 = self.atrous_block1(x)
        # print('size 1',atrous_block1.shape)
        atrous_block6 = self.atrous_block6(x)
        # print('size 6', atrous_block6.shape)
        atrous_block12 = self.atrous_block12(x)
        # print('size 12', atrous_block12.shape)
        atrous_block18 = self.atrous_block18(x)
        # print('size 18', atrous_block18.shape)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        # print('size net', net.shape)
        return net

class ASPP_UNet(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='3d'):
        super(ASPP_UNet, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        self.sample = nn.Upsample(scale_factor=2, mode='nearest')

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1])
        self.down2 = unet3dDown(filter_num_list[1], filter_num_list[2])
        self.down3 = ASPP(filter_num_list[2], filter_num_list[2])

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up

        # self.up1 = unet3dUp(filter_num_list[3], filter_num_list[2])
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1],4)
        self.up3 = unet3dUp(filter_num_list[1], filter_num_list[0],2)

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


        # bridgex = self.bridge(dowx4)

        # upx1 = self.up1(dowx3,dowx2)
        # upx1 = self.sample(dowx3)
        # upx1 = torch.cat((upx1,dowx2),dim=1)

        upx2 = self.up2(dowx3,dowx1)

        upx3 = self.up3(upx2,x1)


        final_x = self.class_conv(upx3)

        out_x = nn.Softmax(1)(final_x)

        return out_x

class ASPP_UNet2(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='3d'):
        super(ASPP_UNet2, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        self.sample = nn.Upsample(scale_factor=2, mode='nearest')

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1])
        self.down2 = ASPP(filter_num_list[1], filter_num_list[2])
        self.down3 = ASPP(filter_num_list[2], filter_num_list[3])

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up

        self.up1 = unet3dUp(filter_num_list[3], filter_num_list[2],2)
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1],2)
        self.up3 = unet3dUp(filter_num_list[1], filter_num_list[0],2)

        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def forward(self, input):

        x = input

        x1 = self.inc(x)

        dowx1 = self.down1(x1)
        # print('dowx1', dowx1.shape)

        dowx2 = self.down2(dowx1)
        print('dowx2',dowx2.shape)

        dowx3 = self.down3(dowx2)
        print('dowx3', dowx3.shape)


        # bridgex = self.bridge(dowx4)

        upx1 = self.up1(dowx3,dowx2)
        # upx1 = self.sample(dowx3)
        # upx1 = upx1 + dowx2
        print('upx1',upx1.shape)

        upx2 = self.up2(upx1,dowx1)

        upx3 = self.up3(upx2,x1)


        final_x = self.class_conv(upx3)

        out_x = nn.Softmax(1)(final_x)

        return out_x










