#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/11/10 14:55
#@Author: dada
#@File  : 2018Juarez.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_util
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

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv(x)
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

class unet3dDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dDown, self).__init__()
        self.pub =pub(in_channels, out_channels)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        self.pool = nn.MaxPool3d(2, stride=2)
        # self.pool = nn.Conv3d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        # self.pub =pub(in_channels//2  + in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        self.pub = BasicConv3d(in_channels//2  + in_channels, out_channels,kernel_size=3)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
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

class UNetJuarez(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(UNetJuarez, self).__init__()

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
        self.down4 = unet3dDown(filter_num_list[3], filter_num_list[4])
        self.down5 = unet3dDown(filter_num_list[4], filter_num_list[5])


        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up
        self.up1 = unet3dUp(filter_num_list[5], filter_num_list[4])
        self.up2 = unet3dUp(filter_num_list[4], filter_num_list[3])
        self.up3 = unet3dUp(filter_num_list[3], filter_num_list[2])
        self.up4 = unet3dUp(filter_num_list[2], filter_num_list[1])
        self.up5 = unet3dUp(filter_num_list[1], filter_num_list[0])

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
        dowx4 = self.down4(dowx3)

        dowx5 = self.down5(dowx4)


        # bridgex = self.bridge(dowx4)
        upx1 = self.up1(dowx5, dowx4)

        upx2 = self.up2(upx1, dowx3)

        upx3 = self.up3(upx2,dowx2)

        upx4 = self.up4(upx3,dowx1)

        upx5 = self.up5(upx4,x1)


        final_x = self.class_conv(upx5)

        out_x = nn.Softmax(1)(final_x)

        return out_x

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(2, 1, 32, 32, 32))
    coor = Variable(torch.randn(2, 3, 32, 32, 32))
    print(img.shape)
    # inputs = [img,img2,img3]
    # # inputs = tuple(inputs)
    # net2 = ChannelAttention(16,4)
    # net = SpatialAttention(7)
    net = UNetJuarez(1, [16, 32, 64, 128, 256,512], 2, net_mode='3d')
    # net = ASPP(1,8)
    init_util.print_network(net)
    out = net(img)
    print(out.shape)




