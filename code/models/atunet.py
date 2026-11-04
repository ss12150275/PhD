#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/9/7 19:33
#@Author: dada
#@File  : atunet.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models.non_local import *

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

####这个给 nonlocal_unet用
class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub =pub(in_channels//2+in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
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
        print(x1.shape,x.shape)
        x = torch.cat((x, x1), dim=1)
        # print("卷积前通道数",x.shape)
        x = self.pub(x)
        return x
# class unet3dUp(nn.Module):
#     def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
#         super(unet3dUp, self).__init__()
#         self.pub =pub(in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
#         if sample:
#             self.sample = nn.Upsample(scale_factor=2, mode='nearest')
#         else:
#             self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
#
#     def forward(self, x, x1):
#         # print(x.shape)
#         # print(x1.shape)
#         x = self.sample(x)
#         # c1 = (x1.size(2) - x.size(2)) // 2
#         # c2 = (x1.size(3) - x.size(3)) // 2
#         # # print(x1.size(2), x.size(2))
#         # x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
#         print(x1.shape,x.shape)
#         x = torch.cat((x, x1), dim=1)
#         # print("卷积前通道数",x.shape)
#         x = self.pub(x)
#         return x

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int, net_mode='3d'):
        super(Attention_block, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
            batch = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            batch = nn.BatchNorm3d
        else:
            conv = None
            batch = None

        self.W_g = nn.Sequential(
            conv(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            batch(F_int)
        )

        self.W_x = nn.Sequential(
            conv(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            batch(F_int)
        )

        self.psi = nn.Sequential(
            conv(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            batch(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch,net_mode='3d'):
        super(up_conv, self).__init__()
        if net_mode == '2d':
            conv = nn.Conv2d
            batch = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            batch = nn.BatchNorm3d
        else:
            conv = None
            batch = None
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            batch(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Atten_UNet(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='3d'):
        super(Atten_UNet, self).__init__()

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
        # up
        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)
        self.Aup4 = up_conv(filter_num_list[4], filter_num_list[3])
        self.Att4 = Attention_block(F_g=filter_num_list[3], F_l=filter_num_list[3], F_int=filter_num_list[2])
        self.up4 = unet3dUp(filter_num_list[4], filter_num_list[3])

        self.Aup3 = up_conv(filter_num_list[3], filter_num_list[2])
        self.Att3 = Attention_block(F_g=filter_num_list[2], F_l=filter_num_list[2], F_int=filter_num_list[1])
        self.up3 = unet3dUp(filter_num_list[3], filter_num_list[2])

        self.Aup2 = up_conv(filter_num_list[2], filter_num_list[1])
        self.Att2 = Attention_block(F_g=filter_num_list[1], F_l=filter_num_list[1], F_int=filter_num_list[0])
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1])

        self.Aup1 = up_conv(filter_num_list[1], filter_num_list[0])
        self.Att1 = Attention_block(F_g=filter_num_list[0], F_l=filter_num_list[0], F_int=filter_num_list[0]//2)
        self.up1 = unet3dUp(filter_num_list[1], filter_num_list[0])

        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def forward(self, input):

        x = input

        x1 = self.inc(x)

        d1 = self.down1(x1)

        d2 = self.down2(d1)

        d3 = self.down3(d2)

        d4 = self.down4(d3)

        ut4 = self.Aup4(d4)
        at4 = self.Att4(g = ut4,x= d3)
        # print("at4: ",at4.shape)
        upx4 = self.up4(ut4,at4)
        # print("upx4",upx4.shape)

        ut3 = self.Aup3(upx4)
        at3 =self.Att3(g = ut3,x= d2)
        # print("at3",at3.shape)
        upx3 = self.up3(ut3,at3)

        ut2 = self.Aup2(upx3)
        at2 = self.Att2(g=ut2, x=d1)
        # print("at2", at2.shape)
        upx2 = self.up2(ut2, at2)

        ut1 = self.Aup1(upx2)
        at1= self.Att1(g=ut1, x=x1)
        # print("at1", at1.shape)
        upx1 = self.up1(ut1, at1)


        final_x = self.class_conv(upx1)

        out_x = nn.Softmax(1)(final_x)

        return out_x

class Nonlocal_UNet(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='3d'):
        super(Nonlocal_UNet, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
            nl = NONLocalBlock2D
        elif net_mode == '3d':
            conv = nn.Conv3d
            nl = NONLocalBlock3D
        else:
            conv = None
            nl = None

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1])
        self.down2 = unet3dDown(filter_num_list[1], filter_num_list[2])
        self.down3 = unet3dDown(filter_num_list[2], filter_num_list[3])
        self.down4 = unet3dDown(filter_num_list[3], filter_num_list[4])
        # up
        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)
        self.nlup = nl(filter_num_list[4])
        self.nl4 = nl(filter_num_list[3])
        self.up4 = unet3dUp(filter_num_list[4], filter_num_list[3])

        self.nl3 = nl(filter_num_list[2])
        self.up3 = unet3dUp(filter_num_list[3], filter_num_list[2])

        self.nl2 = nl(filter_num_list[1])
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1])

        self.nl1 = nl(filter_num_list[0])
        self.up1 = unet3dUp(filter_num_list[1], filter_num_list[0])

        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def forward(self, input):

        x = input

        x1 = self.inc(x)

        d1 = self.down1(x1)

        d2 = self.down2(d1)

        d3 = self.down3(d2)

        d4 = self.down4(d3)

        nx4 = self.nlup(d4)
        print("nx4", nx4.shape)
        nlx4 = self.nl4(d3)
        print("nlx3",nlx4.shape)
        upx4 = self.up4(nx4,nlx4)
        print("upx4",upx4.shape)

        nlx3 = self.nl3(d2)
        print("nlx3",nlx3.shape)
        upx3 = self.up3(upx4,nlx3)
        print("upx3", upx3.shape)

        nlx2 = self.nl2(d1)
        print("nlx2", nlx2.shape)
        upx2 = self.up2(upx3, nlx2)
        print("upx2", upx2.shape)

        nlx1 = self.nl1(x1)
        print("nlx1", nlx1.shape)
        upx1 = self.up1(upx2, nlx1)
        print("upx2", upx1.shape)


        final_x = self.class_conv(upx1)

        out_x = nn.Softmax(1)(final_x)

        return out_x


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(2, 1, 32,32, 32))
    net = Nonlocal_UNet(1, [4, 8, 16, 32, 64], 2,net_mode='3d')
    out = net(img)
    print(out.size())

    tst_noblocalblock()



