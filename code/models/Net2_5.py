#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/11/23 10:16
#@Author: dada
#@File  : 2.5DNet.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import init_util
import time

class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True,net_mode='2d'):
        super(pub, self).__init__()
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        inter_channels = in_channels//2 + in_channels if in_channels > out_channels else out_channels//2
        layers = [
                    conv(in_channels, inter_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    conv(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, bn(inter_channels))
            layers.insert(len(layers)-1, bn(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)

class unetDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True,net_mode='2d'):
        super(unetDown, self).__init__()
        if net_mode == '2d':
            pool = nn.MaxPool2d
        elif net_mode == '3d':
            pool = nn.MaxPool3d
        else:
            pool = None
        # self.pub =pub(in_channels, out_channels,net_mode=net_mode)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        self.pub = conv_single(in_channels, out_channels,5,1,2,1, net_mode=net_mode)
        self.pool = pool(2, stride=2)

    def forward(self, x):
        x = self.pub(x)
        x = self.pool(x)
        return x

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True,net_mode = '2d'):
        super(unet3dUp, self).__init__()
        self.pub =self.pub =pub(in_channels//2  + in_channels, out_channels, batch_norm,net_mode=net_mode)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            if net_mode == '2d':
                self.sample = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
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

class Net(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(Net, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None


        # self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        # down
        self.down1 = unetDown(in_channels, filter_num_list[0],net_mode = net_mode)
        self.down2 = unetDown(filter_num_list[0], filter_num_list[1],net_mode = net_mode)
        # self.down3 = unetDown(filter_num_list[2], filter_num_list[3],net_mode = net_mode)

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up
        self.fc1 = nn.Linear(3*filter_num_list[1]*8*8, 128)
        self.fc2 = nn.Linear(128, 2)


    def forward(self, input1,input2,input3):

        x1 = input1
        x2 = input2
        x3 = input3
        print(x1.shape,x2.shape,x3.shape)
        batch = x1.shape[0]

        # x1 = self.inc(x)

        x1_1 = self.down1(x1)
        # print('dowx1', x1_1.shape)
        x1_2 = self.down2(x1_1)
        # print('dowx2',x1_2.shape)
        x1_2 = x1_2.view(batch,-1)
        # print('x1_2',x1_2.shape)

        x2_1 = self.down1(x2)
        x2_2 = self.down2(x2_1)
        x2_2 = x2_2.view(batch,-1)

        x3_1 = self.down1(x3)
        x3_2 = self.down2(x3_1)
        x3_2 = x3_2.view(batch,-1)

        merge = torch.cat((x1_2,x2_2,x3_2),dim =1)
        # print('merge',merge.shape)


        final_x = self.fc1(merge)

        out_x = self.fc2(final_x)
        print(out_x.shape)
        out_x = nn.Softmax(1)(out_x)

        return out_x

class Net2(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(Net2, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        # self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        # down
        self.down1 = unetDown(in_channels, filter_num_list[0],net_mode = net_mode)
        self.down2 = unetDown(filter_num_list[0], filter_num_list[1],net_mode = net_mode)
        # self.down3 = unetDown(filter_num_list[2], filter_num_list[3],net_mode = net_mode)

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, class_num)


    def forward(self, input):

        x1 = input

        # x1 = self.inc(x)

        x1_1 = self.down1(x1)
         #print('dowx1', x1_1.shape)
        x1_2 = self.down2(x1_1)
        print('dowx2',x1_2.shape)
        x1_2 = x1_2.view(4,-1)
        print('x1_2',x1_2.shape)


        final_x = self.fc1(x1_2)

        out_x = self.fc2(final_x)
        out_x = nn.Softmax(1)(out_x)

        return out_x

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    import numpy as np

    device = torch.device('cuda')

    image = Variable(torch.randn(2, 3, 32, 32)).float().to(device)
    # target = Variable(torch.randn(4, 2))
    net = Net(3, [16, 32, 64, 128, 256], 2,2, net_mode='2d').to(device)
    # net2 = Net2(1, [16, 32, 64, 128, 256], 2, net_mode='3d')
    init_util.print_network(net)
    time1 = time.time()
    for i in range(12):
        # out = net2(image)
        out = net(image, image, image)

        # for output1 in zip(out):
        #     print(output1)
        output1 = out.squeeze().cpu().detach().numpy()
        # print(output1)
        # output1 = np.argmax(output1, axis=1).astype("float")
        print(output1)
    time2 = time.time()
    print(time2 - time1)



