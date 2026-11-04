#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/11/1 14:59
#@Author: dada
#@File  : ThreeNet.py


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_util


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

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class unet3dDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dDown, self).__init__()
        self.pub =pub(in_channels, out_channels)   # RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel,net_mode = '3d'):
        super(ASPP, self).__init__()
        rate = [6,12,18]
        if net_mode == '2d':
            conv = nn.Conv2d
            avpool = nn.AdaptiveAvgPool2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            avpool = nn.AdaptiveAvgPool3d
        else:
            conv = None
            avpool = None

        self.pool = conv_single(in_channel,in_channel,3,1,0,2)

        self.mean = avpool((1, 1, 1))  # (1,1)means ouput_dim
        self.conv_1 = conv_single(in_channel, out_channel, 1, 1,1,1)
        self.atrous_block1 = conv_single(in_channel, out_channel, 1, 1,0,1)
        self.atrous_block6 = conv_single(in_channel, out_channel, 3, 1, rate[0],rate[0])
        self.atrous_block12 = conv_single(in_channel, out_channel, 3, 1, rate[1],rate[1])
        self.atrous_block18 = conv_single(in_channel, out_channel, 3, 1, rate[2],rate[2])
        self.conv_1x1_output = conv_single(out_channel * 5, out_channel, 1, 1,0,1)

    def forward(self, x):
        # x = self.pool(x)
        # print('空洞卷积代替池化后的大小',x.shape)
        size = x.shape[2:]

        image_features = self.mean(x)
        # print('全局平均池化后的大小', image_features.shape)
        image_features = self.conv_1(image_features)
        image_features = F.interpolate(image_features, size=size, mode='nearest')
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

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub =self.pub =pub(in_channels//2  + in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
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

class unet3dUpmerge(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUpmerge, self).__init__()
        self.pub =self.pub =pub(in_channels//2  + in_channels + 16, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1,x2):
        # print(x.shape)
        # print(x1.shape)
        x = self.sample(x)
        # c1 = (x1.size(2) - x.size(2)) // 2
        # c2 = (x1.size(3) - x.size(3)) // 2
        # # print(x1.size(2), x.size(2))
        # x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        # print(x1.shape,x.shape)
        x = torch.cat((x, x1,x2), dim=1)
        # print("卷积前通道数",x.shape)
        x = self.pub(x)
        return x

class asppUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(asppUp, self).__init__()
        self.pub =pub(in_channels//2 + in_channels + 16, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1,x2):
        # print('x',x.shape)
        # print('x1',x1.shape)
        # x = x + x1
        # x = self.sample(x)
        # c1 = (x1.size(2) - x.size(2)) // 2
        # c2 = (x1.size(3) - x.size(3)) // 2
        # print(x1.size(2), x.size(2))
        # x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        # print('相加后大小',x.shape)
        x = torch.cat((x, x1, x2), dim=1)
        # print("卷积前通道数",x.shape)
        x = self.pub(x)
        return x

class unet3dUptop(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUptop, self).__init__()
        self.pub =self.pub =pub(in_channels//2  + in_channels + 19, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1,x2,coor):
        # print(x.shape)
        # print(x1.shape)
        x = self.sample(x)
        # c1 = (x1.size(2) - x.size(2)) // 2
        # c2 = (x1.size(3) - x.size(3)) // 2
        # # print(x1.size(2), x.size(2))
        # x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        # print(x1.shape,x.shape)
        x = torch.cat((x, x1,x2, coor), dim=1)
        # print("卷积前通道数",x.shape)
        x = self.pub(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(UNet3D, self).__init__()

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

        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def sample(self, mu, logvar):
        std = logvar.exp().sqrt()
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):

        x = input

        x1 = self.inc(x)

        dowx1 = self.down1(x1)
        # print('dowx1', dowx1.shape)

        dowx2 = self.down2(dowx1)
        # print('dowx2',dowx2.shape)

        dowx3 = self.down3(dowx2)
        # print('dowx3', dowx3.shape)
        logvar = dowx3
        z = self.sample(dowx3, logvar)

        # bridgex = self.bridge(dowx4)

        upx1 = self.up1(z,dowx2)

        upx2 = self.up2(upx1,dowx1)

        upx3 = self.up3(upx2,x1)


        final_x = self.class_conv(upx3)

        out_x = nn.Softmax(1)(final_x)

        return out_x

class asppUNetcoor1(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(asppUNetcoor1, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.sample = nn.Upsample(scale_factor=2,mode='nearest')

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1])
        self.down2 = unet3dDown(filter_num_list[1], filter_num_list[2])
        self.down3 = ASPP(filter_num_list[2], filter_num_list[3])

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up

        self.up1 = asppUp(filter_num_list[3], filter_num_list[2])
        self.up2 = unet3dUpmerge(filter_num_list[2], filter_num_list[1])
        self.up3 = unet3dUptop(filter_num_list[1], filter_num_list[0])

        # ---- edge branch ----
        self.edge_conv1 = BasicConv3d(filter_num_list[1], filter_num_list[0], kernel_size=1)
        self.edge_conv2 = BasicConv3d(filter_num_list[0], filter_num_list[0], kernel_size=3, padding=1)
        self.edge_conv3 = BasicConv3d(filter_num_list[0], filter_num_list[0], kernel_size=3, padding=1)
        self.edge_conv4 = BasicConv3d(filter_num_list[0], 2, kernel_size=3, padding=1)
        self.edge_sample = nn.Upsample(scale_factor=2, mode='nearest')

        self.edge_sample1 = nn.Upsample(scale_factor=1/2, mode='nearest')
        self.edge_sample2 = nn.Upsample(scale_factor=1/4, mode='nearest')


        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def sample(self,mu,logvar):
        std = logvar.exp().sqrt()
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, input,coor):

        x = input
        coor_x = coor

        x1 = self.inc(x)

        dowx1 = self.down1(x1)
        # print('dowx1', dowx1.shape)
        edge1 = self.edge_conv1(dowx1)
        edge2 = self.edge_conv2(edge1)
        edge_guidance = self.edge_conv3(edge2)
        # print('edge',edge_guidance.shape)
        later_edge = self.edge_conv4(edge_guidance)
        later_edge = self.edge_sample(later_edge)
        out_edge = nn.Softmax(1)(later_edge)


        dowx2 = self.down2(dowx1)
        # print('dowx2',dowx2.shape)

        dowx3 = self.down3(dowx2)
        # print('dowx3', dowx3.shape)

        logvar = dowx3
        z = self.sample(dowx3,logvar)
        # print('dowx3', dowx3.shape)
        # print('z', z.shape)


        # bridgex = self.bridge(dowx4)

        upx1 = self.up1(z,dowx2,self.edge_sample1(edge_guidance))

        upx2 = self.up2(upx1,dowx1,edge_guidance)

        upx3 = self.up3(upx2,x1,self.edge_sample(edge_guidance),coor_x)


        final_x = self.class_conv(upx3)

        out_x = nn.Softmax(1)(final_x)

        return out_x, out_edge

class asppUNetcoor2(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(asppUNetcoor2, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.sample = nn.Upsample(scale_factor=2,mode='nearest')

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1])
        self.down2 = ASPP(filter_num_list[1], filter_num_list[2])
        # self.down3 = ASPP(filter_num_list[2], filter_num_list[3])

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up

        self.up1 = asppUp(filter_num_list[2], filter_num_list[1])
        self.up2 = unet3dUptop(filter_num_list[1], filter_num_list[0])
        # self.up3 = unet3dUptop(filter_num_list[1], filter_num_list[0])

        # ---- edge branch ----
        self.edge_conv1 = BasicConv3d(filter_num_list[1], filter_num_list[0], kernel_size=1)
        self.edge_conv2 = BasicConv3d(filter_num_list[0], filter_num_list[0], kernel_size=3, padding=1)
        self.edge_conv3 = BasicConv3d(filter_num_list[0], filter_num_list[0], kernel_size=3, padding=1)
        self.edge_conv4 = BasicConv3d(filter_num_list[0], class_num, kernel_size=3, padding=1)
        self.edge_sample = nn.Upsample(scale_factor=2, mode='nearest')

        self.edge_sample1 = nn.Upsample(scale_factor=1/2, mode='nearest')
        self.edge_sample2 = nn.Upsample(scale_factor=1/4, mode='nearest')


        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def forward(self, input,coor):

        x = input
        coor_x = coor

        x1 = self.inc(x)

        dowx1 = self.down1(x1)
        # print('dowx1', dowx1.shape)
        edge1 = self.edge_conv1(dowx1)
        edge2 = self.edge_conv2(edge1)
        edge_guidance = self.edge_conv3(edge2)
        # print('edge',edge_guidance.shape)
        later_edge = self.edge_conv4(edge_guidance)
        later_edge = self.edge_sample(later_edge)
        out_edge = nn.Softmax(1)(later_edge)


        dowx2 = self.down2(dowx1)
        # print('dowx2',dowx2.shape)

        # dowx3 = self.down3(dowx2)
        # print('dowx3', dowx3.shape)


        # bridgex = self.bridge(dowx4)

        upx1 = self.up1(dowx2,dowx1,edge_guidance)

        upx2 = self.up2(upx1,x1,self.edge_sample(edge_guidance),coor_x)

        # upx3 = self.up3(upx1,x1,self.edge_sample(edge_guidance),coor_x)


        final_x = self.class_conv(upx2)

        out_x = nn.Softmax(1)(final_x)

        return out_x, out_edge

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
    net = asppUNetcoor2(1, [16, 32, 64, 128, 256], 2, net_mode='3d')
    # net = ASPP(1,8)
    init_util.print_network(net)
    # out = net(img)
    out,laedge = net(img,coor)
    print(out.shape)




















