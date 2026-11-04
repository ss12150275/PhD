#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/9/15 15:33
#@Author: dada
#@File  : resunet.py

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

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.bottle = self._make_layer(Bottleneck,in_channels,3, stride=1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        # 这里的blocks就是该大layer中的残差块数
        # out_channels表示的是这个块中3*3卷积层的输入输出深度
        downsample = None  # shortcut内部的跨层实现
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 判断步长是否为1，判断当前块的输入深度和当前块卷积层深度乘于残差块的扩张
            # 为何用步长来判断，我现在还不明白，感觉没有也行
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion)
            )
            # 一旦判断条件成立，那么给downsample赋予一层1*1卷积层和一层批标准化层。并且这一步将伴随这特征图缩小1/2
            # 而为何要在shortcut中再进行卷积操作？是因为在残差块之间，比如当要从64深度的3*3卷积层阶段过渡到128深度的3*3卷积层阶段，主分支为64深度的输入已经通过128深度的3*3卷积层变成了128深度的输出，而shortcut分支中x的深度仍为64，而主分支和shortcut分支相加的时候，深度不一致会报错。这就需要进行升维操作，使得shortcut分支中的x从64深度升到128深度。
            # 而且需要这样操作的其实只是在基础块BasicBlock中，在瓶颈块Bottleneck中主分支中自己就存在升维操作，那么Bottleneck还在shortcut中引入卷积层的目的是什么？能带来什么帮助？

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        # block()生成上面定义的基础块和瓶颈块的对象，并将dowsample传递给block

        self.in_channels = out_channels * block.expansion  # 改变下面的残差块的输入深度
        # 这使得该阶段下面blocks-1个block，即下面循环内构造的block与下一阶段的第一个block的在输入深度上是相同的。
        for i in range(1, blocks):  # 这里面所有的block
            layers.append(block(self.in_channels, out_channels))
        # 一定要注意，out_channels一直都是3*3卷积层的深度
        return nn.Sequential(*layers)  # 这里表示将layers中的所有block按顺序接在一起

    def forward(self, x):
        return self.bottle(x)

class ResBlock1(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ResBlock1, self).__init__()
        self.bottle1 = Bottleneck(in_channels,out_channels)
        self.bottle2 = Bottleneck(out_channels*4, out_channels)
        self.bottle3 = Bottleneck(out_channels*4, out_channels)
    def forward(self, x):
        out = self.bottle1(x)
        out = self.bottle2(out)
        out = self.bottle3(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels,stride =1, downsample = None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels,kernel_size=3,padding=1)  # 3*3卷积操作
        self.bn2 = nn.BatchNorm3d(out_channels)
        # 这层3*3卷积层的channels是下面_make_layer中的第二个参数规定的

        self.conv3 = nn.Conv3d(out_channels, out_channels* self.expansion , kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        # 这层1*1卷积层，是在升维，四倍的升
        self.conv = nn.Conv3d(in_channels, out_channels * self.expansion , kernel_size=1, bias=False)

        self.relu = nn.ReLU(True)  # 激活函数
        self.downsample = downsample  # shortcut信号
        self.stride = stride  # 获取步长

    def forward(self, x):
        residual = x
        # print(x.shape)
        residual = self.conv(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # 连接一个激活函数
        # print(out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print(out.shape)

        out = self.conv3(out)
        out = self.bn3(out)

        # print(out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)  # 目的同上

        out += residual
        out = self.relu(out)
        # print(out.shape)

        return out

class conv_single(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch,k_s):
        super(conv_single, self).__init__()
        if(k_s == 1):
            pad = 0
        if(k_s == 3):
            pad = 1

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_s, stride=1, padding=pad, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv(x)
        return x

# class unet3dDown(nn.Module):
#
#     def __init__(self, in_channels, out_channels, batch_norm=True):
#         super(unet3dDown, self).__init__()
#         self.pub =pub(in_channels, out_channels)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
#         self.pool = nn.MaxPool3d(2, stride=2)
#
#     def forward(self, x):
#         x = self.pool(x)
#         x = self.pub(x)
#         return x
#
# class unet3dUp(nn.Module):
#     def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
#         super(unet3dUp, self).__init__()
#         self.pub =self.pub =pub(in_channels//2  + in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
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
#         # print(x1.shape,x.shape)
#         x = torch.cat((x, x1), dim=1)
#         # print("卷积前通道数",x.shape)
#         x = self.pub(x)
#         return x

class unet3dDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dDown, self).__init__()
        self.pub =ResBlock1(in_channels, out_channels)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        # print('池化后大小',x.shape)
        x = self.pub(x)
        return x

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub =self.pub =ResBlock1(in_channels//2  + in_channels, out_channels)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        # print('上采样前',x.shape)
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

class unet3dUptop(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUptop, self).__init__()
        self.pub =self.pub =ResBlock1(in_channels//8  + in_channels, out_channels)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        # print('上采样前',x.shape)
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

class resUNet3D(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(resUNet3D, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1])
        self.down2 = unet3dDown(filter_num_list[1]*4, filter_num_list[2])
        self.down3 = unet3dDown(filter_num_list[2]*4, filter_num_list[3])

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up

        self.up1 = unet3dUp(filter_num_list[3]*4, filter_num_list[2])
        self.up2 = unet3dUp(filter_num_list[2]*4, filter_num_list[1])
        self.up3 = unet3dUptop(filter_num_list[1]*4, filter_num_list[0])

        self.class_conv = conv(filter_num_list[0]*4, class_num, 1)

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

        upx1 = self.up1(dowx3,dowx2)
        # print('upx1', upx1.shape)

        upx2 = self.up2(upx1,dowx1)
        # print('upx2', upx2.shape)

        upx3 = self.up3(upx2,x1)
        # print('upx3', upx3.shape)


        final_x = self.class_conv(upx3)

        out_x = nn.Softmax(1)(final_x)

        return out_x

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(2, 32, 28,28, 28))
    # img2 = nn.Conv3d(1,16,3,1)
    net = ResBlock1(32,64)
    # net = resUNet3D(1, [16, 32, 64, 128, 256], 2, net_mode='3d')
    out = net(img)
    print(out.size())




