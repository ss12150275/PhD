#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/9/7 19:38
#@Author: dada
#@File  : SCNN.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

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
        self.pub =pub(in_channels//2  + in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
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

class message_pass(nn.Module):
    def __init__(self, in_channel,ms_ks):
        super(message_pass, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channel, 2 * in_channel, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm3d(2 * in_channel),
            nn.ReLU(),
            nn.Conv3d(2 * in_channel, in_channel, 1, bias=False),
            nn.BatchNorm3d(in_channel),
            nn.ReLU()  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down',
                                        nn.Conv3d(in_channel, in_channel, (3, 3, ms_ks), padding=(1, 1, ms_ks // 2),
                                                  bias=False))
        self.message_passing.add_module('down_up',
                                        nn.Conv3d(in_channel, in_channel, (3, 3, ms_ks), padding=(1, 1, ms_ks // 2),
                                                  bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv3d(in_channel, in_channel, (3, ms_ks, 3), padding=(1, ms_ks // 2, 1),
                                                  bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv3d(in_channel, in_channel, (3, ms_ks, 3), padding=(1, ms_ks // 2, 1),
                                                  bias=False))

    def forward(self, img, seg_gt=None, exist_gt=None):
        x = img
        print('original',x.shape)
        x = self.layer1(x)
        print('layer1',x.shape)
        x = self.message_passing_forward(x)
        print('after massage passing',x.shape)
        return x

    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, S, H, W = x.shape


        if vertical:
            print('before veritical',x.size())
            slices = [x[:, :, :, i:(i + 1), :] for i in range(H)]
            print('after veritical',len(slices),slices[0].shape)
            dim = 3
        else:
            slices = [x[:, :, :, :, i:(i + 1)] for i in range(W)]
            dim = 4
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)


class message_passing(nn.Module):
    def __init__(self, in_channels, filter_num_list,class_num,net_mode='3d'):
        super(message_passing, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None
        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1
        self.ms_ks = 9
        # self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        # self.bce_loss = nn.BCELoss()


        self.inc = conv(in_channels, filter_num_list[0], 3, stride=1, padding=1)

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1])
        self.down2 = unet3dDown(filter_num_list[1], filter_num_list[2])
        self.down3 = unet3dDown(filter_num_list[2], filter_num_list[3])
        self.down4 = unet3dDown(filter_num_list[3], filter_num_list[4])

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)
        self.mp1 = message_pass(filter_num_list[1], 9)
        self.mp2 = message_pass(filter_num_list[2], 9)
        self.mp3 = message_pass(filter_num_list[3], 9)
        self.mp4 = message_pass(filter_num_list[4], 9)
        # up

        self.up4 = unet3dUp(filter_num_list[4], filter_num_list[3])
        self.up3 = unet3dUp(filter_num_list[3], filter_num_list[2])
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1])
        self.up1 = unet3dUp(filter_num_list[1], filter_num_list[0])



        self.class_conv = conv(filter_num_list[0], class_num, 1)



    def forward(self, img, seg_gt=None, exist_gt=None):
        x = img

        x1 = self.inc(x)

        dowx1 = self.down1(x1)

        dowx2 = self.down2(dowx1)

        dowx3 = self.down3(dowx2)

        dowx4 = self.down4(dowx3)


        m4 = self.mp4(dowx4)
        # m3 = self.mp3(dowx3)
        # m2 = self.mp2(dowx2)
        # m1 = self.mp1(dowx1)




        # bridgex = self.bridge(dowx4)
        upx4 = self.up4(m4, dowx3)

        upx3 = self.up3(upx4,dowx2)

        upx2 = self.up2(upx3, dowx1)

        upx1 = self.up1(upx2, x1)


        # seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        # x = self.layer3(x)
        # x = x.view(-1, self.fc_input_feature)
        # exist_pred = self.fc(x)

        # if seg_gt is not None and exist_gt is not None:
        #     loss_seg = self.ce_loss(seg_pred, seg_gt)
        #     loss_exist = self.bce_loss(exist_pred, exist_gt)
        #     loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        # else:
        #     loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
        #     loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
        #     loss = torch.tensor(0, dtype=img.dtype, device=img.device)
        final_x = self.class_conv(upx1)

        out_x = nn.Softmax(1)(final_x)

        return out_x

        # return seg_pred, exist_pred




if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(2, 1, 64, 64, 64))
    print(img.shape)
    # inputs = [img,img2,img3]
    # # inputs = tuple(inputs)
    net = message_passing(1, [8,16,32, 64, 128], 2, net_mode='3d')
    print(net)
    out = net(img)
    print(out.size())
