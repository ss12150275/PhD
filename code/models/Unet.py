# -*- coding:utf-8 -*-
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_util

# 未测试
class SEBlock(nn.Module):
    def __init__(self,in_channels,out_channels,net_mode='2d'):
        super(SEBlock,self).__init__()

        if net_mode == '2d':
            self.gap=nn.AdaptiveAvgPool2d(1)
            conv=nn.Conv2d
        elif net_mode == '3d':
            self.gap=nn.AdaptiveAvgPool3d(1)
            conv=nn.Conv3d
        else:
            self.gap=None
            conv=None

        self.conv1=conv(in_channels,out_channels,1)
        self.conv2=conv(in_channels,out_channels,1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        inpu=x
        x=self.gap(x)
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.sigmoid(x)

        return inpu*x

# 未测试
class DenseBlock(nn.Module):
    def __init__(self,channels,conv_num,net_mode='2d'):
        super(DenseBlock,self).__init__()
        self.conv_num=conv_num
        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.relu=nn.ReLU()
        self.conv_list=[]
        self.bottle_conv_list=[]
        for i in conv_num:
            self.bottle_conv_list.append(conv(channels*(i+1),channels*4,1))
            self.conv_list.append(conv(channels*4,channels,3,padding=1))


    def forward(self,x):

        res_x=[]
        res_x.append(x)

        for i in self.conv_num:
            inputs=torch.cat(res_x,dim=1)
            x=self.bottle_conv_list[i](inputs)
            x=self.relu(x)
            x=self.conv_list[i](x)
            x=self.relu(x)
            res_x.append(x)

        return x

class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True,net_mode = '3d'):
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

class conv_single(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch,k_s,model = '3d'):
        super(conv_single, self).__init__()
        if(k_s == 1):
            pad = 0
        elif (k_s == 3):
            pad = 1
        else:
            pad = None
        if(model == '3d'):
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        elif (model == '2d'):
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.conv = nn.Sequential(
            conv(in_ch, out_ch, kernel_size=k_s, stride=1, padding=pad, bias=True),
            bn(out_ch),
            nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv(x)
        return x

class pub2(nn.Module):

    def __init__(self, in_channels, out_channels,net_model = '2d'):
        super(pub2, self).__init__()

        self.conv1 = conv_single(in_channels,in_channels//4,1,model=net_model)
        self.conv2 = conv_single(in_channels//4 + in_channels, in_channels // 2, 3,model=net_model)
        self.conv3 = conv_single(in_channels//4 + in_channels + in_channels // 2, in_channels, 3,model=net_model)
        self.conv4 = conv_single(in_channels//4 + in_channels + in_channels // 2 + in_channels , out_channels, 3,model=net_model)
        self.skip = conv_single(in_channels,out_channels,1,model=net_model)

    def forward(self, x):
        x1 = self.conv1(x)
        # print('x1 ', x1.shape)
        x11 = torch.cat((x,x1), dim=1)
        # print('x1 ', x11.shape)

        x2 = self.conv2(x11)
        # print('x2 ', x2.shape)
        x22 = torch.cat((x, x1, x2), dim=1)
        # print('x2 ', x22.shape)

        x3 = self.conv3(x22)
        # print('x3 ', x3.shape)
        x33 = torch.cat((x, x1, x2, x3), dim=1)
        # print('x3 ', x33.shape)

        x4 = self.conv4(x33)
        # print('x4 ', x4.shape)
        # s = self.skip(x)
        # print('s ', s.shape)
        #
        # final_x = torch.stack((x4,x3,x2,s), dim= 1)
        # print(final_x.shape)
        return x4

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, net_mode='2d'):
        super(ResBlock, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.conv1 = conv(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = bn(out_channels)

        if in_channels!=out_channels:
            self.res_conv=conv(in_channels,out_channels,1,stride=stride)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            res=self.res_conv(x)
        else:
            res=x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        out = x + res
        out = self.relu(out)

        return out

class Up(nn.Module):
    def __init__(self, down_in_channels, in_channels, out_channels, conv_block, interpolation=True, net_mode='2d'):
        super(Up, self).__init__()

        if net_mode == '2d':
            inter_mode = 'bilinear'
            trans_conv = nn.ConvTranspose2d
        elif net_mode == '3d':
            inter_mode = 'trilinear'
            trans_conv = nn.ConvTranspose3d
        else:
            inter_mode = None
            trans_conv = None

        if interpolation == True:
            self.up = nn.Upsample(scale_factor=2, mode=inter_mode, align_corners=True)
        else:
            self.up = trans_conv(down_in_channels, down_in_channels, 2, stride=2)

        self.conv = RecombinationBlock(in_channels + down_in_channels, out_channels, net_mode=net_mode)

    def forward(self, down_x, x):
        up_x = self.up(down_x)

        x = torch.cat((up_x, x), dim=1)

        x = self.conv(x)

        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block, net_mode='2d'):
        super(Down, self).__init__()
        if net_mode == '2d':
            maxpool = nn.MaxPool2d
        elif net_mode == '3d':
            maxpool = nn.MaxPool3d
        else:
            maxpool = None

        self.conv = RecombinationBlock(in_channels, out_channels, net_mode=net_mode)

        self.down = maxpool(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        out = self.down(x)

        return x, out

# class unet3dDown(nn.Module):
#
#     def __init__(self, in_channels, out_channels, batch_norm=True):
#         super(unet3dDown, self).__init__()
#         self.pub =pub(in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
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
#         self.pub =pub(in_channels//2  + in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
#         if sample:
#             self.sample = nn.Upsample(scale_factor=2, mode='nearest')
#         else:
#             self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
#
#     def forward(self, x, x1):
#         # print(x.shape)
#         # print(x1.shape)
#         # x = self.sample(x)
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

    def __init__(self, in_channels, out_channels, batch_norm=True,net_mode= '3d'):
        super(unet3dDown, self).__init__()
        if net_mode == '2d':
            pool = nn.MaxPool2d
        elif net_mode == '3d':
            pool = nn.MaxPool3d
        else:
            pool = None
        self.pub =pub(in_channels, out_channels,net_mode=net_mode)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        self.pool = pool(2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True,net_mode= '3d'):
        super(unet3dUp, self).__init__()
        self.pub = pub(in_channels  + out_channels, out_channels, batch_norm,net_mode=net_mode)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            if net_mode == '2d':
                self.sample = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
            elif net_mode == '3d':
                self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
            else:
                self.sample = None


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

class unet3dUptop(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUptop, self).__init__()
        self.pub =self.pub =pub(in_channels//2  + in_channels + 3, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1,coor):
        # print(x.shape)
        # print(x1.shape)
        x = self.sample(x)
        # c1 = (x1.size(2) - x.size(2)) // 2
        # c2 = (x1.size(3) - x.size(3)) // 2
        # # print(x1.size(2), x.size(2))
        # x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        # print(x1.shape,x.shape)
        x = torch.cat((x, x1, coor), dim=1)
        # print("卷积前通道数",x.shape)
        x = self.pub(x)
        return x

class SegSEBlock(nn.Module):
    def __init__(self, in_channels, rate=2, net_mode='2d'):
        super(SegSEBlock, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.in_channels = in_channels
        self.rate = rate
        self.dila_conv = conv(self.in_channels, self.in_channels // self.rate, 3, padding=2, dilation=self.rate)
        self.conv1 = conv(self.in_channels // self.rate, self.in_channels, 1)

    def forward(self, input):
        x = self.dila_conv(input)
        x = self.conv1(x)
        x = nn.Sigmoid()(x)

        return x

class RecombinationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3, net_mode='2d'):
        super(RecombinationBlock, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization
        self.kerenl_size = kernel_size
        self.rate = 2
        self.expan_channels = self.out_channels * self.rate

        self.expansion_conv = conv(self.in_channels, self.expan_channels, 1)
        self.skip_conv = conv(self.in_channels, self.out_channels, 1)
        self.zoom_conv = conv(self.out_channels * self.rate, self.out_channels, 1)

        self.bn = bn(self.expan_channels)
        self.norm_conv = conv(self.expan_channels, self.expan_channels, self.kerenl_size, padding=1)

        self.segse_block = SegSEBlock(self.expan_channels, net_mode=net_mode)

    def forward(self, input):
        x = self.expansion_conv(input)

        for i in range(1):
            if self.bach_normalization:
                x = self.bn(x)
            x = nn.ReLU6()(x)
            x = self.norm_conv(x)

        se_x = self.segse_block(x)

        x = x * se_x

        x = self.zoom_conv(x)

        skip_x = self.skip_conv(input)
        out = x + skip_x

        return out

class UNet(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, conv_block=RecombinationBlock, net_mode='2d'):
        super(UNet, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv(in_channels, 16, 1)

        # down
        self.down1 = Down(16, filter_num_list[0], conv_block=conv_block, net_mode=net_mode)
        self.down2 = Down(filter_num_list[0], filter_num_list[1], conv_block=conv_block, net_mode=net_mode)
        self.down3 = Down(filter_num_list[1], filter_num_list[2], conv_block=conv_block, net_mode=net_mode)
        self.down4 = Down(filter_num_list[2], filter_num_list[3], conv_block=conv_block, net_mode=net_mode)

        self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up
        self.up1 = Up(filter_num_list[4], filter_num_list[3], filter_num_list[3], conv_block=conv_block,
                      net_mode=net_mode)
        self.up2 = Up(filter_num_list[3], filter_num_list[2], filter_num_list[2], conv_block=conv_block,
                      net_mode=net_mode)
        self.up3 = Up(filter_num_list[2], filter_num_list[1], filter_num_list[1], conv_block=conv_block,
                      net_mode=net_mode)
        self.up4 = Up(filter_num_list[1], filter_num_list[0], filter_num_list[0], conv_block=conv_block,
                      net_mode=net_mode)

        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def forward(self, input):

        x = input

        x = self.inc(x)

        conv1, x = self.down1(x)

        conv2, x = self.down2(x)

        conv3, x = self.down3(x)

        conv4, x = self.down4(x)

        x = self.bridge(x)

        x = self.up1(x, conv4)

        x = self.up2(x, conv3)

        x = self.up3(x, conv2)

        x = self.up4(x, conv1)

        x = self.class_conv(x)

        # x = nn.Softmax(1)(x)

        return x,x,x

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
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1],net_mode=net_mode)
        self.down2 = unet3dDown(filter_num_list[1], filter_num_list[2],net_mode=net_mode)
        self.down3 = unet3dDown(filter_num_list[2], filter_num_list[3],net_mode=net_mode)

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up

        self.up1 = unet3dUp(filter_num_list[3], filter_num_list[2],net_mode=net_mode)
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1],net_mode=net_mode)
        self.up3 = unet3dUp(filter_num_list[1], filter_num_list[0],net_mode=net_mode)

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

        upx1 = self.up1(dowx3,dowx2)

        upx2 = self.up2(upx1,dowx1)

        upx3 = self.up3(upx2,x1)


        final_x = self.class_conv(upx3)

        out_x = nn.Softmax(1)(final_x)
        # out_x = final_x
        return out_x

class UNet2D(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(UNet2D, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1],net_mode=net_mode)
        self.down2 = unet3dDown(filter_num_list[1], filter_num_list[2],net_mode=net_mode)
        self.down3 = unet3dDown(filter_num_list[2], filter_num_list[3],net_mode=net_mode)

        self.down4 = unet3dDown(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up
        self.up4 = unet3dUp(filter_num_list[4], filter_num_list[3], net_mode=net_mode)

        self.up1 = unet3dUp(filter_num_list[3], filter_num_list[2],net_mode=net_mode)
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1],net_mode=net_mode)
        self.up3 = unet3dUp(filter_num_list[1], filter_num_list[0],net_mode=net_mode)

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

        upx4 = self.up4(dowx4,dowx3)


        # bridgex = self.bridge(dowx4)

        upx1 = self.up1(upx4,dowx2)

        upx2 = self.up2(upx1,dowx1)

        upx3 = self.up3(upx2,x1)


        final_x = self.class_conv(upx3)

        out_x = nn.Softmax(1)(final_x)
        # out_x = final_x
        return out_x

class mergenet(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(mergenet, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.pool = nn.MaxPool3d(2, stride=2)
        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        # down
        self.down1 = pub(filter_num_list[0],filter_num_list[0])
        self.down2 = conv(filter_num_list[0], filter_num_list[0], 1)
        self.down3 = conv(filter_num_list[0], filter_num_list[0], 3,stride=1,padding=1)

        # up

        self.up1 = self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = pub(filter_num_list[0], filter_num_list[0])


        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def forward(self, input):

        x = input

        x1 = self.inc(x)

        dowx1 = self.down1(x1)
        # print('dowx1', dowx1.shape)
        pool1 = self.pool(dowx1)

        dowx2 = self.down2(pool1)
        # print('dowx2',dowx2.shape)

        dowx3 = self.down3(dowx2)
        # print('dowx3', dowx3.shape)


        # bridgex = self.bridge(dowx4)

        upx1 = self.up1(dowx3)

        upx2 = self.up2(upx1)

        final_x = self.class_conv(upx2)

        out_x = nn.Softmax(1)(final_x)

        return out_x

class message_passing(nn.Module):
    def __init__(self, input_size, net_mode='3d'):
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
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None):
        x = self.layer1(img)
        x = self.message_passing_forward(x)
        x = self.layer2(x)

        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input_feature)
        exist_pred = self.fc(x)

        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss

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
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def net_init(self, input_size, ms_ks):
        self.layer1 = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv3d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv3d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv3d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv3d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv3d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # (nB, 128, 36, 100)

        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            nn.Dropout3d(0.1),
            nn.Conv3d(128, 5, 1)  # get (nB, 5, 36, 100)
        )

        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool3d(2, 2),  # (nB, 5, 18, 50)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

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

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch,filter_list, output_ch,):
        super(AttU_Net, self).__init__()

        filters = filter_list

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out




# import config
# def main():
#     args = config.args
#     torch.cuda.set_device(1)
#     device = torch.device('cpu' if args.cpu else 'cuda')
#     model = Atten_UNet(1,[32,48,64,96,128],2,net_mode='3d').to(device)
#     x=torch.rand(2,1,64,64,64)
#     x=x.to(device)
#     model.forward(x)
#
if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(4, 1, 32, 32))
    print(img.shape)
    # inputs = [img,img2,img3]
    # # inputs = tuple(inputs)
    net = UNet2D(1, [32, 64, 96, 128, 256], 2, net_mode='2d')
    init_util.print_network(net)
    out = net(img)
    print(out.size())

