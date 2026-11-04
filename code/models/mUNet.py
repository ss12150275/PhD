from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import init_util
import config

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

class pub2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(pub2, self).__init__()

        self.conv1 = conv_single(in_channels,in_channels//4,1)
        self.conv2 = conv_single(in_channels//4 + in_channels, in_channels // 2, 3)
        self.conv3 = conv_single(in_channels//4 + in_channels + in_channels // 2, in_channels, 3)
        self.conv4 = conv_single(in_channels//4 + in_channels + in_channels // 2 + in_channels , out_channels, 3)
        self.skip = conv_single(in_channels,out_channels,1)

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

class unet3dDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dDown, self).__init__()
        self.pub =pub(in_channels, out_channels)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub =self.pub =pub(in_channels//2  + in_channels, out_channels, batch_norm)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
        self.skip_conv = conv_single(in_channels//2, out_channels,3)

    def forward(self, x, x1,dex):
        # print(x.shape)
        # print(x1.shape)
        x = self.sample(x)
        addx = torch.add(x1,dex)
        # print(addx.shape)
        addx = self.skip_conv(addx)

        # c1 = (x1.size(2) - x.size(2)) // 2
        # c2 = (x1.size(3) - x.size(3)) // 2
        # # print(x1.size(2), x.size(2))
        # x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        # print(x1.shape,x.shape)
        x = torch.cat((x, addx), dim=1)
        # print("卷积前通道数",x.shape)
        x = self.pub(x)
        return x


class mUNet3D(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(mUNet3D, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv_single(in_channels, filter_num_list[0], 3)

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1])
        self.down2 = unet3dDown(filter_num_list[1], filter_num_list[2])
        self.down3 = unet3dDown(filter_num_list[2], filter_num_list[3])

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)
        # deconvolution
        self.deconv1 = nn.ConvTranspose3d(filter_num_list[1], filter_num_list[0], kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose3d(filter_num_list[2], filter_num_list[1], kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose3d(filter_num_list[3], filter_num_list[2], kernel_size=2, stride=2)

        # up

        self.up1 = unet3dUp(filter_num_list[3], filter_num_list[2])
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1])
        self.up3 = unet3dUp(filter_num_list[1], filter_num_list[0])

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

        deconv1 = self.deconv1(dowx1)
        # print('de1', deconv1.shape)
        deconv2 = self.deconv2(dowx2)
        # print('de2', deconv2.shape)
        deconv3 = self.deconv3(dowx3)
        # print('de3', deconv3.shape)


        # bridgex = self.bridge(dowx4)

        upx1 = self.up1(dowx3,dowx2,deconv3)

        upx2 = self.up2(upx1,dowx1,deconv2)

        upx3 = self.up3(upx2,x1,deconv1)


        final_x = self.class_conv(upx3)

        out_x = nn.Softmax(1)(final_x)

        return out_x


def main():
    args = config.args
    # torch.cuda.set_device(1)
    device = torch.device('cpu' if args.cpu else 'cuda')
    model = mUNet3D(1, [16, 32, 64, 128, 256], 2, net_mode='3d').to(device)
    x=torch.rand(2,1,64,64,64)
    x=x.to(device)
    # model.forward(x)
    out = model(x)
    print('the shape of out',out.shape)

if __name__ == '__main__':
    main()



