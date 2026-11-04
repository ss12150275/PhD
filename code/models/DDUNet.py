from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.init_util as init_util

class input_block(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True,net_mode = '3d'):
        super(input_block, self).__init__()
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
                    conv(in_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    conv(out_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, bn(out_channels))
            layers.insert(len(layers)-1, bn(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)

class conv_single(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch,k_s,dilated,model = '3d'):
        super(conv_single, self).__init__()
        pad = dilated
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
            conv(in_ch, out_ch, kernel_size=k_s,dilation=dilated, stride=1, padding=pad, bias=True),
            bn(out_ch),
            nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv(x)
        return x

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
        # self.pub = as_block(in_channels, out_channels,net_mode=net_mode)
        self.pool = pool(2, stride=2)

    def forward(self, x):
        x = self.pub(x)
        return x


class input(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True,net_mode= '3d'):
        super(input, self).__init__()
        if net_mode == '2d':
            pool = nn.MaxPool2d
        elif net_mode == '3d':
            pool = nn.MaxPool3d
        else:
            pool = None
        self.pub =pub(in_channels, out_channels,net_mode=net_mode)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        # self.pub = as_block(in_channels, out_channels,net_mode=net_mode)
        self.pool = pool(2, stride=2)

    def forward(self, x):
        x = self.pub(x)
        return x

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True,net_mode= '3d'):
        super(unet3dUp, self).__init__()
        self.pub = pub(in_channels , out_channels, batch_norm,net_mode=net_mode)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        # self.pub = as_block(in_channels  + out_channels, out_channels,net_mode=net_mode)
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

class DDUNet3D(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(DDUNet3D, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.avp = nn.AvgPool3d(2,stride=2)
        self.pool = nn.MaxPool3d(2, stride=2)

        self.inc = input(in_channels, filter_num_list[0],net_mode=net_mode)

        # down
        self.down1 = unet3dDown(in_channels+filter_num_list[0], filter_num_list[1],net_mode=net_mode)
        self.down2 = unet3dDown(in_channels+filter_num_list[0]+filter_num_list[1], filter_num_list[2],net_mode=net_mode)
        self.down3 = unet3dDown(in_channels+filter_num_list[0]+filter_num_list[1]+filter_num_list[2], filter_num_list[3],net_mode=net_mode)

        self.bridge = unet3dDown(481, filter_num_list[3], net_mode=net_mode)

        # up

        self.up1 = unet3dUp(filter_num_list[3]*2, filter_num_list[2],net_mode=net_mode)
        self.up2 = unet3dUp(filter_num_list[2]*2, filter_num_list[1],net_mode=net_mode)
        self.up3 = unet3dUp(filter_num_list[1]*2, filter_num_list[0],net_mode=net_mode)

        self.up4 = unet3dUp(filter_num_list[0] * 2, filter_num_list[0], net_mode=net_mode)

        self.class_conv = conv(filter_num_list[0], class_num, 1)

    def forward(self, input):

        x = input

        x1 = self.inc(x)

        x = self.avp(x)
        x1_a = torch.cat((x,self.pool(x1)),dim=1)

        dowx1 = self.down1(x1_a)
        # print('dowx1', dowx1.shape)

        dowx_2a = self.avp(x1_a)
        dowx_2a = torch.cat((self.pool(dowx1),dowx_2a),dim=1)

        dowx2 = self.down2(dowx_2a)
        # print('dowx2',dowx2.shape)

        dowx_3a = self.avp(dowx_2a)
        dowx_3a = torch.cat((self.pool(dowx2), dowx_3a), dim=1)

        dowx3 = self.down3(dowx_3a)
        # print('dowx3', dowx3.shape)

        dowx_4a = self.avp(dowx_3a)
        dowx_4a = torch.cat((self.pool(dowx3), dowx_4a), dim=1)


        bridgex = self.bridge(dowx_4a)

        # print(bridgex.shape,dowx3.shape)

        upx1 = self.up1(bridgex,dowx3)

        upx2 = self.up2(upx1,dowx2)

        upx3 = self.up3(upx2,dowx1)

        upx4 = self.up4(upx3,x1)


        final_x = self.class_conv(upx4)

        out_x = nn.Softmax(1)(final_x)
        # out_x = final_x
        return out_x

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(2, 1, 32, 32, 32))
    print(img.shape)
    # inputs = [img,img2,img3]
    # # inputs = tuple(inputs)
    # net = ChannelAttention(128,4)
    # net = SpatialAttention(7)
    net = DDUNet3D(1, [32, 64, 128, 256, 16], 2, net_mode='3d')
    init_util.print_network(net)
    out = net(img)
    print(out.size())
