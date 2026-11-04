from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_util

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

class as_block(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True,net_mode = '3d'):
        super(as_block, self).__init__()
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
        layers2 = [
            conv(in_channels, inter_channels,(3,3,1) , stride=1, padding= (1,1,0)),
            nn.ReLU(True),
            conv(inter_channels, out_channels, (3,3,1), stride=1, padding=(1,1,0)),
            nn.ReLU(True)
        ]
        if batch_norm:
            layers.insert(1, bn(inter_channels))
            layers.insert(len(layers)-1, bn(out_channels))
            layers2.insert(1, bn(inter_channels))
            layers2.insert(len(layers) - 1, bn(out_channels))
        self.pub = nn.Sequential(*layers)
        self.as_conv = nn.Sequential(*layers2)
        self.rasidual = conv(in_channels, out_channels,1)


        self.conv_1x1 = conv(out_channels * 2, out_channels, 1)

    def forward(self, x):
        print("initial-x.shape",x.shape)

        rasidual = self.rasidual(x)
        print('rasidual.shape',rasidual.shape)

        spatial_x = self.pub(x)
        print('spatial_x.shape', spatial_x.shape)

        palne_x = self.as_conv(x)
        print('palne_x.shape', palne_x.shape)

        confuse_x = self.conv_1x1(torch.cat((spatial_x,palne_x),dim=1))
        print('confuse_x.shape', confuse_x.shape)

        out = rasidual + confuse_x
        print('out.shape', out.shape)

        return out

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

class unet3dDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True,net_mode= '3d'):
        super(unet3dDown, self).__init__()
        if net_mode == '2d':
            pool = nn.MaxPool2d
        elif net_mode == '3d':
            pool = nn.MaxPool3d
        else:
            pool = None
        # self.pub =pub(in_channels, out_channels,net_mode=net_mode)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        self.pub = as_block(in_channels, out_channels,net_mode=net_mode)
        self.pool = pool(2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True,net_mode= '3d'):
        super(unet3dUp, self).__init__()
        # self.pub = pub(in_channels  + out_channels, out_channels, batch_norm,net_mode=net_mode)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        self.pub = as_block(in_channels  + out_channels, out_channels,net_mode=net_mode)
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
        print('x1.shape',x1.shape,x.shape)
        x = torch.cat((x, x1), dim=1)
        print("卷积(pub)前通道数",x.shape)
        x = self.pub(x)  #self.pub = as_block(in_channels  + out_channels, out_channels,net_mode=net_mode)
        print("卷积后通道数", x.shape)
        return x

class as_UNet3D(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(as_UNet3D, self).__init__()

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

        print('dowx1', dowx1.shape)

        dowx2 = self.down2(dowx1)

        print('dowx2',dowx2.shape)

        dowx3 = self.down3(dowx2)

        print('dowx3', dowx3.shape)


        # bridgex = self.bridge(dowx4)
        print('-----------------------------------------')
        upx1 = self.up1(dowx3,dowx2)
        print('upx1', upx1.shape,'dowx3',dowx3.shape,'dowx2', dowx2.shape)
        print('-----------------------------------------')

        print('+++++++++++++++++++++++++++++++++++++++++')
        upx2 = self.up2(upx1,dowx1)
        print('up2', upx2.shape, 'upx1', upx1.shape, 'dowx1', dowx1.shape)
        print('+++++++++++++++++++++++++++++++++++++++++')

        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        upx3 = self.up3(upx2,x1)
        print('up3', upx3.shape, 'upx2', upx2.shape, 'x1', x1.shape)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        final_x = self.class_conv(upx3)
        print('final_x', final_x.shape)

        out_x = nn.Softmax(1)(final_x)
        print('out_x', out_x.shape)
        # out_x = final_x
        return out_x

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(1, 1, 32, 32, 32))
    print(img.shape)
    # inputs = [img,img2,img3]
    # # inputs = tuple(inputs)
    net = as_UNet3D(1, [32, 64, 96, 128, 256], 2, net_mode='3d')
    init_util.print_network(net)
    out = net(img)
    print(out.size())