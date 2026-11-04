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

class conv_dilated(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch,kernel,stride,pad,rate,net_mode='3d'):
        super(conv_dilated, self).__init__()
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

    def __init__(self, in_channels, out_channels, batch_norm=True,net_mode= '3d'):
        super(unet3dDown, self).__init__()
        if net_mode == '2d':
            pool = nn.MaxPool2d
        elif net_mode == '3d':
            pool = nn.MaxPool3d
        else:
            pool = None
        self.pub =pub(in_channels, out_channels,net_mode=net_mode)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        # self.pub = PDSE(in_channels, out_channels,net_mode=net_mode)
        self.pool = pool(2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True,net_mode= '3d'):
        super(unet3dUp, self).__init__()
        self.pub = pub(in_channels  + out_channels, out_channels, batch_norm,net_mode=net_mode)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        # self.pub = PDSE(in_channels  + out_channels, out_channels,net_mode=net_mode)
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

class PDRD2(nn.Module):
    def __init__(self, in_channel, out_channel,net_mode = '3d'):
        super(PDRD2, self).__init__()
        rate = [1,2,3]
        if net_mode == '2d':
            conv = nn.Conv2d
            avpool = nn.AdaptiveAvgPool2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            avpool = nn.AdaptiveAvgPool3d
        else:
            conv = None
            avpool = None
        inter_channels = in_channel // 2 + in_channel if in_channel > out_channel else out_channel // 2

        # self.pool = nn.MaxPool3d(2, stride=2)

        self.rasidual = conv(in_channel, out_channel, 3, stride=1, padding=1)

        self.atrous_block1 = conv_dilated(in_channel, inter_channels, 3, 1, rate[0],rate[0])
        self.atrous_block2 = conv_dilated(in_channel, inter_channels, 3, 1, rate[1],rate[1])
        self.atrous_block3 = conv_dilated(in_channel, inter_channels, 3, 1, rate[2],rate[2])

        self.conv = conv(inter_channels, out_channel, 3, stride=1, padding=1)

    def forward(self, x):
        # x = self.pool(x)
        rasidual = self.rasidual(x)

        atrous_block1 = self.atrous_block1(x)
        # print('size 1',atrous_block1.shape)
        atrous_block2 = self.atrous_block2(x)
        # print('size 6', atrous_block6.shape)
        atrous_block3 = self.atrous_block3(x)
        # print('size 12', atrous_block12.shape)

        # pd = atrous_block1 + atrous_block2 + atrous_block3
        pd = torch.add(torch.add(atrous_block1,atrous_block2),atrous_block3)
        pd2 = self.conv(pd)

        out = pd2 + rasidual
        # print('size net', net.shape)
        return out

class PDRD(nn.Module):
    def __init__(self, in_channel, out_channel,net_mode = '3d'):
        super(PDRD, self).__init__()
        rate = [1,2,3]
        if net_mode == '2d':
            conv = nn.Conv2d
            avpool = nn.AdaptiveAvgPool2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            avpool = nn.AdaptiveAvgPool3d
        else:
            conv = None
            avpool = None
        inter_channels = in_channel // 2 + in_channel if in_channel > out_channel else out_channel // 2

        # self.pool = nn.MaxPool3d(2, stride=2)

        self.rasidual = conv(in_channel, out_channel, 1)

        self.conv_1 = conv(in_channel, inter_channels, 1)

        self.atrous_block1 = conv_dilated(inter_channels, out_channel, 3, 1, rate[0],rate[0])
        self.atrous_block2 = conv_dilated(inter_channels, out_channel, 3, 1, rate[1],rate[1])
        self.atrous_block3 = conv_dilated(inter_channels, out_channel, 3, 1, rate[2],rate[2])

        self.conv = conv(out_channel, out_channel, 3, stride=1, padding=1)

    def forward(self, x):
        # x = self.pool(x)
        rasidual = self.rasidual(x)
        # print('size ra',rasidual.shape)

        atrous_block1 = self.atrous_block1(self.conv_1(x))
        # print('size 1',atrous_block1.shape)
        atrous_block2 = self.atrous_block2(self.conv_1(x))
        # print('size 6', atrous_block2.shape)
        atrous_block3 = self.atrous_block3(self.conv_1(x))
        # print('size 12', atrous_block3.shape)

        # pd = atrous_block1 + atrous_block2 + atrous_block3
        pd = torch.add(torch.add(atrous_block1,atrous_block2),atrous_block3)
        pd2 = self.conv(pd)

        out = pd2 + rasidual
        # print('size net', net.shape)
        return out

class PDSE(nn.Module):
    def __init__(self, in_channel, out_channel,net_mode = '3d'):
        super(PDSE, self).__init__()
        rate = [1,2,3]
        if net_mode == '2d':
            conv = nn.Conv2d
            avpool = nn.AdaptiveAvgPool2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            avpool = nn.AdaptiveAvgPool3d
        else:
            conv = None
            avpool = None
        inter_channels = in_channel // 2 + in_channel if in_channel > out_channel else out_channel // 2

        # self.pool = nn.MaxPool3d(2, stride=2)

        self.rasidual = conv(in_channel, out_channel, 1)

        self.conv_1 = conv(in_channel, inter_channels, 1)

        self.atrous_block1 = conv_dilated(inter_channels, out_channel, 3, 1, rate[0],rate[0])
        self.atrous_block2 = conv_dilated(inter_channels, out_channel, 3, 1, rate[1],rate[1])
        self.atrous_block3 = conv_dilated(inter_channels, out_channel, 3, 1, rate[2],rate[2])

        self.se = SEBlock(out_channel*3,out_channel*3,net_mode=net_mode)

        self.conv_1x1 = conv(out_channel*3, out_channel, 1)

        # self.conv = conv_dilated(out_channel, out_channel, 3, 1, 1, 1)

    def forward(self, x):
        # x = self.pool(x)
        rasidual = self.rasidual(x)
        # print('size ra',rasidual.shape)

        atrous_block1 = self.atrous_block1(self.conv_1(x))
        # print('size 1',atrous_block1.shape)
        atrous_block2 = self.atrous_block2(self.conv_1(x))
        # print('size 6', atrous_block2.shape)
        atrous_block3 = self.atrous_block3(self.conv_1(x))
        # print('size 12', atrous_block3.shape)

        # pd = atrous_block1 + atrous_block2 + atrous_block3
        se = self.se(torch.cat((atrous_block1,atrous_block2,atrous_block3),dim = 1))
        pd = self.conv_1x1(se)
        # pd2 = self.conv(pd)

        out = pd + rasidual
        # print('size net', net.shape)
        return out

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

class N_UNet3D(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(N_UNet3D, self).__init__()

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






if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(4, 1, 32, 32, 32))
    print(img.shape)
    # inputs = [img,img2,img3]
    # # inputs = tuple(inputs)
    net = N_UNet3D(1, [32, 64, 96, 128, 256], 2, net_mode='3d')
    init_util.print_network(net)
    out = net(img)
    print(out.size())