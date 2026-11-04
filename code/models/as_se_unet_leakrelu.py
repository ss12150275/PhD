from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.init_util as init_util
from torchsummary import summary

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

        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        inpu=x
        x=self.gap(x)
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.sigmoid(x)

        return inpu*x

class as_block(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True,net_mode = '3d'):
        super(as_block, self).__init__()
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.InstanceNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.InstanceNorm3d
        else:
            conv = None
            bn = None
        inter_channels = in_channels//2 + in_channels if in_channels > out_channels else out_channels//2
        print('inter_channels',inter_channels)
        layers = [
                    conv(in_channels, inter_channels, 3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    conv(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.LeakyReLU(True)
                 ]
        layers2 = [
                    conv(in_channels, inter_channels,(1,3,3) , stride=1, padding= (0,1,1)),
                    nn.LeakyReLU(True),
                    conv(inter_channels, out_channels, (1,3,3), stride=1, padding=(0,1,1)),
                    nn.LeakyReLU(True)
        ]
        if batch_norm:
            layers.insert(1, bn(inter_channels))
            layers.insert(len(layers)-1, bn(out_channels))
            layers2.insert(1, bn(inter_channels))
            layers2.insert(len(layers) - 1, bn(out_channels))
        self.pub = nn.Sequential(*layers)
        self.as_conv = nn.Sequential(*layers2)
        self.cbam = CBAM(out_channels * 2)
        self.rasidual = conv(in_channels, out_channels, 1,stride = 1)

        self.conv_1x1 = conv(out_channels * 2, out_channels, 1)

    def forward(self, x):
        # print('x.shape',x.shape)

        rasidual = self.rasidual(x) #先卷积
        # print('rasidual.shape', rasidual.shape)

        spatial_x = self.pub(x)
        # print('spatial_x.shape', spatial_x.shape)

        palne_x = self.as_conv(x)
        # print('palne_x.shape', palne_x.shape)

        confuse_x = torch.cat((spatial_x,palne_x),dim=1)
        # print('confuse_x.shape', confuse_x.shape)

        at_x = self.cbam(confuse_x)
        # print('at_x.shape', at_x.shape)

        refine_x = self.conv_1x1(at_x)
        # print('refine_x.shape', refine_x.shape)

        # print('rasidual + refine_x',rasidual.shape, refine_x.shape)
        out = rasidual + refine_x
        return out

# class as_block(nn.Module):
#
#     def __init__(self, in_channels, out_channels, batch_norm=True,net_mode = '3d'):
#         super(as_block, self).__init__()
#         if net_mode == '2d':
#             conv = nn.Conv2d
#             bn = nn.BatchNorm2d
#         elif net_mode == '3d':
#             conv = nn.Conv3d
#             bn = nn.BatchNorm3d
#         else:
#             conv = None
#             bn = None
#         inter_channels = in_channels//2 + in_channels if in_channels > out_channels else out_channels//2
#         layers = [
#                     conv(in_channels, inter_channels, 3, stride=1, padding=1),
#                     nn.ReLU(True),
#                     conv(inter_channels, out_channels, 3, stride=1, padding=1),
#                     nn.ReLU(True)
#                  ]
#         layers2 = [
#             conv(in_channels, inter_channels,(1,3,3) , stride=1, padding= (0,1,1)),
#             nn.ReLU(True),
#             conv(inter_channels, out_channels, (1,3,3), stride=1, padding=(0,1,1)),
#             nn.ReLU(True)
#         ]
#         layers3 = [
#             conv(in_channels, inter_channels, (3, 3, 1), stride=1, padding=(1, 1, 0)),
#             nn.ReLU(True),
#             conv(inter_channels, out_channels, (3, 3, 1), stride=1, padding=(1, 1, 0)),
#             nn.ReLU(True)
#         ]
#         if batch_norm:
#             layers.insert(1, bn(inter_channels))
#             layers.insert(len(layers)-1, bn(out_channels))
#             layers2.insert(1, bn(inter_channels))
#             layers2.insert(len(layers) - 1, bn(out_channels))
#             layers3.insert(1, bn(inter_channels))
#             layers3.insert(len(layers) - 1, bn(out_channels))
#         self.pub = nn.Sequential(*layers)
#         self.as_conv = nn.Sequential(*layers2)
#         self.as_conv2 = nn.Sequential(*layers3)
#         self.cbam = CBAM(out_channels * 3)
#         self.rasidual = conv(in_channels, out_channels, 1)
#
#         self.conv_1x1 = conv(out_channels * 3, out_channels, 1)
#
#     def forward(self, x):
#         rasidual = self.rasidual(x)
#         spatial_x = self.pub(x)
#         palne_x = self.as_conv(x)
#         palne_x2 = self.as_conv2(x)
#         confuse_x = torch.cat((spatial_x,palne_x,palne_x2),dim=1)
#         at_x = self.cbam(confuse_x)
#         refine_x = self.conv_1x1(at_x)
#         out = rasidual + refine_x
#         return out

class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = in_channels//2 + in_channels if in_channels > out_channels else out_channels//2
        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.LeakyReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.InstanceNorm3d(inter_channels))
            layers.insert(len(layers)-1, nn.InstanceNorm3d(out_channels))
        self.pub = nn.Sequential(*layers)


    def forward(self, x):
        out = self.pub(x)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1   = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2   = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * x

class CBAM(nn.Module):
    def __init__(self,in_channels):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_channels, 16)
        self.sa = SpatialAttention(7)

        self.conv_du = nn.Sequential(
            # nn.Conv3d(in_planes, in_planes//ratio, 1,stride=1, padding=0),
            nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        sx = self.sa(x)
        x = x * sx
        cx = self.ca(x)
        y = x * cx
        out = self.conv_du(y)
        return out

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
        # self.pool = pool(2, stride=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True,net_mode= '3d'):
        super(unet3dUp, self).__init__()
        # self.pub = pub(in_channels  + out_channels, out_channels, batch_norm,net_mode=net_mode)# RecombinationBlock(in_channels//2+in_channels, out_channels, net_mode='3d')##
        self.pub = pub(in_channels  + out_channels, out_channels)
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

class as_at_UNet3D(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(as_at_UNet3D, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1, padding=1)

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

class as_at_UNet3D_deep(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(as_at_UNet3D_deep, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1, padding=1)

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1], net_mode=net_mode)
        self.down2 = unet3dDown(filter_num_list[1], filter_num_list[2], net_mode=net_mode)
        self.down3 = unet3dDown(filter_num_list[2], filter_num_list[3], net_mode=net_mode)

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up

        self.up1 = unet3dUp(filter_num_list[3], filter_num_list[2], net_mode=net_mode)
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1], net_mode=net_mode)
        self.up3 = unet3dUp(filter_num_list[1], filter_num_list[0], net_mode=net_mode)

        self.class_conv = conv(filter_num_list[0], class_num, 1)

        #####深监督 上采样
        self.deep_sample1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.deep_sample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deep_class_conv1 = conv(filter_num_list[2], class_num, 1)
        self.deep_class_conv2 = conv(filter_num_list[1], class_num, 1)


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

        # out_x = nn.Softmax(1)(final_x)
        out_x = final_x

        output1 = self.deep_class_conv1(self.deep_sample1(upx1))
        output2 = self.deep_class_conv2(self.deep_sample2(upx2))

        return out_x, output2, output1

class as_at_UNet3D_deep_leakrelu(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(as_at_UNet3D_deep_leakrelu, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.inc = conv(in_channels, filter_num_list[0], 3,stride=1, padding=1)

        # down
        self.down1 = unet3dDown(filter_num_list[0], filter_num_list[1], net_mode=net_mode)
        self.down2 = unet3dDown(filter_num_list[1], filter_num_list[2], net_mode=net_mode)
        self.down3 = unet3dDown(filter_num_list[2], filter_num_list[3], net_mode=net_mode)

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up

        self.up1 = unet3dUp(filter_num_list[3], filter_num_list[2], net_mode=net_mode)
        self.up2 = unet3dUp(filter_num_list[2], filter_num_list[1], net_mode=net_mode)
        self.up3 = unet3dUp(filter_num_list[1], filter_num_list[0], net_mode=net_mode)

        self.class_conv = conv(filter_num_list[0], class_num, 1)

        #####深监督 上采样
        self.deep_sample1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.deep_sample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deep_class_conv1 = conv(filter_num_list[2], class_num, 1)
        self.deep_class_conv2 = conv(filter_num_list[1], class_num, 1)


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

        # out_x = nn.Softmax(1)(final_x)
        out_x = final_x

        output1 = self.deep_class_conv1(self.deep_sample1(upx1))
        output2 = self.deep_class_conv2(self.deep_sample2(upx2))

        return out_x, output2, output1


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(1, 1, 192, 192, 192))
    print(img.shape)
    # inputs = [img,img2,img3]
    # # inputs = tuple(inputs)
    # net = ChannelAttention(128,4)
    # net = SpatialAttention(7)
    net = as_at_UNet3D_deep_leakrelu(1, [16, 64, 96, 128, 256], 2, net_mode='3d')
    init_util.print_network(net)
    out1, out2, out3 = net(img)



    # net = as_at_UNet3D_deep(1, [16, 64, 96, 128, 256], 2, net_mode='3d')
    # y = net(torch.randn(2, 1, 3, 224, 224))
    # print(y.size())
    # summary(net, (2, 1, 3, 224, 224))

