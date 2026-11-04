from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


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
        inter_channels = in_channels // 2 + in_channels if in_channels > out_channels else out_channels // 2
        # self.pub =pub(in_channels, out_channels,net_mode=net_mode)# RecombinationBlock(in_channels, out_channels, net_mode='3d')#
        self.pub1 = conv_single(in_channels, inter_channels,3,1,1,1, net_mode=net_mode)
        self.pub2 = conv_single(inter_channels, out_channels, 3, 1, 1, 1, net_mode=net_mode)
        self.pool = pool(2, stride=2)

    def forward(self, x):
        x1 = self.pub1(x)
        x2 = self.pub2(x1)
        out_x = self.pool(x2)
        return out_x

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
        # print('relu之后的结果',x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, net_mode='2d'):
        super(Discriminator, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None


        # self.inc = conv(in_channels, filter_num_list[0], 3,stride=1,padding=1)

        # down
        self.down1 = unetDown(in_channels, filter_num_list[0],net_mode = net_mode)
        self.down2 = unetDown(filter_num_list[0], filter_num_list[1],net_mode = net_mode)
        self.down3 = unetDown(filter_num_list[1], filter_num_list[2],net_mode = net_mode)

        # self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(filter_num_list[2]*8*8*8, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, input1):

        x1 = input1
        # print(x1.shape,x2.shape,x3.shape)
        batch = x1.shape[0]

        # x1 = self.inc(x)

        x1_1 = self.down1(x1)
        # print('dowx1', x1_1.shape)
        x1_2 = self.down2(x1_1)
        # print('dowx2',x1_2.shape)
        x1_3 = self.down3(x1_2)
        # print('dowx2',x1_2.shape)
        x1_3 = x1_3.view(batch,-1)
        # print('x1_2',x1_2.shape)

        # print('merge',merge.shape)


        f1_x = self.fc1(x1_3)
        f1_x = self.bn(f1_x)
        # f1_x = self.dropout(f1_x)
        # print(f1_x)

        f2_x = self.fc2(f1_x)
        # print('f2_x',f2_x)
        out_x = nn.Sigmoid()(f2_x)

        return out_x


def main():
    args = config.args
    # torch.cuda.set_device(1)
    device = torch.device('cpu' if args.cpu else 'cuda')
    model = Discriminator(1, [16, 32, 64, 128, 256], 2, net_mode='3d').to(device)
    x=torch.rand(4,1,64,64,64)
    x=x.to(device)
    # model.forward(x)
    out = model(x)
    print(out,out.shape)

    # value = 1
    # batch = x.shape[0]
    # target = []
    # for i in range(batch):
    #     target.append([value])
    # a = torch.tensor(target)
    # print(a)

if __name__ == '__main__':
    main()