import torch
import torch.nn as nn
from torch.nn import functional as F

# class base_unet(nn.Module):
#     def __init__(self,in_channels,out_channels,mode_choose = '3d'):
#         if mode_choose == '2d':
#             conv =nn.Conv2d
#             bn = nn.BatchNorm2d
#         elif mode_choose =='3d':
#             conv = nn.Conv3d
#             bn = nn.BatchNorm3d
#         else:
#             conv = None
#             bn = None
#
# 基本卷积块
# class Conv(nn.Module):
#     def __init__(self, C_in, C_out,mode_choose='3d'):
#         super(Conv, self).__init__()
#         self.layer = nn.Sequential(
#
#             nn.Conv2d(C_in, C_out, 3, 1, 1),
#             nn.BatchNorm2d(C_out),
#             nn.Dropout(0.3),
#             nn.LeakyReLU(),
#
#             nn.Conv2d(C_out, C_out, 3, 1, 1),
#             nn.BatchNorm2d(C_out),
#             nn.Dropout(0.4),
#             nn.LeakyReLU(),
#         )
#
#     def forward(self, x):
#         return self.layer(x)
class Factory(dict):

    def __call__(self, _module, *_Args, **_KWargs):

        print(_module)
        print(_Args)
        print(_KWargs)
        return self[_module](*_Args, **_KWargs)

    @classmethod
    def Get2DFactory(cls):
        fac = cls()
        fac.update({
            "Conv": nn.Conv2d,
            "BatchNorm": nn.BatchNorm2d,


        })
        return fac

    def register(self, _value):
        self.update({ _value.__name__ : _value })
        return _value

    def name_reigter(self, _name):
        def wrapper(_module):
            self.update({_name: _module})
            return  _module
        return wrapper

fac2d = Factory()
fac2d.update({
    "Conv": nn.Conv2d,
    "BatchNorm": nn.BatchNorm3d
})

fac3d = Factory({
    "Conv": nn.Conv3d,
    "BatchNorm": nn.BatchNorm3d

})
fac3d.update({})
fac3d.name_reigter("kk")(nn.MaxPool3d)
fac3d.register(nn.ReLU)


@fac3d.name_reigter("kk")
class ModeChoose(nn.Module):
    def __int__(self, mode='3d'):
        super(ModeChoose, self).__int__()
        self._mode = mode
        if self._mode == '3d':
            self._fac = Factory.Get2DFactory()

        elif self._mode == '2d':
            self._conv = nn.Conv2d
            self._bn = nn.BatchNorm2d

        conv = self._fac["Conv"]()

class base_conv(ModeChoose):
    def __init__(self, in_channel, out_channel):

        super(base_conv, self).__init__()

        self.layer = nn.Sequential(
            self._fac["Conv"](in_channel, out_channel, 3, 1, 1),
            self._bn(out_channel),
            nn.Dropout3d(0.3),
            nn.LeakyReLU(),

            ModeChoose.conv(out_channel, out_channel, 3, 1, 1),
            ModeChoose.bn(out_channel),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )
    def forwad(self,x):
        return self.layer(x)

class downsampling(ModeChoose):
    def __int__(self,in_channel,out_channel):
        super(downsampling, self).__int__()
        self.down = nn.Sequential(

        )

class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)

if __name__ == "__main__":
    print(fac3d("Conv", 1, 2, 3, stride = (3, 1, 3), padding = 2))
    print(fac3d("BatchNorm", 100))
    print(fac3d)
    print(ModeChoose)
    print("pass")