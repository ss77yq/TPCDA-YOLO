import numpy as np
import torch
from torch import nn
from torch.nn import init

# HardSigmoid
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

# HardSwish激活函数
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)



# lCAM通道注意力模块
class LCAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 共享卷积
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, stride=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

# LDSAM空间注意力模块
class LDSAM(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        # self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class LCBHAMBlock(nn.Module):
    def __init__(self, c, reduction=16, kernel_size=3):
        super().__init__()
        # 代替CBS作用（激活函数改进）
        self.CBS_new = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            h_swish()
        )
        # 通道注意力
        self.ca = LCAM(channel=c, reduction=reduction)
        # 空间注意力
        self.sa = LDSAM(kernel_size=kernel_size)

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.kaiming_normal_(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.normal_(m.weight, std=0.001)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.CBS_new(x)
        x = x * self.ca(x)
        out = x * self.sa(x)
        return out


if __name__ == '__main__':
    input = torch.randn(1, 192, 32, 32)
    # 应用
    cbam = LCBHAMBlock(c=192)

    output = cbam(input)
    print(output.shape)