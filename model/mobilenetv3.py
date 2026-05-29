"""
MobileNetV3-Small Backbone for YOLOv8
À placer dans : E:/yolov8_mobilenetv3/model/mobilenetv3.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    """Hard Sigmoid activation function"""
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish activation function (h-swish)"""
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer pour MobileNetV3"""
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBNHS(nn.Module):
    """Convolution + BatchNorm + h-swish"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = h_swish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MobileNetV3_InvertedResidual(nn.Module):
    """Inverted Residual Block de MobileNetV3"""
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            # Même canal d'entrée
            self.conv = nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if use_hs else nn.ReLU(inplace=True),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # Pointwise convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            # Expansion puis depthwise
            self.conv = nn.Sequential(
                # Expansion
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if use_hs else nn.ReLU(inplace=True),
                # Depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                nn.Hardswish() if use_hs else nn.ReLU(inplace=True),
                # Projection
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV3Stem(nn.Module):
    """Couche Stem (première convolution) de MobileNetV3"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = ConvBNHS(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)