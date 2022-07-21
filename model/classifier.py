import torch
import torch.nn.functional as F
import random
import numpy as np
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

class ASPP_Classifier(nn.Module):
    def __init__(self, in_channels, dilation_series, padding_series, num_classes):
        super(ASPP_Classifier, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x, size=None):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out


class ASPP_Classifier_Gen(nn.Module):
    '''Generalized version of ASPP head'''
    def __init__(self, in_channels, dilation_series, padding_series, num_classes, hidden_dim=128):
        super(ASPP_Classifier_Gen, self).__init__()
        self.head = ASPP_Classifier(in_channels, dilation_series, padding_series, hidden_dim)
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1) # Generalize DeepLabv2 to backbone + classifier structure (make classifier independent)

    def forward(self, x, size=None):
        out = self.head(x)
        out = self.classifier(out)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out
