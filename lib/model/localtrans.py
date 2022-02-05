import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lib.model.network_utils import *
from ATT.attention_layer import Correlation, AttentionLayer
from lib.model.network_utils import *
from lib.image.warping import *


class conv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv3x3, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU())
    
    def forward(self, x):
        return self.conv(x)

class LocalTrans(nn.Module):
    def __init__(self):
        super(LocalTrans, self).__init__()
        self.conv1 = nn.Sequential(conv3x3(3, 32), conv3x3(32, 32), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(conv3x3(32, 64), conv3x3(64, 64), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(conv3x3(64, 64), conv3x3(64, 64), nn.MaxPool2d(2, 2))
        self.conv4 = nn.Sequential(conv3x3(64, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2))

        self.transformer1 = AttentionLayer(128, 4, 32, 32, 5, 2)
        self.transformer2 = AttentionLayer(64, 2, 32, 32, 7, 3)
        self.transformer3 = AttentionLayer(64, 2, 32, 32, 9, 4)
        self.transformer4 = AttentionLayer(64, 2, 32, 32, 9, 4)
        self.transformer5 = AttentionLayer(64, 2, 32, 32, 9, 4)
        self.transformer = [self.transformer1, self.transformer2, self.transformer3, self.transformer4, self.transformer5]

        self.homo1 = nn.Sequential(conv3x3(25, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2), 
            conv3x3(256, 256), conv3x3(256, 256), nn.AvgPool2d(2, 2), nn.Conv2d(256, 8, 1))
        self.homo2 = nn.Sequential(conv3x3(49, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2), 
            conv3x3(256, 256), conv3x3(256, 256), nn.AvgPool2d(2, 2), nn.Conv2d(256, 8, 1))
        self.homo3 = nn.Sequential(conv3x3(81, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2), 
            conv3x3(256, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2),
            conv3x3(256, 256), conv3x3(256, 256), nn.AvgPool2d(2, 2), nn.Conv2d(256, 8, 1))
        self.homo4 = nn.Sequential(conv3x3(81, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2), 
            conv3x3(256, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2),
            conv3x3(256, 256), conv3x3(256, 256), nn.AvgPool2d(2, 2), nn.Conv2d(256, 8, 1))
        self.homo5 = nn.Sequential(conv3x3(81, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2), 
            conv3x3(256, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2),
            conv3x3(256, 256), conv3x3(256, 256), nn.AvgPool2d(2, 2), nn.Conv2d(256, 8, 1))

        self.homo_estim = [self.homo1, self.homo2, self.homo3, self.homo4, self.homo5]

        self.kernel_list = [5, 7, 9, 9, 9]
        self.pad_list = [2, 3, 4, 4, 4]
        self.scale_list = [16, 8, 4, 4, 4]
        self.bias_list = [2, 1, 0.5, 0.25, 0.125]

    def forward(self, x, y, L, show=False):
        device = x.device
        B, C, H, W = x.shape
        
        x, y = self.conv2(self.conv1(x)), self.conv2(self.conv1(y))
        if L <= 1:
            x, y = self.conv3(x), self.conv3(y)
        if L <= 0:
            x, y = self.conv4(x), self.conv4(y)

        transformer = self.transformer[L]
        x, y = transformer(x, y)
            
        scale = self.scale_list[L]
        corr = Correlation.apply(x.contiguous(), y.contiguous(), self.kernel_list[L], self.pad_list[L])
        corr = corr.permute(0, 3, 1, 2) / x.shape[1]
        homo_flow = self.homo_estim[L](corr) * scale * self.bias_list[L]
        if show:
            return homo_flow.reshape(B, 2, 2, 2), corr, x, y
        else:
            return homo_flow.reshape(B, 2, 2, 2)