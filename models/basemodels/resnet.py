# encoding: utf-8
"""
@author: Haoxu, Ming Cheng
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basemodels.base import ResidualBlock3D, ResidualBlock3D_SimAM, ResidualBlock2D, ResidualBlock2D_SimAM


class resnet18_2D(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(1,1), padding=(0,0))
        )

        self.layer1 = self._make_layer(32,   32, block_num=2, pool_size=(2,2))
        self.layer2 = self._make_layer(32,   64, block_num=2, pool_size=(2,2)) 
        self.layer3 = self._make_layer(64,  128, block_num=2, pool_size=(2,2))
        self.layer4 = self._make_layer(128, 256, block_num=2, pool_size=(2,2))

        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def freeze(self):
        for p in self.named_parameters():
            p[1].requires_grad = False

    def _make_layer(self, in_channels, out_channels, block_num, pool_size=1):
        
       
        layers = []
        layers.append(ResidualBlock2D(in_channels, out_channels, pool_size))
       
        for i in range(1, block_num):
            layers.append(ResidualBlock2D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # # (B, 64, 256)

        if len(x.size()) == 3:
            x = x.unsqueeze(1)
            # (B, 1, 64, 256)

        feats = {}

        x = self.pre(x)
        #feats['pred'] = x
        # (B, 32, 64, 256)

        x = self.layer1(x)
        feats['layer1'] = x
        # (B, 32, 32, 128)

        x = self.layer2(x)
        feats['layer2'] = x 
        # (B, 64, 16, 64)

        x = self.layer3(x)
        feats['layer3'] = x 
        # (B, 128, 8, 32)

        x = self.layer4(x)
        feats['layer4'] = x 
        # (B, 256, 4, 16)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        feats['embd'] = x 

        return x, feats


class resnet18_2D_SimAMBlock(resnet18_2D):
    def __init__(self, input_channels):
        super().__init__(input_channels)
    
    def _make_layer(self, in_channels, out_channels, block_num, pool_size=1):
        
       
        layers = []
        layers.append(ResidualBlock2D_SimAM(in_channels, out_channels, pool_size))
       
        for i in range(1, block_num):
            layers.append(ResidualBlock2D_SimAM(out_channels, out_channels))

        return nn.Sequential(*layers)



class resnet18_3D(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        )

        self.layer1 = self._make_layer(32,   32, block_num=2, pool_size=(1,2,2))
        self.layer2 = self._make_layer(32,   64, block_num=2, pool_size=(1,2,2)) 
        self.layer3 = self._make_layer(64,  128, block_num=2, pool_size=(1,2,2))
        self.layer4 = self._make_layer(128, 256, block_num=2, pool_size=(1,2,2))

        self.gap = nn.AdaptiveAvgPool3d((64,1,1))

    def _make_layer(self, in_channels, out_channels, block_num, pool_size=1):
        
       
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, pool_size))
       
        for i in range(1, block_num):
            layers.append(ResidualBlock3D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # (B, 1, 64, 80, 80) or (B, 3, 64, 112, 112)

        feats = {}

        x = self.pre(x)
        #feats['pred'] = x 

        x = self.layer1(x)
        feats['layer1'] = x 

        x = self.layer2(x)
        feats['layer2'] = x 

        x = self.layer3(x)
        feats['layer3'] = x 

        x = self.layer4(x)
        feats['layer4'] = x 

        x = self.gap(x)
        # (B, C, T, 1, 1)
        x = x.view(x.size(0), x.size(1), -1)

        x = x.permute(0,2,1)

        return x, feats

class resnet18_3D_SimAMBlock(resnet18_3D):
    def __init__(self, input_channels):
        super().__init__(input_channels)
    
    def _make_layer(self, in_channels, out_channels, block_num, pool_size=1):
        
       
        layers = []
        layers.append(ResidualBlock3D_SimAM(in_channels, out_channels, pool_size))
       
        for i in range(1, block_num):
            layers.append(ResidualBlock3D_SimAM(out_channels, out_channels))

        return nn.Sequential(*layers)