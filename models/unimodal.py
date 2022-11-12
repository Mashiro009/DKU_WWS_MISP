# encoding: utf-8
"""
@author: Haoxu, Ming Cheng
"""



import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodels.resnet import resnet18_3D, resnet18_2D, resnet18_3D_SimAMBlock, resnet18_2D_SimAMBlock


class UniModel(nn.Module):
    def __init__(self, input_channels, num_class):
        super().__init__()

        self.num_class = num_class

        self.encoder = resnet18_3D(input_channels)
        self.decoder = resnet18_2D(1)

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,  out_features=self.num_class),
        )

    def forward(self, x):
        feats = {}
        x, _ = self.encoder(x)
        feats['encoder_feats'] = _
        feats['encoder_embd'] = x
        x, _ = self.decoder(x)
        feats['decoder_feats'] = _
        x = self.fc(x)

        return x, feats

class UniModel_SimAMBlock(UniModel):
    def __init__(self, input_channels, num_class):
        super().__init__(input_channels, num_class)

        self.encoder = resnet18_3D_SimAMBlock(input_channels)
        self.decoder = resnet18_2D_SimAMBlock(1)