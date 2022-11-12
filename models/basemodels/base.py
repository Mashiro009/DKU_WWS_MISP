# encoding: utf-8
"""
@author: Haoxu, Ming Cheng
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimAM3D(nn.Module):
    # X: input feature [N, C, T, H, W]
    # lambda: coefficient λ in Eqn (5)
    def forward(self, X, lambda_w=1e-4):
        # spatial size
        n = X.shape[2] * X.shape[3] * X.shape[4] - 1
        # square of (t - u)
        d = (X - X.mean(dim=[2,3,4], keepdim=True)).pow(2)
        # d.sum() / n is channel variance
        v = d.sum(dim=[2,3,4], keepdim=True) / n
        # E_inv groups all importance of X
        E_inv = d / (4 * (v + lambda_w)) + 0.5
        # return attended features
        return X * torch.sigmoid(E_inv)

class SimAM(nn.Module):
    # X: input feature [N, C, H, W]
    # lambda: coefficient λ in Eqn (5)
    def forward(self, X, lambda_w=1e-4):
        # spatial size
        n = X.shape[2] * X.shape[3] - 1
        # square of (t - u)
        d = (X - X.mean(dim=[2,3], keepdim=True)).pow(2)
        # d.sum() / n is channel variance
        v = d.sum(dim=[2,3], keepdim=True) / n
        # E_inv groups all importance of X
        E_inv = d / (4 * (v + lambda_w)) + 0.5
        # return attended features
        return X * torch.sigmoid(E_inv)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=1, kernel_size=3, padding=1):
        super().__init__()
       
        self.left = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=pool_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels,out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm3d(out_channels)
        )
        
        self.shortcut = None
        if in_channels != out_channels or (isinstance(pool_size, tuple)):
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=pool_size, padding=0),
                nn.BatchNorm3d(out_channels)
            )
            
            
        return

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)

        out = self.left(x) + identity
        out = F.relu(out)
        
        return out

class ResidualBlock3D_SimAM(ResidualBlock3D):

    def __init__(self, in_channels, out_channels, pool_size=1, kernel_size=3, padding=1):
        super().__init__(in_channels, out_channels, pool_size, kernel_size, padding)

        self.left = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=pool_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels,out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            SimAM3D(),
            nn.BatchNorm3d(out_channels)
        )
        

class ResidualBlock2D(nn.Module):
    def __init__(self,in_channels, out_channels, pool_size=1):
        super().__init__()
       
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=pool_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = None
        if in_channels != out_channels or (isinstance(pool_size, tuple)):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=pool_size, padding=0),
                nn.BatchNorm2d(out_channels)
            )
            
            
        return

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)

        out = self.left(x) + identity
        out = F.relu(out)
        
        return out

class ResidualBlock2D_SimAM(ResidualBlock2D):
    def __init__(self, in_channels, out_channels, pool_size=1):
        super().__init__(in_channels, out_channels, pool_size)

        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=pool_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels, kernel_size=3, stride=1, padding=1),
            SimAM(),
            nn.BatchNorm2d(out_channels)
        )