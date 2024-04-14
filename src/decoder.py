# ../src/decoder.py

import os
import torch

from torch import nn
from torch.nn import functional as F
from dotenv import load_dotenv
load_dotenv()

from constants import Constant as c
from attention import SelfAttention


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, features, height, width)
        residue = x
        n, c, h, w = x.shape
        
        # (batch_size, features, height, width) -> (batch_size, features, height * width)
        x = x.view(n, c, h * w)
        
        # (batch_size, features, height * width) -> (batch_size, height * width, features)
        x = x.transpose(-1, -2)
        
        # (batch_size, height * width, features) -> (batch_size, height * width, features)
        x = self.attention(x)
        
        # (batch_size, height * width, features) -> (batch_size, features, height * width)
        x = x.transpose(-1, -2)
        
        # (batch_size, features, height * width) -> (batch_size, features, height, width) 
        x = x.view((n, c, h, w))

        x += residue
        
        return x

class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # skip connection if input channel and output channel dims should
        # be the same
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, 
                                            out_channels, 
                                            kernel_size=1,
                                            padding=0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, height, width)
        
        residue = x
        x = self.group_norm1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        
        x = self.group_norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return x + self.residual_layer(residue)


class VAEDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            
            nn.Conv2d(4, 512, kernel_shape=3, padding=1),
            
            VAEResidualBlock(512, 512),
            
            VAEAttentionBlock(512),
            # (batch_size, 512, height/8, height/8) -> (batch_size, 512, height/8, height/8)
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            # (batch_size, 512, height/8, height/8) -> (batch_size, 512, height/4, height/4)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            
            # (batch_size, 512, height/4, height/4) -> (batch_size, 512, height/2, height/2)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),
            # (batch_size, 512, height/2, height/2) -> (batch_size, 512, height, height)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128),
            
            nn.SiLU(),
            
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Torch) -> torch.Tensor:
        # x: (batch_size, 4, height/8, width/8):
        
        x /= c.SCALING_CONST
        
        for module in self:
            x = module(x)
            
        return x