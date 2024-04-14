# ../src/encoder.py

import os
import torch

from torch import nn
from torch.nn import functional as F
from dotenv import load_dotenv
from decoder import VAEAttentionBlock, VAEResidualBlock

load_dotenv()

from constants import Constant as c

class VAEEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size, channel, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 125, kernel_size=3, padding=1),
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAEResidualBlock(128, 128),
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAEResidualBlock(128, 128),
            # (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (batch_size, 128, height/2, width/2)) -> (batch_size, 256, height/2, width/2)
            VAEResidualBlock(128, 256),
            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAEResidualBlock(256, 256),
            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAEResidualBlock(256, 512),
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAEResidualBlock(512, 512),
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAEAttentionBlock(512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAEResidualBlock(512, 512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.GroupNorm(32, 512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.SiLU(),
            # padding = 1, so the width and height will increase by 2 each
            # out_height = in_height + pad_top + pad_bot
            # out_width = in_width + pad_left + pad_right, pad_dim = 1
            # since out_height/width = in_height/width + 2, will compensate for kernel_size
            # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel, height, width)
        # noise: (batch_size, out_channels, height/8, width/8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (pad_left, pad_right, pad_top, pad_bot)
                # asym padding: add a layer of padding on the right and bottom of img
                x = F.pad(x, (0, 1, 0, 1))
                x = module(x)

        # divide along channel dimension (batch_size, 8, height/8, width/8) -> 
        # 2 tensors (batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # clamp the variance
        log_variance = torch.clamp(
            log_variance,
            c.LOG_VAR_CLAMP_LOWER,
            c.LOG_VAR_CLAMP_UPPER
        )
        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        # Z = N(0, 1) -> X = N(mean, variance), what is X?
        # X = mean + stdev * Z
        x = mean + stdev * noise
        
        # scale output by constant, constant from original paper
        x *= c.SCALING_CONST
        
        return x
