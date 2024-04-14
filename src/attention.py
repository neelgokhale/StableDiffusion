# ../src/attention.py

import os
import torch
import math

from torch import nn
from torch.nn import functional as F
from dotenv import load_dotenv
load_dotenv()

from constants import Constant as c


class SelfAttention(nn.Module):
    def __init__(self, 
                 n_heads: int, 
                 d_embed: int, 
                 in_proj_bias: bool=True, 
                 out_proj_bias: bool=True) -> None:
        super().__init__()
        
        # W_q, W_k, W_v  represented by a linear matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, causal_mask: bool=False) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 3) ->
        # 3 tensors of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, h, dim/h) ->
        # (batch_size, h, seq_len, dim/h)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        # calculate attention
        # (batch_size, h, seq_len, dim/h) @ (batch_size, h, dim/h, seq_len) -> 
        # (batch_size, h, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)
        
        # create and apply causal mask
        if causal_mask:
            # mask where upper triangle is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        
        # apply softmax
        weight = F.softmax(weight, dim=-1)
        
        # generate output matrix
        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, dim/h) -> 
        # (batch_size, h, seq_len, dim/h)
        output = weight @ v
        
        # (batch_size, h, seq_len, dim/h) -> (batch_size, seq_len, h, dim/h)
        output = output.transpose(1, 2)
        
        # (batch_size, h, seq_len, dim/h) -> (batch_size, seq_len, dim)
        output = output.reshape(input_shape)
        
        # (batch_size, seq_len, dim)
        output = self.out_proj(output)
