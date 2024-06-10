# ../src/clip.py

import os
import torch

from torch import nn
from torch import LongTensor, FloatTensor
from torch.nn import functional as F
from dotenv import load_dotenv
load_dotenv()

from attention import SelfAttention
from constants import Constant as c

class Clip(nn.Module):
    
    def __init__(self):
        self.embedding = CLIPEmbedding(
            c.CLIP_VOCAB_SIZE,
            c.CLIP_EMBEDDING_SIZE,
            c.CLIP_MAX_SEQ_LEN
        )
        
        self.layers = nn.Module([
            CLIPLayer(12, c.CLIP_EMBEDDING_SIZE) for i in range(12)
        ])
        
        self.layernorm = nn.LayerNorm(c.CLIP_EMBEDDING_SIZE)
        
    def forward(self, tokens: LongTensor) -> FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        # (batch_size, seq_len, dim)
        output = self.layernorm(state)
        
        return output
    

class CLIPEmbedding(nn.Module):
    
    def __init__(
        self,
        n_vocab: int,
        n_embeddings: int,
        n_tokens: int
        ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embeddings)
        self.positional_embedding = nn.Parameter(
            torch.zeros(n_tokens, n_embeddings)
        )
        
    def forward(self, tokens: LongTensor):
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.token_embedding(tokens)
        x += self.positional_embedding
        
        # (batch_size, seq_len, dim)
        return x
    

class CLIPLayer(nn.Module):
    
    def __init__(self, n_head: int, n_embeddings: int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embeddings)
        self.attention = SelfAttention(
            n_heads=n_head,
            d_embed=n_embeddings
        )
        self.layernorm_2 = nn.LayerNorm(n_embeddings)
        self.linear_1 = nn.Linear(n_embeddings, 4 * n_embeddings)
        self.linear_2 = nn.Linear(4 * n_embeddings, n_embeddings)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, dim)
        residue = x
        
        # SELF ATTENTION
        x = self.layernorm_1(x)
        # apply a causal mask as this is a text transformer
        z = self.attention(x, causal_mask=True)
        
        x += residue
        
        # FEED FORWARD
        residue = x
        x = self.layernorm_2(x)
        # apply first linear layer of FF
        x = self.linear_1(x)
        # Quick GELU activation function
        x = x * torch.sigmoid(x * c.GELU_CONST)
        x = self.linear_2(x)
        
        x += residue
        
        return x
