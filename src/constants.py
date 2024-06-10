# ../src/constants.py

import os
import torch

from dotenv import load_dotenv

load_dotenv()


class Constant:
    """Misc. constants used throughout library"""
    
    # torch
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() \
        else torch.device("cpu")
    
    # encoder / decoder
    LOG_VAR_CLAMP_LOWER: int = -30
    LOG_VAR_CLAMP_UPPER: int = 20    
    SCALING_CONST: float = 0.18215
    
    # CLIP
    CLIP_VOCAB_SIZE: int = 49408
    CLIP_EMBEDDING_SIZE: int = 768
    CLIP_MAX_SEQ_LEN: int = 77

    GELU_CONST = 1.702
    
    # diffusion
    TIME_EMBEDDING_SIZE = 320
    