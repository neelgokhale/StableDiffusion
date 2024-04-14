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
    LOG_VAR_CLAMP_LOWER = -30
    LOG_VAR_CLAMP_UPPER = 20    
    SCALING_CONST = 0.18215
    