import random

import numpy as np
import torch
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def nhwc_to_nchw(x):
    # Input:
        # x: (N, H, W, C)
    # Output:
        # tensor: (N, C, H, W)
    return torch.from_numpy(x).permute(0, 3, 1, 2).float()