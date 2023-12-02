
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

import data

def slerp(p0: Tensor, p1: Tensor, t: float):
    # p0: b x n tensor
    # p1: b x n tensor
    coss = p0 * p1
    coss = torch.sum(coss, dim=1)
    sigma = torch.acos(coss)
    
    return torch.div(torch.sin((1-t)*sigma), torch.sin(sigma)).unsqueeze(-1) * p0 + \
            torch.div(torch.sin(t*sigma), torch.sin(sigma)).unsqueeze(-1) * p1


def lerp(p0: Tensor, p1: Tensor, t: float):
    return F.normalize((1-t)*p0 + t*p1, dim=0 if len(p0.shape)==1 else 1)
