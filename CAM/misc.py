import numpy as np
import torch


def convert_to_gray(x, percentile=99):
    """
    Args:
        x: torch tensor with shape of (1, 3, H, W)
        percentile: int
    Return:
        result: shape of (1, 1, H, W)
    """
    x_2d = torch.abs(x).sum(dim=1).squeeze(0)
    v_max = np.percentile(x_2d, percentile)
    v_min = torch.min(x_2d)
    torch.clamp_((x_2d - v_min) / (v_max - v_min), 0, 1)
    return x_2d.unsqueeze(0).unsqueeze(0)