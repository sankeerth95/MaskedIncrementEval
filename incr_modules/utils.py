import torch
import torch.nn.functional as F
from incr_modules.masked_types import Masked
from metrics.structural_sparsity import field_channel_sparsity


def print_sparsity(x:Masked, prefix: str = ""):
    pass
    # print(prefix, float(field_channel_sparsity(x[0], field_size=3, threshold=0.0001).cpu().numpy())  )


def compute_mask(x: torch.Tensor, field_size=5, threshold=0.0001):
    mask = F.max_pool2d(torch.abs(x), field_size, stride=(1, 1),
                        padding=((field_size - 1) // 2)).le(threshold)
    return mask
