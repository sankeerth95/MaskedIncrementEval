import torch
import torch.nn.functional as F
from metrics.structural_sparsity import field_channel_sparsity


def print_sparsity(x, prefix: str = ""):
    print(prefix, float(field_channel_sparsity(x, field_size=5, threshold=0.0001).cpu().numpy())  )


def compute_mask(x: torch.Tensor, field_size=5, threshold=0.00001):
    mask = F.max_pool2d(torch.abs(x), field_size, stride=(1, 1),
                        padding=((field_size - 1) // 2)).le(threshold)
    return mask
