import torch
import torch.nn.functional as F
from incr_modules.masked_types import Masked
from metrics.structural_sparsity import field_channel_sparsity

total_ops = 0
def print_sparsity(x, prefix: str = ""):
    # total_ops += (x[0].numel()*3)
    print(prefix, torch.norm(x) )
    # print(prefix, float(field_channel_sparsity(x[0], field_size=1, threshold=0.0001).cpu().numpy()))

def count_ops(x: Masked, ksize):
    global total_ops
    print("sp: ", float(field_channel_sparsity(x[0], field_size=3, threshold=0.0001).cpu().numpy()),  total_ops   )
    total_ops += x[0].count_nonzero()*ksize
    # total_ops += x[0].numel()*ksize

    print(total_ops)

def compute_mask(x: torch.Tensor, field_size=5, threshold=0.0001):
    mask = F.max_pool2d(torch.abs(x), field_size, stride=(1, 1),
                        padding=((field_size - 1) // 2)).le(threshold)
    return mask
