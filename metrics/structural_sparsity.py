import torch
import torch.nn.functional as F

# def blockwise_sparsity(x: torch.Tensor, field_size: int, threshold: float) -> float:
#     return 0.

# NCHW assumpyion
def field_channel_sparsity(x: torch.Tensor, field_size: int, threshold: float=0.000001) -> float:

    # number of nonzeros
    mask = F.max_pool2d(torch.abs(x), field_size, stride=(1, 1),
                        padding=((field_size - 1) // 2)).le(threshold).int()

    return mask.sum()/mask.numel()
