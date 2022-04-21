import torch
import torch.nn.functional as F

# gives the places to apply convolution
def get_mask(x, field_size: int=3, threshold: float=0.0001):
    assert len(x.shape) == 4 # NCHW input

    # number of nonzeros: places to apply convolution
    mask = F.max_pool2d(torch.abs(x), field_size, stride=(1, 1),
                        padding=((field_size - 1) // 2)).ge(threshold).int()

    # places to apply convolution
    return mask


