import torch
import torch.nn as nn

from utils.str_sparse_utils import get_mask
from .model_handlers import BaselineModelhandler




class Conv2dBaseline(BaselineModelhandler):
    def __init__(self, shape=(32, 64), kernel = 3, padding=0, device='cuda'):
        super().__init__(nn.Conv2d(shape[0], shape[1], kernel_size=kernel, padding=padding).to(device))




# from ext.DeltaCNN.src.deltacnn.cuda_kernels import sparse_conv
# class DeltaConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super().__init__()
#         # self.filter = torch.randn(out_channels, kernel_size, kernel_size, in_channels).cuda()
#         self.filter = torch.randn(in_channels, kernel_size, kernel_size, out_channels).cuda()
    
#     def forward(self, x, mask=None):
#         return sparse_conv(x, self.filter, mask=mask)


# # only works on cuda
# class DeltaConvBaseline(BaselineModelhandler):
#     def __init__(self, shape=(32, 64), kernel=3, padding=0, device='cuda'):
#         op = DeltaConv(shape[0], shape[1], kernel).cuda()
#         super().__init__(op)

#     def run_once(self, input_sample):
#         return self.op(input_sample[0], input_sample[1])


