import torch
import torch.nn.functional as F
from incr_modules.fenced_module_masked import IncrementMaskModule
from ._C_ext.pointops.pointops_ext import activation_increment, conv3x3_increment, conv3x3_increment_ext

def activation_incr(x, input_incr):
    output_= torch.empty(x.shape, dtype=torch.float, device='cuda')
    activation_increment(x, input_incr, output_)
    return output_


class ActivationIncr(IncrementMaskModule):

    def __init__(self, shape, device='cuda'):
        super().__init__()
        self.reservoir = torch.zeros(shape).to(device)

    def forward(self, x_incr):
        activation_incr(self.reservoir, x_incr[0])

    def forward_refresh_reservoir(self, x):
        self.reservoir = x.clone().detach()
        return torch.relu(x)


# convolution wrapper
# filter: (out, in, f_h, f_w)
def conv3x3_incr(x, filter):
    output_= torch.empty((1, filter.shape[0], x.shape[2], x.shape[3]), dtype=torch.float, device='cuda')
    conv3x3_increment(x[0], filter, output_[0])
    return output_


# x should be an nhwc non-contiguous tensor; 
def conv3x3_incr_ext(x, filter, c_out, mask=None):
    # NHWC format (noncontiguous input))
    output_= torch.empty((x.shape[0], c_out, x.shape[2], x.shape[3]), dtype=torch.float, device='cuda', memory_format=torch.channels_last)
    conv3x3_increment_ext(x, mask, filter, output_)
    return output_

def convert_filter_out_channels_last(filter, transposed=False):
    if transposed:
        return torch.transpose(torch.transpose(filter, 2, 1), 3, 2).contiguous().clone()
    return torch.transpose(torch.transpose(torch.transpose(filter, 1, 0), 2, 1), 3, 2).contiguous().clone()


class Conv3x3Incr(IncrementMaskModule):

    def __init__(self, in_shape, c_in, c_out, weight, padding=0, device='cuda'):
        super().__init__()
        self.weight = weight
        self.weight_t = convert_filter_out_channels_last(weight)
        self.c_in = c_in
        self.c_out = c_out
        self.mask = torch.rand(in_shape, device=device).le(.1)

    # expects nhwc tensor
    def forward(self, x_incr):
        # print(x_incr.shape, self.weight.shape, self.c_out)
        conv3x3_incr_ext(x_incr, self.weight_t, self.c_out, self.mask)

    def forward_refresh_reservoir(self, x):
        return F.conv2d(x, self.weight)


