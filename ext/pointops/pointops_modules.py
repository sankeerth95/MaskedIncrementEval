import torch
import torch.nn.functional as F
from incr_modules.mask_incr_modules import IncrementMaskModule
from ._C_ext.pointops.pointops_ext import conv3x3_increment_ext, conv5x5_increment_ext, conv1x1_increment_ext
from .pointops_functional import activation_incr


class ActivationIncr(IncrementMaskModule):
    def __init__(self, shape, device='cuda'):
        super().__init__()
        self.reservoir = torch.zeros(shape).to(device)

    def forward(self, x_incr):
        activation_incr(self.reservoir, x_incr[0])

    def forward_refresh_reservoir(self, x):
        self.reservoir = x.clone().detach()
        return torch.relu(x)



class IncrementMaskConvModule(IncrementMaskModule):
    
    def convert_filter_out_channels_last(self, filter, transposed=True):
        return torch.permute(filter, (1,2,3,0)).contiguous().clone()
        if transposed:
            return torch.transpose(torch.transpose(filter, 2, 1), 3, 2).contiguous().clone()
        return torch.transpose(torch.transpose(torch.transpose(filter, 1, 0), 2, 1), 3, 2).contiguous().clone()

    def __init__(self, in_shape, c_in, c_out, weight, stride = 1, padding=0, device='cuda'):
        super().__init__()
        self.weight = weight
        self.weight_t = self.convert_filter_out_channels_last(weight)
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.mask = torch.rand(in_shape, device=device).le(.1)

    def forward_refresh_reservoir(self, x):
        return F.conv2d(x, self.weight, padding='same')


class Conv3x3Incr(IncrementMaskConvModule):
    def forward(self, x_incr):
        output_= torch.empty((x_incr.shape[0], self.c_out, x_incr.shape[2], x_incr.shape[3]), dtype=torch.float, device='cuda', memory_format=torch.channels_last)
        conv3x3_increment_ext(x_incr, self.mask, self.weight_t, output_, self.stride)
        return output_



class Conv5x5Incr(IncrementMaskConvModule):
    def forward(self, x_incr):
        output_= torch.empty((x_incr.shape[0], self.c_out, x_incr.shape[2], x_incr.shape[3]), dtype=torch.float, device='cuda', memory_format=torch.channels_last)
        conv5x5_increment_ext(x_incr, self.mask, self.weight_t, output_, self.stride)
        return output_



class Conv1x1Incr(IncrementMaskConvModule):
    def forward(self, x_incr):
        output_= torch.empty((x_incr.shape[0], self.c_out, x_incr.shape[2], x_incr.shape[3]), dtype=torch.float, device='cuda', memory_format=torch.channels_last)
        conv1x1_increment_ext(x_incr, self.mask, self.weight_t, output_, self.stride)
        return output_





