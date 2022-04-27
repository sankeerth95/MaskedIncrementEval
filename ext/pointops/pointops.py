import torch
import torch.nn.functional as F
from incr_modules.fenced_module_masked import IncrementMaskModule
from ._C_ext.pointops.pointops_ext import activation_increment, conv3x3_increment

def activation_incr(X, input_incr):
    output_= torch.zeros_like(X)
    activation_increment(X, input_incr, output_)
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

def conv3x3_incr(x, filter, c_in, c_out):
    output_= torch.zeros(c_out, x.shape[2], x.shape[3])
    conv3x3_increment(x[0], filter, output_)
    return output_

class Conv3x3Incr(IncrementMaskModule):

    def __init__(self, c_in, c_out, weight, device='cuda'):
        super().__init__()
        self.weight = weight
        self.c_in = c_in
        self.c_out = c_out

    def forward(self, x_incr):
        conv3x3_incr(x_incr, self.weight, self.c_in, self.c_out)

    def forward_refresh_reservoir(self, x):
        return F.conv2d(x, self.weight)




