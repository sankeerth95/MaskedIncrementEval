import torch
from incr_modules.fenced_module_masked import IncrementMaskModule
from ._C_ext.pointops.pointops_ext import activation_increment

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


