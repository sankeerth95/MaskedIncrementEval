import torch
from ._C_ext.pointops.pointops_ext import activation_increment

def activation_incr(X, input_incr):
    output_= torch.zeros_like(X)
    activation_increment(X, input_incr, output_)
    return output_

