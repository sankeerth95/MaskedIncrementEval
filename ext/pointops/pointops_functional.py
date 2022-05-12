from time import sleep
import torch
from ._C_ext.pointops.pointops_ext import activation_increment, conv7x7_increment_ext, conv3x3_increment_ext, conv5x5_increment_ext, conv1x1_increment_ext




def activation_incr(x, input_incr):
    output_= torch.empty(x.shape, dtype=torch.float, device='cuda')
    activation_increment(x, input_incr, output_)
    return output_


def convert_filter_out_channels_last(conv_weights):
    return torch.permute(conv_weights, (1,2,3,0)).contiguous().clone()


# filter dimensions: [cout, kx, ky, cin]
def functional_conv_module(x_incr, conv_weights, mask=None, stride=(1,1), padding=(1,1)):

    if mask == None:
        mask = x_incr.ge(.00001)

    #  TODO: remove later; checks only
    if x_incr.is_contiguous():
        raise AssertionError('received non NHWC tensor')
    if padding[0] != (conv_weights.shape[1] - 1 ) // 2:
        print(padding[0], conv_weights.shape[1])
        raise AssertionError('not implemented for non-same type padding')


    out_H = int((x_incr.shape[2] + 2*padding[0] - conv_weights.shape[1] ) // stride[0] + 1)
    out_W = int((x_incr.shape[3] + 2*padding[1] - conv_weights.shape[2] ) // stride[1] + 1)
    out_C = conv_weights.shape[3]
    batches = x_incr.shape[0]
    output_= torch.empty((batches, out_C, out_H, out_W), dtype=torch.float, device='cuda', memory_format=torch.channels_last)
    if conv_weights.shape[1] == 3:
        conv3x3_increment_ext(x_incr, mask, conv_weights, output_, stride[0])
    elif conv_weights.shape[1] == 5:
        conv5x5_increment_ext(x_incr, mask, conv_weights, output_, stride[0])
    elif conv_weights.shape[1] == 1:
        conv1x1_increment_ext(x_incr, mask, conv_weights, output_, stride[0])
    elif conv_weights.shape[1] == 7:
        conv7x7_increment_ext(x_incr, mask, conv_weights, output_, stride[0])
    else:
        raise NotImplementedError("not Implemented convolution for these dimensions!")


    output_mask = torch.ones_like(output_, dtype=bool)
    return [output_, output_mask]


