import torch
import torch.nn as nn
import torch.nn.functional as F
import ext.pointops.pointops_functional as pf
from .masked_types import Masked, DenseT
from typing import overload

from metrics.structural_sparsity import field_channel_sparsity


def print_sparsity(x, prefix: str = ""):
    return
    print(prefix, float(field_channel_sparsity(x, field_size=5, threshold=k_init).cpu().numpy())  )


# singleton class; not the best way but whatev
class AccumStreamManager:
    def __init__(self):
        self.s = torch.cuda.Stream()
    
    def get_stream(self):
        return self.s

    instance = None
    @classmethod
    def createAccumStream(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance



# accumulates inputs: have to make this conditional
class IncrementReserve:
    def __init__(self, x_init = None):
        # self.accum_stream = AccumStreamManager.createAccumStream() 
        if x_init == None:
            self.reservoir = None
        else:
            self.reservoir = x_init.clone().detach()

    # dense/sparse accumulate accumulate
    def accumulate(self, incr: Masked):
        self.reservoir.add_(incr)
        # return
        # with torch.cuda.stream(self.accum_stream.get_stream()):
        #     self.reservoir.add_(incr)

    def update_reservoir(self, x: torch.Tensor):
        self.reservoir = x.clone().detach() # not in place right now :(

# input output increments: interface for incremental modules! 
class IncrementMaskModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Masked) -> Masked: ...

    # called at the end of non-increment forward pass, if needed.
    def forward_refresh_reservoirs(self, x: DenseT) -> DenseT: 
        return x


# filter: only allow significant elements to pass through
# secret sauce: MASKIFY!
class KFencedMaskModule(IncrementMaskModule):
    def __init__(self, k: float=0.1, field=5):
        super().__init__()
        # internally defined reserves
        self.in_reserve = IncrementReserve()
        self.delta = IncrementReserve() # input tensor dimensions: dense tensor

        self.field = field
        self.k = k

    def floor_by_k(self, T: Masked) -> Masked:  # TODO: implement for sparse inputs.
        return (self.k*torch.floor(0.5 + T/self.k))  

    # accumulate operations: sparsed
    def forward(self, incr: Masked) -> Masked:
        return incr
        T1 = self.delta.reservoir + incr                    # critical path; could be sparse
        f_delta = self.floor_by_k(T1)                   # critical path; could be sparse
        self.delta.update_reservoir(T1 - f_delta)                       # out of order; sparse
        self.in_reserve.accumulate(f_delta)         # out of order; sparse: conditional: doesn't need to be done
        return f_delta

    def forward_refresh_reservoirs(self, x: DenseT) -> DenseT:
        self.in_reserve.update_reservoir(x)
        self.delta.update_reservoir(torch.zeros_like(x))
        return x




def IncrPointwiseMultiply(x1_incr: Masked, x1: IncrementReserve, x2_incr: Masked, x2: IncrementReserve) -> Masked:
    # return x1_incr    
    return [x1_incr[0]*x2_incr[0] + x2.reservoir*x1_incr[0] + x1.reservoir*x2_incr[0], x1_incr[0]|x1_incr[1]]


class PointwiseMultiplyIncr(IncrementMaskModule):
    def __init__(self, x1res_module: IncrementReserve, x2res_module: IncrementReserve):
        super().__init__()
        # define reserves:
        self.x1_res: IncrementReserve = x1res_module
        self.x2_res: IncrementReserve = x2res_module

    def forward(self, x1_incr: Masked, x2_incr: Masked) -> Masked:
        output_incr = IncrPointwiseMultiply(x1_incr, self.x1_res, x2_incr, self.x2_res)
        return output_incr

    def forward_refresh_reservoirs(self, x1: DenseT, x2: DenseT) -> DenseT:
        return x1*x2


def conv2d_from_module(x: Masked, conv_weights, conv_bias=None, stride=(1,1), padding=(1, 1), forward_refresh=False) -> Masked:
    if not forward_refresh:
        output_ = pf.functional_conv_module(x[0], conv_weights, mask=x[1], stride=stride, padding=padding)
        return output_
    else:
        output_ = F.conv2d(x, torch.permute(conv_weights, (3, 0, 1, 2)), bias=conv_bias, 
        stride=stride, padding=padding)
        return output_, None


def transposed_conv2d_from_module(x: Masked, gates: nn.ConvTranspose2d, bias=True) -> Masked:
    gate_bias = gates.bias if bias else None
    return F.conv_transpose2d(x, gates.weight, bias=gate_bias, stride=gates.stride, \
            padding=gates.padding, output_padding=gates.output_padding, dilation=gates.dilation, groups=gates.groups)


def bn2d_from_module(x: Masked, bnm: nn.BatchNorm2d, bias=True) -> Masked:
    bnm_running_mean = bnm.running_mean if bias else None
    bnm_bias = bnm.bias if bias else None
    return F.batch_norm(x, running_mean=bnm_running_mean, running_var=bnm.running_var, weight=bnm.weight, bias=bnm_bias, training=bnm.training, momentum=bnm.momentum, eps=bnm.eps)



# nonlinear operations which are point ops; dense modules only
class NonlinearPointOpIncr(IncrementMaskModule):
    def __init__(self, res_in: IncrementReserve, op=torch.tanh):
        super().__init__()
        self.reservoir_in = res_in
        self.op = op

    def forward(self, x_incr: Masked) -> Masked:
        # compute only for these inputs.
        return self._nonlin(x_incr)

    # x is like an input: don't update external reservoirs
    def forward_refresh_reservoirs(self, x: DenseT):
        return self.op(x)

    # need to be replaced with c++ binding
    def _nonlin(self, x_incr: Masked):
        output_incr = self.op(self.reservoir_in.reservoir + x_incr[0]) - self.op(self.reservoir_in.reservoir)
        return [output_incr, x_incr[1]]


class nnLinearIncr(IncrementMaskModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear()
    
    
    # fully connected implementation
    def forward(self, x_incr: Masked) -> Masked:
        return [F.linear(x_incr[0], self.linear.weight), True|x_incr[1]]

    def forward_refresh_reservoirs(self, x: DenseT) -> DenseT:
        return F.linear(x, self.linear.weight, self.linear.bias)


def interpolate_from_module(x: Masked) -> Masked:
    out1 = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=False)
    return out1, torch.ones_like(out1, dtype=bool)




