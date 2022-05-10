import torch
import torch.nn as nn
import torch.nn.functional as F
from .masked_types import Masked, DenseT


from .mask_incr_functional import IncrPointwiseMultiply, IncrementReserve, bn2d_from_module, conv2d_from_module



# input output increments: interface for incremental modules! 
class IncrementMaskModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Masked) -> Masked: 
        raise NotImplementedError

    # called at the end of non-increment forward pass, if needed.
    def forward_refresh_reservoirs(self, x: DenseT) -> DenseT: 
        return x


# filter: only allow significant elements to pass through
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
        self.reservoir_in.reservoir + x_incr[0]
        output_incr = self.op(self.reservoir_in.reservoir + x_incr[0]) - self.op(self.reservoir_in.reservoir)
        return [output_incr, x_incr[1]]


class nnLinearIncr(nn.Linear):
    
    # fully connected implementation
    def forward(self, x_incr: Masked) -> Masked:
        out = F.linear(x_incr[0], self.weight, bias=None)
        return out, torch.ones_like(out, dtype=bool)

    def forward_refresh_reservoirs(self, x: DenseT) -> DenseT:
        return super().forward(x)

        # return F.linear(x, self.weight, self.bias)


#linear module
class nnConvIncr(nn.Conv2d):

    def forward(self, x_incr):
        # print('x_incr: ', x_incr.shape)
        out = F.conv2d(x_incr[0], self.weight, bias=None, padding=self.padding, stride=self.stride)
        return out, torch.ones_like(out, dtype=bool)
        return conv2d_from_module(x_incr, self.weight)


    def forward_refresh_reservoirs(self, x):
        return super().forward(x)

#linear module
class nnBatchNorm2dIncr(nn.BatchNorm2d):

    def forward(self, x_incr):
        return bn2d_from_module(x_incr, self)


    def forward_refresh_reservoirs(self, x):
        return super().forward(x)


class nnMaxPool2dIncr(nn.MaxPool2d):

    def __init__(self, kernel_size, stride=None, padding=0):
        nn.MaxPool2d.__init__(self, kernel_size=kernel_size, stride=stride, padding=padding)
        self.functional = lambda x: F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.mp_res = IncrementReserve()

    def forward(self, x_incr):
        out = self.functional(self.mp_res.reservoir+x_incr[0]) - self.functional(self.mp_res.reservoir)
        self.mp_res.accumulate(x_incr)
        return out

    def forward_refresh_reservoirs(self, x):
        self.mp_res.update_reservoir(x)
        return super().forward(x)


#linear module
class nnAdaptiveAvgPool2dIncr(nn.AdaptiveAvgPool2d):

    def forward(self, x_incr):
        raise NotImplementedError

    def forward_refresh_reservoirs(self, x):
        return super().forward(x)


class nnSequentialIncr(nn.Sequential):

    def forward_refresh_reservoirs(self, x):
        for module in self:
            x = module.forward_refresh_reservoirs(x)
        return x


class nnReluIncr(NonlinearPointOpIncr):
    def __init__(self):
        self.in_res = IncrementReserve() 
        NonlinearPointOpIncr.__init__(self, self.in_res, torch.relu)

    def forward(self, x_incr: Masked):
        out = NonlinearPointOpIncr.forward(self, x_incr)
        self.in_res.accumulate(x_incr)
        return out

    def forward_refresh_reservoirs(self, x: DenseT):
        out = super().forward_refresh_reservoirs(x)
        self.in_res.update_reservoir(x)
        return out
