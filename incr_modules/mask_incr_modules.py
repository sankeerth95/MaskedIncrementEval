import torch
import torch.nn as nn
import torch.nn.functional as F
from .masked_types import Masked, DenseT


from .mask_incr_functional import IncrPointwiseMultiply, IncrementReserve



# input output increments: interface for incremental modules! 
class IncrementMaskModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Masked) -> Masked: ...

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
        output_incr = self.op(self.reservoir_in.reservoir + x_incr[0]) - self.op(self.reservoir_in.reservoir)
        return [output_incr, x_incr[1]]


class nnLinearIncr(nn.Linear, IncrementMaskModule):

    
    # fully connected implementation
    def forward(self, x_incr: Masked) -> Masked:
        return [F.linear(x_incr[0], self.weight, bias=None), True|x_incr[1]]

    def forward_refresh_reservoirs(self, x: DenseT) -> DenseT:
        return F.linear(x, self.linear.weight, self.linear.bias)


#linear module
class nnConvIncr(nn.Conv2d):

    def forward(self, x_incr):
        return x_incr


    def forward_refresh_reservoirs(self, x):
        return x

#linear module
class nnBatchNorm2dIncr(nn.BatchNorm2d):

    def forward(self, x_incr):
        return x_incr


    def forward_refresh_reservoirs(self, x):
        return x


class nnMaxPool2dIncr(nn.MaxPool2d):

    def forward(self, x_incr):
        return x_incr


    def forward_refresh_reservoirs(self, x):
        return x



#linear module
class nnAdaptiveAvgPool2dIncr(nn.AdaptiveAvgPool2d):

    def forward(self, x_incr):
        return x_incr

    def forward_refresh_reservoirs(self, x):
        return x




class nnSequentialIncr(nn.Sequential):

    def __init__(self, *module_list):
        nn.Sequential.__init__(self, *module_list)


    def forward_refresh_reservoirs(self, x):
        
        for module in self:
            input = module.forward_refresh_reservoirs(input)
        return input




