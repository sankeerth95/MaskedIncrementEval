import torch
from ..sparse_types import Sp, DenseT, SpOrDense
from ._C_ext.pointops.pointops_ext import pointwise_add

def pointwise_incr(X: DenseT, input_incr: Sp) -> Sp:
    return pointwise_add(X, input_incr.values(), input_incr.indices().int())
    


