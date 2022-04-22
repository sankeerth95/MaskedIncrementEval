import torch

from ._C_ext.pointops.pointops_ext import pointwise_add


def pointwise_incr(X, input_incr):
    return pointwise_add(X, input_incr.values(), input_incr.indices().int())
    


