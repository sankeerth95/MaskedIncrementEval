import torch
from typing import Union

_device='cuda'

SparseTypes = {
    'cpu': torch.sparse.FloatTensor,
    'cuda': torch.cuda.sparse.FloatTensor
}

Sp = SparseTypes[_device]
DenseT = torch.Tensor
SpOrDense = Union[Sp, DenseT]
Masked = torch.Tensor

