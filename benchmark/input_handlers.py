from typing import Tuple
import torch
import torch.nn.functional as F
from torch.nn.modules import ReflectionPad2d
from torch.utils.data import DataLoader
from ev_projs.rpg_e2depth.utils.event_tensor_utils import EventPreprocessor
from data_fetchers.continuous_event_datasets import ContinuousEventsDataset
from abc import abstractmethod
from metrics.structural_sparsity import field_channel_sparsity

from utils.str_sparse_utils import get_mask

class InputHandler:
    def __init__(self):
        ...

    @abstractmethod
    def get_single_sample(self, index: int):
        raise NotImplementedError

class IncrInputHandler(InputHandler):
    def __init__(self):
        super().__init__()
        self.prev_x = None

class RandomInputHandler(InputHandler):
    def __init__(self, shape, scale: float=1., device='cuda') -> None:
        super().__init__()
        self.shape = shape
        self.scale = scale
        self.device = device

    def get_single_sample(self, index: int):
        return torch.randn(self.shape, device=self.device).to(memory_format=torch.channels_last)


class ZeroInputHandler(InputHandler):
    def __init__(self, shape, device='cuda') -> None:
        super().__init__()
        self.shape = shape
        self.device = device

    def get_single_sample(self, index: int):
        return torch.zeros(self.shape, device=self.device)


class SparseRandomInputHandler(RandomInputHandler):
    def __init__(self, shape, scale: float=1., sparsity = .9, device='cuda') -> None:
        super().__init__(shape, scale, device)
        self.sparsity = sparsity

    def get_single_sample(self, index: int):
        x = (torch.rand(self.shape, device=self.device) > self.sparsity)
        ret = self.scale*x*(torch.randn(self.shape).to(self.device))
        # print("field channel sparsity = ", field_channel_sparsity(ret, 3))
        return ret, get_mask(ret)

# thin wrapper around previous one
class SparseCOORandomInputHandler(SparseRandomInputHandler):
    def __init__(self, shape, scale: float=1., sparsity = .9, device='cuda') -> None:
        super().__init__(shape, scale, sparsity, device)

    def get_single_sample(self, index: int):
        ret = super().get_single_sample(index)
        return ret.to_sparse_coo()


# not rtelly accurate: real structural sparsity is lower.
class StructurallySparseRandomInputHandlerNCHW(RandomInputHandler):
    def __init__(self, shape, scale: float=1., field_size: int=3, strsparsity = .9, device='cuda') -> None:
        assert len(shape) == 4 # NCHW tensor
        super().__init__(shape, scale, device)
        self.strsparsity = strsparsity
        self.field_size = field_size


    def get_single_sample(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # points where it is zero
        mask = torch.rand(self.shape).to(self.device).le(self.strsparsity)
        nbd_mask = F.max_pool2d(torch.abs(mask.float()), self.field_size, stride=(1, 1), padding=(self.field_size - 1)//2 ).ge(0.5)

        T = self.scale*torch.randn(self.shape).to(self.device)
        T[nbd_mask] = 0.
        # ~mask: points where it is nonzero
        return T, (~mask).int()
        # return T, torch.ones_like(mask).to('cuda').int()


class DatasetInputHandler(InputHandler):
    def __init__(self, start_index=0, device='cuda') -> None:
        super().__init__()
        self.start_index = start_index
        self.device = device
        class options:
            hot_pixels_file= None
            flip = False
            no_normalize = False

        self.event_preprocessor = EventPreprocessor(options)
        self.pad = ReflectionPad2d((3, 3, 2, 2))

        base_folder = '/home/sankeerth/ev/rpg_e2depth/data/test/'
        self.dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
            width=346, height=260, window_size = 0.05, time_shift = 0.001) # 1 ms time shift, 

        self.get_data_i = lambda i: torch.unsqueeze(
                self.pad(self.event_preprocessor(
                    torch.Tensor(self.dataset[i])
                )), dim=0 ).to(device)
        self.data = DataLoader(self.dataset, shuffle=False, batch_size=1)


class IncrDatasetInputHandler(DatasetInputHandler, IncrInputHandler):
    def __init__(self, start_index = 0, device='cuda'):
        super().__init__(start_index, device)
        self.prev_x = self.get_data_i(start_index)

    def get_single_sample(self, index):
        x = self.get_data_i(self.start_index + index)
        input_ = x-self.prev_x
        self.prev_x = x
        return input_

class IncrDatasetInputHandler_2args(DatasetInputHandler, IncrInputHandler):
    def __init__(self, start_index = 0, device='cuda'):
        super().__init__(start_index, device)
        self.prev_x = (self.get_data_i(start_index), None)

    def get_single_sample(self, index):
        x = self.get_data_i(self.start_index + index)
        input_ = x-self.prev_x[0]
        self.prev_x = (x, None)
        return input_, None



class BaselineInputHandler(DatasetInputHandler):
    def __init__(self, start_index = 0, device='cuda'):
        super().__init__(start_index, device)

    def get_single_sample(self, index):
        x = self.get_data_i(self.start_index + index)
        return x


