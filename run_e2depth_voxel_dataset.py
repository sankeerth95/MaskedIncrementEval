import torch
from torch.nn import ReflectionPad2d

from benchmark.input_handlers import IncrDatasetInputHandler, IncrDatasetInputHandler_2args
from benchmark.model_getter import ModelGetter
from benchmark.model_handlers import IncrModelHandler
from data_fetchers.continuous_event_datasets import ContinuousEventsDataset
from data_fetchers.voxel_event_dataset import VoxelGridDataset
from ev_projs.rpg_e2depth.utils.event_tensor_utils import EventPreprocessor
from utils.input_view import show_tensor_image


if __name__ == '__main__':
    device='cuda'
    base_folder = '/home/sankeerth/ev/depth_proj/data/test/'
    event_folder='events/voxels'
    vd = VoxelGridDataset(base_folder, event_folder, transform=None, normalize=False)


    get_data_i = lambda i: torch.unsqueeze(vd[i]['events'], dim=0).to(device, memory_format=torch.channels_last)


    for i in range(1000):
        show_tensor_image(get_data_i(i))

