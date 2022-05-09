import argparse
import torch
from torch.nn import ReflectionPad2d

from data_fetchers.continuous_event_datasets import ContinuousEventsDataset
from data_fetchers.voxel_event_dataset import VoxelGridDataset
from ev_projs.rpg_e2depth.utils.event_tensor_utils import EventPreprocessor
from ev_projs.rpg_e2depth.utils.loading_utils import load_model
from ev_projs.rpg_e2depth.depth_prediction import DepthEstimator
from utils.input_view import show_tensor_image
from ev_projs.rpg_e2depth.options.inference_options import set_depth_inference_options


if __name__ == '__main__':
    device='cuda'

    base_folder = '/home/sankeerth/ev/depth_proj/data/test/'
    dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
            width=346, height=260, evframe_type='voxelgrid', window_size = 0.05, time_shift = 0.05) # 1 ms time shift


    base_folder = '/home/sankeerth/ev/rpg_asynet/data/NCaltech101_ObjectDetection/Caltech_data/'
    dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
            width=240, height=180, evframe_type='histogram', window_size = 0.05, time_shift = 0.001) # 1 ms time shift



    # class options:
    #     hot_pixels_file=None
    #     flip = False
    #     no_normalize=False
    #     no_recurrent=False
    #     use_gpu=True
    #     output_folder=None

    # event_preprocessor = EventPreprocessor(options)
    # pad = ReflectionPad2d((3, 3, 2, 2))

    get_cont_data_i = lambda i: torch.Tensor(dataset[i])

    # event_folder='events/voxels'
    # vd = VoxelGridDataset(base_folder, event_folder, transform=None, normalize=False)
    # get_voxel_data_i = lambda i: vd[i]['events']
    print("length of dataset = ", len(dataset))
    for i in range(len(dataset)):
        print(i)
        show_tensor_image(get_cont_data_i(i))






