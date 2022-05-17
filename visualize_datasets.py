import torch
from torch.nn import ReflectionPad2d

from data_fetchers.continuous_event_datasets import ContinuousEventsDataset
from data_fetchers.voxel_event_dataset import VoxelGridDataset
from ev_projs.event_flow.dataloader.h5 import H5Loader
from ev_projs.rpg_e2depth.utils.event_tensor_utils import EventPreprocessor
from ev_projs.rpg_e2depth.utils.loading_utils import load_model
from ev_projs.rpg_e2depth.depth_prediction import DepthEstimator
from utils.input_view import show_tensor_image, show_tensor_image_CHW, show_two_tensor_image
from ev_projs.rpg_e2depth.options.inference_options import set_depth_inference_options


if __name__ == '__main__':
    device='cuda'

# Caltech101:
    evframe_type = 'histogram'
    base_folder = '/home/sankeerth/ev/data/Caltech_data/'
    dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
            width=232, height=101, evframe_type=evframe_type, window_size = 0.05, time_shift = 0.001) # 1 ms time shift

# mcsec data:
    evframe_type = 'voxelgrid'
    base_folder = '/home/sankeerth/ev/depth_proj/data/test/'
    dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
            width=346, height=260, evframe_type=evframe_type, window_size = 0.05, time_shift = 0.001)


    evframe_type = 'voxelgrid'
    base_folder = '/home/sankeerth/ev/depth_proj/data/test/'
    dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
            width=346, height=260, evframe_type=evframe_type, window_size = 0.05, time_shift = 0.001)


    evframe_type = 'voxelgrid'
    base_folder = '/home/sankeerth/ev/depth_proj/data/test/'
    dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
            width=346, height=260, evframe_type=evframe_type, window_size = 0.05, time_shift = 0.001)

    evframe_type = 'event_voxel'
    config = {'data': {'mode': 'gtflow_dt1', 'window': 15000, 'path': 'ev_projs/event_flow/datasets/data/ECD/', 'window_loss': 10000, 'window_eval': 15000}, 'loader': {'resolution': [180, 240], 'batch_size': 1, 'augment': [], 'gpu': 0, 'seed': 0, 'n_epochs': 100, 'augment_prob': [0.5, 0.5, 0.5]}, 'hot_filter': {'enabled': True, 'max_px': 100, 'min_obvs': 5, 'max_rate': 0.8}, 'experiment': 'Default', 'prev_runid': '', 'loss': {'flow_regul_weight': 0.001, 'clip_grad': 100.0, 'overwrite_intermediate': False}, 'optimizer': {'name': 'Adam', 'lr': 0.0002}, 'model': {'name': 'RecEVFlowNet', 'encoding': 'cnt', 'round_encoding': False, 'norm_input': False, 'num_bins': 2, 'base_num_channels': 32, 'kernel_size': 3, 'activations': ['relu', None], 'mask_output': True, 'spiking_neuron': 'None'}, 'vis': {'bars': True, 'verbose': True, 'enabled': False, 'px': 400, 'store_grads': False, 'activity': False, 'store': False}, 'metrics': {'name': ['FWL', 'RSAT'], 'flow_scaling': 128}}
    dataset_path = '/home/sankeerth/ev/rpg_asynet/data/NCaltech101_ObjectDetection/'
    dataset = H5Loader(config, config["model"]["num_bins"])

    # print(dataset[1] - dataset[0])
    get_cont_data_i = lambda i: torch.Tensor(dataset[i]['event_cnt'])


    # class options:
    #     hot_pixels_file=None
    #     flip = False
    #     no_normalize=False
    #     no_recurrent=False
    #     use_gpu=True
    #     output_folder=None

    # event_preprocessor = EventPreprocessor(options)
    # pad = ReflectionPad2d((3, 3, 2, 2))


    # vd = VoxelGridDataset(base_folder, event_folder='events/voxels', transform=None, normalize=False)
    # get_voxel_data_i = lambda i: vd[i]['events']
    print("length of dataset = ", len(dataset))
    for i in range(len(dataset)):
        # show_tensor_image(get_cont_data_i(i))
        if evframe_type =="voxelgrid":
            show_two_tensor_image(get_cont_data_i(i), get_cont_data_i(i))
        elif evframe_type == 'histogram':
            show_tensor_image(get_cont_data_i(i))
        elif evframe_type == 'event_voxel':
            show_tensor_image_CHW(get_cont_data_i(i))
        else:
            raise NotImplementedError



