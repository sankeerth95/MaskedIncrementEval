import argparse
import torch
from torch.nn import ReflectionPad2d

from data_fetchers.continuous_event_datasets import ContinuousEventsDataset
from data_fetchers.voxel_event_dataset import VoxelGridDataset
from ev_projs.rpg_e2depth.utils.event_tensor_utils import EventPreprocessor
from ev_projs.rpg_e2depth.utils.loading_utils import load_model, load_model_incr
from ev_projs.rpg_e2depth.depth_prediction import DepthEstimator, DepthEstimatorIncr
from ev_projs.rpg_e2depth.options.inference_options import set_depth_inference_options


if __name__ == '__main__':
    device='cuda'
    visualize=True


    width, height = 346, 260
    base_folder = '/home/sankeerth/ev/depth_proj/data/test/'
    # memory_format
    # dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
    #         width=width, height=height, window_size = 0.05, time_shift = 0.05) # 1 ms time shift
    dataset = VoxelGridDataset(base_folder, 'events/voxels', transform=None, normalize=False)

    pth = '/home/sankeerth/ev/experiments/pretrained_models/E2DEPTH_si_grad_loss_mixed.pth.tar'    
    model = load_model_incr(pth).to(device).eval()

    if visualize:
        # for visualize 
        parser = argparse.ArgumentParser(description='Evaluating a trained network')
        set_depth_inference_options(parser)
        args = parser.parse_args()
        estimator = DepthEstimatorIncr(model, height, width, model.num_bins, args)
        for i in range(1000):
            estimator.update_reconstruction(dataset[i], i)

    else:
        class options:
            hot_pixels_file=None
            flip = False
            no_normalize=False
            no_recurrent=False
            use_gpu=True
            output_folder=None
        event_preprocessor = EventPreprocessor(options)
        pad = ReflectionPad2d((3, 3, 2, 2))
        get_data_i = lambda i: torch.unsqueeze(
            pad(event_preprocessor(torch.Tensor(dataset[i]))),
            dim=0).to(device)#, memory_format=memory_format)

        for i in range(1):
            event_tensor = get_data_i(i)
            # c,h = model.forward_refresh_reservoirs(event_tensor, None)
            c,h = model(event_tensor, None)
        print(h)




