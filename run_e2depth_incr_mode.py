import argparse
import torch
from data_fetchers.continuous_event_datasets import ContinuousEventsDataset
from ev_projs.rpg_e2depth.utils.loading_utils import load_model_incr
from ev_projs.rpg_e2depth.depth_prediction import DepthEstimatorIncr
from ev_projs.rpg_e2depth.options.inference_options import set_depth_inference_options

if __name__ == '__main__':
    device='cuda'
    memory_format=torch.channels_last
    base_folder = '/home/sankeerth/ev/depth_proj/data/test/'

    dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
            width=346, height=260, window_size = 0.05, time_shift = 0.05) # 1 ms time shift

    # class options:
    #     hot_pixels_file=None
    #     flip = False
    #     no_normalize=False
    #     no_recurrent=False
    #     use_gpu=True
    #     output_folder=None

    # event_preprocessor = EventPreprocessor(options)
    # pad = ReflectionPad2d((3, 3, 2, 2))

    # get_data_i = lambda i: torch.unsqueeze(
    #         pad(event_preprocessor(
    #             torch.Tensor(dataset[i])
    #         )), dim=0 ).to(device, memory_format=memory_format)



    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    

    set_depth_inference_options(parser)
    args = parser.parse_args()

    pth = '/home/sankeerth/ev/experiments/pretrained_models/E2DEPTH_si_grad_loss_mixed.pth.tar'    
    model = load_model_incr(pth)
    width, height = 346, 260

    estimator = DepthEstimatorIncr(model, height, width, model.num_bins, args)

    for i in range(1000):
        event_tensor = torch.Tensor(dataset[i])
        estimator.update_reconstruction(event_tensor, i)


        # show_tensor_image()




