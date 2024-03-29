import argparse, subprocess
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

from torch.nn import ReflectionPad2d
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
from data_fetchers.continuous_event_datasets import ContinuousEventsDataset
from data_fetchers.voxel_event_dataset import VoxelGridDataset, VoxelGridDatasetEval
from ev_projs.rpg_e2depth.utils.event_tensor_utils import EventPreprocessor
from ev_projs.rpg_e2depth.utils.loading_utils import load_model, load_model_incr
from ev_projs.rpg_e2depth.depth_prediction import DepthEstimator, DepthEstimatorIncr
from ev_projs.rpg_e2depth.options.inference_options import set_depth_inference_options


if __name__ == '__main__':
    device='cuda'
    visualize=False
    continuous_data = False

    width, height = 346, 260
    base_folder = '/home/sankeerth/ev/depth_proj/data/test/'
    # memory_format
    # if continuous_data:
    #     dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
    #             width=width, height=height, window_size = 0.05, time_shift = 0.001) # 1 ms time shift
    # else:
    dataset = VoxelGridDatasetEval(base_folder, 'events/voxels', 'depth/data', transform=None, normalize=False)
    pth = '/home/sankeerth/ev/experiments/pretrained_models/E2DEPTH_si_grad_loss_mixed.pth.tar'    
    model = load_model_incr(pth).to(device)
    model.eval()

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
            no_normalize=True
            no_recurrent=False
            use_gpu=True
            output_folder=None
        event_preprocessor = EventPreprocessor(options)
        pad = ReflectionPad2d((3, 3, 2, 2))
        get_data_i = lambda i: [torch.unsqueeze(
            pad(event_preprocessor(torch.Tensor(dataset[i][0]))),
            dim=0).to(device), dataset[i][1] ]#, memory_format=memory_format)


        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True) as prof:
            with torch.no_grad():

                for i in range(100):
                    event_tensor, gt_depth = get_data_i(i)

                    if i % 1 == 0:
                        torch.cuda.synchronize()
                        with record_function("model_inference_base"):
                            c,h = model.forward_refresh_reservoirs(event_tensor, None)
                            torch.cuda.synchronize()
                        event_tensor_prev = event_tensor

                        # print(event_tensor - c)
                        print(i, torch.sqrt(torch.mean((gt_depth.cuda() - F.interpolate(c, size=(260, 346))**2)))   )

                    else:
                        x = (event_tensor - event_tensor_prev).to(memory_format=torch.channels_last)
                        print(x.count_nonzero(), x.numel())
                        torch.cuda.synchronize()
                        with record_function("model_inference"):
                            # c_incr, h_incr = model((x, None), None)
                            torch.cuda.synchronize()

                        event_tensor_prev = event_tensor
                        # c += c_incr[0]
    
        print(prof.key_averages().table(sort_by="{}_time_total".format(device), row_limit=15))

        write=False
        if write:
            prof.export_chrome_trace("trace_{}.json".format(device))
            prof.export_stacks("/tmp/profiler_stacks.txt", "self_{}_time_total".format(device))
            with open('stack_flame_{}.svg'.format(device), 'w') as fp:
                subprocess.run(
                    [ '/home/sankeerth/FlameGraph/flamegraph.pl', 
                    '--title', 
                    "{} time_total".format(device), 
                    '--countname',
                    "us.", 
                    '--reverse', 
                    '/tmp/profiler_stacks.txt'],
                    stdout=fp
                )



