import argparse

import numpy as np
import torch
from torch.optim import *

from ev_projs.SysSSLE2VID.configs.parser import YAMLParser
from ev_projs.SysSSLE2VID.dataloader.h5 import H5Loader
from ev_projs.SysSSLE2VID.utils.iwe import deblur_events, compute_pol_iwe
from ev_projs.SysSSLE2VID.utils.utils import load_model
from ev_projs.SysSSLE2VID.utils.visualization import Visualization

from torch.profiler import profile, record_function, ProfilerActivity
from ev_projs.rpg_e2depth.utils.timers import CudaTimer

from incr_modules.fireflownet_incr import FireFlowNetIncr

import subprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trained_model", 
        default="./ev_projs/SysSSLE2VID/trained_models/model_29072022_132353_92e71bd3d8244494b350b3add8a55b82/FireFlowNet.pt", 
        help="model to be evaluated"
    )

    parser.add_argument(
        "--config",
        default="./ev_projs/SysSSLE2VID/configs/eval_flow_fireflow.yml",
        help="config file, overwrites model settings",
    )

    args = parser.parse_args()
    config_parser = YAMLParser(args.config)

    config = config_parser.merge_configs(args.trained_model)
    config["loader"]["batch_size"] = 1

    # store validation settings
    # eval_id = config_parser.log_eval_config(config)


    # initialize settings
    device = 'cuda'
    kwargs = config_parser.loader_kwargs

    # visualization tool
    # config["vis"]["enabled"] = False
    # config["vis"]["store"] = False


    # optical flow settings
    num_bins = config["data"]["num_bins"]
    flow_scaling = config["model_flow"]["flow_scaling"]
    model = eval(config["model_flow"]["name"])(config["model_flow"], num_bins).to(device)
    model = load_model(config["trained_model"], model, device)
    model.eval()

    print(config)

    # data loader
    data = H5Loader(config, num_bins)
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs
    )
    _v = torch.ones([10]).cuda() # initialize
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True) as prof:
        for i_batch, inputs in enumerate(dataloader):

            if i_batch == 100:
                break

            with torch.no_grad():
                print(inputs.keys())
                event_voxels = inputs["inp_voxel"].to(device, memory_format=torch.channels_last)
                event_cnt = inputs["inp_cnt"].to(device=device, memory_format=torch.channels_last)

                if i_batch%20 == 0:
                    with CudaTimer('model inference base'):
                        with record_function("model_inference_base"):
                            torch.cuda.synchronize()
                        # model_output = model(
                        #     event_voxels, event_cnt, log=config["vis"]["activity"]
                        # )
                            model_output = model.forward_refresh_reservoirs(event_voxels)['flow'][-1][0]
                            torch.cuda.synchronize()

                else:
                    event_voxel_incr = (event_voxels-event_voxels_prev).to(memory_format=torch.channels_last)
                    event_cnt_incr = (event_cnt-event_cnt_prev).to(memory_format=torch.channels_last)
                    print(event_cnt_incr.count_nonzero(), event_cnt.numel())

                    with CudaTimer('model inference'):
                        with record_function("model_inference"):
                            torch.cuda.synchronize()
                            preds = model((event_voxel_incr, None))
                            torch.cuda.synchronize()

                    model_output += preds['flow'][-1][0][0]

                event_voxels_prev = event_voxels
                event_cnt_prev = event_cnt


    # print(model_output)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device), row_limit=30))



    write=True
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
        # show_tensor_image()

exit()





