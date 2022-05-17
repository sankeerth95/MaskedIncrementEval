import argparse, os, subprocess
import torch
import torch.nn.functional as F
import numpy as np

from ev_projs.event_flow.dataloader.h5 import H5Loader
from ev_projs.event_flow.eval_flow import test
from ev_projs.event_flow.configs.parser import YAMLParser
from ev_projs.event_flow.models.model import RecEVFlowNet

from torch.utils.data import DataLoader
from data_fetchers.continuous_event_datasets import ContinuousEventsDataset
from torch.profiler import profile, record_function, ProfilerActivity

import mlflow
from ev_projs.event_flow.utils.utils import load_model
from incr_modules.evflownet_incr import MultiResUNetRecurrentIncr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", default='EVFlowNet', help="sdf")
    parser.add_argument(
        "--config",
        default="./eval_MVSEC.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="don't save stuff",
    )
    args = parser.parse_args()

    # launch testing
    
    aee_measurement = False    
    if aee_measurement:
        test(args, YAMLParser(args.config))
        exit()
# exit()

    device='cuda'
    height, width = 256, 256
    continuous_dataset = False
    if continuous_dataset:
        memory_format=torch.channels_last
        base_folder = '/home/sankeerth/ev/data/Caltech_data/'
        dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
                width=width, height=height, evframe_type='histogram', window_size = 0.05, time_shift = 0.001) # 1 ms time shift
    else:
        yamlconfig = YAMLParser(args.config)
        mlflow.set_tracking_uri(args.path_mlflow)
        run = mlflow.get_run(args.runid)
        config = yamlconfig.merge_configs(run.data.params)        
        dataset = H5Loader(config, config["model"]["num_bins"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=dataset.custom_collate,
    )

    model1 = RecEVFlowNet(config['model']).to(device)
    model1 = load_model(args.runid, model1, device)
    model1 = model1.eval()

    model = MultiResUNetRecurrentIncr().to(device)
    model.load_state_dict(model1.multires_unetrec.state_dict())


    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True) as prof:
        for i_batch, inputs in enumerate(dataloader):

            if i_batch == 40:
                break

            with torch.no_grad():
                event_voxels = inputs["event_voxel"].to(device, memory_format=torch.channels_last)
                event_cnt = inputs["event_cnt"].to(device=device, memory_format=torch.channels_last)

                if i_batch%40 == 0:
                    with record_function("model_inference_base"):
                        # model_output = model(
                        #     event_voxels, event_cnt, log=config["vis"]["activity"]
                        # )
                        model_output = model.forward_refresh_reservoirs(
                            event_cnt
                        )
                else:
                    event_voxel_incr = (event_voxels-event_voxels_prev).to(memory_format=torch.channels_last)
                    event_cnt_incr = (event_cnt-event_cnt_prev).to(memory_format=torch.channels_last)
                    print(event_cnt_incr.count_nonzero(), event_cnt.numel())

                    with record_function("model_inference"):
                        preds = model((event_cnt_incr, None))

                    model_output += preds[-1][0]

                event_voxels_prev = event_voxels
                event_cnt_prev = event_cnt



            # loss = yoloLoss(model_output, bounding_box, model_input_size)[0]


            # detected_bbox = yoloDetect(model_output, self.model_input_size.to(model_output.device),
            #                             threshold=0.3)
            # detected_bbox = nonMaxSuppression(detected_bbox, iou=0.6)
            # detected_bbox = detected_bbox.cpu().numpy()

            # sum_loss += loss


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
    print(f"Test Loss: {sum_loss}")
        # show_tensor_image()

exit()


