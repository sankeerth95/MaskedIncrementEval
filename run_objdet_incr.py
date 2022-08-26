import os, subprocess
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from data_fetchers.continuous_event_datasets import ContinuousEventsDataset
# import Caltech101ContinuousEvDataset
from incr_modules.obj_detection_incr import DenseObjectDetIncr
from ev_projs.rpg_async.models.yolo_loss import yoloLoss
from ev_projs.rpg_async.dataloader.dataset import NCaltech101_ObjectDetection
from torch.profiler import profile, record_function, ProfilerActivity



if __name__ == '__main__':

    device='cuda'
    height, width = 180, 240
    continuous_dataset = True
    if continuous_dataset:
        memory_format=torch.channels_last
        base_folder = '/home/sankeerth/ev/data/Caltech_data/'
        dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
                width=width, height=height, evframe_type='histogram', window_size = 0.05, time_shift = 0.001) # 1 ms time shift
    else:
        dataset_path = '/home/sankeerth/ev/rpg_asynet/data/NCaltech101_ObjectDetection/'
        dataset = NCaltech101_ObjectDetection(
            dataset_path, 
            'all', 
            height, 
            width, 
            25000, 
            mode='validation', 
            event_representation='histogram', 
            shuffle=False
        )

    val_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    model = DenseObjectDetIncr(101, in_c=2, small_out_map=True)
    lp = '/home/sankeerth/ev/rpg_asynet/log/20220508-215001/checkpoints/model_step_49.pth'
    m = torch.load(lp)
    model.load_state_dict(m['state_dict'], strict=False)
    model = model.eval().to(device)

    model_input_size = torch.tensor([191, 255])
    sum_accuracy = 0
    sum_loss = 0
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True) as prof:

        for i_batch, sample_batched in enumerate(val_loader):

            if i_batch == 100:
                break

            if continuous_dataset:
                histogram = sample_batched
            else:
                event, bounding_box, histogram = sample_batched

                bounding_box = bounding_box.to(device)
                bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * model_input_size[1].float()
                                                / width).long()
                bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * model_input_size[0].float()
                                                / height).long()

            histogram = histogram.to(device)
            histogram = F.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(model_input_size))

            with torch.no_grad():
                if i_batch%100 == 0:
                    print('refresh: ', torch.count_nonzero(histogram))
                    with record_function("model_inference_base"):
                        model_output = model.forward_refresh_reservoirs(histogram)
                        histogram_prev = histogram
                else:
                    # 
                    # m(torch.randn((1, 2, 191, 255)))                    
                    x = (histogram-histogram_prev).to(memory_format=torch.channels_last)
                    print(torch.count_nonzero(x), x.numel(), x.shape)
                    with record_function("model_inference"):
                        out, mask = model((x, None))
                    model_output += out
                    histogram_prev = histogram

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




