import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
# import Caltech101ContinuousEvDataset
from incr_modules.obj_detection_incr import DenseObjectDetIncr
from ev_projs.rpg_async.models.yolo_loss import yoloLoss
from ev_projs.rpg_async.dataloader.dataset import NCaltech101_ObjectDetection



if __name__ == '__main__':

    device='cuda'
    # memory_format=torch.channels_last
    # base_folder = '/home/sankeerth/ev/depth_proj/data/test/'

    # dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
    #         width=346, height=260, window_size = 0.05, time_shift = 0.05) # 1 ms time shift

    height, width = 180, 240
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

    model = DenseObjectDetIncr(dataset.nr_classes, in_c=2,
                                    small_out_map=(True))
    lp = '/home/sankeerth/ev/rpg_asynet/log/20220508-215001/checkpoints/model_step_49.pth'
    m = torch.load(lp)
    model.load_state_dict(m['state_dict'])
    model = model.to(device)
    model.eval()

    val_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    model_input_size = torch.tensor([191, 255])
    sum_accuracy = 0
    sum_loss = 0

    for i_batch, sample_batched in enumerate(val_loader):
        event, bounding_box, histogram = sample_batched

        bounding_box = bounding_box.to(device)
        histogram = histogram.to(device) 

        # Convert spatial dimension to model input size
        histogram = F.interpolate(histogram.permute(0, 3, 1, 2),
                                                    torch.Size(model_input_size))

        # Change x, width and y, height
        bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * model_input_size[1].float()
                                        / width).long()
        bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * model_input_size[0].float()
                                        / height).long()

        with torch.no_grad():

            if i_batch%1000 == 0:
                model_output = model.forward_refresh_reservoirs(histogram)
                histogram_prev = histogram
            else:
                out, mask = model(histogram - histogram_prev)
                model_output += out
                histogram_prev = histogram

            loss = yoloLoss(model_output, bounding_box, model_input_size)[0]
            # detected_bbox = yoloDetect(model_output, self.model_input_size.to(model_output.device),
            #                             threshold=0.3)
            # detected_bbox = nonMaxSuppression(detected_bbox, iou=0.6)
            # detected_bbox = detected_bbox.cpu().numpy()

        sum_loss += loss


    print(f"Test Loss: {sum_loss}")

        # show_tensor_image()




