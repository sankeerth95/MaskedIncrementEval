import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from data_fetchers.continuous_event_datasets import ContinuousEventsDataset


from ev_projs.rpg_event_representation_learning.utils.dataset import NCaltech101
from ev_projs.rpg_event_representation_learning.utils.loader import Loader
from incr_modules.event_representation_incr import ClassifierIncrEval
from ev_projs.rpg_event_representation_learning.utils.loss import cross_entropy_loss_and_accuracy
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np



def collate_events(data):
    labels = []
    events = []
    for i, d in enumerate(data):
        labels.append(d[1])
        ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events,0))
    labels = default_collate(labels)
    return events, labels



if __name__ == '__main__':
    device='cuda'

    height, width = 180, 240
    continuous_dataset = False
    if continuous_dataset:
        memory_format=torch.channels_last
        base_folder = '/home/sankeerth/ev/data/Caltech_data/'
        dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
                width=width, height=height, evframe_type='histogram', window_size = 0.05, time_shift = 0.001) # 1 ms time shift
    else:
        dataset_path = '/home/sankeerth/ev/rpg_event_representation_learning/N-Caltech101/testing/'
        dataset = NCaltech101(
            dataset_path
        )


    test_loader = DataLoader(dataset, collate_fn=collate_events)


    model = ClassifierIncrEval()
    m = torch.load('/home/sankeerth/ev/rpg_event_representation_learning/log/checkpoint_27225_0.7328.pth')
    model.load_state_dict(m['state_dict'])
    model = model.to(device).eval()
 
    sum_accuracy = 0
    sum_loss = 0


    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True) as prof:

        for i_batch, sample in enumerate(test_loader):
            events, labels = sample

    model = ClassifierIncrEval()
    m = torch.load('/home/sankeerth/ev/rpg_event_representation_learning/log/checkpoint_27225_0.7328.pth')
    model.load_state_dict(m['state_dict'])
    model = model.to(device).eval()
 
    sum_accuracy = 0
    sum_loss = 0


    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True) as prof:

        for i_batch, sample in enumerate(test_loader):
            events, labels = sample
            events = events.to(device)
            labels = labels.to(device)

            vox = model.quantization_layer.forward(events)
            vox_cropped = model.crop_and_resize_to_resolution(vox, model.crop_dimension)

            with torch.no_grad():
                if i_batch%20 == 0:
                    with record_function("model_inference_base"):
                        pred_labels = model.classifier.forward_refresh_reservoirs(vox_cropped)
                    vox_cropped_prev = vox_cropped
                else:
                    x = (vox_cropped - vox_cropped_prev).to(memory_format=torch.channels_last)
                    with record_function("model_inference"):
                        out, mask = model.classifier((x, None))
                    pred_labels += out
                    vox_cropped_prev = vox_cropped        
                    # loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

            # sum_accuracy += accuracy
            # sum_loss += loss

            if i_batch == 20:
                break

    # print(pred_labels)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device), row_limit=15))
    # print(f"Test Loss: {sum_loss}")

    # test_loss = sum_loss.item() / len(test_loader)
    # test_accuracy = sum_accuracy.item() / len(test_loader)
    # print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # show_tensor_image()


