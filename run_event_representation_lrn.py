import argparse, os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_fetchers.continuous_event_datasets import ContinuousEventsDataset
from ev_projs.rpg_async.dataloader.dataset import NCaltech101_ObjectDetection
from ev_projs.rpg_event_representation_learning.utils.dataset import NCaltech101
from ev_projs.rpg_event_representation_learning.utils.loader import Loader
from incr_modules.event_representation_incr import ClassifierIncrEval
from ev_projs.rpg_event_representation_learning.utils.loss import cross_entropy_loss_and_accuracy
from torch.profiler import profile, record_function, ProfilerActivity



if __name__ == '__main__':
    device='cuda'

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
            dataset_path, 
            'all', 
            height, 
            width, 
            25000, 
            mode='validation', 
            event_representation='histogram', 
            shuffle=False
        )

    test_loader = DataLoader(dataset)


    model = ClassifierIncrEval()
    m = torch.load('/home/sankeerth/ev/rpg_event_representation_learning/log/checkpoint_27225_0.7328.pth')

    # m.state_dict())

    model.load_state_dict(m['state_dict'])
    model = model.to(device)
    model.eval()
 
    sum_accuracy = 0
    sum_loss = 0


    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True) as prof:

        for i_batch, sample in tqdm(test_loader):
            events, labels = sample
            with torch.no_grad():
                if i_batch%1 == 0:
                    with record_function("model_inference_base"):
                        pred_labels, _ = model.forward_refresh_reservoir(events)
        
                else:
                    with record_function("model_inference_base"):
                        out, mask = model(events)
                        model_out += out
        
                    # loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)


            # sum_accuracy += accuracy
            # sum_loss += loss

    # test_loss = sum_loss.item() / len(test_loader)
    # test_accuracy = sum_accuracy.item() / len(test_loader)
    # print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # show_tensor_image()




