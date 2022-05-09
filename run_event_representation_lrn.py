import argparse, os
import torch
from tqdm import tqdm
from ev_projs.rpg_event_representation_learning.utils.dataset import NCaltech101
from ev_projs.rpg_event_representation_learning.utils.loader import Loader
from incr_modules.event_representation_incr import ClassifierIncrEval
from ev_projs.rpg_event_representation_learning.utils.loss import cross_entropy_loss_and_accuracy


def FLAGS():
    parser = argparse.ArgumentParser(
        """Deep Learning for Events. Supply a config file.""")

    # can be set in config
    parser.add_argument("--checkpoint", default="", required=False)
    parser.add_argument("--test_dataset", default="", required=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)

    flags = parser.parse_args()

    # assert os.path.isdir(os.path.dirname(flags.checkpoint)), f"Checkpoint{flags.checkpoint} not found."
    # assert os.path.isdir(flags.test_dataset), f"Test dataset directory {flags.test_dataset} not found."

    # print(f"----------------------------\n"
    #       f"Starting testing with \n"
    #       f"checkpoint: {flags.checkpoint}\n"
    #       f"test_dataset: {flags.test_dataset}\n"
    #       f"batch_size: {flags.batch_size}\n"
    #       f"device: {flags.device}\n"
    #       f"----------------------------")

    return flags



if __name__ == '__main__':
    device='cuda'
    # memory_format=torch.channels_last
    # base_folder = '/home/sankeerth/ev/depth_proj/data/test/'

    # dataset = ContinuousEventsDataset(base_folder=base_folder, event_folder='events/data',\
    #         width=346, height=260, window_size = 0.05, time_shift = 0.05) # 1 ms time shift


    flags = FLAGS()
    test_dataset = NCaltech101('/home/sankeerth/ev/rpg_event_representation_learning/N-Caltech101/testing/')
    test_loader = Loader(test_dataset, flags, device)


    model = ClassifierIncrEval()
    m = torch.load('/home/sankeerth/ev/rpg_event_representation_learning/log/checkpoint_27225_0.7328.pth')

    # m.state_dict())

    model.load_state_dict(m['state_dict'])
    model = model.to(device)
    model.eval()
 
    sum_accuracy = 0
    sum_loss = 0

    for events, labels in tqdm(test_loader):
        with torch.no_grad():
            pred_labels, _ = model.forward_refresh_reservoir(events)
            loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)
        sum_accuracy += accuracy
        sum_loss += loss

    test_loss = sum_loss.item() / len(test_loader)
    test_accuracy = sum_accuracy.item() / len(test_loader)

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # show_tensor_image()




