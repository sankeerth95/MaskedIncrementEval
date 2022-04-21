import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join

class DepthDataset(Dataset):

    def __init__(self, base_folder, depth_folder, max_length, reg_factor=5.7, clip_distance=100., initial_stamp=0, transform=None):
      
        self.depth_folder = join(base_folder, depth_folder)
        dummy_frame = np.load(join(self.depth_folder, 'depth_0000000000.npy')).astype(np.float32)
        self.height, self.width = dummy_frame.shape
        self.stamps = np.loadtxt(join(self.depth_folder, 'timestamps.txt'))[:, 1]

        self.stamps -= initial_stamp
        self.length = max_length
        self.transform = transform
        self.clip_distance = clip_distance
        self.reg_factor = reg_factor

    def __len__(self):
        return self.length


    def __getitem__(self, frame_idx):

        frame = np.load(join(self.depth_folder, 'depth_{:010d}.npy'.format(frame_idx))).astype(np.float32)
        frame = np.clip(frame, 0.0, self.clip_distance)/self.clip_distance # Clip to maximum distance and Normalize
        frame = 1.0 + np.log(frame) / self.reg_factor # Convert to log depth
        frame = frame.clip(0, 1.0) # Clip between 0 and 1.0
        if len(frame.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            frame = np.expand_dims(frame, -1)
        frame = np.moveaxis(frame, -1, 0)  # H x W x C -> C x H x W
        frame = torch.from_numpy(frame)  # numpy to tensor

        if self.transform:
            frame = self.transform(frame)

        return frame



