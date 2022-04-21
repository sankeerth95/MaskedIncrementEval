import torch
import numpy as np
import random, imageio
from torch.utils.data import Dataset
from os.path import join
from .utils.util import rgb2gray

class FrameDataset(Dataset):

    def __init__(self, base_folder, frame_folder, maxlength, transform=None, normalize=True):
        super().__init__()
        self.base_folder = base_folder
        self.frame_folder = join(base_folder, frame_folder)
        self.normalize = normalize
        self.transform = transform
        self.length = maxlength

    def __len__(self):
        return self.length

    def parse_event_folder(self):
        pass

    def __getitem__(self, frame_idx, transform_seed=None):

        if transform_seed is None:
            transform_seed = random.randint(0, 2**32)

        rgb_frame = imageio.imread(join(self.frame_folder, 'frame_{:010d}.png'.format(frame_idx)),
                                                            as_gray=False).astype(np.float32)
        if len(rgb_frame.shape) > 2:
            if rgb_frame.shape[2] > 1:
                gray_frame = rgb2gray(rgb_frame)  # [H x W]
        else:
            gray_frame = rgb_frame

        if self.normalize:
            gray_frame /= 255.0  # normalize
            gray_frame = np.expand_dims(gray_frame, axis=0)  # expand to [1 x H x W]

        gray_frame = torch.from_numpy(gray_frame)
        if self.transform:
            random.seed(transform_seed)
            gray_frame = self.transform(gray_frame)

        return gray_frame # [1 x H x W] tensor


