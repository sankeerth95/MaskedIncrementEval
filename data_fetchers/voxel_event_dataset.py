import torch
import random
import numpy as np
from .base_event_dataset import EventDataset
from .utils.util import normalize_voxelgrid

from os.path import join


class VoxelGridDataset(EventDataset):

    def parse_event_folder(self):
        self.num_bins = None

    def num_channels(self):
        return self.num_bins

    def __getitem__(self, i, transform_seed=None):
        assert(i >= 0)
        assert(i < self.length)

        if transform_seed is None:
            transform_seed = random.randint(0, 2**32)

        # event_tensor will be a [num_bins x H x W] floating point array
        event_tensor = np.load(join(self.event_folder, 'event_tensor_{:010d}.npy'.format(self.first_valid_idx + i)))
        if self.normalize:
            event_tensor = normalize_voxelgrid(event_tensor)
        self.num_bins = event_tensor.shape[0]

        events = torch.from_numpy(event_tensor)  # [C x H x W]
        if self.transform:
            random.seed(transform_seed)
            events = self.transform(events)

        return {'events': events}  # [num_bins x H x W] tensor






