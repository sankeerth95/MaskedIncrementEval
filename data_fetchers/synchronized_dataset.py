import torch
import numpy as np
from torch.utils.data import Dataset
from data_fetchers.base_event_dataset import EventDataset
from .utils.util import first_element_greater_than
from .frames_dataset import FrameDataset
from .depth_dataset import DepthDataset


#EDF: events, depth, frames
class SynchronizedEDFDataset(Dataset):

    def __init__(self, 
                event_dataset: EventDataset,
                depth_dataset: DepthDataset,
                frame_dataset: FrameDataset,
                every_x_rgb_frame=1,
                use_phased_arch=False):
        super().__init__()
        self.eps = 1e-06
        self.use_phased_arch = use_phased_arch
        self.length = event_dataset.length
        self.every_x_rgb_frame = every_x_rgb_frame
        self.depth_dataset = depth_dataset
        self.frame_dataset = frame_dataset
        self.event_dataset = event_dataset
        self.stamps = self.event_dataset.stamps

    def __len__(self):
        return self.length

    def get_frameidx(self, j):
        event_timestamp = self.event_dataset.get_stamp_at(j)            
        (frame_idx, frame_timestamp) = first_element_greater_than(self.stamps, event_timestamp)
        assert(frame_idx >= 0)
        assert(frame_idx < len(self.stamps))
        assert(frame_timestamp - event_timestamp < 0.00001)
        return frame_idx, event_timestamp

    # TODO: figure out what transform seed is.
    def __getitem__(self, i, seed=None):
        assert(i >= 0)
        assert(i < (self.length // self.every_x_rgb_frame))

        item = {}
        for k in range(0, self.every_x_rgb_frame):
            j = i * self.every_x_rgb_frame + k
            voxelgrid, shifted_voxelgrid = self.event_dataset.__getitem__(j)

            # latest depth frame
            frame_idx, event_timestamp = self.get_frameidx(j)
            frame = self.depth_dataset.__getitem__(frame_idx)

            # latest rgb
            if self.frame_dataset is not None:
                try:
                    gray_frame = self.frame_dataset.__getitem__(frame_idx)
                except FileNotFoundError:
                    gray_frame = None

            # recordable features
            item['events{}'.format(k)] = voxelgrid
            item['shifted_events{}'.format(k)] = shifted_voxelgrid
            if self.use_phased_arch:
                timestamp = torch.from_numpy(np.asarray([event_timestamp]).astype(np.float32))
                item['times_events{}'.format(k)] = timestamp

        # depth and rgb!
        item['depth_image'.format(k)] = frame # overwritable
        item['image'] = gray_frame
        if self.use_phased_arch:
            item['times_image'] = timestamp

        return item

