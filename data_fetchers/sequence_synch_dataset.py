from torch.utils.data import Dataset
import numpy as np
import random
import torch.nn.functional as f


class SequenceSynchronizedDataset(Dataset):

    def __init__(self,
                 sync_dataset,
                 sequence_length=2,
                 proba_pause_when_running=0.0, 
                 proba_pause_when_paused=0.0,
                 step_size=20,
                 scale_factor=1.0,
                 every_x_rgb_frame=1):

        assert(sync_dataset is not None)
        assert(sequence_length > 0)
        assert(step_size > 0)

        self.dataset = sync_dataset
        self.L = sequence_length
        self.step_size = step_size
        self.every_x_rgb_frame = every_x_rgb_frame
        self.proba_pause_when_running = proba_pause_when_running
        self.proba_pause_when_paused = proba_pause_when_paused
        self.scale_factor = scale_factor

        if self.L * self.every_x_rgb_frame >= self.dataset.length:
            self.length = 0
        else:
            self.length = (self.dataset.length - self.L * self.every_x_rgb_frame) // self.step_size\
                          // self.every_x_rgb_frame + 1

    def __getitem__(self, i):

        seed = random.randint(0, 2**32)
        sequence = []
        j = i * self.step_size
        k = 0
        paused = False
        for n in range(self.L):

            probability_pause = self.proba_pause_when_paused if paused else self.proba_pause_when_running
            paused = (np.random.rand() < probability_pause) and (n>0)

            item = self.dataset.__getitem__(j + k, seed)
            if paused:
                item['events'].fill_(0.0)
            sequence.append(item)

            k += 0 if paused else 1


        if self.scale_factor < 1.0:
            self._scale(sequence)

        return sequence


    def __len__(self):
        return self.length


    def _scale(self, sequence):
        for data_items in sequence:
            for k, item in data_items.items():
                if k != "times" and k != "batchlength_events":
                # item = item[None]
                    item = f.interpolate(item, scale_factor=self.scale_factor, mode='bilinear',
                                        recompute_scale_factor=False, align_corners=False)
                    item = item[0]
                    data_items[k] = item


