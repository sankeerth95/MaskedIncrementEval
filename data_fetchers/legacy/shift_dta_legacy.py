import torch
from torch_sparse import coalesce
from torch.utils.data import Dataset
from rospy import Duration

import torchvision.transforms.functional as F
from PIL import Image
import os
import numpy as np
import cv2

import h5py

_MAX_SKIP_FRAMES = 6
_TEST_SKIP_FRAMES = 4
_N_SKIP = 1



# self.n_ima: cumulatively increasing image counts
# convention: n_ima: [0, left, left+right]
class DataLoader:
    def __init__(self):
        self.n_ima = [0]

    def derive_cam_and_image_iter(self, index, n_ima):
        cam = -1
        image_iter = index
        for i in n_ima:
            if index < i:
                break
            cam += 1
            image_iter = image_iter - i
        return cam, image_iter

class NumpyDataLoader(DataLoader):
    def __init__(self, data_folder_path, shift_duration, split):
        super().__init__()
        self.event_data_paths, self.n_ima = self.read_file_paths(data_folder_path, split)
        self.shift_duration = shift_duration

    def get_n_ima(self):
        return self.n_ima

    def read_file_paths(self,
                        data_folder_path,
                        split,
                        sequence=None):

        event_data_paths = []
        n_ima = 0
        if sequence is None:
            bag_list_file = open(os.path.join(data_folder_path, "{}_bags.txt".format(split)), 'r')
            lines = bag_list_file.read().splitlines()
            bag_list_file.close()
        else:
            if isinstance(sequence, (list, )):
                lines = sequence
            else:
                lines = [sequence]
        
        n_ima = [0]
        for line in lines:
            bag_name = line

            event_data_paths.append(os.path.join(data_folder_path,bag_name))
            num_ima_file = open(os.path.join(data_folder_path, bag_name, 'n_images.txt'), 'r')
            num_imas = num_ima_file.read()
            num_ima_file.close()
            num_imas_split = num_imas.split(' ')
            n_left_ima = int(num_imas_split[0]) - _MAX_SKIP_FRAMES
            n_ima.append(n_left_ima + n_ima[-1])
            
            n_right_ima = int(num_imas_split[1]) - _MAX_SKIP_FRAMES
            if n_right_ima > 0 and split != 'test':
                n_ima.append(n_right_ima + n_ima[-1])
            else:
                n_ima.append(n_ima[-1])
            event_data_paths.append(os.path.join(data_folder_path,bag_name))

        return event_data_paths, n_ima

    def load_image(self, index, n_frames):        
        cam, image_iter = self.derive_cam_and_image_iter(index, self.n_ima)
        prefix = self.event_data_paths[cam]
        cam = 'left' if cam % 2 == 0 else 'right' # convention

        event_count_images, event_time_images, image_times = np.load(prefix + "/events/"+ cam +"_event" + \
            str(image_iter).rjust(5,'0') + ".npy", encoding='bytes', allow_pickle=True)
        event_count_images = torch.from_numpy(event_count_images.astype(np.int16))
        event_time_images = torch.from_numpy(event_time_images.astype(np.float32))
        image_times = torch.from_numpy(image_times.astype(np.float64))

        #load shifted events
        shifted_event_count_images, shifted_event_time_images, _ = np.load(prefix + "/events_"+ str(self.shift_duration) +"/"+cam+"_event" +\
             str(image_iter).rjust(5,'0') + ".npy", encoding='bytes', allow_pickle=True)
        shifted_event_count_images = torch.from_numpy(shifted_event_count_images.astype(np.int16))
        shifted_event_time_images = torch.from_numpy(shifted_event_time_images.astype(np.float32))

        # get prev and next image : only required for training
        prev_img_path = prefix + "/images/" + cam + "_image" + str(image_iter).rjust(5,'0') + ".png"
        next_img_path = prefix + "/images/" + cam +  "_image" + str(image_iter+n_frames).rjust(5,'0') + ".png"
        prev_image = Image.open(prev_img_path)
        next_image = Image.open(next_img_path)
        
        return event_count_images, event_time_images, prev_image, next_image, image_times, shifted_event_count_images, shifted_event_time_images


class H5pyDataLoader(DataLoader):
 
    def __init__(self, data_paths, rows, cols, split):
        super().__init__()
        self.rows = rows
        self.cols = cols

        self.data = []
        for data_path in data_paths:
            instance = h5py.File(data_path)
            self.data.append(instance)

            if 'image_raw' in instance['davis']['left'].keys():
                n_left_ima = len(instance['davis']['left']['image_raw'])
            else:
                n_left_ima = 0
            if 'image_raw' in instance['davis']['right'].keys():        
                n_right_ima = len(instance['davis']['right']['image_raw'])
            else:
                n_right_ima = 0

            self.n_ima.append(self.n_ima[-1] + n_left_ima)
            if n_right_ima > 0 and split != 'test':
                self.n_ima.append(self.n_ima[-1] + n_right_ima)
            else:
                self.n_ima.append(self.n_ima[-1])


    def accumulate_event_images(self, events):

        events_torch = torch.tensor(events).cuda()

        coords = events_torch[:, [1, 0, 3]].T
        coords[2, :] = (coords[2, :]+1)/2 
        n_points = coords.shape[-1]

        event_count_image = torch.sparse_coo_tensor(coords, torch.ones(n_points).cuda(), (self.rows, self.cols, 2)).to_dense()

        t = events_torch[:, 2]
        t -= torch.min(t)
        t_pos = coalesce(coords[:2,:].long(), coords[2, :]*t, m=self.rows, n=self.cols, op='max')
        t_neg = coalesce(coords[:2,:].long(), (1.-coords[2, :])*t, m=self.rows, n = self.cols, op='max')

        t_pos = torch.sparse_coo_tensor(t_pos[0], t_pos[1], (self.rows, self.cols)).to_dense()
        t_neg = torch.sparse_coo_tensor(t_neg[0], t_neg[1], (self.rows, self.cols)).to_dense()
        event_time_image = torch.stack([t_neg, t_pos], dim=2)

        return event_count_image, event_time_image

    def load_image(self, index, n_frames, max_aug=6, n_skip=1):
        
        cam, image_iter = self.derive_cam_and_image_iter(index, self.n_ima)
        lr = 'left' if cam % 2 == 0 else 'right' # convention

        events = self.data[cam]['davis'][lr]['events']        
        event_indices = self.data[cam]['davis'][lr]['image_raw_event_inds']
        
        event_count_images, event_time_images = [], []
        for i in range(n_frames):
            start = min(image_iter+i, len(event_indices)-2)
            end = min(image_iter+i+max_aug*n_skip, len(event_indices)-1)
            event_count_image, event_time_image = self.accumulate_event_images(events[event_indices[start]+1:event_indices[end]])
            event_count_images.append(event_count_image)
            event_time_images.append(event_time_image)
        
        event_count_images, event_time_images = torch.stack(event_count_images), torch.stack(event_time_images)

        # need to  construct event time and event count images
        prev_image = self.data[cam]['davis'][lr]['image_raw'][image_iter]
        next_image = self.data[cam]['davis'][lr]['image_raw'][image_iter+n_frames*n_skip]
        image_ts = self.data[cam]['davis'][lr]['image_raw_ts'][image_iter:image_iter+n_frames*n_skip+1]

        return event_count_images, event_time_images, Image.fromarray(prev_image), Image.fromarray(next_image), image_ts
        
    def close(self):
        self.data.close()


class MyEventData(Dataset):
    """
    args:
    data_folder_path:the path of data
    split:'train' or 'test'
    """
    def __init__(self, data_folder_path, split, image_height, image_width, \
        shift_duration = Duration(0, 20000000), count_only=False, time_only=False, skip_frames=False, use_h5=True):
        self._split = split
        self._count_only = count_only
        self._time_only = time_only
        self._skip_frames = skip_frames
        self.image_height = image_height
        self.image_width = image_width
        self.shift_duration = shift_duration
        if use_h5:
            self.data_loader = H5pyDataLoader(['/home/sankeerth/Downloads/outdoor_day1_data.hdf5'], 260, 346, split)
        else:
            self.data_loader = NumpyDataLoader(data_folder_path, shift_duration, split)


    def __getitem__(self, index):

        # get skip frames for test
        if self._split == 'test':
            if self._skip_frames:
                n_frames = _TEST_SKIP_FRAMES
            else:
                n_frames = 1
        else:
            n_frames = np.random.randint(low=1, high=_MAX_SKIP_FRAMES+1) * _N_SKIP

        # load events
        event_count_images, event_time_images, prev_image, next_image, image_times \
            = self.data_loader.load_image(index, n_frames)

        # combine n frames of events read
        timestamps = [image_times[0], image_times[n_frames]]
        event_count_image, event_time_image = self._read_events(event_count_images, event_time_images, n_frames)

        if self._split == 'train':
            # transforms: augment dataset on the flow for training
            rand_flip = np.random.randint(low=0, high=2)
            rand_rotate = np.random.randint(low=-30, high=30)
            x = np.random.randint(low=1, high=(event_count_image.shape[1]-self.image_height))
            y = np.random.randint(low=1, high=(event_count_image.shape[2]-self.image_width))

            event_image = self._augment_events(event_count_image, event_time_image, x,y, rand_rotate, rand_flip)
            # shifted_event_image = self._augment_events(shifted_event_count_image, shifted_event_time_image, x,y, rand_rotate, rand_flip)
            prev_image = self._augment_image(prev_image, x, y, rand_rotate, rand_flip)
            next_image = self._augment_image(next_image, x, y, rand_rotate, rand_flip)
        else:
            # get testing image
            event_image = self._get_test_event_image(event_count_image, event_time_image)
            # shifted_event_image = self._get_test_event_image(shifted_event_count_image, shifted_event_time_image)
            prev_image = self._get_test_image(prev_image)
            next_image = self._get_test_image(next_image)

        return event_image, prev_image, next_image, timestamps

    def __len__(self):
        return self.data_loader.n_ima[-1] - 6 # hack: so that we can refer to the next image successfully

    def _get_test_image(self, img):
        return F.to_tensor(F.center_crop(img, (self.image_height, self.image_width)))

    def _get_test_event_image(self, event_count_image, event_time_image):
        if self._count_only:
            event_image = F.to_tensor(F.center_crop(F.to_pil_image(event_count_image / 255.), 
                                        (self.image_height, self.image_width)))
            event_image = event_image * 255.
        elif self._time_only:
            event_image = F.to_tensor(F.center_crop(F.to_pil_image(event_time_image), 
                                        (self.image_height, self.image_width)))
        else:
            event_image = torch.cat((event_count_image / 255.,event_time_image), dim=0)
            event_image = F.to_tensor(F.center_crop(F.to_pil_image(event_image), 
                                        (self.image_height, self.image_width)))
            event_image[:2,...] = event_image[:2,...] * 255.
        return event_image

    def _augment_image(self, 
                        img,  # clobbered
                        x, y, 
                        rand_rotate, 
                        rand_flip):
        if rand_flip == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rand_rotate)
        img = F.to_tensor(img)
        img = img[...,x:x+self.image_height,y:y+self.image_width]
        return img

    def _augment_events(self,
                        event_count_image, # clobbered 
                        event_time_image, # clobbered
                        x,y,
                        rand_rotate, 
                        rand_flip):
        if self._count_only:
            event_count_image = F.to_pil_image(event_count_image / 255.)
            # random_flip
            if rand_flip == 0:
                event_count_image = event_count_image.transpose(Image.FLIP_LEFT_RIGHT)
            # random_rotate
            event_image = event_count_image.rotate(rand_rotate)
            # random_crop
            event_image = F.to_tensor(event_image) * 255.
            event_image = event_image[:,x:x+self.image_height,y:y+self.image_width]
        elif self._time_only:
            event_time_image = F.to_pil_image(event_time_image)
            # random_flip
            if rand_flip == 0:
                event_time_image = event_time_image.transpose(Image.FLIP_LEFT_RIGHT)
            # random_rotate
            event_image = event_time_image.rotate(rand_rotate)
            # random_crop
            event_image = F.to_tensor(event_image)
            event_image = event_image[:,x:x+self.image_height,y:y+self.image_width]
        else:
            event_count_image = F.to_pil_image(event_count_image / 255.)
            event_time_image = F.to_pil_image(event_time_image)
            # random_flip
            if rand_flip == 0:
                event_count_image = event_count_image.transpose(Image.FLIP_LEFT_RIGHT)
                event_time_image = event_time_image.transpose(Image.FLIP_LEFT_RIGHT)
            # random_rotate
            event_count_image = event_count_image.rotate(rand_rotate)
            event_time_image = event_time_image.rotate(rand_rotate)
            # random_crop
            event_count_image = F.to_tensor(event_count_image) * 255.
            event_time_image = F.to_tensor(event_time_image) 
            event_image = torch.cat((event_count_image,event_time_image), dim=0)
            event_image = event_image[...,x:x+self.image_height,y:y+self.image_width]
 
        return event_image

    def _read_events(self,
                     event_count_images,
                     event_time_images,
                     n_frames):
        #event_count_images = event_count_images.reshape(shape).type(torch.float32)
        event_count_image = event_count_images[:n_frames, :, :, :]
        event_count_image = torch.sum(event_count_image, dim=0).type(torch.float32)
        event_count_image = event_count_image.permute(2,0,1)

        #event_time_images = event_time_images.reshape(shape).type(torch.float32)
        event_time_image = event_time_images[:n_frames, :, :, :]
        event_time_image = torch.max(event_time_image, dim=0)[0]

        event_time_image /= torch.max(event_time_image)
        event_time_image = event_time_image.permute(2,0,1)

        '''
        if self._count_only:
            event_image = event_count_image
        elif self._time_only:
            event_image = event_time_image
        else:
            event_image = torch.cat([event_count_image, event_time_image], dim=2)

        event_image = event_image.permute(2,0,1).type(torch.float32)
        '''

        return event_count_image, event_time_image


if __name__ == "__main__":
    data = MyEventData('../data/mvsec/', 'test', 256, 256, shift_duration = Duration(0, 20000000), use_h5=True)
    EventDataLoader = torch.utils.data.DataLoader(dataset=data, batch_size=1,shuffle=False)
    it = 0
    for i in EventDataLoader:
        a = i[0][0].numpy()
        b = i[1][0].numpy()
        c = i[2][0].numpy()
        d = i[3][0].numpy()


        print(np.count_nonzero(a[:2,...]))
        print(np.count_nonzero(a[2:,...]))

#        a = a[2,...]+a[3,...]
        a = a[0,...]+a[1,...]

        # d = d[2,...]+d[3,...]
        print(np.max(a))

        a = (a-np.min(a))/(np.max(a)-np.min(a))
        b = np.transpose(b,(1,2,0))
        c = np.transpose(c,(1,2,0))

#        d = (d-np.min(d))/(np.max(d)-np.min(d))

        cv2.imshow('a',a)
        cv2.imshow('b',b)
        cv2.imshow('c',c)
#        cv2.imshow('d',d)
#        cv2.imshow('a+d',a+d)

        cv2.waitKey(1)

