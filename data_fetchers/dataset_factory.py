from data_fetchers.sequence_synch_dataset import SequenceSynchronizedDataset
from .frames_dataset import FrameDataset
from .depth_dataset import DepthDataset
from .synchronized_dataset import SynchronizedEDFDataset
from .continuous_event_datasets import ShiftedRawEventsDataset

class SequenceDatasetFactory:

    @staticmethod
    def createFrameDataset(base_folder, frame_folder, maxlength, transform=None, normalize=None):
        return FrameDataset(base_folder, frame_folder,
                            maxlength,
                            transform=transform,
                            normalize=normalize)

    @staticmethod
    def createDepthDataset(base_folder, depth_folder, maxlength, clip_distance=100., reg_factor=5.7, transform=None):
        return DepthDataset(base_folder, depth_folder, 
                            maxlength, reg_factor, clip_distance, 
                            initial_stamp=0, # event_dataset.initial_stamp, # TODO: wtf is initial stamp
                            transform=transform)

    @staticmethod
    def createShiftedrawDataset(base_folder, event_folder,
                                width, height,
                                start_time=0.0, 
                                stop_time=0.0,
                                transform=None,
                                normalize=None,
                                delta=0.01):
        return ShiftedRawEventsDataset(base_folder, event_folder,
                                    width=width, height=height,
                                    start_time=start_time, 
                                    stop_time=stop_time,
                                    transform=transform,
                                    normalize=normalize,
                                    delta=delta)

    @staticmethod
    def createSyncFED(base_folder, event_folder, depth_folder, frame_folder, clip_distance=100., reg_factor=5.7, start_time=0.0, stop_time=0.0, transform=None, normalize=False, delta=0.01):
        
        event_dataset = SequenceDatasetFactory.createShiftedrawDataset(base_folder, event_folder, width=346, height=260, 
            start_time=start_time, stop_time=stop_time, transform=transform, normalize=normalize, delta=delta)
        frame_dataset = SequenceDatasetFactory.createFrameDataset(base_folder, frame_folder, event_dataset.length, transform=transform, normalize=normalize)
        depth_dataset = SequenceDatasetFactory.createDepthDataset(base_folder, depth_folder, event_dataset.length, clip_distance, reg_factor, transform=transform)
        return SynchronizedEDFDataset(event_dataset, depth_dataset, frame_dataset)

    @staticmethod
    def createSeqSyncFED(base_folder, event_folder, depth_folder, frame_folder, sequence_length=2, clip_distance=100., reg_factor=5.7, start_time=0.0, stop_time=0.0, transform=None, normalize=False, delta=0.01):
        sync_dataset = SequenceDatasetFactory.createSyncFED(base_folder, event_folder, depth_folder, frame_folder, clip_distance=clip_distance, \
            reg_factor=reg_factor, start_time=start_time, stop_time=stop_time, transform=transform, normalize=normalize, delta=delta)
        return SequenceSynchronizedDataset(sync_dataset, sequence_length)


