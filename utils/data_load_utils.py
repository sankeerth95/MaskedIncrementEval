import numpy as np


def readBoundingBox(file_path):
    with open(file_path) as f:
        annotations = np.fromfile(f, dtype=np.int16)
    return annotations[2:10]

def loadEventsFile(file_name):
    with open(file_name, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint8)

    raw_data = np.uint32(raw_data)
    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    all_p = all_p.astype(np.float64)
    all_p[all_p == 0] = -1

    return np.column_stack((all_x, all_y, all_ts, all_p))



