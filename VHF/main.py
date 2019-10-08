import os

import numpy as np

from vhf.io import save_result
from vhf.process import process_all_files


def main():
    time_step = 1  # ns
    path = os.path.join(".", "data")
    # path = "/home/zelenyy/storage/data/sri-thunderstorm/VHF/first/"
    observed_point = np.array([1e3,0,0]) # meter
    signal_electric, signal_magnetic, time = process_all_files(path, observed_point, time_step)
    print(save_result("result.hdf5", signal_electric, signal_magnetic, time))

if __name__ == '__main__':
    main()