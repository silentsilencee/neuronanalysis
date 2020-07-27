import numpy as np 
import pandas as pd
import h5py
import os

def read_file_setup(path, tracker):
    num_frames = 1
    tracker = {}
    with h5py.File(path, 'r') as f1:
        data = f1["frames"].value
    data = data[:num_frames]
    shape = data.shape
    data=data.reshape(-1, data.shape[2]).T[0]
    new_info = data.astype('float64') 
    new_info = new_info/150
    offset = -1* np.min(new_info) + 1
    new_info = new_info + offset
    new_info = np.log(new_info)
    mean = np.mean(new_info)
    std = np.std(new_info)
    tracker[val[i_folder]] = [offset,mean,std]
    return tracker
    

if __name__ == "__main__":
    list_of_cells = os.listdir("/global/cfs/cdirs/m2043/balewski/neuronBBP-data_67pr")
    base_path = "/global/cfs/cdirs/m2043/balewski/neuronBBP-data_67pr/"
    paths_new = []
    tracker = {}
    for cell in list_of_cells:
        paths_new.append(base_path + cell + '/' + cell + '.cellSpike.data_0.h5')
    for path in paths_new:
        read_file_setup(path, tracker)
    