import numpy as np 
import pandas as pd
import h5py
import os
import time
import umap.plot
from multiprocessing import Pool
import pickle
from functools import partial
import collections
import scipy


def read_file(path):
    num_frames = 6000
    tracker = curr_mappings
    with h5py.File(path, 'r') as f1:
        data = f1["frames"].value
    data = data[:num_frames]
    shape = data.shape
    data=data.reshape(-1, data.shape[2]).T[0]
    new_info = data.astype('float64') 
    new_info = np.mean(new_info.reshape(-1, 5), 1)
    new_info = new_info.reshape(6000,-1)
    new_info = new_info/150
    new_info += tracker[path[len(path) -25 :len(path) -20]][0]
    stims_deleted = len(np.where((new_info<0).any(axis=1))[0])
    new_info = np.delete(new_info, np.where((new_info<0).any(axis=1)), axis = 0)
    new_info = np.log(new_info)
    new_info -= tracker[path[len(path) -25 :len(path) -20]][1]
    new_info /= tracker[path[len(path) -25 :len(path) -20]][2]
    cell = np.copy(new_info)
    return (cell, stims_deleted)
                
def read_file_l2(curr_name,cell_cur, path):
    final_vals = {}
    num_frames = 6000
    tracker = curr_mappings
    with h5py.File(path, 'r') as f1:
        data = f1["frames"].value
    data = data[:num_frames]
    shape = data.shape
    data=data.reshape(-1, data.shape[2]).T[0]
    new_info = data.astype('float64') 
    new_info = np.mean(new_info.reshape(-1, 5), 1)
    new_info = new_info.reshape(6000,-1)
    new_info = new_info/150
    new_info += tracker[path[len(path) -25 :len(path) -20]][0]
    stims_deleted = len(np.where((new_info<0).any(axis=1))[0])
    new_info = np.delete(new_info, np.where((new_info<0).any(axis=1))[0], axis = 0)
    new_info = np.log(new_info)
    new_info -= tracker[path[len(path) -25 :len(path) -20]][1]
    new_info /= tracker[path[len(path) -25 :len(path) -20]][2]
    cell = np.copy(new_info)
    final_vals[curr_name+' '+path[len(path) -25 :len(path) -20]] = np.average(scipy.spatial.distance.cdist(cell_cur,cell))
    return final_vals
    
    
if __name__ == "__main__":
    list_of_cells = os.listdir("/global/cfs/cdirs/m2043/balewski/neuronBBP-data_67pr")
    base_path = "/global/cfs/cdirs/m2043/balewski/neuronBBP-data_67pr/"
    paths_new = []
    for cell in list_of_cells:
        paths_new.append(base_path + cell + '/' + cell + '.cellSpike.data_2.h5')
    with open('/global/homes/s/shivam/mappings.pickle', 'rb') as handle:
        curr_mappings = pickle.load(handle)
    
    mapper = {}
    stims_deleted_map = {}
    for i in range(len(paths_new)):
        path_1 = paths_new[i]
        cell_1, stims_deleted = read_file(path_1)
        stims_deleted_map[path_1[len(path_1) -25 :len(path_1) -20]] = stims_deleted
        paths = paths_new[i+1:]
        func = partial(read_file_l2, path_1[len(path_1) -25 :len(path_1) -20], cell_1)
        try:
            pool = Pool(15)
            data_outputs = pool.map(func, iterable=paths)
        finally: 
            pool.close()
            pool.join()
        for d in data_outputs:
            for k, v in d.items():
                mapper[k] = v
    
    with open('/global/homes/s/shivam/final_mappings.py', 'wb') as handle:
        pickle.dump(mapper, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('/global/homes/s/shivam/final_stims_deleted.py', 'wb') as handle:
        pickle.dump(stims_deleted_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(tracker)
    print(stims_deleted_map)
    
    