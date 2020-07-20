import numpy as np
import h5py
import os
import time

def get_l2_mappings(number_of_frames):
    list_of_cells = os.listdir("/global/cfs/cdirs/m2043/balewski/neuronBBP-packMay1/probe_orig/")
    num_frames = number_of_frames
    cache_of_old_data = {}
    final_vals = {}
    for i_folder in range(len(list_of_cells)):
        index = 0
        while (os.listdir("/global/cfs/cdirs/m2043/balewski/neuronBBP-packMay1/probe_orig/"+list_of_cells[i_folder]+"/")[index][-3:] != ".h5"):
            index += 1
        file = os.listdir("/global/cfs/cdirs/m2043/balewski/neuronBBP-packMay1/probe_orig/"+list_of_cells[i_folder]+"/")[index]
        f1 = h5py.File("/global/cfs/cdirs/m2043/balewski/neuronBBP-packMay1/probe_orig/"+list_of_cells[i_folder]+"/"+file, 'r')
        data = f1["frames"].value
        data = data[:num_frames]
        data=data.reshape(-1, data.shape[2]).T[0]
        new_info = data.astype('float64')
        new_info = new_info/150
        if list_of_cells[i_folder] not in cache_of_old_data.keys():
            offset = -1 * np.min(new_info) + 1
            new_info = new_info + offset
            new_info = np.log(new_info)
            mean = np.mean(new_info)
            std = np.std(new_info)
            cache_of_old_data[list_of_cells[i_folder]] = [offset,mean,std]
        new_info += cache_of_old_data[list_of_cells[i_folder]][0]
        new_info = np.log(new_info)
        new_info -= cache_of_old_data[list_of_cells[i_folder]][1]
        new_info /= cache_of_old_data[list_of_cells[i_folder]][2]
        cell_1 = np.copy(new_info)
        cell_1 = cell_1.reshape(-1, 8000)
        print("stage 1")
        for j_folder in range(i_folder+1, len(list_of_cells)):
            index = 0
            while os.listdir("/global/cfs/cdirs/m2043/balewski/neuronBBP-packMay1/probe_orig/"+list_of_cells[j_folder]+"/")[index][-3:] != ".h5":
                index+= 1
            file = os.listdir("/global/cfs/cdirs/m2043/balewski/neuronBBP-packMay1/probe_orig/"+list_of_cells[j_folder]+"/")[index]
            f1 = h5py.File("/global/cfs/cdirs/m2043/balewski/neuronBBP-packMay1/probe_orig/"+list_of_cells[j_folder]+"/"+file, 'r')
            data = f1["frames"].value
            data = data[:num_frames]
            data=data.reshape(-1, data.shape[2]).T[0]
            new_info = data.astype('float64')
            new_info = new_info/150
            if list_of_cells[j_folder] not in cache_of_old_data.keys():
                offset = -1* np.min(new_info) + 1
                new_info = new_info + offset
                new_info = np.log(new_info)
                mean = np.mean(new_info)
                std = np.std(new_info)
                cache_of_old_data[list_of_cells[j_folder]] = [offset,mean,std]
            new_info += cache_of_old_data[list_of_cells[j_folder]][0]
            new_info = np.log(new_info)
            new_info -= cache_of_old_data[list_of_cells[j_folder]][1]
            new_info /= cache_of_old_data[list_of_cells[j_folder]][2]
            cell_2 = np.copy(new_info)
            cell_2 = cell_2.reshape(-1, 8000)
            print("stage 2")
            curr = []
            for val_1 in cell_1:
                for val_2 in cell_2:
                    # we can use the cuda implementation of linalg norm which is
                    #import pycuda.gpuarray as gpuarray
                    #import numpy as np
                    #import skcuda.linalg as linalg
                    # nrm = linalg.norm(gpuarray.to_gpu(val_1-val_2))
                    curr.append(np.linalg.norm(val_1-val_2))
            final_vals[list_of_cells[i_folder]+list_of_cells[j_folder]] = np.mean(curr)
            print(np.mean(curr))
    return final_vals, cache_of_old_data
