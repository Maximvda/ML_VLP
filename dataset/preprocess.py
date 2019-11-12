import scipy.io
import numpy as np
import random
import os
import pickle

def preprocess(dataroot):
    #Load matlab file
    mat = scipy.io.loadmat(os.path.join(dataroot,'dataset.mat'))
    #Initialising some variables
    rx_id = mat['rx_id'][0]-1
    num_meas = mat['number_of_meas'][0][0]
    offset = mat['offset']
    resolution = mat['resolution'][0][0]

    channel_data = mat['channel_data']

    #New data variable List with element = [Value,k=48 nog vragen, position]
    data = []

    for id in rx_id:
        for i in range(0,channel_data.shape[4]):
            for j in range(0,channel_data.shape[5]):
                for measurement_id in range(0,num_meas):
                    tmp_data = channel_data[:,:,id,0,i,j,measurement_id]
                    x = offset[id][0] + i*resolution
                    y = offset[id][1] + j*resolution
                    position = [x, y]
                    tmp_data = [tmp_data, position]
                    data.append(tmp_data)

    #Randomly shuffling and splitting data in train and test set
    random.shuffle(data)
    train_data = data[:int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]

    with open(os.path.join(dataroot,'train_data.data'), 'wb') as f:
        pickle.dump(train_data, f)

    with open(os.path.join(dataroot,'test_data.data'), 'wb') as f:
        pickle.dump(test_data, f)
