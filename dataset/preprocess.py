import scipy.io
import numpy as np
import random
import os
import pickle

def preprocess(dataroot, normalise=False):
    #Load matlab file
    mat = scipy.io.loadmat(os.path.join(dataroot,'dataset.mat'))
    #Initialising some variables
    rx_id = mat['rx_id'][0]-1
    no_it = mat['no_it'][0][0]
    offset = mat['offset']
    resolution = mat['resolution'][0][0]

    swing = 1
    channel_data = mat['swing'] if swing else np.mean(mat['channel_data'],axis=1)
    input_norm = np.max(channel_data)/2

    #New data variable List with element = [Value,k=48 nog vragen, position]
    data = []

    for id in rx_id:
        for i in range(0,channel_data.shape[3]):
            for j in range(0,channel_data.shape[4]):
                for it in range(0,no_it):
                    #Select 1 measurement and reshape in the 6,6 grid of the LEDS
                    tmp_data = channel_data[:,id,it,i,j].reshape((6,6))
                    #Calculate position of measurement device
                    x = offset[id][0] + i*resolution
                    y = offset[id][1] + j*resolution

                    #Normalisation for input and output
                    if normalise:
                        tmp_data = (tmp_data-input_norm)/input_norm
                        x = x/3000
                        y = y/3000

                    position = [x, y]
                    tmp_data = [tmp_data, position]
                    data.append(tmp_data)

    #Randomly shuffling and splitting data in train, val and test set
    #train test split 0.8 and 0.2 then train val split again 0.8 and 0.2 from train split -> 0.8*0.8 = 0.64 of data
    random.shuffle(data)
    train_data = data[:int(0.64*len(data))]
    val_data = data[int(0.64*len(data)):int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]

    #Writing db to file
    with open(os.path.join(dataroot,'train_data.data'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(dataroot,'val_data.data'), 'wb') as f:
        pickle.dump(test_data, f)
    with open(os.path.join(dataroot,'test_data.data'), 'wb') as f:
        pickle.dump(test_data, f)
