import scipy.io
import numpy as np
import random
import os
import pickle

from utils.utils import getDensity
#Preprocess the matlab and store necessary variables into files for training
def preprocess(dataroot, TX_density, TX_input, normalise=False):
    #Load matlab file
    mat = scipy.io.loadmat(os.path.join(dataroot,'dataset.mat'))
    #Initialising some variables
    rx_id = mat['rx_id'][0]-1
    no_it = mat['no_it'][0][0]
    offset = mat['offset']
    resolution = mat['resolution'][0][0]

    #choose which type of data to use swing or channel_data
    swing = 1
    channel_data = mat['swing'] if swing else np.mean(mat['channel_data'],axis=1)
    input_norm = np.max(channel_data)/2

    #Set the LEDs which are not used for the desired density to 0
    for led in getDensity(TX_density):
        break if led is None
        channel_data[led,:,:,:,:] = 0

    print(channel_data[:,0,0,0,0])
    print(x)


    #Shuffles the received signals in such a way that the LEDs
    #are not in the correct position as they are in the testbed.
    #So the led on position 2,4 of the 6x6 grid can be replaced to position 5,1 in the 6x6 matrix
    #This decouples the LEDs measurement from its spatial position in the testbed
    np.random.shuffle(channel_data)

    #pos_x_mm = mat['pos_x_mm'][0]
    #pos_y_mm = mat['pos_y_mm'][0]
    #arr = []
    #for id in rx_id:
    #    x = (offset[id][0] + pos_x_mm)/3000
    #    y = (offset[id][1] + pos_y_mm)/3000
    #    tmp = (channel_data-input_norm)/input_norm
    #    arr.append([[tmp[:,id,it,i,j], [x[i], y[j]]] for i in range(0,len(x)-2) for j in range(0,len(y)) for it in range(0,no_it)][0])


    #New data variable, list of all data points
    data = []
    #Iterate over each measurement (each RX, each x and y position and each iteration)
    for id in rx_id:
        for x in range(0,channel_data.shape[3]):
            for y in range(0,channel_data.shape[4]):
                for it in range(0,no_it):
                    #Select 1 measurement
                    tmp_data = channel_data[:,id,it,x,y]
                    #Sort measurement from high to low and select TX_input highest element
                    high_el = np.sort(tmp_data)[::-1][TX_input-1]
                    #Set all values lower then the high_el to 0 and reshape in 6x6 grid for convolution
                    tmp_data = np.array([0 if el < high_el else el for el in tmp_data]).reshape((6,6))

                    #Calculate position of the RX for this measurement
                    pos_x = offset[id][0] + x*resolution
                    pos_y = offset[id][1] + y*resolution

                    #Normalisation for input and output
                    if normalise:
                        tmp_data = (tmp_data-input_norm)/input_norm
                        pos_x = pos_x/3000
                        pos_y = pos_y/3000

                    position = [pos_x, pos_y]
                    tmp_data = [tmp_data, position]
                    data.append(tmp_data)

    #Randomly shuffling and splitting data in train, val and test set
    #train test split 0.8 and 0.2 then train val split again 0.8 and 0.2 from train split -> 0.8*0.8 = 0.64 of data
    random.shuffle(data)
    train_data = data[:int(0.64*len(data))]
    val_data = data[int(0.64*len(data)):int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]

    #Writing db to file
    extension = '_'.join((str(TX_density),str(TX_input))) + '.data'
    with open(os.path.join(dataroot,'train_data_' + extension), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(dataroot,'val_data_' + extension), 'wb') as f:
        pickle.dump(test_data, f)
    with open(os.path.join(dataroot,'test_data_' + extension), 'wb') as f:
        pickle.dump(test_data, f)
