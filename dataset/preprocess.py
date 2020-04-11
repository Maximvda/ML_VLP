import scipy.io
import numpy as np
import random
import os
import pickle

from utils.utils import printProgBar

#Preprocess the Matlab database and store necessary variables into files for training
def preprocess(dataroot, normalise, verbose):
    #Initialise variables and directory of matfiles
    data = np.empty((0,2)); heatmap_data = np.empty((0,2)); counter = 0
    pth = os.path.join(dataroot,'mat_files')
    #List mat files and select only files with row in name
    files = os.listdir(pth)
    files = [file for file in files if 'row' in file]

    #Get the random state such that data points are shuffled similar over different mat files
    rng_state = np.random.get_state()
    index_map = np.arange(36)
    np.random.shuffle(index_map)
    np.savetxt(os.path.join(dataroot,'index_map.txt'), index_map, delimiter=",")
    np.random.set_state(rng_state)

    #Read all the matlab files
    for file in files:
        counter += 1
        if verbose:
            print(printProgBar(counter,len(files)), end='\r')
        tmp_data, tmp_heatmap_data = read_mat_file(os.path.join(pth,file), normalise, rng_state)
        heatmap_data = np.append(heatmap_data,tmp_heatmap_data, axis=0)
        data = np.append(data,tmp_data, axis=0)
    print("")
    save_data(data, dataroot, normalise, heatmap_grid=heatmap_data)

    process_simulation(dataroot, rng_state, normalise, verbose)

#Read a matlab file and return needed variables to store data
def read_mat_file(file, normalise, rng_state):
    #Load matlab file
    mat = scipy.io.loadmat(file)
    #Initialising some variables
    data = np.empty((0,2)); heatmap_data = np.empty((0,2));
    rx_id = mat['rx_id'][0]-1
    no_it = mat['no_it'][0][0]
    offset = mat['offset']
    resolution = mat['resolution'][0][0]
    #Normalise the height by dividing with 200cm as this is maximum ceiling height
    height = (mat['height']/200)[0][0]

    #Swing is a measure for RSS
    channel_data = mat['swing']
    input_norm = np.max(channel_data)/2

    #Shuffles the received signals in such a way that the LEDs
    #are not in the correct position as they are in the testbed.
    #So the led on position 2,4 of the 6x6 grid can be replaced to position 5,1 in the 6x6 matrix
    #This decouples the LEDs measurement from their spatial position in the testbed
    np.random.set_state(rng_state)
    np.random.shuffle(channel_data)

    #Iterate over each measurement (each RX, each x position and each iteration)
    for id in rx_id:
        for x in range(0,channel_data.shape[3]):
            for it in range(0,no_it):
                #Select 1 measurement
                tmp_data = channel_data[:,id,it,x]

                #Calculate position of the RX for this measurement
                #Also normalise position by dividing with 3000mm max size of test grid
                y = int(file.split("_")[-1][:-4])
                pos_x = (offset[id][0] + y*resolution)/3000
                pos_y = (offset[id][1] + x*resolution)/3000

                #Normalisation for input and output
                if normalise:
                    tmp_data = (tmp_data-input_norm)/input_norm

                position = np.array([pos_x, pos_y, height])
                tmp_data = np.expand_dims(np.array((tmp_data, position)), axis=0)

                #If it is data sample from first iteration and from the first height
                #Then add it to the data of heatmap
                if it == 0 and mat['height']==176:
                    heatmap_data = np.append(heatmap_data,tmp_data, axis=0)
                    data = np.append(data,tmp_data, axis=0)
                else:
                    data = np.append(data,tmp_data, axis=0)
    return data, heatmap_data

#Pre-process the simulation data similar to real data -> see comments in read_mat_file
def process_simulation(dataroot,rng_state, normalise, verbose):
    file = os.path.join(dataroot,'simulationdata.data')
    if os.path.exists(file):
        print("Preprocessing simulation data")
        #Open the file with data
        with open(file, 'rb') as f:
            dict = pickle.load(f)
        channel_data = dict['channel_data']
        pos_RX = dict['pos_RX']
        pos_TX = dict['pos_TX']

        input_norm = np.max(channel_data)/2

        np.random.set_state(rng_state)
        np.random.shuffle(channel_data)
        data = []; counter = 0
        for it in range(0,3):
            for RX in pos_RX:
                counter += 1
                if verbose:
                    print(printProgBar(counter,3*len(pos_RX)), end='\r')
                for i in range(len(pos_TX)):
                    tmp_data = channel_data[:,it, int(RX[0]/10), int(RX[1]/10), i]
                    if normalise:
                        tmp_data = (tmp_data-input_norm)/input_norm

                    position = np.array([RX[0]/3000, RX[1]/3000, pos_TX[i][0][2]/2000])
                    tmp_data = np.expand_dims(np.array((tmp_data, position)), axis=0)

                    data.append(tmp_data)

        data = np.reshape(data, (-1,2))
        print("")
        save_data(data, dataroot,normalise, simulate=True)
    else:
        print("Simulation data not generated yet")

#Save pre-processed data to files
def save_data(data, dataroot, normalise, simulate=False, heatmap_grid=None):
    #Randomly shuffling and splitting data in train, val and test set
    #train test split 0.8 and 0.2 then train val split again 0.8 and 0.2 from train split -> 0.8*0.8 = 0.64 of data
    np.random.shuffle(data)
    if simulate:
        #Writing db to file
        with open(os.path.join(dataroot,'simulation_data_{}_{}.data'.format(normalise, 'train')), 'wb') as f:
            pickle.dump(data[:int(0.8*len(data))], f)
        with open(os.path.join(dataroot,'simulation_data_{}_{}.data'.format(normalise, 'val')), 'wb') as f:
            pickle.dump(data[int(0.8*len(data)):], f)
    else:
        #Writing train data to file
        with open(os.path.join(dataroot,'data_{}_{}.data'.format(normalise, 'train')), 'wb') as f:
            pickle.dump(data[:int(0.64*len(data))], f)
        #Writing validation data to file
        with open(os.path.join(dataroot,'data_{}_{}.data'.format(normalise, 'val')), 'wb') as f:
            pickle.dump(data[int(0.64*len(data)):int(0.8*len(data))], f)
        #Writing test data to file
        with open(os.path.join(dataroot,'data_{}_{}.data'.format(normalise, 'test')), 'wb') as f:
            pickle.dump(data[int(0.8*len(data)):], f)
        #Writing heatmap data to file
        with open(os.path.join(dataroot,'data_{}_{}.data'.format(normalise, 'heatmap')), 'wb') as f:
            pickle.dump(heatmap_grid, f)


#Get offset of the first LED (X = 230mm and y =170mm)
#random shuffle needs to be off and run on all data
def get_offsets(data):
    max = 0
    for item in data:
        val = item[0][0][0]
        if val > max:
            max = val
            best = item

    print(best)
