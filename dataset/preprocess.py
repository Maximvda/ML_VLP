import scipy.io
import numpy as np
import random
import os
import pickle

from simulation.simulation import testbed_simulation

#Sets all RX signals from TX that are not in the chosen configuartion to 0
#The different TX configuartions can be found on Github page
def setConfiguartion(channel_data, TX_config, dynamic):
    all_TX = [i for i in range(0,36)]
    #list TXs needed for specific configuartion
    list_dict = {
    1: all_TX,
    2: [0,2,4,12,14,16,24,26,28],
    3: [7,10,25,28],
    4: [0,2,3,5,12,14,15,17,18,20,21,23,30,32,33,35],
    5: [0,5,14,15,20,21,30,35],
    6: [14,15,20,21]}
    #Remove needed TX from all TX to get all TX which need to be set to 0
    #Select appropriate configuartion acoording to TX_config and iterate over TX
    #Set all these TX to 0
    list = [] if TX_config == 1 else [id for id in all_TX if id not in list_dict[TX_config]]
    if dynamic:
        return np.delete(channel_data, list, 0), len(list_dict[TX_config])
    else:
        for it in list:
            channel_data[it] = 0
        return channel_data, len(list_dict[TX_config])


def readMatFile(file, data, TX_config, TX_input, normalise, rng_state, dynamic):
    #Load matlab file
    mat = scipy.io.loadmat(file)
    #Initialising some variables
    rx_id = mat['rx_id'][0]-1
    no_it = mat['no_it'][0][0]
    offset = mat['offset']
    resolution = mat['resolution'][0][0]

    #Choose which type of data to use swing or channel_data
    #Swing is a measure for RSS
    swing = 1
    channel_data = mat['swing'] if swing else np.mean(mat['channel_data'],axis=1)
    input_norm = np.max(channel_data)/2

    #Set the TX which are not used for the desired density to 0
    channel_data, lenData = setConfiguartion(channel_data, TX_config, dynamic)
    TX_input = lenData if TX_input >= lenData else TX_input
    shape = int(np.ceil(np.sqrt(channel_data.shape[0]))) if dynamic else 6
    #Shuffles the received signals in such a way that the LEDs
    #are not in the correct position as they are in the testbed.
    #So the led on position 2,4 of the 6x6 grid can be replaced to position 5,1 in the 6x6 matrix
    #This decouples the LEDs measurement from its spatial position in the testbed
    np.random.set_state(rng_state)
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
    #Iterate over each measurement (each RX, each x and y position and each iteration)
    for id in rx_id:
        for x in range(0,channel_data.shape[3]):
            if len(channel_data.shape) == 4:
                for it in range(0,no_it):
                    #Select 1 measurement
                    tmp_data = channel_data[:,id,it,x]
                    #Sort measurement from high to low and select TX_input highest element
                    high_el = np.sort(tmp_data)[::-1][TX_input-1]
                    #Set all values lower then the high_el to 0 and reshape in 6x6 grid for convolution
                    tmp_data = np.array([0 if el < high_el else el for el in tmp_data]).reshape((shape,shape))

                    #Calculate position of the RX for this measurement
                    y = int(file.split("_")[-1][:-4])
                    pos_x = offset[id][0] + y*resolution
                    pos_y = offset[id][1] + x*resolution

                    #Normalisation for input and output
                    if normalise:
                        tmp_data = (tmp_data-input_norm)/input_norm
                        pos_x = pos_x/3000
                        pos_y = pos_y/3000

                    position = [pos_x, pos_y]
                    tmp_data = [tmp_data, position, mat['height']]
                    data.append(tmp_data)
            else:
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
                        tmp_data = [tmp_data, position, mat['height']]
                        data.append(tmp_data)

def process_simulation(dataroot, TX_config, TX_input,rng_state, normalise, dynamic):
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
        channel_data, iets = setConfiguartion(channel_data, TX_config, dynamic)
        TX_input = lenData if TX_input >= lenData else TX_input
        shape = int(np.ceil(np.sqrt(channel_data.shape[0]))) if dynamic else 6
        np.random.set_state(rng_state)
        np.random.shuffle(channel_data)
        data = []
        for RX in pos_RX:
            tmp_data = channel_data[:, int(RX[0]/10), int(RX[1]/10)]
            #Sort measurement from high to low and select TX_input highest element
            high_el = np.sort(tmp_data)[::-1][TX_input-1]
            #Set all values lower then the high_el to 0 and reshape in 6x6 grid for convolution
            tmp_data = np.array([0 if el < high_el else el for el in tmp_data]).reshape((shape,shape))
            tmp_data = (tmp_data-input_norm)/input_norm
            #Still have to implement multiple heights for simulation
            data.append([tmp_data, [RX[0]/3000, RX[1]/3000], 1870])

        saveData(data, dataroot, TX_config, TX_input, dynamic, simulate=True)
    else:
        print("Simulation data doesn't exist")

def saveData(data, dataroot, TX_config, TX_input, dynamic, simulate=False):
    #Randomly shuffling and splitting data in train, val and test set
    #train test split 0.8 and 0.2 then train val split again 0.8 and 0.2 from train split -> 0.8*0.8 = 0.64 of data
    random.shuffle(data)
    dict = {'train': data[:int(0.64*len(data))],
            'val':   data[int(0.64*len(data)):int(0.8*len(data))],
            'test':  data[int(0.8*len(data)):]}

    #Writing db to file
    pretension = 'simulation_data_' if simulate else 'data_'
    extension = '_'.join((str(TX_config), str(TX_input), str(dynamic))) + '.data'
    with open(os.path.join(dataroot,pretension + extension), 'wb') as f:
        pickle.dump(dict, f)

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

#Preprocess the Matlab database and store necessary variables into files for training
def preprocess(dataroot, TX_config, TX_input, normalise, dynamic):
    data = []
    pth = os.path.join(dataroot,'mat_files')
    files = os.listdir(pth)
    rng_state = np.random.get_state()
    for file in files:
        if 'row' in file:
            print(file)
            readMatFile(os.path.join(pth,file), data, TX_config, TX_input, normalise, rng_state, dynamic)
    saveData(data, dataroot, TX_config, TX_input, dynamic)

    process_simulation(dataroot, TX_config, TX_input,rng_state, normalise, dynamic)
