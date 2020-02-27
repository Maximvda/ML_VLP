import scipy.io
import numpy as np
import random
import os
import pickle

from utils.utils import convolution2d

def getCelPosition(cel):
    return {7:[720, 725], 8:[1230, 670], 9:[1735, 670], 10:[2225, 725],
            13:[730, 1170], 14:[1240, 1170], 15:[1745, 1170], 16:[2245, 1170],
            19:[730, 1670], 20:[1240, 1670], 21:[1745, 1670], 22:[2245, 1670],
            25:[720, 2225], 26:[1235, 2170], 27:[1720, 2170], 28:[2220, 2225]}[cel]

def celToData(data, cel):
    return [data[cel-7], data[cel-6], data[cel-5],
            data[cel-1], data[cel], data[cel+1],
            data[cel+5], data[cel+6], data[cel+7]]

def getCelData(measurement, position, train, test, map_grid, map_7, map_25):
    for i in [7,8,9,10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28]:
        pos = getCelPosition(i)
        cel = getRealUnitCell(position)
        dist = np.sqrt((pos[0]-position[0])**2+(pos[1]-position[1])**2)
        if dist <=750:
            cell_measurement = celToData(measurement,i)
            rel_pos = [(position[0]-pos[0])/750, (position[1]-pos[1])/750]
            data = [cell_measurement, rel_pos]
            if i in [19,13,25,26,27,20,14,24]:
                test.append(data)
                if i == 25:
                    map_25.append(len(test)-1)
                if i == cel:
                    map_grid['test'].append(len(test)-1)
            else:
                train.append([cell_measurement, rel_pos])
                if i == map_7:
                    map_7.append(len(train)-1)
                if i == cel:
                    map_grid['train'].append(len(train)-1)


def getRealUnitCell(pos):
    distances = []
    pos_LEDs = [[230, 170], [730, 175], [1240, 170], [1740, 170], [2240, 170], [2740, 170],
            [230, 670], [720, 725], [1230, 670], [1735, 670], [2225, 725], [2725, 670],
            [230, 1170], [730, 1170], [1240, 1170], [1745, 1170], [2245, 1170], [2735, 1170],
            [230, 1670], [730, 1670], [1240, 1670], [1745, 1670], [2245, 1670], [2735, 1670],
            [230, 2170], [720, 2225], [1235, 2170], [1720, 2170], [2220, 2225], [2710, 2170],
            [215, 2670], [715, 2670], [1245, 2670], [1730, 2670], [2245, 2670], [2730, 2670]]
    for i in range(0,len(pos_LEDs)):
        LED = pos_LEDs[i]
        dist = np.sqrt((LED[0]-pos[0])**2+(LED[1]-pos[1])**2)
        distances.append(dist)
    distances = np.asarray(distances)
    cell = convolution2d(distances, max=False)
    return cell

def readMatFile(file, train, test, map_grid, map_7, map_25, normalise):
    #Load matlab file
    mat = scipy.io.loadmat(file)
    #Initialising some variables
    rx_id = mat['rx_id'][0]-1
    no_it = mat['no_it'][0][0]
    offset = mat['offset']
    resolution = mat['resolution'][0][0]
    height = mat['height']/200 if normalise else mat['height']

    #Choose which type of data to use swing or channel_data
    #Swing is a measure for RSS
    swing = 1
    channel_data = mat['swing'] if swing else np.mean(mat['channel_data'],axis=1)
    input_norm = np.max(channel_data)/2
    counter = 0

    #New data variable, list of all data points
    #Iterate over each measurement (each RX, each x and y position and each iteration)
    for id in rx_id:
        for x in range(0,channel_data.shape[3]):
            for it in range(0,no_it):
                #Select 1 measurement
                tmp_data = channel_data[:,id,it,x]

                #Calculate position of the RX for this measurement
                y = int(file.split("_")[-1][:-4])
                pos_x = offset[id][0] + y*resolution
                pos_y = offset[id][1] + x*resolution

                #Normalisation for input and output
                if normalise:
                    tmp_data = (tmp_data-input_norm)/input_norm

                getCelData(tmp_data, [pos_x, pos_y], train, test, map_grid, map_7, map_25)


def saveData(train, test, map_grid, map_7, map_25, dataroot, normalise):
    #Randomly shuffling and splitting data in train, val and test set
    #train test split 0.8 and 0.2 then train val split again 0.8 and 0.2 from train split -> 0.8*0.8 = 0.64 of data
    random.shuffle(train)
    dict = {'train': train[:int(0.8*len(train))],
            'val':   train[int(0.8*len(train)):],
            'test':  test,
            'map_grid': map_grid,
            'map_7': map_7,
            'map_25': map_25}

    #Writing db to file
    extension = str(normalise) + '.data'
    with open(os.path.join(dataroot,'data_' + extension), 'wb') as f:
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
def preprocess(dataroot, normalise):
    train = []; test = []; map_grid = {}; map_7 = []; map_25 = []
    map_grid['train'] = []; map_grid['test'] = []
    pth = os.path.join(dataroot,'mat_files')
    files = os.listdir(pth)
    for file in files:
        if 'row' in file:
            print(file)
            readMatFile(os.path.join(pth,file), train, test, map_grid,map_7,map_25, normalise)
    saveData(train, test, map_grid, map_7, map_25, dataroot, normalise)
