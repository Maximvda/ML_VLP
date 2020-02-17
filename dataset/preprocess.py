import scipy.io
import numpy as np
import random
import os
import pickle

def LED_to_cell(LED):
    return {0: [0], 1: [0, 1],  2: [1,2],   3: [2,3],   4: [3,4],   5: [4],
            6: [0,6],  7: [0,6,7,1],    8: [1,2,7,8],   9: [2,3,8,9],   10: [3,4,9,10], 11: [4,10],
            12: [6,12], 13: [6,7,12,13], 14:[7,8,13,14], 15:[8,9,14,15], 16:[9,10,15,16], 17:[10,16],
            18: [12,18], 19:[12,13,18,19], 20:[13,14,19,20],21:[14,15,20,21],22:[15,16,21,22],23:[16,22],
            24:[18,24],25:[18,19,24,25],26:[19,20,25,26],27:[20,21,26,27],28:[21,22,27,28],29:[22,28],
            30:[24],31:[24,25],32:[25,26],33:[26,27],34:[27,28],35:[28]}[LED]

def likelyCell(max_LEDs):
    cells = []
    for LED in max_LEDs:
        cells.extend(LED_to_cell(LED))
    return max(set(cells), key = cells.count)

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
    cell = min(distances.argsort()[:4])
    return cell

def readMatFile(file, data, heatmap_data, normalise, dynamic):
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

                cell = getRealUnitCell([pos_x, pos_y])

                #Normalisation for input and output
                if normalise:
                    tmp_data = (tmp_data-input_norm)/input_norm
                    pos_x = pos_x/3000
                    pos_y = pos_y/3000

                #unit_cell = min(tmp_data.argsort()[::-1][:4])
                unit_cell = likelyCell(tmp_data.argsort()[::-1][:4])

                if unit_cell != cell:
                    counter += 1
                position = [pos_x, pos_y, height]
                tmp_data = [tmp_data, position]
                if it == 0 and mat['height']==176:
                    heatmap_data.append(tmp_data)
                    data.append(tmp_data)
                else:
                    data.append(tmp_data)

    print(counter)

def saveData(data, dataroot, TX_config, TX_input, dynamic, simulate=False, heatmap_grid=None):
    #Randomly shuffling and splitting data in train, val and test set
    #train test split 0.8 and 0.2 then train val split again 0.8 and 0.2 from train split -> 0.8*0.8 = 0.64 of data
    random.shuffle(data)
    dict = {'train': data[:int(0.64*len(data))],
            'val':   data[int(0.64*len(data)):int(0.8*len(data))],
            'test':  data[int(0.8*len(data)):],
            'heatmap_grid': heatmap_grid}

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
    data = []; heatmap_data = []
    pth = os.path.join(dataroot,'mat_files')
    files = os.listdir(pth)
    for file in files:
        if 'row' in file:
            print(file)
            readMatFile(os.path.join(pth,file), data, heatmap_data, normalise, dynamic)
    saveData(data, dataroot, TX_config, TX_input, dynamic, heatmap_grid=heatmap_data)

    process_simulation(dataroot, TX_config, TX_input,rng_state, normalise, dynamic)
