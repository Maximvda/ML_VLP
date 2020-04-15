import scipy.io
import numpy as np
import random
import os
import pickle

from utils.utils import printProgBar
from utils.utils import convolution2d
from utils.config import get_cell_mask
from utils.config import get_cel_center_position
from utils.utils import getCelPosition

#Preprocess the Matlab database and store necessary variables into files for training
def preprocess(dataroot, normalise, cell_type, verbose):
    #init variables needed for unit cell processing
    mask = get_cell_mask()[cell_type]; center_pos = get_cel_center_position()[cell_type]
    cells = get_cells(mask);
    #take 16% of all cells as cells for validation split and 20% for test split
    val_cells = np.random.choice(cells, int(np.floor(len(cells)*0.16)), replace=False).tolist()
    test_cells = np.random.choice([cell for cell in cells  if (not cell in val_cells)], int(np.floor(len(cells)*0.2)), replace=False).tolist()
    train_cells = [cell for cell in cells if (not cell in val_cells and not cell in test_cells)]
    cell_dict = {'all': cells,
                'train': train_cells,
                'val': val_cells,
                'test': test_cells,
                'train_map': np.random.choice(train_cells, 1).tolist()[0],
                'test_map': np.random.choice(test_cells, 1).tolist()[0]}

    #Init data dictionary used to hold all data
    data_dict = {'train': [],
                'val': [],
                'test': [],
                'train_map': [],
                'test_map': [],
                'grid': []}


    #Initialise variables and directory of matfiles
    pth = os.path.join(dataroot,'mat_files')
    #List mat files and select only files with row in name
    files = os.listdir(pth)
    files = [file for file in files if 'row' in file]
    counter = 0
    #Read all the matlab files
    for file in files:
        counter += 1
        if verbose:
            print(printProgBar(counter,len(files)), end='\r')
        read_mat_file(os.path.join(pth,file), normalise, mask, center_pos, cell_dict, data_dict)
    print("")
    save_data(data_dict, dataroot, normalise, cell_type)

#Read a matlab file and return needed variables to store data
def read_mat_file(file, normalise, mask, center_pos, cell_dict, data_dict):
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

    #Iterate over each measurement (each RX, each x position and each iteration)
    for id in rx_id:
        for x in range(0,channel_data.shape[3]):
            for it in range(0,no_it):
                #Select 1 measurement
                tmp_data = channel_data[:,id,it,x]

                #Calculate position of the RX for this measurement
                #Also normalise position by dividing with 3000mm max size of test grid
                y = int(file.split("_")[-1][:-4])
                pos_x = (offset[id][0] + y*resolution)
                pos_y = (offset[id][1] + x*resolution)

                #Normalisation for input and output
                if normalise:
                    tmp_data = (tmp_data-input_norm)/input_norm

                bool = (it == 0 and (mat['height']==176)[0][0])
                process_measurement(tmp_data, height, [pos_x, pos_y], mask, center_pos, cell_dict, data_dict, bool)

#Process one measurement and return all data that is used for training of unit cell approach
def process_measurement(measurement, height, measurement_pos, mask, center_pos, cell_dict, data_dict, bool):
    #Max distance from center of cell
    max = 550+np.sqrt(center_pos[0]**2+center_pos[1]**2)
    #Get the most likely cell where measurement was taken
    cel = convolution2d(measurement, mask)

    #Iterate over all the possible unit cells for particular cell type
    for i in cell_dict['all']:
        #Get cell position and translate it to center of cel position
        pos = getCelPosition(i)
        pos = [pos[0]+center_pos[0], pos[1]+center_pos[1]]
        dist = np.sqrt((pos[0]-measurement_pos[0])**2+(pos[1]-measurement_pos[1])**2)

        if dist <= max:
            data = data_from_cel(measurement, i)
            rel_pos = [(measurement_pos[0]-pos[0])/max, (measurement_pos[1]-pos[1])/max, height]
            data = [data, rel_pos]
            #Correctly assign the data to correct splits
            if i in cell_dict['train']:
                data_dict['train'].append(data)
            elif i in cell_dict['val']:
                data_dict['val'].append(data)
            elif i in cell_dict['test']:
                data_dict['test'].append(data)

            if bool:
                if i == cell_dict['train_map']:
                    data_dict['train_map'].append(data)
                if i == cell_dict['test_map']:
                    data_dict['test_map'].append(data)
                if i == cel:
                    data_dict['grid'].append({'data': data, 'cel': cel})



#Save pre-processed data to files
def save_data(data_dict, dataroot, normalise, cell_type):
    print("Saving data")
    #Write data for each split to separate file
    for key in data_dict:
        with open(os.path.join(dataroot,'data_{}_{}_{}.data'.format(cell_type,normalise,key)), 'wb') as f:
            pickle.dump(data_dict[key], f)

#This function returns all the possible cells for a particular unit cell
def get_cells(mask):
    cell_list = []
    #Iterate over rows
    for i in range(0,6):
        #iterate over columns
        for j in range(0,6):
            index_list = []
            for led in mask:
                col = j+led['col']; row = i+led['row']
                index_list.append(col+6*row)
                if (col >= 6) or (row >= 6):
                    index_list = []
                    break
            if len(index_list) > 0:
                cell_list.append(min(index_list))
    return cell_list


#Retrieve the data for a particular given cel
def data_from_cel(data, cel):
    return [data[cel-7], data[cel-6], data[cel-5],
            data[cel-1], data[cel], data[cel+1],
            data[cel+5], data[cel+6], data[cel+7]]
