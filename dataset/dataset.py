import pickle
import torch
import numpy as np
import os
import random

from torch.utils.data import Dataset
from utils.config import cell_rotation

#Expansion of the Dataset class to fit our dataset
class Data(Dataset):
    def __init__(self, path, blockage, rotations, cell_type, output_nf, real_block=False):
        #init variables
        self.blockage = blockage
        self.rotations = rotations
        self.cell_type = cell_type
        self.real_block = real_block
        #Open the file with data
        with open(path, 'rb') as f:
            data = pickle.load(f)

        #If height is not predicted remove the z coordinate from the data samples
        if output_nf == 2:
            set_output(data[:,1])

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        cel = None
        if type(data) is dict:
            cel = data["cel"]
            data = data["data"]
        #Load a specific data item
        input = data[0]
        output = data[1]

        input, output = augment_data(input, output, self.rotations, self.blockage, self.real_block, self.cell_type)

        #Transform to torch tensor and to desired dimension and type
        input = torch.FloatTensor(input)
        output = torch.FloatTensor(output)

        if cel != None:
            output = [output, cel]

        return input, output

    def get_data(self):
        return self.data

#Remove the z coordinates from the output
def set_output(position_data):
    for i in range(len(position_data)):
        position_data[i] = position_data[i][0:2]

#Augment the training data by performing rotations and setting the blockage
def augment_data(input, output, rotations, blockage, real_block, cell_type):
    #Get the indices of the TX that are blocked
    #In the more realistic setting always a random amount of TXs are blocked
    #Otherwise it is always a fixed percentage of blocked TXs
    if real_block:
        prob = random.random()
        prob = min(prob, 0.7)
        indices = np.random.choice(np.arange(len(input)), replace=False, size=int(prob*len(input)))
    else:
        indices = np.random.choice(np.arange(len(input)), replace=False, size=int(blockage*len(input)))

    for ind in indices:
        input[ind] = -1

    if rotations:
        prob = random.random()
        #rotate data 90 deg
        if 0.25 < prob <= 0.5:
            input, output_p = cell_rotation(input, output, cell_type)
        #rotate data 180 deg
        elif 0.5 < prob <= 0.75:
            input, output_p = cell_rotation(input, output, cell_type)
            input, output_p = cell_rotation(input, output, cell_type)
        #rotate data 270 deg
        elif 0.75 < prob <= 1:
            input, output_p = cell_rotation(input, output, cell_type)
            input, output_p = cell_rotation(input, output, cell_type)
            input, output_p = cell_rotation(input, output, cell_type)
    if len(output) == 3:
        output = [output_p[0], output_p[1], output[2]]
    else:
        output = output_p
    return input, output
