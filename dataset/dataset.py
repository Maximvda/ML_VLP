import pickle
import os
import torch
import numpy as np
import random

from torch.utils.data import Dataset

def augmentation(input, output, rotations, blockage):
    prob = random.random()
    if 0.12 < prob < blockage:
        indices = np.random.choice(np.arange(9), replace=False, size=int(9*prob))
        for ind in indices:
            input[ind] = -1
    if rotations:
        prob = random.random()
        #rotate data 25deg
        if 0.25 < prob <= 0.5:
            output = [output[1], -output[0]]
            input = [input[2], input[5], input[8],
                    input[1], input[4], input[7],
                    input[0], input[3], input[6]]
        #rotate data 50 deg
        elif prob <= 0.75:
            output = [-output[0], -output[1]]
            input = [input[8], input[7], input[6],
                    input[5], input[4], input[3],
                    input[2], input[1], input[0]]
        elif prob <= 1:
            output = [-output[1], output[0]]
            input = [input[6], input[3], input[0],
                    input[7], input[4], input[1],
                    input[8], input[5], input[2]]
    return input, output



#Expansion of the Dataset class to fit our dataset
class data(Dataset):
    def __init__(self, path, split, rotations=False, blockage=False):
        self.cel = None
        if 'map' in split:
            self.rotations = False
            #Open the file with data
            with open(path, 'rb') as f:
                data = pickle.load(f)
                indices = data[split]
                if 'map_grid' in split:
                    train = data['train']
                    test = data['test']
                    train = train[indices['train'][:][0]]
                    test = test[indices['test'][:][0]]
                    self.data = np.concatenate((train,test))
                    self.cel = np.concatenate((indices['train'][:][1],indices['test'][:][1]))
                elif 'map_7' in split:
                    train = data['train']
                    self.data = train[indices]
                else:
                    test = data['test']
                    self.data = test[indices]

        else:
            #Open the file with data
            with open(path, 'rb') as f:
                self.data = pickle.load(f)[split]

                self.rotations = rotations
        self.blockage = blockage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #Load a specific data item
        input = self.data[idx][0]
        if self.cel == None:
            output = self.data[idx][1]
        else:
            output = [self.data[idx][1], self.cel[idx]]

        input, output = augmentation(input, output, self.rotations, self.blockage)

        #Transform to torch tensor and to desired dimension and type
        input = torch.FloatTensor(input)
        #input = input.type(torch.FloatTensor)
        output = torch.FloatTensor(output)

        return input, output
