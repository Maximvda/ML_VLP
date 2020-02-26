import pickle
import os
import torch
import numpy as np
import random

from torch.utils.data import Dataset

def augmentation(input, output):
    prob = random.random()
    if 0.12 < prob < 0.65:
        indices = np.random.choice(np.arange(9), replace=False, size=int(9*prob))
        for ind in indices:
            input[ind] = -1

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
    def __init__(self, path, split, model_type):
        #Open the file with data
        with open(path, 'rb') as f:
            self.data = pickle.load(f)[split]

        #Variables needed to reshape data when cnn model is used
        self.shape = None
        if 'CNN' in model_type:
            size = len(self.data[0][0])
            self.shape = int(np.ceil(np.sqrt(size)))
            self.padding = self.shape**2 - size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #Load a specific data item
        input = self.data[idx][0]
        output = self.data[idx][1]
        input, output = augmentation(input, output)

        #Reshape the input into a square if the cnn model is used
        if not self.shape == None:
            input = np.pad(input,(0,self.padding), mode='constant')
            input = input.reshape((self.shape,self.shape))

        #Transform to torch tensor and to desired dimension and type
        input = torch.FloatTensor(input)
        #input = input.type(torch.FloatTensor)
        output = torch.FloatTensor(output)

        return input, output
