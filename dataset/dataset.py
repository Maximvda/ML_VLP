import pickle
import os
import torch
import numpy as np

from torch.utils.data import Dataset

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

        #Reshape the input into a square if the cnn model is used
        if not self.shape == None:
            input = np.pad(input,(0,self.padding), mode='constant')
            input = input.reshape((self.shape,self.shape))

        #Transform to torch tensor and to desired dimension and type
        input = torch.FloatTensor(input)
        #input = input.type(torch.FloatTensor)
        output = torch.FloatTensor(output)

        return input, output
