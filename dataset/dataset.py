import pickle
import os
import torch

from torch.utils.data import Dataset

#Expansion of the Dataset class to fit our dataset
class data(Dataset):
    def __init__(self, path, split):
        #Open the file with data
        with open(path, 'rb') as f:
            self.data = pickle.load(f)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #Load a specific data item
        input = self.data[idx][0]
        output = self.data[idx][1]

        #Transform to torch tensor and to desired dimension and type
        input = torch.from_numpy(input)
        input = torch.unsqueeze(input,0).type(torch.FloatTensor)
        output = torch.FloatTensor(output)

        return input, output
