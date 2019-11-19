import pickle
import os
import torch

from torch.utils.data import Dataset
#from torchvision import transforms

class data(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[idx][0]
        output = self.data[idx][1]

        input = torch.from_numpy(input)
        input = torch.unsqueeze(input,0)
        output = torch.FloatTensor(output)

        return input, output
