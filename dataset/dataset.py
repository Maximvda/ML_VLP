import pickle
import os
import torch

from torch.utils.data import Dataset
#from torchvision import transforms

class data(Dataset):
    def __init__(self, dataroot, is_train, device):
        path = os.path.join(dataroot, 'train_data.data' if is_train else 'test_data.data')
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

        #self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[idx][0]
        output = self.data[idx][1]


        input = torch.from_numpy(input).to(self.device)
        input = torch.unsqueeze(input,0)
        output = torch.FloatTensor(output).to(self.device)

        return input, output
