import pickle
import torch
import numpy as np
import os

from torch.utils.data import Dataset

from utils.config import get_configuration_dict

#Expansion of the Dataset class to fit our dataset
class Data(Dataset):
    def __init__(self, path, TX_config, TX_input, blockage, output_nf):
        self.blockage = blockage
        #Open the file with data
        with open(path, 'rb') as f:
            data = pickle.load(f)

        #Load index_map array
        index_map = np.loadtxt(os.path.join("/".join(path.split("/")[0:-1]),'index_map.txt'))

        set_data_configuration(data[:,0], index_map, TX_config, TX_input, blockage)

        #If height is not predicted remove the z coordinate from the data samples
        if output_nf == 2:
            set_output(data[:,1])

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #Load a specific data item
        input = self.data[idx][0]
        output = self.data[idx][1]

        #Get indices of blockage
        indices = np.random.choice(np.arange(len(input)), replace=False, size=int(self.blockage*len(input)))
        for ind in indices:
            input[ind] = -1

        #Transform to torch tensor and to desired dimension and type
        input = torch.from_numpy(input).type(torch.FloatTensor)
        #input = torch.unsqueeze(input,0).type(torch.FloatTensor)
        output = torch.FloatTensor(output)

        return input, output


def set_output(position_data):
    for i in range(len(position_data)):
        position_data[i] = position_data[i][0:2]

#Deletes all RX signals from TX that are not in the chosen configuartion
#Sets all RX signals that are not in top heighest TX_input to zero
def set_data_configuration(channel_data, index_map, TX_config, TX_input, blockage):
    all_TX = [i for i in range(0,36)]
    #list TXs needed for specific configuartion
    list_dict = get_configuration_dict()
    #Remove needed TX from all TX to get all TX which need to be set to 0
    #Select appropriate configuartion acoording to TX_config and iterate over TX
    list = [] if TX_config == 1 else [id for id in all_TX if id not in list_dict[TX_config]]
    #retrieve the correct indexes as the data has been shuffled during preprocessing
    indexes = [np.where(index_map == i)[0][0] for i in list]

    #Set TX_input to length of configuration if its higher then number of TX in configuration
    len_conf = len(list_dict[TX_config])
    TX_input = len_conf if (TX_input >= len_conf) else TX_input

    for i in range(len(channel_data)):
        channel_data[i] = np.delete(channel_data[i], indexes, 0)

        #Sort measurement from high to low and select TX_input highest element
        high_el = np.sort(channel_data[i])[::-1][TX_input-1]
        #Set all values lower then the high_el to -1 (As if nothing was received)
        channel_data[i] = np.array([-1 if el < high_el else el for el in channel_data[i]])
