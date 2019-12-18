import os

from torch.utils.data import DataLoader
from dataset.dataset import data
from dataset.preprocess import preprocess
from simulation.simulation import testbed_simulation

#Sets up a data loader for the requested split
def setup_database(args, split="train"):
    #Choose correct file depending on desired split
    file = 'simulation_data' if args.simulate and split == 'train' else 'data'
    file = '_'.join((file, str(args.TX_config),str(args.TX_input), str(args.dynamic))) + '.data'
    path = os.path.join(args.dataroot, file)

    if args.simulate:
        testbed_simulation(args.dataroot)

    #If file not present it may still need to be preprocessed
    if not os.path.isfile(path):
        print("Pre-processing dataset")
        preprocess(args.dataroot, args.TX_config, args.TX_input, args.normalise, args.dynamic)

    #Initialise dataset and setup a data loader
    dataset = data(path, split)
    dataLoader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, num_workers=8)

    return dataLoader
