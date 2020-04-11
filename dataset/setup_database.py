import os

from dataset.preprocess import preprocess
from dataset.simulation import testbed_simulation

#Checks if dataset is already pre-processed and returns its path
def setup_database(args, split="train"):
    #Init path dictionary
    path_dict = {}

    #Add the paths of the experimental data to dictionary
    for split in ['train', 'val', 'test', 'heatmap']:
        path_dict[split] = os.path.join(args.dataroot, 'data_{}_{}.data'.format(args.normalise,split))

    #Generates the simulation data if it's not yet generated
    #Add paths to dictionary if simulation is generated
    if args.simulate:
        testbed_simulation(args.dataroot, args.verbose)
        path_dict['train'] = os.path.join(args.dataroot, 'simulation_data_{}_train.data'.format(args.normalise))
        path_dict['sim_val'] = os.path.join(args.dataroot, 'simulation_data_{}_val.data'.format(args.normalise))

    #Check if all data files are preprocessed if not rerun preprocessing
    for key in path_dict:
        if not os.path.isfile(path_dict[key]):
            print("Pre-processing dataset")
            preprocess(args.dataroot, args.normalise, args.verbose)

    return path_dict
