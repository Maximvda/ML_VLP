import os

from dataset.preprocess import preprocess

#Checks if dataset is already pre-processed and returns its path
def setup_database(args, split="train"):
    #Init path dictionary
    path_dict = {}

    #Add the paths of the experimental data to dictionary
    for split in ['train', 'val', 'test', 'train_map', 'test_map', 'grid']:
        path_dict[split] = os.path.join(args.dataroot, 'data_{}_{}_{}.data'.format(args.cell_type, args.normalise,split))

    #Check if all data files are preprocessed if not rerun preprocessing
    for key in path_dict:
        if not os.path.isfile(path_dict[key]):
            print("Pre-processing dataset")
            preprocess(args.dataroot, args.normalise, args.cell_type, args.verbose)

    return path_dict
