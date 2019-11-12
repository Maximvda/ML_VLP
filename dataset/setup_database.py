import os

from torch.utils.data import DataLoader
from dataset.dataset import data
from dataset.preprocess import preprocess

def setup_database(args):
    file = 'train_data.data' if args.is_train else 'test_data.data'
    if not os.path.isfile(os.path.join(args.dataroot,file)):
        print("Pre-processing dataset")
        preprocess(args.dataroot)

    dataset = data(args.dataroot, args.is_train, args.device)
    dataLoader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, num_workers=2)

    return dataLoader
