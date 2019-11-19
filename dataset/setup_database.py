import os

from torch.utils.data import DataLoader
from dataset.dataset import data
from dataset.preprocess import preprocess

def setup_database(args, split="train"):
    file = 'train_data.data' if split == 'train' else 'val_data.data' if split == 'val' else 'test_data.data'
    path = os.path.join(args.dataroot,file)
    if not os.path.isfile(path):
        print("Pre-processing dataset")
        preprocess(args.dataroot, args.normalise)

    dataset = data(path)
    dataLoader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, num_workers=2)

    return dataLoader
