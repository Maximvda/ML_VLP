import argparse
import torch
import os

from utils.utils import saveArguments

def str2bool(v):
    if v.lower() in ('true', 'yes', '1', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def str2list(v):
    v = v[1:-1]
    vlist = v.split(',')
    list = []
    for item in vlist:
        list.append(int(item))
    return list

def parse_args():
    parser = argparse.ArgumentParser(description="Visible Light Positioning with Machine Learning")

    parser.add_argument('--is_train', type=str2bool, default='True', help="Set to true if you want to train model or false to evaluate it")
    parser.add_argument('--cuda', type=str2bool, default='True', help="Availability of cuda gpu")

    #Dataset options
    parser.add_argument('--dataroot', default="/home/maxim/Documents/School/Jaar 6/Thesis/Code/dataset/database", required=False, help="Path to the dataset")
    parser.add_argument('--result_root', default="/home/maxim/Documents/School/Jaar 6/Thesis/Code/results", required=False, help="Path to the result directory")
    parser.add_argument('--normalise', type=str2bool, default="True", help="If set to true dataset input and output are normalised")
    parser.add_argument('--TX_density', type=int, default=36, help='Select the led configuartion with certain density.')
    parser.add_argument('--TX_input', type=int, default=1, help="Limit the amount of inputs of the network to only the best received signals.")

    #Model options
    parser.add_argument('--nf', type=int, default=64, help="The numer of features for the model layers")

    #Training options
    parser.add_argument('--batch_size', type=int, default=32, help="The size of the batch for training")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate of the optimiser")
    parser.add_argument('--checkpoint_freq', type=int, default=1, help="Setting checkpoint frequency in number of epochs to store training state")
    parser.add_argument('--visualise', type=str2bool, default='False', help="Visualising the training process with a plot")

    return check_args(parser.parse_args())

def check_args(args):
    if not args.cuda:
        print("Consider running code on gpu enabled device")

    try:
        assert args.TX_input > 0 and args.TX_input <= 36
    except:
        print("TX_input must have a value between one and 36")

    try:
        assert args.batch_size >= 1
    except:
        print("Batch size must be greater or equal to one")

    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)

    args.device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
    saveArguments(args)

    return args
