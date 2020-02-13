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
    parser.add_argument('--gpu_number', type=int, default=None, help="Number of gpu that will be used")
    parser.add_argument('--experiment', type=int, default=None, help='Select a certain experiment to run or leave default None to train a single model.')
    parser.add_argument('--simulate', type=str2bool, default="False")

    #Dataset options
    parser.add_argument('--dataroot', default=None, required=False, help="Path to the dataset")
    parser.add_argument('--result_root', default=None, required=False, help="Path to the result directory")
    parser.add_argument('--normalise', type=str2bool, default="True", help="If set to true dataset input and output are normalised")
    parser.add_argument('--TX_config', type=int, default=1, help='Select the TX configuartion with a certain density. Configurations can be found at https://github.com/Maximvda/ML_VLP')
    parser.add_argument('--TX_input', type=int, default=36, help="Limit the amount of inputs of the network to only the best received signals")
    parser.add_argument('--dynamic', type=str2bool, default="False", help="Sizes the input data according to the number of TX if set to true")

    #Model options
    parser.add_argument('--model_type', type=str, default='FC', choices=['CNN', 'FC, FC_expand'])
    parser.add_argument('--nf', type=int, default=64, help="The numer of features for the model layers")
    parser.add_argument('--extra_layers', type=int, default=4, help="The number of extra layers in the model such that it has more parameters.")

    #Training options
    parser.add_argument('--batch_size', type=int, default=32, help="The size of the batch for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate of the optimiser")
    parser.add_argument('--checkpoint_freq', type=int, default=1, help="Setting checkpoint frequency in number of epochs to store training state")

    #Evaluation options
    parser.add_argument('--visualise', type=str2bool, default='False', help="Visualising the training process with a plot")
    parser.add_argument('--plot', type=str2bool, default='False')

    return check_args(parser.parse_args())

def check_args(args):
    if not args.cuda:
        print("Consider running code on gpu enabled device")
    else:
        if torch.cuda.device_count() > 1:
            print("Check which GPU is not in use and set the gpu_number argument accordingly.")
            try:
                assert args.gpu_number is not None
                print("GPU number set to: {}".format(args.gpu_number))
                args.device = torch.device('cuda', args.gpu_number)
            except:
                print("Set gpu_number argument.")
                raise
        else:
            args.device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")

    try:
        assert 1 <= args.TX_config <= 6
    except:
        print("TX_config must be one of the 6 configuartions so a value between 1 and 6")

    try:
        assert args.TX_input > 0 and args.TX_input <= 36
    except:
        print("TX_input must have a value between one and 36")

    try:
        assert args.batch_size >= 1
    except:
        print("Batch size must be greater or equal to one")
    root = os.getcwd()
    args.dataroot = os.path.join(root,'dataset/database') if args.dataroot == None else args.dataroot
    args.result_root = os.path.join(root,'results') if args.result_root == None else args.result_root
    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)

    saveArguments(args)

    return args
