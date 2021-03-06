import argparse
import os
import numpy as np
import torch

#Here you can define the initial parameters the PBT algorithm uses
#The perturbation factors can also be defined
#The hyperparameter space you want to search is defined here
def get_PBT_choices():
    dict = {
        'lr': np.logspace(-5, -1, base=10),
        'momentum': np.linspace(0.5, .9999),
        'model_type': ['Type_1', 'Type_2'],
        'nf': [2**x for x in range(5,10)],
        'hidden_layers': [x for x in range(1,6)],
        'batch_size': [2**x for x in range(5,12)],
        'optimiser': ['lr', 'betas'],
        'model': ['nf', 'hidden_layers'],
        'parameters': ['batch_size'],
        'perturb_factors': (1.2, 0.8)
    }
    return dict

#TX configurations can be added here
#List the TX you want in your configuration
def get_configuration_dict():
    #The predefined TX configuartions from 1 to 6 can be found on the Github page
    list_dict = {
    1: [i for i in range(0,36)],
    2: [0,2,4,12,14,16,24,26,28],
    3: [7,10,25,28],
    4: [0,2,3,5,12,14,15,17,18,20,21,23,30,32,33,35],
    5: [0,5,14,15,20,21,30,35],
    6: [14,15,20,21]}
    return list_dict

#Get the size of a specific configuration
def config2size(TX_config):
    return len(get_configuration_dict()[TX_config])


def parse_args():
    #Argument parser
    parser = argparse.ArgumentParser(description="Visible Light Positioning with Machine Learning")

    #General/harware options
    parser.add_argument('--gpu_number', type=str2list, default=0, help="Number or list of GPUs that will be used")
    parser.add_argument('--experiment', type=int, default=None, help='Select a certain experiment to run or leave default None to train a single model')
    parser.add_argument('--simulate', type=str2bool, default="False", help='Train model through usage of a simulation')
    parser.add_argument('--seed', type=int, default=1996, help="Set random seed to reproduce results")

    #Dataset options
    parser.add_argument('--dataroot', default=None, required=False, help="Path to the dataset")
    parser.add_argument('--result_root', default=None, required=False, help="Path to the result directory")
    parser.add_argument('--normalise', type=str2bool, default="True", help="If set to true the model inputs are normalised. The output is always normalised")
    parser.add_argument('--TX_config', type=int, default=1, help='Select the TX configuartion you want to use. Different configurations can be found at https://github.com/Maximvda/ML_VLP')
    parser.add_argument('--TX_input', type=int, default=36, help="Limit the amount of inputs of the network to only the best received signals")
    parser.add_argument('--blockage', type=float, default=0.0, help="Percentage of TXs who are blocked")

    #Model options
    parser.add_argument('--model_type', type=str, default='Type_2', choices=['Type_1', 'Type_2'], help="Set the model type to use")
    parser.add_argument('--nf', type=int, default=256, help="The numer of features for the model layers")
    parser.add_argument('--hidden_layers', type=int, default=4, help="The number of hidden layers in the model")
    parser.add_argument('--output_nf', type=int, default=3, choices=[2,3], help="Can be set to two then height is not predicted")

    #Training options
    parser.add_argument('--is_train', type=str2bool, default='True', help="Set to true if you want to train model or false to evaluate it")
    parser.add_argument('--pbt_training', type=str2bool, default="false", help="Set to true if you wish to use PBT algorithm")
    parser.add_argument('--batch_size', type=int, default=32, help="The size of the batch for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate of the optimiser")
    parser.add_argument('--workers', type=int, default=1, help="Set number of workers used, to train multiple models simultanious")
    parser.add_argument('--population_size', type=int, default=120, help='Size of the population when using PBT')

    #Training SVM or RF
    parser.add_argument('--SVM', type=str2bool, default="false", help="Set to true if you wish to train a SVM")
    parser.add_argument('--RF', type=str2bool, default="false", help="Set to true if you wish to train RF")

    #Visual options
    parser.add_argument('--visualise', type=str2bool, default='False', help="Visualising the training process with a plot")
    parser.add_argument('--verbose', type=str2bool, default="True", help="Set to true for more progress updates")

    return check_args(parser.parse_args())

#Check if arguments are valid and initialise
def check_args(args):
    set_GPU(args)
    try:
        assert 1 <= args.TX_config <= len(get_configuration_dict())
        args.size = config2size(args.TX_config)
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

    try:
        assert args.hidden_layers >= 1
    except:
        print("Number of hidden layers must be greater or equal to one")

    if (not args.normalise) and (args.blockage != 0.0):
        print("Configuration of blockage with normalisation=False is not correctly implemented.")
        print("Data processing to set amount of blockage in the Data class in file dataset/dataset.py should be extended for non normalised inputs (value should be set to 0 instead of -1).")
        raise NameError('Configuration not implemented')

    setup_directories(args)

    return args

#Setup the dataset and result directories
def setup_directories(args):
    root = os.getcwd()
    if args.dataroot == None:
        args.dataroot = os.path.join(root,'dataset/database')
        if args.verbose:
            print('Dataroot was not set, default location used: {}'.format(args.dataroot))
    if args.result_root == None:
        args.result_root = os.path.join(root,'results')
        if args.verbose:
            print('Resultroot was not set, default location used: {}'.format(args.result_root))
    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)

#Check GPU arguments and system hardware and set device accordingly
def set_GPU(args):
    if args.gpu_number == None:
        print("Consider running code on gpu enabled device")
        print("PBT_training only works with gpu")
    else:
        if torch.cuda.device_count() > 1:
            try:
                assert args.gpu_number is not None
                print("GPU number set to: {}".format(args.gpu_number))
                args.device = torch.device('cuda', args.gpu_number[0]) if type(args.gpu_number) == list else torch.device('cuda', args.gpu_number)
            except:
                print("Check which GPU is not in use and set the gpu_number argument accordingly.")
                raise
        else:
            args.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

#Converts a string argument to a boolean
def str2bool(v):
    if v.lower() in ('true', 'yes', '1', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
#Converts the string argument to either an int or list
def str2list(v):
    if len(v) == 1:
        return int(v)
    else:
        v = v[1:-1]
        vlist = v.split(',')
        list = []
        for item in vlist:
            list.append(int(item))
        return list
