import random
import numpy as np
import torch

from utils.config import parse_args
from experiments.experiments import experiment
from dataset.setup_database import setup_database

from trainers.trainer import Trainer
from trainers.pbt_trainer import Pbt_trainer
from eval.eval import Eval_obj
from trainers.svm_rf import *

#Trains or evaluates a model
def main(args):
    #Train using the PBT algorithm
    if args.pbt_training:
        Pbt_trainer(args)

    #Train SVM or RF
    elif args.SVM:
        SVM(args)
    elif args.RF:
        RF(args)

    #Get trainer and train model
    elif args.is_train:
        trainer = Trainer(args)
        trainer.train()
    else:
        #Best performing model is loaded and evaluated on the test set
        evalObj = Eval_obj(args)

if __name__ == '__main__':
    print("Script started")
    args = parse_args()
    print("Arguments parsed")

    #Set the random seed to reproduce results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #Pre-process dataset if needed and return correct path
    args.dataset_path = setup_database(args)

    #Reset seeds after dataset is setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #Run experiment if its set else run main function
    if args.experiment != None:
        experiment(args)
    else:
        main(args)
