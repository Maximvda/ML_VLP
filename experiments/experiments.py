import os
import torch.multiprocessing as _mp

from experiments.worker import Worker
from utils.plotscript import *
from eval.eval import Eval_obj
from dataset.setup_database import setup_database

def experiment(args):
    print("Running experiment {}".format(args.experiment))
    hyper_par = []

    #Make sure the datasets are processed.
    args.cell_type = "2x2"
    setup_database(args)
    args.cell_type = "3x3"
    setup_database(args)

    #Experiment 1 looks at the influence on performance of different cell_types in function of the amount of blockage
    if args.experiment == 1:
        setup_dir(args, 'experiment_unit_cell_1')
        #Define the parameters to train models on
        for i in range(11):
            hyper_par.append({'cell_type': '3x3', 'blockage': i*0.1,
                                'rotations': False,
                                'dataset_path': {
                                    'train': '/home/r0579568/ML_VLP/dataset/database/data_3x3_True_train.data',
                                    'val': '/home/r0579568/ML_VLP/dataset/database/data_3x3_True_val.data',
                                    'test': '/home/r0579568/ML_VLP/dataset/database/data_3x3_True_test.data',
                                }})
            hyper_par.append({'cell_type': '2x2', 'blockage': i*0.1,
                                'rotations': False,
                                'dataset_path': {
                                    'train': '/home/r0579568/ML_VLP/dataset/database/data_2x2_True_train.data',
                                    'val': '/home/r0579568/ML_VLP/dataset/database/data_2x2_True_val.data',
                                    'test': '/home/r0579568/ML_VLP/dataset/database/data_2x2_True_test.data',
                                }})

            hyper_par.append({'cell_type': '3x3', 'blockage': i*0.1,
                                'rotations': True,
                                'dataset_path': {
                                    'train': '/home/r0579568/ML_VLP/dataset/database/data_3x3_True_train.data',
                                    'val': '/home/r0579568/ML_VLP/dataset/database/data_3x3_True_val.data',
                                    'test': '/home/r0579568/ML_VLP/dataset/database/data_3x3_True_test.data',
                                }})

        run_experiment(args, hyper_par)

        #Make plots of this experiment
        #And print performance on test set for the three best models
        best_files = plot_exp_1(args.result_root)
        for file in best_files:
            obj = Eval_obj(args, file)
            obj.demo()

        if args.verbose:
            print("Results of experiment 1 saved at {}".format(args.result_root))

    #Experiment 2 looks at the influence on performance by application rotational transformations on the input data
    elif args.experiment == 2:
        setup_dir(args, 'experiment_unit_cell_2')

        #Train one model with rotations as data augmentation and one without
        hyper_par.append({'rotations': True})
        hyper_par.append({'rotations': False})

        run_experiment(args, hyper_par)

        dict = []
        root = os.path.join(args.result_root, 'checkpoints')
        files = os.listdir(root)
        files.sort()
        for file in files:
            if 'best' in file:
                obj = Eval_obj(args, os.path.join('checkpoints',file))
                dict.append(obj.heatMap())

        plot_exp_2(args.result_root, dict)

        if args.verbose:
            print("Results of experiment 2 saved at {}".format(args.result_root))

    else:
        print("Experiment {} is not implemented".format(args.experiment))


#Trains all models for the given hyper_par list
def run_experiment(args, hyper_par):
    #Init variables for workers
    mp = _mp.get_context('forkserver')
    tasks = mp.Queue(maxsize=len(hyper_par))

    #Set Id for the tasks
    for i in range(len(hyper_par)):
        tasks.put(dict(id=i))

    #Setup the worker threads
    workers = [Worker(args,i, tasks, hyper_par)
               for i in range(args.workers)]
    #Start worker processes and wait for them to finish
    [w.start() for w in workers]
    [w.join() for w in workers]
    print("")

#Inits the directories for an experiment
def setup_dir(args, experiment):
    #Setup dir for all results of experiment
    args.result_root = os.path.join(args.result_root, experiment)
    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)
        os.mkdir(os.path.join(args.result_root,'checkpoints'))
