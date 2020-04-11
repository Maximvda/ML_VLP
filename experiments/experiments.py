import os
import torch.multiprocessing as _mp

from experiments.worker import Worker
from utils.plotscript import *
from eval.eval import Eval_obj

from dataset.setup_database import setup_database


def experiment(args):
    print("Running experiment {}".format(args.experiment))
    hyper_par = []

    #Experiment 1 does a hyperparameter search to find a good model architecture
    #The amount of hidden layers, the number of features and model type are explored
    if args.experiment == 1:
        setup_dir(args, 'experiment_1')
        #Define the hyperparameter search
        #First model type
        for i in ['Type_1','Type_2']:
            #Number of features used
            for j in [32, 64, 128, 256]:
                #Number of hidden layers
                for k in [1,2,3,4,5]:
                    if not ((j == 32 or j == 64) and i == 'Type_2'):
                        hyper_par.append({  'model_type': i,
                                        'nf': j,
                                        'hidden_layers': k})
        run_experiment(args, hyper_par)
        print("")
        #Make plots of this experiment
        #And print performance on test set for the three best models
        best_files = plot_exp_1(args.result_root)
        for file in best_files:
            obj = Eval_obj(args, file)
            obj.demo()

    if args.verbose:
        print("Results of experiment 1 saved at {}".format(args.result_root))

    #Experiment 2 performs a sweep over the different configurations
    #Performance is evaluated on the validation set
    #Heatmaps for all different configurations are then plotted to compare performance
    elif args.experiment == 2:
        setup_dir(args, 'experiment_2')
        #Define the hyperparameter list for all TX_config
        for i in range(1,7):
            hyper_par.append({'TX_config': i})

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



    #Experiment 3 runs a sweep over all number of TX_inputs
    #For each possible number of TX_inputs a model is trained
    #The achieved distance on the val set is then plotted in function of the TX_inputs for each model
    elif args.experiment == 3:
        setup_dir(args, 'experiment_3')
        #Define the hyperparameter list for all TX_inputs
        for i in range(1,37):
            hyper_par.append({'TX_input': i})

        run_experiment(args, hyper_par)

        plot_exp_3(args.result_root)


        if args.verbose:
            print("Results of experiment 3 saved at {}".format(args.result_root))

    #Experiment 4 runs a sweep over the amount of blockage
    #For each blockage a new model is trained to compare the performance difference
    #The achieved distance on the val set is plotted in function of amount of blockage for each model
    elif args.experiment == 4:
        setup_dir(args, 'experiment_4')
        #Define the hyperparameter list for all TX_inputs
        for i in range(1,11):
            hyper_par.append({'blockage': i*0.1})

        run_experiment(args, hyper_par)
        plot_exp_4(args.result_root)


    #Experiment 5 investigates if it is possible to train a model on simulation data
    #and then use this model for a real environment
    #the influence of normalisation on the input data is also investigated
    elif args.experiment == 5:
        setup_dir(args, 'experiment_5')

        #Experiment without normalised inputs
        hyper_par.append({'dataset_path': {'train': 'data_False_train.data',
                                            'val': 'data_False_val.data',
                                            'test': 'data_False_test.data'}})

        #Experiment with simulated training data, val and test set on real data
        hyper_par.append({'dataset_path': {'train': 'simulation_data_True_train.data',
                                            'val': 'data_True_val.data',
                                            'test': 'data_True_test.data'}})

        run_experiment(args, hyper_par)
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

#Inits the directories for an experiment
def setup_dir(args, experiment):
    #Setup dir for all results of experiment
    args.result_root = os.path.join(args.result_root, experiment)
    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)
        os.mkdir(os.path.join(args.result_root,'checkpoints'))
