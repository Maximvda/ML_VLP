import os
import torch.multiprocessing as _mp

from experiments.worker import Worker

#Experiment 1 does a hyperparameter search to find a good model architecture
#The amount of hidden layers, the number of features and model type are explored
def experiment1(args):
    print("Performing experiment 1")

    #Setup dir for all results of experiment 1
    args.result_root = os.path.join(args.result_root, 'experiment_1')
    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)
        os.mkdir(os.path.join(args.result_root,'checkpoints'))

    hyper_par = []

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
