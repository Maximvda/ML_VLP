import torch.multiprocessing as _mp
import numpy as np
import torch
import os
import multiprocessing
import time

from trainers.trainer import Trainer
from utils.utils import printMultiLine
from utils.config import get_PBT_choices

mp = _mp.get_context('spawn')

#List used to keep track of best performing tasks of the population
best_ids = [1,2,3,4,5,6]

#PBT training script
def Pbt_trainer(args):
    #Set and create path for the checkpoints of each task of the population
    args.result_root = os.path.join(args.result_root, 'pbt_training')
    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)
    pth = os.path.join(args.result_root,'checkpoints')
    if not os.path.exists(pth):
        os.mkdir(pth)
    #Init variables for MP
    mp = _mp.get_context('spawn')
    mp = mp.get_context('forkserver')
    population = mp.Queue(maxsize=args.population_size)
    finish_tasks = mp.Queue(maxsize=args.population_size)

    #Load the current and best iteration of each task if training was already started
    _iter = 1; _best_iter = 0; _best_score = 3000.0
    for i in range(0,args.population_size):
        file = os.path.join(pth,'task-%03d.pth'% i)
        if os.path.exists(file):
            checkpoint = torch.load(file,map_location=torch.device('cpu'))
            _iter = max(_iter,checkpoint['iter']); _best_iter = max(_best_iter, checkpoint['best_iter'])
            _best_score = min(_best_score, checkpoint['min_dist']['2D'])
            #Delete checkpoint to free up the memory
            del checkpoint

    #multiprocessing shared variables to keep track of progress
    iter = mp.Value('i', _iter)
    best_iter = mp.Value('i',_best_iter)
    best_score = mp.Value('f', _best_score)
    del _iter, _best_iter, _best_score

    #Initialise score for every task in population
    for i in range(args.population_size):
        population.put(dict(id=i, score=0))

    #Setup the worker threads
    workers = [Worker(args,i, iter,best_iter,best_score, population, finish_tasks)
               for i in range(args.workers)]
    workers.append(Explorer(iter, args.result_root, best_iter, population, finish_tasks, args.verbose))

    #Start workers and wait for them to finish
    [w.start() for w in workers]
    [w.join() for w in workers]
    task = []
    while not finish_tasks.empty():
        task.append(finish_tasks.get())
    while not population.empty():
        task.append(population.get())
    #Sort and report the obtained scores
    task = sorted(task, key=lambda x: x['score'], reverse=False)
    printMultiLine(1,"",offset=2, end=True)
    print('best score for task: ', task[0]['id'], ' with score: ', task[0]['score'])

#"""Copy parameters from the better model and the hyperparameters
#   and running averages from the corresponding optimizer."""
def exploit_and_explore(result_root, iter, top_checkpoint_path, bot_checkpoint_path):
    #Init variable
    param_dict = {}
    #Get the PBT choices for the hyperparameter search
    choices = get_PBT_choices()
    # Load best checkpoint and its parameters
    checkpoint = torch.load(os.path.join(result_root,top_checkpoint_path),map_location=torch.device('cpu'))
    #Perform perturbation on optimiser parameters
    optimizer_state_dict = checkpoint['optim']
    for hyperparam_name in choices['optimiser']:
        perturb = np.random.choice(choices['perturb_factors'])
        for param_group in optimizer_state_dict['param_groups']:
            param_dict['betas'] = (min(param_group['betas'][0]*perturb,0.9999),param_group['betas'][1])
            param_dict['lr'] = perturb*param_group['lr']

    #Perform perturbation on parameters
    for parameter in choices['parameters']:
        perturb = np.random.choice(choices['perturb_factors'])
        checkpoint[parameter] = int(np.ceil(perturb * checkpoint[parameter]))

    #Perturb the model parameters for the first 7 iterations
    if iter <= 7:
        checkpoint['model'] = None; checkpoint['iter'] = 0
        perturb = np.random.choice(choices['perturb_factors'])
        checkpoint['nf'] = int(checkpoint['nf']*perturb)
        perturb = np.random.choice([-1,0, 1])
        checkpoint['hidden_layers'] = min(max(checkpoint['hidden_layers'] + perturb,0),7)

    #Retain model from the bad performing task with 90% probability
    elif np.random.uniform() >= 0.1:
        bad_check = torch.load(os.path.join(result_root,bot_checkpoint_path),map_location=torch.device('cpu'))
        checkpoint['model'] = bad_check['model']
        checkpoint['nf'] = bad_check['nf']
        checkpoint['hidden_layers'] = bad_check['hidden_layers']
        checkpoint['model_type'] = bad_check['model_type']
        optimizer_state_dict = bad_check['optim']
        del bad_check

    #Update optimiser parameters with perturbed values
    for hyperparam_name in choices['optimiser']:
        for param_group in optimizer_state_dict['param_groups']:
            param_group[hyperparam_name] = param_dict[hyperparam_name]

    checkpoint['optim'] = optimizer_state_dict

    #Save the checkpoint and free up its memory
    torch.save(checkpoint, os.path.join(result_root,bot_checkpoint_path))
    del checkpoint

#Worker class that trains population task and calculates performance
class Worker(mp.Process):
    def __init__(self, args, worker_id, iter, best_iter, best_score, population, finish_tasks):
        super().__init__()
        self.iter = iter; self.best_iter = best_iter
        self.verbose = args.verbose
        self.best_score = best_score
        self.population = population
        self.finish_tasks = finish_tasks
        if type(args.gpu_number) == list:
            args.device = torch.device('cuda', args.gpu_number[worker_id % len(args.gpu_number)])
        else:
            args.device = torch.device('cuda', args.gpu_number)
        self.trainer = Trainer(args,worker_id=worker_id)

    def run(self):
        while True:
            #Stop training if no improvement since last 5 iterations
            if self.iter.value - self.best_iter.value >= 5:
                break
            # Set the correct parameters for the Trainer class depending on the task id
            task = self.population.get()
            self.trainer.set_id(task['id'], reload=True)
            try:
                #Train task with one iteration
                self.train_iter()

                score = self.trainer.calcPerformance()
                #Save best task if it has the lowest score from entire population
                with self.best_score.get_lock():
                    if score < self.best_score.value:
                        if self.verbose:
                            printMultiLine(0, "Improved best score from: {}\t to: {} by task: {}".format(round(self.best_score.value,3),round(score,3),task['id']),offset=-1)
                        self.trainer.save_checkpoint(save_best=True)
                        self.best_score.value = score
                self.trainer.save_checkpoint()
                #If the task is in the best performing tasks than update the best_iter criteria
                if task['id'] in best_ids:
                    #Get best iteration of task
                    with self.best_iter.get_lock():
                        self.best_iter.value = max(self.trainer.get_best_iter(), self.best_iter.value)
                self.finish_tasks.put(dict(id=task['id'], score=score))
            except KeyboardInterrupt:
                break

    #Train for amount of iterations model is behind of worker iterations
    #one iteration is 45 seconds
    def train_iter(self):
        #Init infinite training process
        process = InfTrainProcess(self.trainer)

        #init stop flag, idle time and thread
        iter = self.trainer.get_iter()
        idle_time = 45*(self.iter.value-iter)

        #Start process and sleep for idle time
        process.start()
        time.sleep(idle_time)
        #Shutdown process and wait for it to finish
        process.shutdown()
        process.join()
        #After training update iter of Trainer class
        self.trainer.set_iter(self.iter.value)


class Explorer(mp.Process):
    def __init__(self, iter, result_root, best_iter, population, finish_tasks, verbose):
        super().__init__()
        self.iter = iter
        self.best_iter = best_iter
        self.population = population
        self.finish_tasks = finish_tasks
        self.result_root = result_root
        self.verbose = verbose

    def run(self):
        while True:
            #Stop training if no improvement since last 5 iterations
            if self.iter.value - self.best_iter.value >= 5:
                break
            if self.population.empty() and self.finish_tasks.full():
                if self.verbose:
                    printMultiLine(0, "Exploit and explore", offset=-1)
                tasks = []
                while not self.finish_tasks.empty():
                    tasks.append(self.finish_tasks.get())
                #Sort tasks according to their score
                tasks = sorted(tasks, key=lambda x: x['score'], reverse=False)
                if self.verbose:
                    printMultiLine(0, 'Best score for task: {} at iter {} is: {}\tbest iter: {}'.format(
                        tasks[0]['id'], self.iter.value, tasks[0]['score'], self.best_iter.value), offset=-1)
                #Fraction of population to exploit and explore
                fraction = 0.2
                cutoff = int(np.ceil(fraction * len(tasks)))
                best_ids = [best['id'] for best in tasks[:int(np.ceil(0.05 * len(tasks)))]]
                tops = tasks[:cutoff]
                bottoms = tasks[len(tasks) - cutoff:]
                for bottom in bottoms:
                    top = np.random.choice(tops)
                    top_checkpoint_path = "checkpoints/task-%03d.pth" % top['id']
                    bot_checkpoint_path = "checkpoints/task-%03d.pth" % bottom['id']
                    exploit_and_explore(self.result_root, self.iter.value, top_checkpoint_path, bot_checkpoint_path)
                #Increase iteration and continue training
                with self.iter.get_lock():
                    self.iter.value += 1
                for task in tasks:
                    self.population.put(task)
            else:
                #Wait a minute before checking again to conserve resources
                time.sleep(60)

#Process that trains for till process is shutdown
class InfTrainProcess(multiprocessing.Process):
    def __init__(self, trainer):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()
        self.trainer = trainer

    def run(self):
        while not self.exit.is_set():
            self.trainer.train_iter()

    def shutdown(self):
        self.exit.set()
