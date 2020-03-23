import torch.multiprocessing as _mp
import torch.optim as optim
import numpy as np
import torch
import os

from trainer.trainer import Trainer
from utils.utils import printMultiLine

mp = _mp.get_context('spawn')

def exploit_and_explore(result_root, epoch, top_checkpoint_path, bot_checkpoint_path, hyper_params,
                        perturb_factors=(1.2, 0.8)):
    #"""Copy parameters from the better model and the hyperparameters
    #   and running averages from the corresponding optimizer."""
    # Copy model parameters
    checkpoint = torch.load(os.path.join(result_root,top_checkpoint_path),map_location=torch.device('cuda',2))
    optimizer_state_dict = checkpoint['optim']
    batch_size = checkpoint['batch_size']
    for hyperparam_name in hyper_params['optimizer']:
        perturb = np.random.choice(perturb_factors)
        for param_group in optimizer_state_dict['param_groups']:
            betas = (min(param_group['betas'][0]*perturb,0.9999),param_group['betas'][1])
            lr = perturb*param_group['lr']

    if hyper_params['batch_size']:
        perturb = np.random.choice(perturb_factors)
        batch_size = int(np.ceil(perturb * batch_size))

    if epoch <= 7:
        checkpoint['model'] = None; checkpoint['epoch'] = 0
        perturb = np.random.choice(perturb_factors)
        checkpoint['nf'] = int(checkpoint['nf']*perturb)
        perturb = np.random.choice([-1,0, 1])
        checkpoint['extra_layers'] = min(max(checkpoint['extra_layers'] + perturb,0),5)
    elif np.random.uniform() >= 0.1:
        bad_check = torch.load(os.path.join(result_root,bot_checkpoint_path),map_location=torch.device('cuda',1))
        checkpoint['model'] = bad_check['model']
        checkpoint['nf'] = bad_check['nf']
        checkpoint['extra_layers'] = bad_check['extra_layers']
        checkpoint['model_type'] = bad_check['model_type']
        optimizer_state_dict = bad_check['optim']
        del bad_check

    for param_group in optimizer_state_dict['param_groups']:
        param_group['lr'] = lr
        param_group['betas'] = betas


    checkpoint['optim'] = optimizer_state_dict
    checkpoint['batch_size'] = batch_size

    torch.save(checkpoint, os.path.join(result_root,bot_checkpoint_path))
    del checkpoint


def Pbt_trainer(args):
    pth = os.path.join(args.result_root,'checkpoints')
    if not os.path.exists(pth):
        os.mkdir(pth)
    mp = _mp.get_context('spawn')
    mp = mp.get_context('forkserver')
    population = mp.Queue(maxsize=args.population_size)
    finish_tasks = mp.Queue(maxsize=args.population_size)
    _epoch = 1; _best_epoch = 0
    for i in range(0,args.population_size):
        file = os.path.join(pth,'task-%03d.pth'% i)
        if os.path.exists(file):
            checkpoint = torch.load(file,map_location=torch.device('cuda',2))
            _epoch = max(_epoch,checkpoint['epoch']); _best_epoch = max(_best_epoch, checkpoint['best_epoch'])
            del checkpoint
    epoch = mp.Value('i', _epoch)
    best_epoch = mp.Value('i',_best_epoch)
    #Moet eigenlijk ook ingeladen worden van het beste_model als er al getrained was
    best_score = mp.Value('f', 3000.0)
    for i in range(args.population_size):
        population.put(dict(id=i, score=0))

    hyper_params = {'optimizer': ["lr", "betas"], "batch_size": True,
                    'model': ['nf', 'extra_layers']}

    workers = [Worker(args,i, epoch,best_epoch,best_score, population, finish_tasks)
               for i in range(5)]
    workers.append(Explorer(epoch, args.result_root, best_epoch, population, finish_tasks, hyper_params))

    [w.start() for w in workers]
    [w.join() for w in workers]
    task = []
    while not finish_tasks.empty():
        task.append(finish_tasks.get())
    while not population.empty():
        task.append(population.get())
    task = sorted(task, key=lambda x: x['score'], reverse=False)
    printMultiLine(1,"",offset=2, end=True)
    print('best score for task: ', task[0]['id'], ' with score: ', task[0]['score'])

class Worker(mp.Process):
    def __init__(self, args, worker_id, epoch, best_epoch, best_score, population, finish_tasks):
        super().__init__()
        self.epoch = epoch
        self.best_epoch = best_epoch
        self.best_score = best_score
        self.population = population
        self.finish_tasks = finish_tasks
        args.device = torch.device('cuda', worker_id % 5+3)
        self.trainer = Trainer(args,worker_id=worker_id)

    def run(self):
        while True:
            if self.epoch.value - self.best_epoch.value >= 25:
            #if self.epoch.value  > 5:
                break
            # Train
            task = self.population.get()
            self.trainer.set_id(task['id'])
            try:
                self.trainer.train_epoch(self.epoch.value)
                score = self.trainer.calcPerformance()
                with self.best_score.get_lock():
                    if score < self.best_score.value:
                        printMultiLine(0, "Improved best score from: {}\t to: {} by task: {}".format(round(self.best_score.value,3),round(score,3),task['id']),offset=-1)
                        self.trainer.save_checkpoint(save_best=True)
                        self.best_score.value = score
                self.trainer.save_checkpoint()
                with self.best_epoch.get_lock():
                    self.best_epoch.value = max(self.trainer.get_bestepoch(), self.best_epoch.value)
                self.finish_tasks.put(dict(id=task['id'], score=score))
            except KeyboardInterrupt:
                break


class Explorer(mp.Process):
    def __init__(self, epoch, result_root, best_epoch, population, finish_tasks, hyper_params):
        super().__init__()
        self.epoch = epoch
        self.best_epoch = best_epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.hyper_params = hyper_params
        self.result_root = result_root

    def run(self):
        while True:
            if self.epoch.value - self.best_epoch.value >= 25:
            #if self.epoch.value  > 5:
                break
            if self.population.empty() and self.finish_tasks.full():
                printMultiLine(0, "Exploit and explore", offset=-1)
                tasks = []
                while not self.finish_tasks.empty():
                    tasks.append(self.finish_tasks.get())
                tasks = sorted(tasks, key=lambda x: x['score'], reverse=False)
                printMultiLine(0, 'Best score for task: {} at epoch {} is: {}\tbest epoch: {}'.format(
                        tasks[0]['id'], self.epoch.value, tasks[0]['score'], self.best_epoch.value), offset=-1)
                #printMultiLine(0, 'Worst score on' + str(tasks[-1]['id']) + 'is' + str(tasks[-1]['score']), offset=-1)
                fraction = 0.2
                cutoff = int(np.ceil(fraction * len(tasks)))
                tops = tasks[:cutoff]
                bottoms = tasks[len(tasks) - cutoff:]
                for bottom in bottoms:
                    top = np.random.choice(tops)
                    top_checkpoint_path = "checkpoints/task-%03d.pth" % top['id']
                    bot_checkpoint_path = "checkpoints/task-%03d.pth" % bottom['id']
                    exploit_and_explore(self.result_root, self.epoch.value, top_checkpoint_path, bot_checkpoint_path, self.hyper_params)
                with self.epoch.get_lock():
                    self.epoch.value += 1
                for task in tasks:
                    self.population.put(task)
