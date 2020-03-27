import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import multiprocessing
import numpy as np
import time

from trainers.model import Model
from utils.modelUtils import setup_model
from utils.modelUtils import save_state
from utils.utils import visualise
from utils.utils import calcDistance
from utils.utils import printMultiLine
from utils.utils import printProgBar

from dataset.dataset import Data

#Class to train model save desired results and evaluate performance on validation
class Trainer(object):
    def __init__(self,args,id=None, worker_id=0):
        #Set the seed for reproducability
        if args.verbose:
            printMultiLine(worker_id, "Setting up neural network")
        #Initialise some variables
        self.worker_id = worker_id; self.verbose = args.verbose;    self.result_root = args.result_root
        self.device = args.device;
        self.learning = True;   self.visualise = args.visualise

        #Initialise model parameters
        self.size = args.size; self.model_type = args.model_type; self.nf = args.nf; self.hidden_layers = args.hidden_layers
        self.criterion = nn.MSELoss(); self.output_nf = args.output_nf; self.learning_rate = args.learning_rate

        #Initialise datasets
        self.train_dataset = Data(args.dataset_path['train'], args.TX_config, args.TX_input, args.blockage, args.output_nf)
        self.val_dataset = Data(args.dataset_path['val'], args.TX_config, args.TX_input, args.blockage, args.output_nf)

        #If population based training is not used
        if not args.pbt_training:
            self.batch_size = args.batch_size
            #Setup dataloaders
            self.data_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=4)
            self.val_data_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=True, num_workers=4)
            self.step = int(len(self.data_loader)/10);
            #Load checkpoint
            setup_model(self, 'checkpoint.pth')

    #Train for one iteration if epoch is set to true then iteration is an epoch
    def train_iter(self, epoch=False):
        if self.verbose:
            text = 'Epoch: {}'.format(self.iter) if epoch else 'Iter: {}\t Task {}'.format(self.iter, self.task_id)
            printMultiLine(self.worker_id, text)
        for i, data in enumerate(self.data_loader):
            if i % self.step == 0 and self.verbose:
                printMultiLine(self.worker_id, printProgBar(i,len(self.data_loader)), offset=1)
            self.model.zero_grad()

            #Get a data sample
            input = data[0].to(self.device)
            output = data[1].to(self.device)

            #Forward through model and calculate loss
            prediction = self.model(input)
            loss = self.criterion(prediction, output)
            loss.backward()
            self.optim.step()

            #Visualise training if visualise is true
            if self.visualise:
                with torch.no_grad():
                    visualise(output, prediction)
        if epoch:
            self.iter += 1

    #Train model till there is no performance improvement anymore
    def train(self, experiment=False):
        #Initialising the loss function Binary Cross Entropy loss
        if self.verbose:
            printMultiLine(self.worker_id, "Starting training loop")
        while self.learning:
            self.train_iter(epoch=True)

            self.calcPerformance(experiment)

            #Stop training if there has been no improvement over the last 25 epochs
            if self.iter - self.best_iter >= 25:
                self.learning = False

            #Store training state
            #If experiment is running store state to specified file
            if experiment:
                save_state(self, self.file)
            else:
                save_state(self, 'checkpoint.pth')

    #Calculate the performance on the validation set and store best performing model
    def calcPerformance(self, experiment):
        if self.verbose:
            printMultiLine(self.worker_id,"Calculating performance on validation set.", offset=1)
        dist_dict = {'2D': [], 'z': [], '3D': []}
        #Set model to evaluation mode
        self.model.eval()
        #Get prediction error for every batch in validation set
        for i, data in enumerate(self.val_data_loader):
            with torch.no_grad():
                input = data[0].to(self.device)
                output = data[1].to(self.device)

                prediction = self.model(input)

                #Calculate the distance between predicted and target points
                dist = calcDistance(prediction, output)
                dist_dict['2D'].append(dist['2D'])
                if len(dist) == 3:
                    dist_dict['z'].append(dist['z']); dist_dict['3D'].append(dist['3D'])

        #The average error over the entire set is calculated
        for key in dist_dict:
            if len(dist_dict[key]) > 0:
                dist_dict[key] = sum(dist_dict[key])/len(dist_dict[key])
            else:
                dist_dict[key] = np.inf
        #The distance is denormalised to cm's
        dist_dict['2D'] = dist_dict['2D']*300; dist_dict['z'] = dist_dict['z']*200
        if self.verbose:
            printMultiLine(self.worker_id,"Distance on val set 2D: {} cm\t Z: {} cm\t 3D: {} cm".format(
                round(dist_dict['2D'],2),round(dist_dict['z'],2), round(dist_dict['3D'],2)), offset=1)

        #If performance of new model is better then all previous ones it is saved
        if dist_dict['2D'] < self.min_distance:
            if self.verbose:
                printMultiLine(self.worker_id,"Dist: {}\tis smaller than min distance: {}".format(round(dist_dict['2D'],2),round(self.min_distance,2)), offset=2)
            self.min_distance = dist_dict['2D']
            self.best_iter = self.iter
            save_state(self,"checkpoints/best-%03d.pth" % self.task_id) if experiment else save_state(self, 'model.pth')
        self.model.train()
        return dist_dict['2D']

    def set_id(self, id, reload=False):
        #set file and id from the Population task
        self.file = "checkpoints/task-%03d.pth" % id
        self.task_id = id
        #Correctly load model for the task
        setup_model(self, self.file, reload_model=reload)
        #Setup dataloaders with correct batch size for the task
        self.data_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=4)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=True, num_workers=4)
        self.step = int(len(self.data_loader)/10);

    def save_checkpoint(self, save_best=False):
        if save_best:
            save_state(self, 'checkpoints/best_model.pth')
        else:
            save_state(self, self.file)

    def set_attribute(self, key, value):
        setattr(self, key, value)

    def get_best_iter(self):
        return self.best_iter

    def set_iter(self, iter):
        self.iter = iter
    def get_iter(self):
        return self.iter
