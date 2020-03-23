import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from trainer.model import Model
from torch.utils.data import DataLoader
from utils.modelUtils import loadCheckpoint
from utils.modelUtils import saveCheckpoint
from utils.modelUtils import saveBestModel
from utils.utils import visualise
from dataset.setup_database import setup_database
from utils.utils import calcDistance
from utils.utils import printMultiLine
from utils.utils import printProgBar

from dataset.dataset import data

def get_optimizer(model):
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    optimiser_class = optim.Adam
    #optimizer_class = optim.SGD
    lr = np.random.choice(np.logspace(-5, -1, base=10))
    momentum = np.random.choice(np.linspace(0.5, .9999))
    #return optimizer_class(model.parameters(), lr=lr, momentum=momentum)
    return optimiser_class(model.parameters(), lr=lr, betas=(momentum,0.999))

def get_model(self):
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    self.model_type = np.random.choice(['FC', 'FC_expand'])
    #optimizer_class = optim.SGD
    self.nf = np.random.choice([2**x for x in range(5,10)])
    self.extra_layers = np.random.choice([x for x in range(0,5)])
    #return optimizer_class(model.parameters(), lr=lr, momentum=momentum)
    return Model(self.size, self.model_type, self.nf, self.extra_layers).to(self.device)

#Class to train cnn architecture and save desired results and statistics during training
class Trainer(object):
    def __init__(self,args,id=None, worker_id=0):
        printMultiLine(worker_id, "Setting up neural network")
        #Initialise some variables
        self.epoch = 0;     self.best_epoch = 0; self.min_distance = 3000
        self.learning = True
        self.batch_size = args.batch_size
        self.task_id = id; self.size = args.size; self.model_type = args.model_type; self.nf = args.nf; self.extra_layers = args.extra_layers
        self.worker_id = worker_id
        #Initialise dataset and setup a data loader
        self.train_dataset = data(args.dataset_path, 'train');
        self.val_dataset = data(args.dataset_path, 'val');
        self.result_root = args.result_root
        self.criterion = nn.MSELoss()
        self.device = args.device; self.visualise = args.visualise

        #self.data_loader = tqdm.tqdm(DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8),
        #                       desc='Train (task {})'.format(self.task_id),
        #                       ncols=80, leave=True)
        self.data_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=4)

        #self.val_data_loader = tqdm.tqdm(DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8),
        #                       desc='Train (task {})'.format(self.task_id),
        #                       ncols=80, leave=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=True, num_workers=4)
        self.step = int(len(self.data_loader)/10)

        if not args.pbt_training:
            #Setup CNN and optimiser
            self.model = Model(args.size, args.model_type, args.nf, args.extra_layers).to(args.device)
            self.optim = optim.Adam(self.model.parameters(), args.learning_rate, betas=(0.5, 0.999))
            #Load the previous training checkpoint
            loadCheckpoint(self, args.device, 'checkpoint.pth')


    def train_epoch(self, epoch):
        timeout_start = time.time()
        #while self.epoch <= epoch:
        while time.time() < (timeout_start + (45*(epoch-self.epoch))):
            printMultiLine(self.worker_id, 'Epoch: {}\t Task {}'.format(self.epoch, self.task_id))
            for i, data in enumerate(self.data_loader):
                if i % self.step == 0:
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

                #Store training stats
                with torch.no_grad():
                    if self.visualise:
                        visualise(output, prediction)
        self.epoch = epoch

    def train(self):
        #Initialising the loss function Binary Cross Entropy loss
        print("Starting training loop")
        while self.learning:
            self.train_epoch()

            #Calculate the performance on the validation set and store best performing model
            self.calcPerformance()

            #If checkpoint freq is reached store training state
            if self.epoch % 1 == 0:
                saveCheckpoint(self, 'checkpoint.pth')

            #Stop training if there has been no improvement over the last 50 epochs
            if self.epoch - self.best_epoch >= 25:
                self.learning = False
                saveCheckpoint(self, 'checkpoint.pth')

    def calcPerformance(self):
        printMultiLine(self.worker_id,"Calculating performance on validation set.", offset=1)
        distance = []
        dist_height = []
        self.model.eval()
        for i, data in enumerate(self.val_data_loader):
            with torch.no_grad():
                input = data[0].to(self.device)
                output = data[1].to(self.device)

                #prediction = self.model(input)[:,:,0,0]
                prediction = self.model(input)

                #Calculate the distance between predicted and target points
                dist, dist_z = calcDistance(prediction, output)
                distance.append(dist); dist_height.append(dist_z)

        #The average distance over the entire test set is calculated
        dist = sum(distance)/len(distance)
        dist_z = sum(dist_height)/len(dist_height)
        #The distance is denormalised to cm's
        dist = dist*300
        dist_z = dist_z*200
        printMultiLine(self.worker_id,"Distance on val set: {}cm\tHeight prediction error: {}cm".format(round(dist,2),round(dist_z,2)), offset=1)

        #If performance of new model is better then all previous ones it is saved
        if dist < self.min_distance:
            printMultiLine(self.worker_id,"Dist: {}\tis smaller than min distance: {}".format(round(dist,2),round(self.min_distance,2)), offset=2)
            self.min_distance = dist
            saveBestModel(self.result_root, self.model, self.task_id)
            self.best_epoch = self.epoch
        self.model.train()
        return dist

    def set_id(self, id):
        self.file = "checkpoints/task-%03d.pth" % id
        self.task_id = id

        self.model = get_model(self)
        self.optim = get_optimizer(self.model)

        loadCheckpoint(self, self.device, self.file, reload_model=True)

        self.data_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=6)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=True, num_workers=6)

    def save_checkpoint(self, save_best=False):
        if save_best:
            saveCheckpoint(self, 'checkpoints/best_model.pth')
        else:
            saveCheckpoint(self, self.file)

    def getModel(self):
        return self.model

    def get_epoch(self):
        return self.epoch

    def get_bestepoch(self):
        return self.best_epoch
