import torch
import torch.nn as nn
import torch.optim as optim

from utils.modelUtils import loadCheckpoint
from utils.modelUtils import saveCheckpoint
from utils.modelUtils import saveBestModel
from utils.utils import visualise
from utils.utils import makePlot
from dataset.setup_database import setup_database
from utils.utils import calcDistance
from models.architecture import model

#Class to train cnn architecture and save desired results and statistics during training
class model_obj(object):
    def __init__(self,args):
        print("Setting up CNN model")
        #Initialise some variables
        self.epoch = 0;     self.best_epoch = 0;
        self.learning = True
        self.data_loader = setup_database(args); self.val_data_loader = setup_database(args, 'val')
        self.loss = [];     self.distance = []
        self.result_root = args.result_root
        self.min_distance = 3000
        self.step = int(len(self.data_loader)/5)

        #Setup CNN and optimiser
        self.model = model(9, args.model_type, args.nf, args.extra_layers).to(args.device)
        self.optim = optim.Adam(self.model.parameters(), args.learning_rate, betas=(0.5, 0.999))

        #Load the previous training checkpoint
        loadCheckpoint(self, args.device)

    def train(self, args):
        #Initialising the loss function Binary Cross Entropy loss
        #criterion = nn.BCELoss()
        criterion = nn.MSELoss()
        crit = nn.MSELoss(reduction='none')

        print("Starting training loop")
        while self.learning:
            print('Epoch: {}'.format(self.epoch))
            for i, data in enumerate(self.data_loader):
                self.model.zero_grad()

                #Get a data sample
                input = data[0].to(args.device)
                output = data[1].to(args.device)
                #Forward through model and calculate loss
                prediction = self.model(input)
                loss = torch.sqrt(torch.sum(crit(prediction[:,0:2].detach(),output),dim=1)).view(-1,1)
                output = torch.cat((output,loss),dim=1)
                #zero = torch.full(distance.size(), 0, device=args.device)
                #loss = criterion(distance, zero)
                loss = criterion(prediction, output)
                loss.backward()
                self.optim.step()

                #Store training stats
                self.loss.append(loss.item())
                with torch.no_grad():
                    if i % self.step == 0:
                        print('[{}/{}]\tPrediction: {}\tTarget: {}\tLoss: {}'.format(
                        i, len(self.data_loader), prediction.cpu().numpy()[0], output.cpu().numpy()[0], loss.item()))
                        if args.visualise:
                            visualise(output, prediction)

            self.epoch += 1

            #Calculate the performance on the validation set and store best performing model
            self.calcPerformance(args.device)

            #If checkpoint freq is reached store training state
            if self.epoch % args.checkpoint_freq == 0:
                saveCheckpoint(self)

            #Stop training if there has been no improvement over the last 50 epochs
            if self.epoch - self.best_epoch >= 25:
                self.learning = False


        print("Training finished\nCreating plots of training and evaluation statistics")
        saveCheckpoint(self)
        self.createPlots()

    def calcPerformance(self, device):
        print("Calculating performance on validation set.")
        distance = []
        self.model.eval()
        for i, data in enumerate(self.val_data_loader):
            with torch.no_grad():
                input = data[0].to(device)
                output = data[1].to(device)

                #prediction = self.model(input)[:,:,0,0]
                prediction = self.model(input)

                #Calculate the distance between predicted and target points
                dist = calcDistance(prediction, output)
                distance.append(dist)

        #The average distance over the entire test set is calculated
        dist = sum(distance)/len(distance)
        #The distance is denormalised to cm's
        dist = dist*75
        print("Distance on val set: {}cm".format(dist))

        #If performance of new model is better then all previous ones it is saved
        if dist < self.min_distance:
            print("Dist: {}\tis smaller then min distance: {}".format(dist,self.min_distance))
            self.min_distance = dist
            saveBestModel(self.result_root, self.model)
            self.best_epoch = self.epoch
        self.distance.append(dist)
        self.model.train()

    def createPlots(self):
        makePlot(self.loss, 'plot_training.pdf', 'Training loss in function of number of iterations.', ['Iteration', 'Loss'], self.result_root)
        #makePlot(loss, 'plot_eval.png', 'Evaluation loss in function of number of iterations.', ['Iteration', 'Loss'], self.result_root)
        print("Min on validation set: ", min(self.distance))
        makePlot(self.distance, 'plot_distance.pdf', 'Average distance of predicted point to actual position in function of epochs on validation set.', ['Epoch', 'Distance (cm)'], self.result_root)

    def getModel(self):
        return self.model

    def get_epoch(self):
        return self.epoch

    def get_distance(self):
        return self.distance
