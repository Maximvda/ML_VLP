import torch
import torch.nn as nn
import torch.optim as optim

from utils.initModel import initModel
from utils.initModel import loadModel
from utils.initModel import saveCheckpoint
from utils.utils import visualise
from eval.eval import eval
from utils.utils import makePlot

class CNN(object):
    def __init__(self,args, data_loader):
        print("Setting up CNN model")
        #Initialise some variables
        self.epoch = 0
        self.data_loader = data_loader
        self.loss = []
        self.result_root = args.result_root

        #Setup CNN and optimiser and initialise their previous checkpoint
        self.model = initModel(data_loader, args.nf)

        self.optim = optim.Adam(self.model.parameters(), args.learning_rate, betas=(0.5, 0.999))

        loadModel(self, args.device)

        self.eval = eval(args)

    def train(self, args):
        #Initialising the loss function Binary Cross Entropy loss
        criterion = nn.BCELoss()
        #criterion = nn.MSELoss()

        print("Starting training loop")
        while self.epoch < args.epochs:
            print('Epoch: {}/{}'.format(self.epoch, args.epochs))
            for i, data in enumerate(self.data_loader):
                self.model.zero_grad()

                #Get a data sample
                input = data[0].type(torch.FloatTensor).to(args.device)
                output = data[1].to(args.device)

                #Forward through model and calculate loss
                prediction = self.model(input)[:,:,0,0]
                loss = criterion(prediction, output)
                loss.backward()
                self.optim.step()

                #Store training stats
                self.loss.append(loss.item())
                with torch.no_grad():
                    if i % 50 == 0:
                        print('[{}/{}]\tPrediction: {}\tTarget: {}\tLoss: {}'.format(
                        i, len(self.data_loader), prediction.cpu().numpy()[0], output.cpu().numpy()[0], loss.item()))
                        if args.visualise:
                            visualise(output, prediction)

            self.epoch += 1

            if self.epoch % args.checkpoint_freq == 0:
                saveCheckpoint(self)

            if args.save_training_stats:
                self.eval.saveStats(self.model, self.epoch)


        saveCheckpoint(self)
        print("Training finished\nCreating plots of training and evaluation statistics")
        self.createPlots()

    def createPlots(self):
        makePlot(self.loss, 'plot_training.png', 'Training loss in function of number of iterations.', ['Iteration', 'Loss'], self.result_root)
        loss, distance = self.eval.getLossDistance()
        makePlot(loss, 'plot_eval.png', 'Evaluation loss in function of number of iterations.', ['Iteration', 'Loss'], self.result_root)
        makePlot(distance, 'plot_distance.png', 'Average distance of predicted point to actual position in function of epochs.', ['Epoch', 'Distance (cm)'], self.result_root)

    def getModel(self):
        return self.model

    def get_epoch(self):
        return self.epoch
