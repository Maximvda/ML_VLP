import torch
import torch.nn as nn
import torch.optim as optim

from models.architecture import cnn
from utils.initModel import initModel
from utils.initModel import saveCheckpoint
from eval.utils import visualise

class CNN(object):
    def __init__(self,args, data_loader):
        print("Setting up CNN model")
        #Initialise some variables
        self.epoch = 0
        self.data_loader = data_loader
        self.loss = []
        self.result_root = args.result_root

        #Setup CNN and optimiser and initialise their previous checkpoint
        input, output = next(iter(data_loader))
        size = [input.size(2), input.size(3)]
        self.model = cnn(size, 1, args.nf)

        self.optim = optim.Adam(self.model.parameters(), args.learning_rate, betas=(0.5, 0.999))

        initModel(self, args.device)

    def train(self, args):
        #Initialising the loss function Mean square error
        #criterion = nn.BCEWithLogitsLoss()
        criterion = nn.BCELoss()

        print("Starting training loop")
        while self.epoch < args.epochs:
            print('Epoch: {}/{}'.format(self.epoch, args.epochs))
            for i, data in enumerate(self.data_loader):
                self.model.zero_grad()

                #Get a data sample
                input = data[0].type(torch.FloatTensor).to(args.device)
                output = data[1].to(args.device)
                batchSize = input.size(0)
                #print(self.model)

                #Forward through model and calculate loss
                prediction = self.model(input)[:,:,0,0]
                loss = criterion(prediction, output)
                loss.backward()
                self.optim.step()

                #Store training stats
                self.loss.append(loss.item())
                visualise(output, prediction)
                with torch.no_grad():
                    if i % 50 == 0:
                        print('[{}/{}]\tPrediction: {}\tTarget: {}\tLoss: {}'.format(
                        i, len(self.data_loader), prediction.cpu().numpy()[0], output.cpu().numpy()[0], loss.item()))
                        if args.visualise:
                            visualise(output, prediction)

            self.epoch += 1

            if self.epoch % args.checkpoint_freq == 0:
                saveCheckpoint(self)


        saveCheckpoint(self)
        print("Training finished")

    def return_Generator(self):
        return self.Gen

    def get_epoch(self):
        return self.epoch
