import torch
import torch.nn as nn
import os

from dataset.setup_database import setup_database
from utils.utils import calcDistance

class eval(object):
    def __init__(self, args):
        print("Setting up eval object")
        args.is_train = False
        self.data_loader = setup_database(args)
        args.is_train =True
        self.loss = []
        self.distance = []
        self.device = args.device
        self.result_root = args.result_root

        #Restoring previous calculated metrics
        self.path = os.path.join(self.result_root,'evalStats.pth')
        if os.path.isfile(self.path):
            checkpoint = torch.load(self.path, map_location=self.device)
            self.loss = checkpoint['loss']
            self.distance = [checkpoint['distance']]

    def calcMetrics(self, model):
        criterion = nn.BCELoss()
        distance = []
        for i, data in enumerate(self.data_loader):
            with torch.no_grad():
                #forward batch of test data through the network
                input = data[0].type(torch.FloatTensor).to(self.device)
                output = data[1].to(self.device)
                prediction = model(input)[:,:,0,0]

                loss = criterion(prediction, output)
                self.loss.append(loss.item())

                #Calculate the distance between predicted and target points
                distance.append(calcDistance(prediction, output))

        #The average distance over the entire test set is calculated
        dist = sum(distance)/len(distance)
        #The distance is denormalised to cm's
        dist = dist*300
        self.distance.append(dist)

    def saveStats(self, model):
        print("Saving training stats")
        #Calculate necessary metrics which need to be saved
        #Performance metric and loss on test data
        self.calcMetrics(model)

        #Save calculated metrics
        torch.save({
                'loss': self.loss,
                'distance': self.distance,
                }, self.path)

    def getLossDistance(self):
        return self.loss, self.distance
