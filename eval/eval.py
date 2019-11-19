import torch
import torch.nn as nn
import os

from dataset.setup_database import setup_database
from utils.initModel import initModel
from utils.utils import calcDistance
from utils.utils import visualise

class eval(object):
    def __init__(self, args):
        print("Setting up eval object")
        self.val_data_loader = setup_database(args, 'val')
        self.test_data_loader = setup_database(args, 'test')
        self.loss = []
        self.distance = []
        self.best_epoch = 0
        self.device = args.device
        self.result_root = args.result_root
        self.min_distance = 3000
        self.best_model = initModel(self.val_data_loader, args.nf).to(args.device)

        #Restoring previous calculated metrics
        self.path = os.path.join(self.result_root,'evalStats.pth')
        if os.path.isfile(self.path):
            checkpoint = torch.load(self.path, map_location=self.device)
            self.loss = checkpoint['loss']
            self.distance = checkpoint['distance']
            self.best_epoch = checkpoint['best_epoch']
            self.min_distance = checkpoint['min_dist']
            self.best_model.load_state_dict(checkpoint['best_model'])
        self.best_model.eval()

    def calcMetrics(self, model, epoch):
        criterion = nn.BCELoss()
        distance = []
        for i, data in enumerate(self.val_data_loader):
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
        print("Distance on val set: {}cm".format(dist))
        if dist < self.min_distance:
            print("Dist: {}\tis smaller then min distance: {}".format(dist,self.min_distance))
            self.min_distance = dist
            self.best_model = model.train()
            self.best_epoch = epoch
            self.demo()
        self.distance.append(dist)

    def saveStats(self, model, epoch):
        print("Saving training stats")
        #Calculate necessary metrics which need to be saved
        #Performance metric and loss on test data
        self.calcMetrics(model.eval(), epoch)
        model.train()

        #Save calculated metrics
        torch.save({
                'best_model': self.best_model.state_dict(),
                'loss': self.loss,
                'distance': self.distance,
                'best_epoch': self.best_epoch,
                'min_dist': self.min_distance,
                }, self.path)

    def demo(self):
        distance = []
        for i, data in enumerate(self.val_data_loader):
            with torch.no_grad():
                #forward batch of test data through the network
                input = data[0].type(torch.FloatTensor).to(self.device)
                output = data[1].to(self.device)
                prediction = self.best_model(input)[:,:,0,0]
                distance.append(calcDistance(prediction, output))
                #visualise(output, prediction, pause=0.1)

        #The average distance over the entire test set is calculated
        dist = sum(distance)/len(distance)
        #The distance is denormalised to cm's
        dist = dist*300
        print("In epoch: {}\tBest distance: {}\nDistance on test set: {}cm".format(self.best_epoch, self.min_distance, dist))


    def getLossDistance(self):
        return self.loss, self.distance
