import torch
import torch.nn as nn
import os

from dataset.setup_database import setup_database
from utils.modelUtils import initModel
from utils.modelUtils import loadBestModel
from utils.utils import calcDistance
from utils.utils import visualise

#Object to evaluate the performance of the model on the test set
class eval_obj(object):
    def __init__(self, args):
        print("Setting up eval object")
        #Initialising some variables
        self.test_data_loader = setup_database(args, 'test')
        self.device = args.device
        self.best_model = initModel(self.test_data_loader, args.nf).to(args.device)
        self.visualise = args.visualise

        loadBestModel(args.result_root, self.best_model, args.device)

        #Setting the model to evaluation mode
        self.best_model.eval()

    #Calculates the distance between predicted and real position within a certain area
    #The area is given through two points p1 and p2 defining a rectangle between them
    def distanceArea(self, p1, p2):
        min_x = min((p1[0],p2[0])); max_x = max((p1[0],p2[0]))
        min_y = min((p1[1],p2[1])); max_y = max((p1[1],p2[1]))

        distance = []
        with torch.no_grad():
            for i, data in enumerate(self.test_data_loader):
                position = data[1]
                if (max_x < position[0] < min_x) or (max_y < position[1] < min_y):
                    continue
                input = data[0].to(self.device)
                prediction = self.best_model(input)[:,:,0,0]
                distance.append(calcDistance(prediction, output))

        #The average distance over the entire test set is calculated
        dist = sum(distance)/len(distance)
        #The distance is denormalised to cm's
        dist = dist*300
        print("Distance on test set within area: p1 {}\tp2 {}\nis: {}cm".format(p1,p2,dist))

    #Calculates the distance between predicted and real position of the samples in the test set
    #If visualise is enables these distances are visualy plotted
    def demo(self):
        distance = []
        for i, data in enumerate(self.test_data_loader):
            with torch.no_grad():
                #forward batch of test data through the network
                input = data[0].to(self.device)
                output = data[1].to(self.device)
                prediction = self.best_model(input)[:,:,0,0]
                distance.append(calcDistance(prediction, output))

                if self.visualise:
                     visualise(output, prediction, pause=0.1)

        #The average distance over the entire test set is calculated
        dist = sum(distance)/len(distance)
        #The distance is denormalised to cm's
        dist = dist*300
        print("Distance on test set is: {}cm".format(dist))
        return dist
