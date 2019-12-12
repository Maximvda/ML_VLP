import torch
import torch.nn as nn
import os

from dataset.setup_database import setup_database
from utils.modelUtils import initModel
from utils.modelUtils import loadBestModel
from utils.utils import calcDistance
from utils.utils import visualise
from utils.utils import calcBias

#Object to evaluate the performance of the model on the test set
class eval_obj(object):
    def __init__(self, args):
        print("Setting up eval object")
        #Initialising some variables
        self.test_data_loader = setup_database(args, 'test')
        self.device = args.device
        self.best_model = initModel(self.test_data_loader, args.nf, args.extra_layers).to(args.device)
        self.visualise = args.visualise

        loadBestModel(args.result_root, self.best_model, args.device)

        #Setting the model to evaluation mode
        self.best_model.eval()

    #Calculates the distance between predicted and real position of the samples in the test set
    #If visualise is enables these distances are visualy plotted
    def demo(self, area1=None, area2=None):
        distance = []
        x = []; y = []
        for i, data in enumerate(self.test_data_loader):
            with torch.no_grad():
                #forward batch of test data through the network
                input = data[0].to(self.device)
                output = data[1].to(self.device)
                prediction = self.best_model(input)[:,:,0,0]
                distance.append(calcDistance(prediction, output, area1, area2))
                x1, y1 = calcBias(prediction, output)
                x.append(x1); y.append(y1)

                if self.visualise:
                     visualise(output, prediction, pause=0.1)

        #The average distance over the entire test set is calculated
        dist = sum(filter(None,distance))/len(distance)
        #The distance is denormalised to cm's
        dist = dist*300
        if area1 is not None:
            print("Distance on test set within area1: {}\tarea2: {}\tis: {}cm".format(area1,area2,dist))
        else:
            print("Distance on test set is: {}cm".format(dist))
        print("Bias on x: {}\ton y: {}".format(sum(x)/len(x), sum(y)/len(y)))
        return dist
