import torch
import torch.nn as nn
import os

from dataset.setup_database import setup_database
from utils.modelUtils import initModel
from utils.modelUtils import loadBestModel
from utils.utils import calcDistance
from utils.utils import visualise

class eval_obj(object):
    def __init__(self, args):
        print("Setting up eval object")
        self.test_data_loader = setup_database(args, 'test')
        self.device = args.device
        self.best_model = initModel(self.test_data_loader, args.nf).to(args.device)
        self.visualise = args.visualise

        loadBestModel(args.result_root, self.best_model, args.device)

        self.best_model.eval()

    def demo(self):
        distance = []
        for i, data in enumerate(self.test_data_loader):
            with torch.no_grad():
                #forward batch of test data through the network
                input = data[0].type(torch.FloatTensor).to(self.device)
                output = data[1].to(self.device)
                prediction = self.best_model(input)[:,:,0,0]
                distance.append(calcDistance(prediction, output))

                if self.visualise:
                     visualise(output, prediction, pause=0.1)

        #The average distance over the entire test set is calculated
        dist = sum(distance)/len(distance)
        #The distance is denormalised to cm's
        dist = dist*300
        print("Distance on test set: {}cm".format(dist))
