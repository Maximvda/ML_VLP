import torch
import torch.nn as nn
import os
import numpy as np

from dataset.setup_database import setup_database
from utils.modelUtils import initModel
from utils.modelUtils import loadBestModel
from utils.utils import calcDistance
from utils.utils import visualise
from utils.utils import calcBias
from utils.utils import makeHeatMap

#Object to evaluate the performance of the model on the test set
class eval_obj(object):
    def __init__(self, args):
        print("Setting up eval object")
        #Initialising some variables
        self.test_data_loader = setup_database(args, 'test')
        if args.experiment==2:
            self.heatMap_data = setup_database(args, 'heatmap_grid')
        self.device = args.device
        self.best_model = initModel(self.test_data_loader, args.model_type, args.nf, args.extra_layers).to(args.device)
        self.visualise = args.visualise

        loadBestModel(args.result_root, self.best_model, args.device)

        #Setting the model to evaluation mode
        self.best_model.eval()
        self.result_root = args.result_root

    #Calculates the distance between predicted and real position of the samples in the test set
    #If visualise is enables these distances are visualy plotted
    def demo(self, area1=None, area2=None):
        distance = []
        dist_height = []
        x = []; y = []
        for i, data in enumerate(self.test_data_loader):
            with torch.no_grad():
                #forward batch of test data through the network
                input = data[0].to(self.device)
                output = data[1].to(self.device)
                prediction = self.best_model(input)
                dist, dist_z = calcDistance(prediction, output, area1, area2)
                distance.append(dist)
                dist_height.append(dist_z)
                x1, y1 = calcBias(prediction, output)
                x.append(x1); y.append(y1)

                if self.visualise:
                     visualise(output, prediction, pause=0.1)

        #The average distance over the entire test set is calculated
        dist = sum(filter(None,distance))/len(distance)
        dist_z = sum(dist_height)/len(dist_height)
        #The distance is denormalised to cm's
        dist = dist*300
        dist_z = dist_z*200
        if area1 is not None:
            print("Distance on test set within area1: {}\tarea2: {}\tis: {}cm\theight: {}".format(area1,area2,dist,dist_z))
        else:
            print("Distance on test set is: {}cm\theight: {}".format(dist, dist_z))
        print("Bias on x: {}\ton y: {}".format(sum(x)/len(x), sum(y)/len(y)))
        return dist

    def heatMap(self, title):
        map = np.full((3000,3000),np.inf)
        mapz = np.full((3000,3000),np.inf)
        for i, data in enumerate(self.heatMap_data):
            with torch.no_grad():
                input = data[0].to(self.device)
                output = data[1].to(self.device)
                prediction = self.best_model(input)
                for it in range(0,len(input)):
                    sample = input[it]
                    pos = sample[1]
                    x = int(pos[0]*3000); y = int(pos[1]*3000)

                    dist, dist_z = calcDistance(prediction[it], output[it])
                    map[x,y] = dist*300
                    mapz[x,y] = dist_z*200

        makeHeatMap(map, title+'.png', 'Prediction error: (cm)', self.result_root)
        makeHeatMap(mapz, title+'_height.png', 'Height prediction error: (cm)', self.result_root)
