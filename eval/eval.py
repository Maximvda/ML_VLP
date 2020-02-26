import torch
import torch.nn as nn
import os
import numpy as np

from dataset.setup_database import setup_database
from utils.modelUtils import loadBestModel
from utils.utils import calcDistance
from utils.utils import visualise
from utils.utils import calcBias
from utils.utils import makeHeatMap
from utils.utils import printProgBar
from models.architecture import model

#Object to evaluate the performance of the model on the test set
class eval_obj(object):
    def __init__(self, args):
        print("Setting up eval object")
        #Initialising some variables
        self.test_data_loader = setup_database(args, 'test')
        if args.experiment==2:
            self.heatMap_data = setup_database(args, 'heatmap_grid')
        self.device = args.device
        self.best_model = model(9, args.model_type, args.nf, args.extra_layers).to(args.device)
        self.visualise = args.visualise

        loadBestModel(args.result_root, self.best_model, args.device)

        #Setting the model to evaluation mode
        self.best_model.eval()
        self.result_root = args.result_root

    #Calculates the distance between predicted and real position of the samples in the test set
    #If visualise is enables these distances are visualy plotted
    def demo(self):
        distance = []
        dist_height = []
        x = []; y = []
        for i, data in enumerate(self.test_data_loader):
            printProgBar(i,len(self.test_data_loader))
            with torch.no_grad():
                #forward batch of test data through the network
                input = data[0].to(self.device)
                output = data[1].to(self.device)
                prediction = self.best_model(input)
                dist = calcDistance(prediction, output)
                distance.append(dist)
                x1, y1 = calcBias(prediction, output)
                x.append(x1); y.append(y1)

                if self.visualise:
                     visualise(output, prediction, pause=0.1)

        #The average distance over the entire test set is calculated
        dist = sum(filter(None,distance))/len(distance)
        #The distance is denormalised to cm's
        dist = dist*75
        print("\nDistance on test set\tis: {}cm".format(dist))
        print("Bias on x: {}\ton y: {}".format(sum(x)/len(x), sum(y)/len(y)))
        return dist

    def heatMap(self, title):
        map = np.full((300,300),np.inf)
        mapz = np.full((300,300),np.inf)
        for i, data in enumerate(self.heatMap_data):
            with torch.no_grad():
                input = data[0].to(self.device)
                output = data[1].to(self.device)
                prediction = self.best_model(input)
                for it in range(0,len(input)):
                    pos = output[it]
                    x = int(round(pos[0].item()*300)); y = int(round(pos[1].item()*300))
                    dist = torch.sqrt((prediction[it][0]-pos[0])**2+(prediction[it][1]-pos[1])**2)
                    dist_z = torch.sqrt((prediction[it][2]-pos[2])**2)
                    map[x,y] = dist*300
                    mapz[x,y] = dist_z*200

        makeHeatMap(map, 'TX_config_'+str(title)+'.pdf', 'Prediction error (cm)', self.result_root)
        makeHeatMap(mapz, 'TX_config_'+str(title)+'_height.pdf', 'Height prediction error (cm)', self.result_root)
        return map
