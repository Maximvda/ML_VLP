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

from utils.utils import getCelPosition

def calcMap(args,map_split):
    error = []
    pred_error = []
    data = setup_database(args.args, map_split)
    if 'map_grid' == map_split:
        map = np.full((300,300),np.inf)
        mapz = np.full((300,300),np.inf)
    else:
        map = np.full((250,250),np.inf)
        mapz = np.full((250,250),np.inf)

    for i, data in enumerate(data):
        with torch.no_grad():
            input = data[0].to(args.device)
            if 'map_grid' == map_split:
                output = data[1][0].to(args.device)
                cels = data[1][1]
            else:
                output = data[1].to(args.device)
            prediction = args.best_model(input)
            for it in range(0,len(input)):
                pos = output[it]
                if 'map_grid' == map_split:
                    pos_cel = getCelPosition(cels[it].item())
                    x = int(round(pos[0].item()*125+pos_cel[0]/10)); y = int(round(pos[1].item()*125+pos_cel[1]/10))
                else:
                    x = int(round(pos[0].item()*125+125+1e-04)); y = int(round(pos[1].item()*125+125+1e-04))

                dist = torch.sqrt((prediction[it][0]-pos[0])**2+(prediction[it][1]-pos[1])**2).item()
                error.append(dist)
                dist_z = prediction[it][2].item()
                pred_error.append(dist_z)
                map[x,y] = dist*125
                mapz[x,y] = dist_z*125

    error = (sum(error)/len(error))*125
    pred_error = (sum(pred_error)/len(pred_error))*125

    makeHeatMap(map, str(map_split)+'.pdf', 'Prediction error (cm)', error, args.result_root)
    makeHeatMap(mapz, str(map_split)+'_height.pdf', 'Height prediction error (cm)',pred_error, args.result_root)

#Object to evaluate the performance of the model on the test set
class eval_obj(object):
    def __init__(self, args):
        print("Setting up eval object")
        #Initialising some variables
        self.test_data_loader = setup_database(args, 'test')
        self.args = args
        self.device = args.device
        output_nc = 3 if args.estimate_error else 2
        self.best_model = model(9,output_nc, args.model_type, args.nf, args.extra_layers).to(args.device)
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
        dist = dist*125
        print("\nDistance on test set\tis: {}cm".format(dist))
        print("Bias on x: {}\ton y: {}".format(sum(x)/len(x), sum(y)/len(y)))
        return dist

    def heatMap(self):
        calcMap(self,'map_grid')
        calcMap(self,'map_7')
        calcMap(self,'map_25')
