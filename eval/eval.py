import torch
from torch.utils.data import DataLoader
import numpy as np

from utils.modelUtils import setup_model
from utils.utils import calcDistance
from utils.utils import visualise
from utils.utils import calcBias
from utils.utils import makeHeatMap
from dataset.dataset import Data
from utils.utils import printProgBar

#Object to evaluate the performance of the model on the test set
class Eval_obj(object):
    def __init__(self, args, file=None):
        if args.verbose:
            print("Setting up eval object")
        #Initialising some variables
        self.device = args.device;  self.result_root = args.result_root
        self.visualise = args.visualise; self.verbose = args.verbose
        if file == None:
            file = 'model.pth'

        #Load model from file
        setup_model(self, file, reload_model=True)

        #Setup dataset and dataloaders
        test_dataset = Data(args.dataset_path['test'], self.blockage, self.rotations, self.cell_type, self.output_nf)
        heatmap_dataset = Data(args.dataset_path['heatmap'], self.blockage, self.rotations, self.cell_type, self.output_nf)
        self.test_data_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True, num_workers=4)
        self.heatmap_loader = DataLoader(heatmap_dataset, batch_size = self.batch_size, shuffle=True, num_workers=4)

        #Setting the model to evaluation mode
        self.model.eval()
        self.result_root = args.result_root

    #Calculates the distance between predicted and real position of the samples in the test set
    #If visualise is enables these distances are visualy plotted
    def demo(self):
        if self.verbose:
            print("Running demo, testing performance on test set.")
        #Init variables to store distances
        dist_dict = {'2D': [], 'z': [], '3D': []}
        x = []; y = []
        #Get prediction error for every batch in validation set
        for i, data in enumerate(self.test_data_loader):
            with torch.no_grad():
                if self.verbose:
                    print(printProgBar(i,len(self.test_data_loader)),end='\r')
                input = data[0].to(self.device)
                output = data[1].to(self.device)
                prediction = self.model(input)

                #Calculate the distance between predicted and target points
                dist = calcDistance(prediction, output)
                x1, y1 = calcBias(prediction, output)
                x.append(x1); y.append(y1)

                dist_dict['2D'].append(dist['2D'])
                if len(dist) == 3:
                    dist_dict['z'].append(dist['z']); dist_dict['3D'].append(dist['3D'])

                if self.visualise:
                     visualise(output, prediction, pause=0.1)
        print("")
        #The average error over the entire set is calculated
        for key in dist_dict:
            if len(dist_dict[key]) > 0:
                dist_dict[key] = sum(dist_dict[key])/len(dist_dict[key])
            else:
                dist_dict[key] = np.inf

        print("Distance on val set 2D: {} cm\t Z: {} cm\t 3D: {} cm".format(
                round(dist_dict['2D'],2),round(dist_dict['z'],2), round(dist_dict['3D'],2)))

        print("Bias on x: {}\ton y: {}".format(sum(x)/len(x), sum(y)/len(y)))


    def heatMap(self):
        dist_2D = []
        if self.verbose:
            print("Creating heatmap")
        map = np.full((300,300),np.inf)
        mapz = np.full((300,300),np.inf)
        for i, data in enumerate(self.heatmap_loader):
            with torch.no_grad():
                if self.verbose:
                    print(printProgBar(i,len(self.heatmap_loader)),end='\r')
                input = data[0].to(self.device)
                output = data[1].to(self.device)
                prediction = self.model(input)
                for it in range(0,len(input)):
                    pos = output[it]
                    x = int(round(pos[0].item()*300)); y = int(round(pos[1].item()*300))
                    dist = torch.sqrt((prediction[it][0]-pos[0])**2+(prediction[it][1]-pos[1])**2)
                    dist_z = torch.sqrt((prediction[it][2]-pos[2])**2)
                    map[x,y] = dist*300
                    mapz[x,y] = dist_z*200
                    dist_2D.append(dist*300)
        print("")
        if self.verbose:
            print("Heatmaps stored at {}".format(self.result_root))
        makeHeatMap(map, 'TX_config_'+str(self.TX_config)+'.pdf', 'Prediction error (cm)', self.result_root)
        makeHeatMap(mapz, 'TX_config_'+str(self.TX_config)+'_height.pdf', 'Height prediction error (cm)', self.result_root)
        return {'map': map, 'TX_conf': self.TX_config, 'dist': sum(dist_2D)/len(dist_2D)}
