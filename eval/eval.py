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
from utils.config import get_cel_center_position
from utils.utils import getCelPosition

#Object to evaluate the performance of the model on the test set
class Eval_obj(object):
    def __init__(self, args, file=None, blockage=False):
        if args.verbose:
            print("Setting up eval object")
        #Initialising some variables
        self.device = args.device;  self.result_root = args.result_root
        self.visualise = args.visualise; self.verbose = args.verbose
        self.dataset_path = args.dataset_path
        if file == None:
            file = 'model.pth'

        #Load model from file
        setup_model(self, file, reload_model=True)

        #Setup dataset and dataloaders
        self.test_dataset = Data(args.dataset_path['test'], self.blockage, self.rotations, self.cell_type, self.output_nf, real_block=blockage)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle=True, num_workers=0)

        #Setting the model to evaluation mode
        self.model.eval()
        self.result_root = args.result_root

    #Prints model properties
    def stateModel(self):
        print("Type: {}".format(self.cell_type))
        print("Blockage: {}".format(self.blockage))
        print("Rotations: {}".format(self.rotations))

    #Calculates the distance between predicted and real position of the samples in the test set
    #If visualise is enables these distances are visualy plotted
    def demo(self):
        if self.verbose:
            print("Running demo, testing performance on test set.")
        #Init variables to store distances
        dist_dict = {'2D': 0, 'z': 0, '3D': 0}
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
                dist = calcDistance(prediction, output, self.cell_type)
                x1, y1 = calcBias(prediction, output)
                x.append(x1); y.append(y1)

                dist_dict['2D'] += dist['2D']
                if len(dist) == 3:
                    dist_dict['z'] += dist['z']; dist_dict['3D'] += dist['3D']

                if self.visualise:
                     visualise(output, prediction, pause=0.1)
        print("")
        #The average error over the entire set is calculated
        for key in dist_dict:
            if dist_dict[key] == 0:
                dist_dict[key] = np.inf
            else:
                dist_dict[key] /= len(self.test_dataset)

        print("Distance on test set 2D: {} cm\t Z: {} cm\t 3D: {} cm".format(
                round(dist_dict['2D'],2),round(dist_dict['z'],2), round(dist_dict['3D'],2)))

        print("Bias on x: {}\ton y: {}".format(sum(x)/len(x), sum(y)/len(y)))


    def heatMap(self):
        if self.verbose:
            print("Creating Heatmaps")

        calcMap(self,'grid')
        calcMap(self,'train_map')
        calcMap(self,'test_map')

        if self.verbose:
            print("Heatmaps stored at {}".format(self.result_root))


def calcMap(args,map_split):
    error = 0
    test_error = []
    heatmap_dataset = Data(args.dataset_path[map_split], args.blockage, args.rotations, args.cell_type, args.output_nf)
    dataLoader = DataLoader(heatmap_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)

    center_pos = get_cel_center_position()[args.cell_type]
    max = int((300+np.sqrt(center_pos[0]**2+center_pos[1]**2))/10)

    if 'grid' == map_split:
        map = np.full((300,300),np.inf)
    else:
        map = np.full((int(max*2+1),int(max*2+1)),np.inf)

    for i, data in enumerate(dataLoader):
        with torch.no_grad():
            if args.verbose:
                print(printProgBar(i,len(dataLoader)),end='\r')
            input = data[0].to(args.device)

            if 'grid' == map_split:
                output = data[1][0].to(args.device)
                cels = data[1][1]
            else:
                output = data[1].to(args.device)

            prediction = args.model(input)
            for it in range(0,len(input)):
                pos = output[it]

                if 'grid' == map_split:
                    pos_cel = getCelPosition(cels[it].item())
                    x = int(round(pos[0].item()*max+pos_cel[0]/10+center_pos[0]/10)); y = int(round(pos[1].item()*max+pos_cel[1]/10+center_pos[1]/10))
                else:
                    x = int(round(pos[0].item()*max+max+1e-04)); y = int(round(pos[1].item()*max+max+1e-04))

                dist = calcDistance(prediction[it].unsqueeze(0), output[it].unsqueeze(0), args.cell_type)
                if map_split == 'test_map' and abs(x - max) < 50 and abs(y-max) < 50:
                    test_error.append(dist['2D'])
                #dist = torch.sqrt((prediction[it][0]-pos[0])**2+(prediction[it][1]-pos[1])**2)
                #dist_z = torch.sqrt((prediction[it][2]-pos[2])**2)
                map[x,y] = dist['2D']
                error += dist['2D']


    print("")
    error /= len(heatmap_dataset)
    if args.verbose:
        print("The average error over the entire heatmap {} is: {}".format(map_split, error))
    if args.verbose and map_split == 'test_map':
        print("The average error in 50x50 square equals: {}".format(sum(test_error)/len(test_error)))
    makeHeatMap(map, str(map_split)+'.pdf', 'Prediction error (cm)', error, args.result_root)
