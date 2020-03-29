import os
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from eval.eval import Eval_obj
from utils.utils import makePlot

#Make plots for experiment 1 showing influence of architecture parameters
def plot_exp_1(result_root):
    #init list to store results
    dist = []
    root = os.path.join(result_root, 'checkpoints')
    files = os.listdir(root)
    #Get the required results
    for key in [32,64,128,256]:
        constraints = {'model_type': 'Type_1', 'nf':key}
        dist.append(getDist(root, files, constraints, 'hidden_layers'))

    data_labels = ['number of features = 32', 'number of features = 64', 'number of features = 128', 'number of features = 256']
    makePlot(dist, 'NF_infl.pdf', 'Error on validation set', ['Number of hidden layers', 'Accuracy 2D (cm)'], result_root, data_labels)

    dist = []
    #Get the required results
    for model in ['Type_1', 'Type_2']:
        for key in [128,256]:
            constraints = {'model_type': model, 'nf':key}
            dist.append(getDist(root, files, constraints, 'hidden_layers'))

    data_labels = ['Type 1: number of features = 128', 'Type 1: number of features = 256', 'Type 2: number of features = 128', 'Type 2: number of features = 256']
    makePlot(dist, 'type_infl.pdf', 'Error on validation set', ['Number of hidden layers', 'Distance (cm)'], result_root, data_labels)


def plot_exp_2(args):
    #Plots for experiment 2
    args.TX_input = 9
    args.nf = 256
    args.extra_layers = 3
    args.model_type = 'FC_expand'
    args.dynamic = True
    args.experiment=2
    pth = os.path.join(args.result_root, 'experiment_4')

    maps = []
    for i in range(1,7):
        args.TX_config = i
        args.result_root = os.path.join(pth, 'TX_config_' + str(i))
        evalObj = Eval_obj(args)
        maps.append(evalObj.heatMap(args.TX_config))

    fig, axs = plt.subplots(nrows=2, ncols=3)
    #fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    for i in range(0,6):
        ax = axs.flat[i]
        img = ax.imshow(maps[i], cmap='viridis', vmin=0, vmax=45, interpolation='nearest')
        if i ==3 or i == 4 or i == 5:
            ax.set_xlabel('x-axis (cm)')
        if i == 0 or i == 3:
            ax.set_ylabel('y-axis (cm)')
        ax.invert_yaxis()
        ax.set_title('Conf {}'.format(i+1))

    fig.suptitle('Prediction error (cm)')
    resultpath = os.path.join(pth, 'heatmap_comparison.pdf')
    plt.tight_layout()
    #fig.text(0.5, -0.05, "X-axis: (cm)", ha='center')
    #fig.text(-0.05, 0.5, "Y-axis: (cm)", va='center', rotation='vertical')
    fig.colorbar(img, ax=list(axs))
    plt.savefig(resultpath, bbox_inches='tight')
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=2)
    #fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    conv = [0,1,1,1]
    for i in [0, 3]:
        ax = axs.flat[conv[i]]
        img = ax.imshow(maps[i], cmap='viridis', vmin=0, vmax=6.5, interpolation='nearest')
        ax.set_xlabel('x-axis (cm)')
        if i == 0:
            ax.set_ylabel('y-axis (cm)')
        ax.invert_yaxis()
        ax.set_title('Conf {}'.format(i+1))

    fig.suptitle('Prediction error (cm)')
    resultpath = os.path.join(pth, 'heatmap_comparison_high.pdf')
    plt.tight_layout()
    #fig.text(0.5, -0.05, "X-axis: (cm)", ha='center')
    #fig.text(-0.05, 0.5, "Y-axis: (cm)", va='center', rotation='vertical')
    fig.colorbar(img, ax=list(axs))
    plt.savefig(resultpath, bbox_inches='tight')
    plt.close()

#Makes plot for experiment 3 showing influence of Number of TX inputs
def plot_exp_3(result_root):
    #init list to store results
    dist = []
    root = os.path.join(result_root, 'checkpoints')
    files = os.listdir(root)
    #Get the required results
    for i in range(1,37):
        constraints = {'TX_input':key}
        dist.append(getDist(root, files, constraints, 'TX_input'))

    makePlot(dist, 'Best_TX_input.pdf', 'Error on validation set', ['Number of TX', 'Accuracy 2D (cm)'],  result_root)


#Makes plot for experiment 4 showing influence of amount of blockage
def plot_exp_4(result_root):
    #init list to store results
    dist = []
    root = os.path.join(result_root, 'checkpoints')
    files = os.listdir(root)
    #Get the required results
    for i in range(1,11):
        constraints = {'blockage':0.1*i}
        dist.append(getDist(root, files, constraints, 'blockage'))

    makePlot(dist, 'influence_blockage.pdf', 'Error on validation set', ['Amount of blockage', 'Accuracy 2D (cm)'],  result_root)


#Sort the first list according to second list
def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z

#Retrieve the min_distance of all files depending on constraints
#also sort retreived distances depending on sorting parameter
def getDist(root, files, constraints, sort_par):
    dist = []; sorter = []
    for file in files:
        if 'task' in file:
            cp = torch.load(os.path.join(root,file))
            #If all constraints are satisfied by checkpoint then add distance to list
            if all([cp[key] == constraints[key] for key in constraints]):
                dist.append(cp['min_distance'])
                sorter.append(cp[sort_par])
    dist = sort_list(dist,sorter)
    dist.insert(0,np.inf)
    return dist
