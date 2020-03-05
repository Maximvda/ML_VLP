import os
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from eval.eval import eval_obj
from utils.utils import makePlot

def getDist(root, list):
    dist = []
    for it in list:
        pth = os.path.join(root, it)
        cp = torch.load(os.path.join(pth,'checkpoint.pth'))
        dist.append(min(cp['distance']))
    return dist

def checktest(args):
    pth = os.path.join(args.result_root, 'experiment_1_unit_cell')
    dist = {}
    for j in [False, True]:
        dist[j] = []
        for i in [0,0.1,0.2,0.3,0.4,0.5,0.6]:
            args.result_root = os.path.join(pth, 'blockage_' + str(i) + '_rot_' + str(j))
            args.blockage = i
            args.rotations = j
            evalObj = eval_obj(args)
            dist[j].append(evalObj.demo())
            evalObj.heatMap()

    distance = [dist[False], dist[True]]
    data_labels = ['Data augmentation: False', 'Data augmentation: True']
    makePlot(distance, 'exp_test.pdf', 'Error on test set', ['Blockage probability', 'Distance (cm)'], pth, data_labels)


def plotExp2(args):
    #Plots for experiment 2
    args.experiment=2
    pth = os.path.join(args.result_root, 'experiment_4')

    maps = []
    for i in range(1,7):
        args.TX_config = i
        args.result_root = os.path.join(pth, 'TX_config_' + str(i))
        evalObj = eval_obj(args)
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

def plotscript(args):
    root = args.result_root
    #Plots for experiment 1
    pth = os.path.join(args.result_root, 'experiment_1_unit_cell')

    list = ['blockage_0_rot_False', 'blockage_0.1_rot_False', 'blockage_0.2_rot_False','blockage_0.3_rot_False',
            'blockage_0.4_rot_False', 'blockage_0.5_rot_False', 'blockage_0.6_rot_False']
    dist_rot_false = getDist(pth, list)
    list = ['blockage_0_rot_True', 'blockage_0.1_rot_True', 'blockage_0.2_rot_True','blockage_0.3_rot_True',
            'blockage_0.4_rot_True', 'blockage_0.5_rot_True', 'blockage_0.6_rot_True']
    dist_rot_true = getDist(pth, list)
    dist = [dist_rot_false, dist_rot_true]
    data_labels = ['Data augmentation: False', 'Data augmentation: True']

    makePlot(dist, 'exp.pdf', 'Error on validation set', ['Blockage probability', 'Distance (cm)'], pth, data_labels)

    checktest(args)
