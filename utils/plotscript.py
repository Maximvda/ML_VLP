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

def plotExp2(args):
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
        evalObj = eval_obj(args)
        maps.append(evalObj.heatMap(args.TX_config))

    fig, axs = plt.subplots(nrows=2, ncols=3)
    #fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    for i in range(0,6):
        ax = axs.flat[i]
        img = ax.imshow(maps[i], cmap='viridis', vmin=0, vmax=45, interpolation='nearest')
        if i ==3 or i == 4 or i == 5:
            ax.set_xlabel('x-axis: (cm)')
        if i == 0 or i == 3:
            ax.set_ylabel('y-axis: (cm)')
        ax.invert_yaxis()
        ax.set_title('Conf {}'.format(i+1))

    fig.suptitle('Prediction error: (cm)',y=1.05)
    resultpath = os.path.join(pth, 'heatmap_comparison.pdf')
    plt.tight_layout()
    #fig.text(0.5, -0.05, "X-axis: (cm)", ha='center')
    #fig.text(-0.05, 0.5, "Y-axis: (cm)", va='center', rotation='vertical')
    fig.colorbar(img, ax=list(axs))
    plt.savefig(resultpath, bbox_inches='tight')
    plt.close()

def plotscript(args):
    #Plots for experiment 1
    pth = os.path.join(args.result_root, 'experiment_1')
    list = [np.inf]
    for i in range(36):
        list.append('TX_input_'+str(i+1))
    dist = getDist(pth, list)
    makePlot(dist, 'Best_TX_input.pdf', 'Error on validation set', ['Number of TX', 'Distance (cm)'], pth)
    #plotExp2(args)

    #Plots for experiment 3
    pth = os.path.join(args.result_root,'experiment_3')

    list = ['FC_32_0', 'FC_32_1', 'FC_32_2', 'FC_32_3', 'FC_32_4']
    dist_FC_32 = getDist(pth, list)
    list = ['FC_64_0', 'FC_64_1', 'FC_64_2', 'FC_64_3', 'FC_64_4']
    dist_FC_64 = getDist(pth, list)
    list = ['FC_128_0', 'FC_128_1', 'FC_128_2', 'FC_128_3', 'FC_128_4']
    dist_FC_128 = getDist(pth, list)
    list = ['FC_256_0', 'FC_256_1', 'FC_256_2', 'FC_256_3', 'FC_256_4']
    dist_FC_256 = getDist(pth, list)
    dist = [dist_FC_32, dist_FC_64, dist_FC_128, dist_FC_256]
    data_labels = ['nf = 32', 'nf = 64', 'nf = 128', 'nf=256']

    makePlot(dist, 'NF_infl.pdf', 'Error on validation set', ['Number of extra layers', 'Distance (cm)'], pth, data_labels)



    list = ['FC_128_0', 'FC_128_1', 'FC_128_2', 'FC_128_3', 'FC_128_4']
    dist_FC_32 = getDist(pth, list)
    list = ['FC_256_0', 'FC_256_1', 'FC_256_2', 'FC_256_3', 'FC_256_4']
    dist_FC_64 = getDist(pth, list)
    list = ['FC_expand_128_0', 'FC_expand_128_1', 'FC_expand_128_2', 'FC_expand_128_3', 'FC_expand_128_4']
    dist_FC_128 = getDist(pth, list)
    list = ['FC_expand_256_0', 'FC_expand_256_1', 'FC_expand_256_2', 'FC_expand_256_3', 'FC_expand_256_4']
    dist_FC_256 = getDist(pth, list)
    dist = [dist_FC_32, dist_FC_64, dist_FC_128, dist_FC_256]
    data_labels = ['Type 1: nf = 128', 'Type 1: nf = 256', 'Type 2: nf = 128', 'Type 2: nf=256']

    makePlot(dist, 'type_infl.pdf', 'Error on validation set', ['Number of extra layers', 'Distance (cm)'], pth, data_labels)
