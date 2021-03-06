import os
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

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
        acc_list = getDist(root, files, constraints, 'hidden_layers')
        dist.append(acc_list)

    data_labels = ['number of features = 32', 'number of features = 64', 'number of features = 128', 'number of features = 256']
    x_ticks = [1,2,3,4,5]
    makePlot(dist, 'NF_infl.pdf', 'Error on validation set', ['Number of hidden layers', 'Accuracy 2D (cm)'], result_root, data_labels, ticks=x_ticks)

    dist = []
    #Get the required results
    for model in ['Type_1', 'Type_2']:
        for key in [128,256]:
            constraints = {'model_type': model, 'nf':key}
            acc_list = getDist(root, files, constraints, 'hidden_layers')
            dist.append(acc_list)

    data_labels = ['Type 1: number of features = 128', 'Type 1: number of features = 256', 'Type 2: number of features = 128', 'Type 2: number of features = 256']
    makePlot(dist, 'type_infl.pdf', 'Error on validation set', ['Number of hidden layers', 'Accuracy 2D (cm)'], result_root, data_labels, ticks=x_ticks)
    print("Plots for experiment 1 saved to {}".format(result_root))

    dict = get_best_three(root, files)
    print("Best:\tType: {}\tNF: {}\tHidden_layers: {}\tScore: 2D {}, 3D {}".format(
        dict['1']['model_type'], dict['1']['nf'],dict['1']['hidden_layers'], dict['1']['min_dist']['2D'], dict['1']['min_dist']['3D']))
    print("Second:\tType: {}\tNF: {}\tHidden_layers: {}\tScore: 2D {}, 3D {}".format(
        dict['2']['model_type'], dict['2']['nf'],dict['2']['hidden_layers'], dict['2']['min_dist']['2D'], dict['2']['min_dist']['3D']))
    print("Third:\tType: {}\tNF: {}\tHidden_layers: {}\tScore: 2D {}, 3D {}".format(
        dict['3']['model_type'], dict['3']['nf'],dict['3']['hidden_layers'], dict['3']['min_dist']['2D'], dict['3']['min_dist']['3D']))

    return dict['files']

#Make plots for experiment 2 plotting difference between TX configurations
def plot_exp_2(result_root, dict_list):
    root = os.path.join(result_root, 'checkpoints')

    #Retrieve maps from dict
    maps = []
    for i in range(1,7):
        for dict in dict_list:
            if dict['TX_conf'] == i:
                maps.append(dict['map'])
                print("2D accuracy on heatmap split for TX_config {} is: {}".format(i,round(dict['dist'].item(),2)))

    fig, axs = plt.subplots(nrows=2, ncols=3)
    for i in range(0,6):
        ax = axs.flat[i]
        img = ax.imshow(maps[i], cmap='viridis', vmin=0, vmax=25, interpolation='nearest')
        if i ==3 or i == 4 or i == 5:
            ax.set_xlabel('x-axis (cm)')
        if i == 0 or i == 3:
            ax.set_ylabel('y-axis (cm)')
        ax.invert_yaxis()
        ax.set_title('Conf {}'.format(i+1))

    fig.suptitle('Prediction error (cm)')
    resultpath = os.path.join(result_root, 'heatmap_comparison.pdf')
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
        img = ax.imshow(maps[i], cmap='viridis', vmin=0, vmax=5.5, interpolation='nearest')
        ax.set_xlabel('x-axis (cm)')
        if i == 0:
            ax.set_ylabel('y-axis (cm)')
        ax.invert_yaxis()
        ax.set_title('Conf {}'.format(i+1))

    fig.suptitle('Prediction error (cm)')
    resultpath = os.path.join(result_root, 'heatmap_comparison_high.pdf')
    plt.tight_layout()
    #fig.text(0.5, -0.05, "X-axis: (cm)", ha='center')
    #fig.text(-0.05, 0.5, "Y-axis: (cm)", va='center', rotation='vertical')
    fig.colorbar(img, ax=list(axs))
    plt.savefig(resultpath, bbox_inches='tight')
    plt.close()

#Makes plot for experiment 3 showing influence of Number of TX inputs
def plot_exp_3(result_root):
    #init list to store results
    root = os.path.join(result_root, 'checkpoints')
    files = os.listdir(root)
    #Get the required results
    dist = getDist(root, files, {}, 'TX_input')
    dist.insert(0,np.inf)

    #x x_ticks
    x_ticks = [i*3 for i in range(0,13)]
    x_ticks[0] = 1

    makePlot(dist, 'Best_TX_input.pdf', 'Error on validation set', ['Number of TXs', 'Accuracy 2D (cm)'],  result_root, ticks=x_ticks)


#Makes plot for experiment 4 showing influence of amount of blockage
def plot_exp_4(result_root):
    #init list to store results
    dist = []
    root = os.path.join(result_root, 'checkpoints')
    files = os.listdir(root)
    #Get the required results
    for i in range(1,11):
        constraints = {'blockage':0.1*i}
        dist.append(getDist(root, files, constraints, 'blockage')[0])

    makePlot(dist, 'influence_blockage.pdf', 'Error on validation set', ['Amount of blockage (%)', 'Accuracy 2D (cm)'],  result_root, ticks=np.linspace(10,100,10))

#Just print the obtained scores of both models
def plot_exp_5(result_root):
    #init list to store results
    root = os.path.join(result_root, 'checkpoints')
    files = os.listdir(root)
    files.sort()
    dist = getDist(root, files, {}, None)
    print("Training model on non-normalised inputs results in 2D accuracy of : {}".format(round(dist[0],3)))
    print("Training model on simulation data and validating on real data results in 2D accuracy of : {}".format(round(dist[1],3)))
    print("Training model on 2D prediction results in 2D accuracy of : {}".format(round(dist[2],3)))




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
            cp = torch.load(os.path.join(root,file),map_location=torch.device('cpu'))
            #If all constraints are satisfied by checkpoint then add distance to list
            if all([cp[key] == constraints[key] for key in constraints]):
                dist.append(cp['min_dist']['2D'])
                if not sort_par == None:
                    sorter.append(cp[sort_par])
    #Sort list according to the sorting parameter
    if not sort_par == None:
        dist = sort_list(dist,sorter)
    return dist

#Retrieve the parameters of three best performing models
#According to their min_distance on validation set
def get_best_three(root, files):
    checkpoints = []; sorter = []; file_names = []
    for file in files:
        if 'task' in file:
            cp = torch.load(os.path.join(root,file),map_location=torch.device('cpu'))
            checkpoints.append(cp)
            sorter.append(cp['min_dist']['2D'])
            file_names.append(file)
    checkpoints = sort_list(checkpoints,sorter)
    file_names = sort_list(file_names,sorter)

    best_files = []
    for file in file_names[0:3]:
        ext = file.split("-")[-1]
        best_files.append("checkpoints/best-"+ext)
    return {'1': checkpoints[0], '2': checkpoints[1], '3': checkpoints[2], 'files': best_files}
