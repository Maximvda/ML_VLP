import os
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from utils.utils import makePlot

#Make plots for experiment 1 showing influence of Unit cel on blockage
#and the influence of rotations on blockage
def plot_exp_1(result_root):
    #init list to store results
    dist = []
    root = os.path.join(result_root, 'checkpoints')
    files = os.listdir(root)
    #Get the required results
    for key in ['3x3', '2x2']:
        constraints = {'cell_type': key, 'rotations': False}
        acc_list = getDist(root, files, constraints, 'blockage')
        dist.append(acc_list)

    data_labels = ['cell type = 3x3', 'cell type = 2x2']
    makePlot(dist, 'type_infl_blockage.pdf', 'Error on validation set', ['Amount of blockage (%)', 'Accuracy 2D (cm)'], result_root, data_labels, ticks=np.linspace(0,1,11))
    print("Plots for experiment 1 saved to {}".format(result_root))


    #init list to store results
    dist = []
    #Get the required results
    for key in [True, False]:
        constraints = {'cell_type': '3x3', 'rotations': key}
        acc_list = getDist(root, files, constraints, 'blockage')
        dist.append(acc_list)

    data_labels = ['Rotations = True', 'Rotations = False']
    makePlot(dist, 'rotation_infl_blockage.pdf', 'Error on validation set', ['Amount of blockage (%)', 'Accuracy 2D (cm)'], result_root, data_labels, ticks=np.linspace(0,1,11))
    print("Plots for experiment 1 saved to {}".format(result_root))

    dict = get_best_three(root, files)
    print("Best:\tType: {}\tNF: {}\tHidden_layers: {}\tScore: 2D {}, 3D {}".format(
        dict['1']['model_type'], dict['1']['nf'],dict['1']['hidden_layers'], dict['1']['min_dist']['2D'], dict['1']['min_dist']['3D']))
    print("Second:\tType: {}\tNF: {}\tHidden_layers: {}\tScore: 2D {}, 3D {}".format(
        dict['2']['model_type'], dict['2']['nf'],dict['2']['hidden_layers'], dict['2']['min_dist']['2D'], dict['2']['min_dist']['3D']))
    print("Third:\tType: {}\tNF: {}\tHidden_layers: {}\tScore: 2D {}, 3D {}".format(
        dict['3']['model_type'], dict['3']['nf'],dict['3']['hidden_layers'], dict['3']['min_dist']['2D'], dict['3']['min_dist']['3D']))

    return dict['files']



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
