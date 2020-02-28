import os

from main import main
from utils.utils import makePlot

#Experiment 1 runs a sweep over all number of TX_inputs
#For each possible number of TX_inputs a model is trained
#The achieved distance on the val set is then plotted in function of the epoch for each model
def experiment1(args):
    print("Performing experiment 1")
    #Initialise some variables
    val_dist = [] #Holds all distances on the val set during training
    test_dist = []
    data_labels = []

    #Setup dir for all results of experiment 1
    pth = os.path.join(args.result_root, 'experiment_1_unit_cell')
    if not os.path.exists(pth):
        os.mkdir(pth)

    #Loop over all possible TX_inputs
    for blockage in [0,0.1,0.2,0.3,0.4,0.5,0.6]:
        #Setup result root
        args.result_root = os.path.join(pth, 'blockage_' + str(blockage))
        if not os.path.exists(args.result_root):
            os.mkdir(args.result_root)

        args.blockage = blockage
        data_labels.append('Blockage percentage: {}'.format(blockage))

        #Train the model for the specific TX_input
        args.is_train = True
        val_dist.append(main(args))

        #If model is trained check achieved distance on test set
        args.is_train = False
        test_dist.append(main(args))

    #Create plot comparing the performance
    filename = 'TX_input_distance.pdf'
    title = 'Performance improvement by using more TX to predict the RX position.'
    labels = ['Epoch', 'Distance (cm)']
    makePlot(val_dist, filename, title, labels, pth, data_labels)
    filename = 'Best_TX_input.pdf'
    title = 'Distance error in function of number of TX'
    labels = ['Number of TX', 'Distance (cm)']
    makePlot(test_dist, filename, title, labels, pth)

    print("Distance on test set for all models: ", test_dist)

def experiment2(args):
    val_dist = []
    test_dist = []
    data_labels = []

    #Setup dir for all results of experiment 1
    pth = os.path.join(args.result_root, 'experiment_2_unit_cell')
    if not os.path.exists(pth):
        os.mkdir(pth)
    for rot in [False]:
        args.result_root = os.path.join(pth, 'rotation_'+str(rot))
        if not os.path.exists(args.result_root):
            os.mkdir(args.result_root)
        args.rotations = rot
        data_labels.append('Rotations: {}'.format(rot))

        args.is_train = True
        val_dist.append(main(args))
        args.is_train = False
        test_dist.append(main(args))

    #Create plot comparing the performance
    filename = 'data_augmentation.pdf'
    title = 'Performance improvement by adding rotation data augmentation.'
    labels = ['Epoch', 'Distance (cm)']
    makePlot(val_dist, filename, title, labels, pth, data_labels)
    filename = 'Best_TX_input.pdf'
    title = 'Distance error in function of number of TX'
    labels = ['Number of TX', 'Distance (cm)']
    makePlot(test_dist, filename, title, labels, pth)

    print("Distance on test set for all models: ", test_dist)

def experiment(args):
    if args.experiment == 1:
        experiment1(args)
    elif args.experiment == 2:
        experiment2(args)
