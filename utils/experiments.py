import os

from main import main
from utils.utils import makePlot

#Experiment 1 runs a sweep over all number of TX_inputs
#For each possible number of TX_inputs a model is trained
#The achieved distance on the val set is then plotted in function of the epoch for each model
def experiment1(args):
    print("Performing experiment 1")
    #Initialise some variables
    args.TX_config = 1
    val_dist = [] #Holds all distances on the val set during training
    test_dist = []
    data_labels = []

    #Setup dir for all results of experiment 1
    pth = os.path.join(args.result_root, 'experiment_1')
    if not os.path.exists(pth):
        os.mkdir(pth)

    #Loop over all possible TX_inputs
    for i in range(1,37):
        #Setup result root
        args.result_root = os.path.join(pth, 'TX_input_' + str(i))
        if not os.path.exists(args.result_root):
            os.mkdir(args.result_root)

        args.TX_input = i
        data_labels.append('Number of TX: {}'.format(i))

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
    print("Performing experiment 2")
    #Initialise some variables
    args.TX_input = 9
    args.nf = 256
    args.extra_layers = 3
    args.model_type = 'FC_expand'
    args.dynamic = True

    val_dist = [] #Holds all distances on the val set during training
    test_dist = []
    data_labels = []

    #Setup dir for all results of experiment 1
    pth = os.path.join(args.result_root, 'experiment_4')
    if not os.path.exists(pth):
        os.mkdir(pth)

    #Loop over all possible TX_inputs
    for i in range(1,7):
        #Setup result root
        args.result_root = os.path.join(pth, 'TX_config_' + str(i))
        if not os.path.exists(args.result_root):
            os.mkdir(args.result_root)

        args.TX_config = i
        data_labels.append('TX config: {}'.format(i))

        #Train the model for the specific TX_config
        args.is_train = True
        val_dist.append(main(args))

        #If model is trained check achieved distance on test set
        args.is_train = False
        test_dist.append(main(args))

    #Create plot comparing the performance
    filename = 'TX_config_distance.pdf'
    title = 'Influence of different TX configuartions on position estimation.'
    labels = ['Epoch', 'Distance (cm)']
    colors = ['blue', 'mediumseagreen', 'red', 'gold', 'orange', 'black']
    makePlot(val_dist, filename, title, labels, pth, data_labels, colors)

    print("Distance on test set for all models: ", test_dist)

def experiment3(args):
    print("Performing experiment 3")
    #Initialise some variables
    args.TX_config = 1
    args.TX_input = 36
    args.dynamic = False

    val_dist = [] #Holds all distances on the val set during training
    test_dist = []
    data_labels = []

    #Setup dir for all results of experiment 1
    pth = os.path.join(args.result_root, 'experiment_3')
    if not os.path.exists(pth):
        os.mkdir(pth)

    #Loop over all possible models
    for i in ['FC','FC_expand']:
        for j in [128, 256]:
            for k in [0,1,2,3,4]:
                args.model_type = i
                args.nf = j
                args.extra_layers = k
                print("Model type: {}\nFeatures: {}\nExtra layers: {}".format(
                args.model_type, args.nf, args.extra_layers
                ))

                label = '{}_{}_{}'.format(i,j,k)
                #Setup result root
                args.result_root = os.path.join(pth, label)
                if not os.path.exists(args.result_root):
                    os.mkdir(args.result_root)

                data_labels.append(label)

                #Train the model for the specific TX_config
                args.is_train = True
                val_dist.append(main(args))

                #If model is trained check achieved distance on test set
                args.is_train = False
                test_dist.append(main(args))

    #Create plot comparing the performance
    filename = 'Experiment3.pdf'
    title = 'Influence of different model architectures on performance.'
    labels = ['Epoch', 'Distance (cm)']
    makePlot(val_dist, filename, title, labels, pth, data_labels)

    print("Distance on test set for all models: ", test_dist)

def experiment(args):
    if args.experiment == 1:
        experiment1(args)
    elif args.experiment == 2:
        experiment2(args)
    elif args.experiment == 3:
        experiment3(args)
