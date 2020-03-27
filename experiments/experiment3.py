#Experiment 1 runs a sweep over all number of TX_inputs
#For each possible number of TX_inputs a model is trained
#The achieved distance on the val set is then plotted in function of the epoch for each model
def experiment3(args):
    print("Performing experiment 3")
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
