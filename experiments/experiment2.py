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
