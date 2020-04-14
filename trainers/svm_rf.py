import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from dataset.dataset import Data

#Train a random forest for the regression task
def RF(args):
    input_train, output_train = process_data(args,'train')
    input_val, output_val = process_data(args,'val')

    print("Start fitting")
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, n_jobs=8)
    rf.fit(input_train, output_train)

    print("Predicting and calculating performance")
    prediction = rf.predict(input_val)
    distance = np.mean(np.sqrt((output_val-prediction)**2))
    print("Distance on validation set with random forest: ", distance)

#Train three SVM for each position coordinate one
def SVM(args):
    input_train, output_train = process_data(args,'train')
    input_val, output_val = process_data(args,'val')

    print("Start fitting")
    svm_x = SVR();svm_y = SVR();svm_z = SVR();
    svm_x.fit(input_train, output_train[:,0])
    svm_y.fit(input_train, output_train[:,1])
    svm_z.fit(input_train, output_train[:,2])

    print("Predicting and calculating performance")
    pred_x = svm_x.predict(input_val);
    pred_y = svm_y.predict(input_val);
    pred_z = svm_z.predict(input_val);

    prediction = []
    for i in range(pred_x.shape[0]):
        prediction.append([pred_x[i], pred_y[i], pred_z[i]])

    prediction = np.array(prediction)
    distance = np.mean(np.sqrt((output_val-prediction)**2))
    print("Distance on validation set with SVM: ", distance)


def process_data(args,split):
    dataset = Data(args.dataset_path[split], args.TX_config, args.TX_input, args.blockage, args.output_nf)
    data = dataset.get_data()

    input_list = []; output_list = [];

    print("Creating {} data".format(split))
    for sample in data:
        input = sample[0]
        output_list.append(sample[1])

        #Get indices of blockage
        indices = np.random.choice(np.arange(len(input)), replace=False, size=int(args.blockage*len(input)))
        for ind in indices:
            input[ind] = -1

        input_list.append(input)

    return np.array(input_list), np.array(output_list)
