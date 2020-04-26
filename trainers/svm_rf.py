import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from dataset.dataset import Data
from dataset.dataset import augment_data
from utils.config import get_cel_center_position

#Train a random forest for the regression task
def RF(args):
    input_train, output_train = process_data(args,'train')
    input_val, output_val = process_data(args,'val')

    print("Started fitting random forest")
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, n_jobs=8)
    rf.fit(input_train, output_train)

    print("Predicting and calculating performance")
    prediction = rf.predict(input_val)
    dist_dict = calc_dist(output_val, prediction)
    print("Distance on validation set with random forest: 2D {}\t 3D {}".format(dist_dict['2D'], dist_dict['3D']))

#Train three SVM for each position coordinate one
def SVM(args):
    input_train, output_train = process_data(args,'train')
    input_val, output_val = process_data(args,'val')

    print("Started fitting a SVM")
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
    dist_dict = calc_dist(output_val, prediction)
    print("Distance on validation set with SVM: 2D {}\t 3D {}".format(dist_dict['2D'], dist_dict['3D']))


def process_data(args,split):
    dataset = Data(args.dataset_path[split], args.blockage, args.rotations, args.cell_type, args.output_nf)
    data = dataset.get_data()

    input_list = []; output_list = [];

    print("Creating {} data".format(split))
    for sample in data:
        input = sample[0]
        output = sample[1]

        input, output = augment_data(input, output, args.rotations, args.blockage, False, args.cell_type)

        input_list.append(input)
        output_list.append(output)

    return np.array(input_list), np.array(output_list)

def calc_dist(x,y):
    center_pos = get_cel_center_position()[cell_type]
    norm = (550+np.sqrt(center_pos[0]**2+center_pos[1]**2))/10

    dist_2D = np.sqrt((x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2)*norm
    dist_2D = np.mean(dist_2D)
    if len(x[0]) == 3:
        dist_3D = np.sqrt(((x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2)*norm**2+((x[:,2]-y[:,2])**2)*200**2)
        dist_3D = np.mean(dist_3D)
        return {'2D': dist_2D, '3D': dist_3D}
    else:
        return {'2D': dist_2D, '3D': np.inf}
