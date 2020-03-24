import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

root = os.getcwd()
dataroot = os.path.join(root,'dataset/database/data_True.data')

with open(dataroot, 'rb') as f:
    dict = pickle.load(f)

data = np.array(dict['data'])

print("Creating train data")
train_data = data[dict['train']]
input_train = np.array(train_data[:,0][:].tolist())
output_train = np.array(train_data[:,1][:].tolist())

print("Creating val data")
val_data = data[dict['val']]
input_val = np.array(val_data[:,0][:].tolist())
output_val = np.array(val_data[:,1][:].tolist())

print("Creating test data")
test_data = data[dict['test']]
input_test = np.array(test_data[:,0][:].tolist())
output_test = np.array(test_data[:,1][:].tolist())

print("Start fitting")

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, n_jobs=32)
rf.fit(input_train, output_train)

prediction = rf.predict(input_val)
distance = np.mean(np.sqrt((output_val-prediction)**2))
print("Distance on validation set first tree: ", distance)


prediction = rf.predict(input_test)
distance = np.mean(np.sqrt((output_test-prediction)**2))
print("Distance on test set first tree: ", distance)


clf = svm.SVR()
>>> clf.fit(X, y)
SVR()
>>> clf.predict([[1, 1]])
array([1.5])
