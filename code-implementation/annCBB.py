from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data reading
cbb = pd.read_csv('Basketball Data Mining/finalCBB.csv')
cbb = cbb.drop(cbb.columns[0:5],axis=1)
cbb = cbb.drop(['YEAR'],axis=1)
cbb_inputs = cbb.drop(['WR'], axis=1)
cbb_outputs = cbb.drop(cbb.columns[0:-1], axis=1)
# model creation
clf = MLPRegressor(solver='lbfgs', max_iter=1000, alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

cbb_inputs_np = cbb_inputs.to_numpy() #conversion into numpy array
cbb_outputs_np = cbb_outputs.to_numpy().ravel()

#creates testing and training splits
def get_testing_split(list_to_split, testing_index, n_split):
    testing_index_beginning = testing_index*n_split
    testing_index_ending = testing_index_beginning+n_split
    testing_fold = list_to_split[testing_index_beginning:testing_index_ending]
    training_fold = np.concatenate([list_to_split[:testing_index], list_to_split[testing_index_ending:]])
    return testing_fold, training_fold


def cross_val(input_data, out_data, n_folds):
    n_split = int(len(out_data)/n_folds)
    rmses = []
    for i in range(n_folds):
        testing_input, training_input = get_testing_split(input_data, i,n_split)
        testing_output, training_output = get_testing_split(out_data, i,n_split)
        clf.fit(training_input, training_output)
        predictions = clf.predict(testing_input)
        # processing rmse
        squared_diff = (testing_output-predictions)**2 
        rmses.append(np.sqrt(np.sum(squared_diff)/len(testing_input))) #rmse formula
    return rmses

np.savetxt("ann_rmses.csv", cross_val(cbb_inputs_np, cbb_outputs_np, 20), delimiter=',')