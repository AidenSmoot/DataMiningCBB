import numpy as np
import pandas as pd
import numpy as np

offset = 2013

cbb = pd.read_csv('finalCBB.csv')
cbb = cbb.drop(cbb.columns[0:5],axis=1)
thetas = []

# SGD Regression with L2 penalty (Ridge Regression) from scratch
def sgd_regression_l2(X, y, learning_rate=0.01, n_epochs=100, alpha=0.1, max_iter=1000):
    m, n = X.shape
    theta = np.random.randn(n, 1)  # initialize weights randomly
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi) + 2 * alpha * theta
            theta = theta - learning_rate * gradients
        if epoch == max_iter:
            break
    return theta

# Fit the model with max_iter parameter
alpha = 0.1  # L2 regularization parameter (lambda)
max_iter = 500  # Maximum number of iterations

for year in range(2013, 2024):
    cbb_year = cbb[cbb['YEAR'] == year]
    cbb_year = cbb_year.drop(['YEAR'],axis=1)
    cbb_year_input = cbb_year.drop(['WR'], axis=1)
    cbb_year_input_array = cbb_year_input.values
    cbb_year_array_bias =  np.c_[np.ones((len(cbb_year_input_array), 1)), cbb_year_input_array]
    year_output = cbb_year.drop(cbb_year.columns[0:-1], axis=1)
    cbb_year_output_array = year_output.values
    thetas.append(sgd_regression_l2(cbb_year_array_bias, cbb_year_output_array, learning_rate=0.01, n_epochs=100, alpha=alpha, max_iter=max_iter))

inputs = cbb.drop(['WR'], axis=1)
output = cbb.drop(cbb.columns[0:-1], axis=1)
inputs = inputs.drop(['YEAR'],axis=1)
cbb_array = inputs.values
output_array = output.values
# Add bias term to the features
X_train_bias = np.c_[np.ones((len(cbb_array), 1)), cbb_array]
theta = sgd_regression_l2(X_train_bias, output_array, learning_rate=0.01, n_epochs=100, alpha=alpha, max_iter=max_iter)
thetas.append(theta)

# here, year 2024 will be the prediction taking into account every year
def make_prediction(thetas, x, year):
    x_bias = np.insert(x, 0, 1.0)
    y = x_bias.dot(thetas[year-offset])
    return y[0]

#enter stats here
x = np.array([0.7063,-0.0979183,2.31598,1.75049])
# y = make_prediction(theta, x)
# print(y)

print(make_prediction(thetas, x, 2014))
