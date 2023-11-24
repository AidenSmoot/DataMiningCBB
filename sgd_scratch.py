import numpy as np
import pandas as pd
import numpy as np

cbb = pd.read_csv('finalCBB.csv')
cbb = cbb.drop(cbb.columns[0:5],axis=1)
cbb = cbb.drop(['YEAR'],axis=1)
inputs = cbb.drop(['WR'], axis=1)
rowCount = inputs.shape[0]
output = cbb.drop(cbb.columns[0:-1], axis=1)

cbb_array = cbb.values
output_array = output.values

# Add bias term to the features
X_train_bias = np.c_[np.ones((len(cbb_array), 1)), cbb_array]

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
theta = sgd_regression_l2(X_train_bias, output_array, learning_rate=0.01, n_epochs=100, alpha=alpha, max_iter=max_iter)

def make_prediction(theta, x):
    x_bias = np.insert(x, 0, 1.0)
    y = x_bias.dot(theta)
    return y[0]

#enter stats here
x = np.array([0.7063,-0.0979183,2.31598,1.75049,0.6875])
# y = make_prediction(theta, x)
# print(y)

print(make_prediction(theta, x))
