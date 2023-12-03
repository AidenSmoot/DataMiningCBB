import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

offset = 2013

cbb = pd.read_csv('finalCBB.csv')
cbb = cbb.drop(cbb.columns[0],axis=1)
thetas = []

#compute the mean and standard deviation of each column
means = [] 
stdevs = []

for column in cbb:
    means.append(np.mean(cbb[column]))
    stdevs.append(np.std(cbb[column]))

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
    if year == 2020:
        cbb20 = pd.read_csv('finalCBB20.csv')
        cbb20 = cbb20.drop(cbb20.columns[0],axis=1)
        cbb20_input = cbb20.drop(['WR'], axis=1)
        cbb20_input_array = cbb20_input.values
        cbb20_array_bias =  np.c_[np.ones((len(cbb20_input_array), 1)), cbb20_input_array]
        cbb20_output = cbb20.drop(cbb20.columns[0:-1], axis=1)
        cb20_output_array = cbb20_output.values
        thetas.append(sgd_regression_l2(cbb20_array_bias, cb20_output_array, learning_rate=0.01, n_epochs=100, alpha=alpha, max_iter=max_iter))
    else:
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
attributes = inputs.columns.tolist()
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
newData = np.array([0.881005,-0.686403,-1.56669,-0.189683,2.70316,0.149391,-0.404236,-0.688248])
# y = make_prediction(theta, x)
# print(y)


years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
weights = np.array([theta.T[0] for theta in thetas][:len(thetas)-1]).T

# print(thetas_t[0])

def standardize(newData):
    newDataStd = []
    for i in range(len(newData)):
        newDataStd.append((newData[i] - means[i]) / stdevs[i])
    return np.array(newDataStd)


def predict_and_plot(input, years, thetas):
    standardized_input = standardize(input)
    predictions = []
    for i in range(len(years)):
        print("year :", years[i])
        predictions.append(make_prediction(thetas, input, years[i]))
    plt.plot(years, predictions)
    plt.show()
    return predictions


# for i in range(0, len(attributes)):
#     plt.plot(years, weights[i], label = attributes[i])
# plt.legend()
# plt.xlabel("Year")
# plt.ylabel("Weight")
# plt.show()

# print(thetas[0])

print(predict_and_plot(newData, years, thetas))
