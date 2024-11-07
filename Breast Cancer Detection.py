# performing linear algebra
import numpy as np

# data processing
import pandas as pd

# visualisation
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
data.head()

data.info()

# Removing unwanted data to increase performance speed
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Convert diagnosis to binary (1 for 'M', 0 for 'B')
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]

data.head()

y = data.diagnosis.values
X_data = data.drop(['diagnosis'], axis=1)

# Normalization of features
X = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Transposing for easier matrix manipulation
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)

def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.00
    return w, b

def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head

def forward_backward_propagation(w, b, X_train, y_train):
    z = np.dot(w.T, X_train) + b
    y_head = sigmoid(z)
    loss = - y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = np.sum(loss) / X_train.shape[1]

    # Backward propagation
    derivative_weight = np.dot(X_train, (y_head - y_train).T) / X_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / X_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost, gradients

def update(w, b, X_train, y_train, learning_rate, num_iterations):
    cost_list = []
    cost_list2 = []
    index = []

    # Updating (learning) parameters for num_iterations times
    for i in range(num_iterations):
        # Forward and backward propagation to find cost and gradients
        cost, gradients = forward_backward_propagation(w, b, X_train, y_train)
        cost_list.append(cost)

        # Update weights and bias
        w -= learning_rate * gradients["derivative_weight"]
        b -= learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration {}: {}".format(i, cost))

    # Update (learn) parameters: weights and bias
    parameters = {"weight": w, "bias": b}
    
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients, cost_list

def predict(w, b, X_test):
    # Forward propagation for prediction
    z = sigmoid(np.dot(w.T, X_test) + b)
    Y_prediction = np.zeros((1, X_test.shape[1]))

    # If z is bigger than 0.5, predict 1 (malignant), otherwise 0 (benign)
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction

def logistic_regression(X_train, y_train, X_test, y_test, learning_rate, num_iterations):
    dimension = X_train.shape[0]
    w, b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, X_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], X_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], X_train)

    # Train/test Errors
    print("Train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

logistic_regression(X_train, y_train, X_test, y_test, learning_rate=1, num_iterations=100)
