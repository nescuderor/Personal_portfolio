'''
The goal of this exercise is to perform a minibatch stochastic gradient descent algorithm for a linear model.

The results of this script can be validated against the results in https://ufal.mff.cuni.cz/courses/npfl129/2425-winter#assignments:~:text=every%20such%20configuration.-,linear_regression_l2,-Deadline%3A%20Oct%2021
'''
# --- Base libraries ---
import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import matplotlib.pyplot as plt

# --- Global variables ---
BATCH_SIZE = 50
DATA_SIZE = 100
EPOCHS = 500
L2 = 0.1
LEARNING_RATE = 0.01
SEED = 92
TEST_SIZE = 0.5
PLOT = True

# --- Data generation ---
#Generating base data based on random simmulated data
data, target = sklearn.datasets.make_regression(n_samples=DATA_SIZE, random_state=SEED)
generator = np.random.RandomState(SEED) #Base generator for the random values iterations

#Addition of constant term for the bias
data = np.pad(data, ((0, 0), (0,1)), mode='constant', constant_values=1)
target = target.reshape(-1, 1)

#Creating the train-test split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size=TEST_SIZE, random_state=SEED)

# --- Fitting via minibatch stochastic gradient descent ---
#Creation of initial weights
weights = generator.uniform(size=x_train.shape[1], low = -0.1, high=0.1).reshape(-1, 1)

#Performing the minibatches SGD
train_rmses, test_rmses = [], [] #List for storing evolution of RMSE
for epoch in range(EPOCHS):
    #Base permutation for the epoch
    permutation = generator.permutation(x_train.shape[0])

    #Processing the permuted data in batches
    for i in range(0, len(permutation), BATCH_SIZE):
        minibatch = permutation[i:i+BATCH_SIZE]

        #Estimating the components for the correction of the weights
        forecast_correction = (x_train[minibatch].T @ ((x_train[minibatch] @ weights) - y_train[minibatch])) / BATCH_SIZE #Batch error correction
        ridge_correction = weights * L2 #Ridge (L_2) penalty
        ridge_correction[-1, :] = 0 #Weight for the bias

        #Weights correction
        weights = weights - LEARNING_RATE * (forecast_correction + ridge_correction) #Weights correction

    #Storing the evolution of the RMSE for evaluating the generalization gap
    train_rmses.append(sklearn.metrics.root_mean_squared_error(y_train, x_train @ weights))
    test_rmses.append(sklearn.metrics.root_mean_squared_error(y_test, x_test @ weights))

# --- Establishing a direct comparison for a model without gradient descent ---
#Setting a linear regression base model
comparative_model = sklearn.linear_model.LinearRegression()
comparative_model = comparative_model.fit(x_train, y_train)
prediction = comparative_model.predict(x_test)

explicit_rmse = sklearn.metrics.root_mean_squared_error(y_test, prediction)

# --- Pushing the results ---
print(f'Test RMSE: {test_rmses[-1]}, explicit {explicit_rmse}',
      f'Learned weights: {weights[:12,:]}',
      sep='\n')

#Plotting the evolution of the RMSEs in each, training and testing data.
fig, axs = plt.subplots()
axs.plot(range(EPOCHS), train_rmses, label="Train")
axs.plot(range(EPOCHS), test_rmses, label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()