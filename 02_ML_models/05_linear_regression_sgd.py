'''
Title: Linear Regression with Mini-Batch Stochastic Gradient Descent (SGD)

Description:
This script implements a linear regression model from scratch using the mini-batch
stochastic gradient descent (SGD) optimization algorithm. It includes L2 regularization
(Ridge penalty) to help prevent overfitting.

The core of the script is the SGD training loop, which iteratively updates the model's
weights to minimize the Mean Squared Error (MSE) loss function. The process is as follows:
1.  Synthetic data is generated for a regression task.
2.  A bias (intercept) term is added to the feature matrix.
3.  The data is split into training and testing sets.
4.  Model weights are initialized with small random values.
5.  The script iterates for a fixed number of epochs. In each epoch:
    a. The training data is shuffled.
    b. The data is processed in small groups (mini-batches).
    c. For each mini-batch, the gradient of the loss function (including the L2
       penalty) is calculated.
    d. The model weights are updated by taking a small step in the opposite
       direction of the gradient.
6.  The script tracks the Root Mean Squared Error (RMSE) on both the training and
    test sets after each epoch to monitor learning progress.
7.  Finally, it compares the result to a standard scikit-learn Linear Regression
    model and plots the learning curves.

The results of this script can be validated against the results in https://ufal.mff.cuni.cz/courses/npfl129/2425-winter#assignments:~:text=best%20of%20them.-,linear_regression_sgd,-Deadline%3A%20Oct%2021
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

# --- Data Generation and Preparation ---
# Generating base data based on a random simulated regression problem.
data, target = sklearn.datasets.make_regression(n_samples=DATA_SIZE, random_state=SEED)
generator = np.random.RandomState(SEED) # Base generator for reproducible random values.

# Add a constant column of 1s to the feature matrix to act as the bias (intercept) term.
data = np.pad(data, ((0, 0), (0,1)), mode='constant', constant_values=1)
target = target.reshape(-1, 1)

# Creating the train-test split for model training and evaluation.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size=TEST_SIZE, random_state=SEED)

# --- Fitting via Mini-Batch Stochastic Gradient Descent ---
# Creation of initial weights with small random values to break symmetry.
weights = generator.uniform(size=x_train.shape[1], low = -0.1, high=0.1).reshape(-1, 1)

# Performing the mini-batch SGD loop.
train_rmses, test_rmses = [], [] # List for storing the evolution of RMSE over epochs.
for epoch in range(EPOCHS):
    # Shuffle the training data indices at the beginning of each epoch.
    permutation = generator.permutation(x_train.shape[0])

    # Processing the permuted data in mini-batches.
    for i in range(0, len(permutation), BATCH_SIZE):
        minibatch = permutation[i:i+BATCH_SIZE]

        # Estimating the components of the gradient for the weight update.
        # 1. Gradient of the Mean Squared Error for the current mini-batch.
        forecast_correction = (x_train[minibatch].T @ ((x_train[minibatch] @ weights) - y_train[minibatch])) / BATCH_SIZE
        # 2. Gradient of the L2 regularization term.
        ridge_correction = weights * L2
        # The bias term is typically not regularized.
        ridge_correction[-1, :] = 0

        # Update the weights by taking a step in the opposite direction of the combined gradient.
        weights = weights - LEARNING_RATE * (forecast_correction + ridge_correction)

    # Storing the evolution of the RMSE after each full pass (epoch) to evaluate the generalization gap.
    train_rmses.append(sklearn.metrics.root_mean_squared_error(y_train, x_train @ weights))
    test_rmses.append(sklearn.metrics.root_mean_squared_error(y_test, x_test @ weights))

# --- Scikit-learn Baseline Comparison ---
# Train a standard Linear Regression model as a baseline for comparison.
comparative_model = sklearn.linear_model.LinearRegression()
comparative_model = comparative_model.fit(x_train, y_train)
prediction = comparative_model.predict(x_test)

explicit_rmse = sklearn.metrics.root_mean_squared_error(y_test, prediction)

# --- Results and Visualization ---
# Print the final test RMSE from our SGD model and the baseline model.
print(f'Test RMSE: {test_rmses[-1]}, explicit {explicit_rmse}',
      f'Learned weights: {weights[:12,:]}',
      sep='\n')

# Plot the training and testing RMSE over epochs to visualize learning and check for overfitting.
if PLOT:
    fig, axs = plt.subplots()
    axs.plot(range(EPOCHS), train_rmses, label="Train")
    axs.plot(range(EPOCHS), test_rmses, label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("Learning Curves (RMSE vs. Epochs)")
    plt.show()