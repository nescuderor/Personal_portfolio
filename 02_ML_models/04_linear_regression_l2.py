'''
Title: Hyperparameter Tuning for L2 Regularization in Ridge Regression

Description:
This script performs a manual grid search to find the optimal L2 regularization
strength (lambda, or `alpha` in scikit-learn) for a Ridge Regression model.

Ridge Regression is a linear model that includes a penalty term to shrink the
magnitude of the model coefficients, which helps to prevent overfitting. The
strength of this penalty is controlled by the hyperparameter lambda.

The script executes the following steps:
1.  Loads the diabetes dataset.
2.  Splits the data into training and testing sets.
3.  Defines a range of lambda values to test, spaced logarithmically.
4.  Iterates through each lambda value:
    a. Fits a Ridge Regression model.
    b. Makes predictions on the test set.
    c. Calculates the Root Mean Squared Error (RMSE).
5.  Identifies the lambda that resulted in the lowest RMSE.
6.  Plots the RMSE as a function of the regularization strength lambda to
    visualize the impact of regularization on model performance.

The results of this script can be validated against the results in https://ufal.mff.cuni.cz/courses/npfl129/2425-winter#assignments:~:text=every%20such%20configuration.-,linear_regression_l2,-Deadline%3A%20Oct%2021
'''
# --- Libraries ---
import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import matplotlib.pyplot as plt

# --- Global Variables ---
PLOT = True
SEED = 13
TEST_SIZE = 0.80

# --- Data Loading and Splitting ---
# Load the dataset and split it into training and testing sets.
dataset = sklearn.datasets.load_diabetes()
data = dataset.data
target = dataset.target

#Train-test split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size=TEST_SIZE, random_state=SEED)

# --- Hyperparameter Grid Search ---
# Define a range of lambda values (regularization strength) to test.
# np.geomspace is used to create values spaced evenly on a log scale.
lambdas = np.geomspace(0.01, 10, num=500)

#Doing a manual gridsearch for tunning the lambda parameter
rmses = []
for lambda_value in lambdas:
    model = sklearn.linear_model.Ridge(alpha=lambda_value)
    model = model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    rmse = sklearn.metrics.root_mean_squared_error(y_test, predictions)
    rmses.append(rmse)

# --- Find and Display Best Result ---
# Find the index of the best score (minimum RMSE) and the corresponding lambda.
best_score = np.where(np.array(rmses) == min(rmses))[0][0]
print(lambdas[best_score], rmses[best_score])

# Plotting results for identify the relation between hyperparameters and the rmse
fig, ax = plt.subplots()
ax.plot(lambdas, rmses)

plt.xscale("log")
plt.xlabel("L2 regularization strength $\\lambda$")
plt.ylabel("RMSE")
plt.show()