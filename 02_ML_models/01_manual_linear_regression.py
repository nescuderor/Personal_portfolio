'''
Title: Manual Linear Regression with NumPy

Description:
This script implements a linear regression model from scratch using only the NumPy library.
The primary goal is to demonstrate a fundamental understanding of the Ordinary Least Squares (OLS)
method by manually computing the model's weights using the normal equation.

The normal equation is a closed-form solution for finding the optimal weights (w) that minimize
the sum of squared residuals. The formula is:
w = (X^T * X)^-1 * X^T * y

The script performs the following steps:
1. Loads the scikit-learn diabetes dataset.
2. Appends a constant term to the feature matrix to account for the model's intercept (bias).
3. Splits the data into training and testing sets.
4. Calculates the optimal weights `w` on the training data using the normal equation.
5. Makes predictions on the test set.
6. Evaluates the model's performance by computing the Root Mean Squared Error (RMSE).

The results of this script can be validated against the results in https://ufal.mff.cuni.cz/courses/npfl129/2425-winter#assignments:~:text=publish%20your%20solutions.-,linear_regression_manual,-Deadline%3A%20Oct%2014
'''
# Base libraries
import numpy as np
import sklearn.datasets
import sklearn.model_selection

# --- Global Variables ---
# SEED: A random seed for reproducibility of the train/test split.
SEED = 42
# TEST_SIZE: The proportion of the dataset to allocate to the test set.
TEST_SIZE = 0.1

# --- Data Loading and Preparation ---
# Unpacking the scikit-learn diabetes dataset
dataset = sklearn.datasets.load_diabetes()
data = dataset.data
targets = dataset.target

# Add a constant term (a column of ones) to the feature matrix.
# This is necessary to allow the linear regression model to learn an intercept (bias) term.
# The shape of `data` changes from (n_samples, n_features) to (n_samples, n_features + 1).
data = np.hstack([data, np.ones((data.shape[0], 1))])

# --- Train/Test Split ---
# Performing a standard split of the data into training and validation sets.
# The model will be trained on the training set and evaluated on the unseen test set.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, targets, test_size=TEST_SIZE, random_state=SEED)

# --- Model Computation (Normal Equation) ---
# Computing the linear regression weights `w` using the closed-form normal equation.
# This single line is the core of the OLS implementation.
# @ is the operator for matrix multiplication in NumPy.
# .T is the transpose of a matrix.
# np.linalg.inv() computes the inverse of a matrix.
w = np.linalg.inv((x_train.T @ x_train)) @ (x_train.T @ y_train)

# --- Prediction and Evaluation ---
# Make predictions on the test set by computing the dot product of the test features and the learned weights.
y_estim = x_test @ w

# Computing the Root Mean Squared Error (RMSE) to evaluate model performance.
# RMSE is a standard metric for regression tasks that measures the square root of the average of squared differences
# between predicted values and actual values.
rmse = (sum((y_estim - y_test)**2)/y_test.shape[0])**(1/2)
print(f"Root Mean Squared Error: {rmse}")