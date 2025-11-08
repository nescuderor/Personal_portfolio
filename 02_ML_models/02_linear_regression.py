'''
Title: Polynomial Regression with Scikit-Learn

Description:
This script demonstrates how to perform polynomial regression by incrementally adding features
to a scikit-learn `LinearRegression` model. It visualizes how the model fit improves and
how the Root Mean Squared Error (RMSE) changes as the complexity of the polynomial increases.

The core idea is to start with a simple linear model (order 1) and progressively add
higher-order polynomial features (x^2, x^3, etc.) in a loop. For each order, the script:
1. Creates the new polynomial feature.
2. Splits the data into training and testing sets.
3. Fits a linear regression model on the expanded feature set.
4. Calculates the RMSE on the test set.
5. Plots the resulting model fit against the training and test data.

This process effectively shows the trade-off between model complexity and its ability to fit the data,
providing a practical look at the concept of overfitting.

The results of this script can be validated against the results in https://ufal.mff.cuni.cz/courses/npfl129/2425-winter#assignments:~:text=the%20test%20set.-,linear_regression_features,-Deadline%3A%20Oct%2014
'''
# Base libraries
import numpy as np
import sklearn.linear_model
import sklearn.model_selection
import matplotlib.pyplot as plt


# --- Global Variables ---
# Configuration parameters for the experiment.
SIZE = 40      # Number of data points to generate.
RANGE = 3      # The maximum polynomial order to test.
SEED = 42      # Random seed for reproducibility.
TEST_SIZE = 0.5# Proportion of the data to use for the test set.
PLOT = True    # Flag to control whether plots are generated.

# --- Data Generation ---
# Define a simple, non-linear dataset to model.
# A sine wave is used with some random noise to simulate real-world data.
xs = np.linspace(0, 7, num=SIZE)
ys = np.sin(xs) + np.random.RandomState(SEED).normal(0, 0.2, size=SIZE)

# --- Plotting Setup ---
# Prepare a grid of subplots to display the results for each polynomial order.
if PLOT is True:
    nrows = int(np.ceil(RANGE/3))
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))

# --- Feature Creation, Model Training, and Evaluation Loop ---
rmses = []
# The `incremental` array will store the polynomial features (x, x^2, x^3, ...).
incremental = np.empty([SIZE, 1], dtype='float64')

for order in range(1, RANGE+1):
    # Add the next polynomial feature to the feature matrix.
    if order == 1:
        incremental[:, order-1] = xs
    else:
        incremental = np.hstack([incremental, xs.reshape(-1, 1)**order])

    # Split the data into training and test sets for the current feature set.
    train_xs, test_xs, train_ys, test_ys = sklearn.model_selection.train_test_split(incremental, ys, test_size=TEST_SIZE, random_state=SEED)

    # Fit a linear regression model on the training data.
    model = sklearn.linear_model.LinearRegression(fit_intercept= True).fit(train_xs, train_ys)

    # Make predictions and calculate the RMSE on the test set.
    prediction = model.predict(test_xs)
    rmse = (sum((prediction - test_ys)**2)/test_ys.shape[0])**(1/2)
    rmses.append(rmse)

    # --- Plotting Results for the Current Order ---
    if PLOT is True:
        try:
            axs.shape[1]
            # Calculate the row and column for the subplot
            row = (order - 1) // ncols #Floor operation for determining the row
            col = (order - 1) % ncols #Module for determining the column

            # Select the correct subplot axis using 2D indexing
            ax = axs[row, col]

            # Plot training data, test data, and the model's prediction line on the selected axis
            ax.plot(train_xs[:, 0], train_ys, "go", label="Train Data")
            ax.plot(test_xs[:, 0], test_ys, "ro", label="Test Data")
            line_x = np.linspace(xs.min(), xs.max(), num=100)
            line_features = np.power.outer(line_x, np.arange(1, order + 1))
            ax.plot(line_x, model.predict(line_features), "b-", label=f"Order {order} Fit")

            # Add title and legend for clarity
            ax.set_title(f"Polynomial Order {order}")
            ax.legend()

        except:
            # Select the correct subplot axis using 2D indexing
            ax = axs[order-1]

            # Plot training data, test data, and the model's prediction line on the selected axis
            ax.plot(train_xs[:, 0], train_ys, "go", label="Train Data")
            ax.plot(test_xs[:, 0], test_ys, "ro", label="Test Data")
            line_x = np.linspace(xs.min(), xs.max(), num=100)
            line_features = np.power.outer(line_x, np.arange(1, order + 1))
            ax.plot(line_x, model.predict(line_features), "b-", label=f"Order {order} Fit")

            # Add title and legend for clarity
            ax.set_title(f"Polynomial Order {order}")
            ax.legend()

# --- Final Output ---
# Display the plots and print the list of RMSE values.
if PLOT is True:
    plt.tight_layout()
    plt.show()
print(f"RMSEs for orders 1 to {RANGE}: {rmses}")





