'''
The goal for this script is to perform a linear regression via scikit-learn via its linear_model module.
And additional factor, it tries to demonstrate the development of basic features.

The results of this script can be validated against the results in linear_regression_features
'''
#Base libraries
import numpy as np
import sklearn.linear_model
import sklearn.model_selection
import matplotlib.pyplot as plt


#Global variables
SIZE = 40
RANGE = 3
SEED = 42
TEST_SIZE = 0.5
PLOT = True

#Defining a random noisy data
xs = np.linspace(0, 7, num=SIZE)
ys = np.sin(xs) + np.random.RandomState(SEED).normal(0, 0.2, size=SIZE)

#Basic setting for plotting
if PLOT is True:
    nrows = int(np.ceil(RANGE/3))
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

#Creation of features and estimation of the model
rmses = []
incremental = np.empty([SIZE, 1], dtype='float64')
for order in range(1, RANGE+1):
    if order == 1:
        incremental[:, order-1] = xs
    else:
        incremental = np.hstack([incremental, xs.reshape(-1, 1)**order])

    #Generating training/test split:
    train_xs, test_xs, train_ys, test_ys = sklearn.model_selection.train_test_split(incremental, ys, test_size=TEST_SIZE, random_state=SEED)

    #Fitting the model:
    model = sklearn.linear_model.LinearRegression(fit_intercept= True).fit(train_xs, train_ys)
    prediction = model.predict(test_xs)
    rmse = (sum((prediction - test_ys)**2)/test_ys.shape[0])**(1/2)
    rmses.append(rmse)

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

if PLOT is True:
    plt.show()
print(rmses)





