'''
Title: Dynamic Feature Engineering Pipeline with Scikit-Learn

Description:
This script demonstrates how to build a sophisticated feature engineering pipeline
using scikit-learn. The pipeline is designed to be dynamic, automatically
distinguishing between continuous and categorical features in a dataset and
applying appropriate transformations.

The process is as follows:
1.  A function `create_features` encapsulates the entire pipeline.
2.  Inside the function, columns are programmatically identified as either
    continuous (floats) or categorical (integers).
3.  A `ColumnTransformer` applies `OneHotEncoder` to categorical features and
    `StandardScaler` to continuous features.
4.  A `PolynomialFeatures` transformer is then used to generate second-degree
    and interaction features from the processed data.
5.  The script loads a dataset, splits it, and then runs both the training and
    test sets through the feature engineering pipeline to prepare them for a
    machine learning model.

The results of this script can be validated against the results in https://ufal.mff.cuni.cz/courses/npfl129/2425-winter#assignments:~:text=linear%20regression%20solver.-,feature_engineering,-Deadline%3A%20Oct%2021
'''

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

# --- Global Variables ---
DATABASE = "diabetes"
SEED = 42
TEST_SIZE = 0.5

# --- Feature Engineering Pipeline ---
dataset = getattr(sklearn.datasets, "load_{}".format(DATABASE))()

x_train, x_test = sklearn.model_selection.train_test_split(dataset.data, test_size=TEST_SIZE, random_state=SEED)

# Determining the pipeline for the type of data
def create_features(database):
    """
    Builds and applies a feature engineering pipeline to a given dataset.

    This function programmatically distinguishes between continuous (float) and
    categorical (integer-like) columns in the input data. It then applies
    different transformations to each type:
    - Categorical columns are one-hot encoded.
    - Continuous columns are standardized (scaled to have zero mean and unit variance).

    Finally, it generates second-degree polynomial and interaction features from the
    newly transformed feature set.

    Parameters:
    ----------
    database : np.ndarray
        The input dataset to be transformed, with shape (n_samples, n_features).

    Returns:
    -------
    np.ndarray
        The transformed dataset with one-hot encoding, scaling, and polynomial
        features applied.
    """
    # Programmatically distinguish between integer and float columns.
    # A column is considered float if any of its values has a non-zero fractional part.
    integers_comparison = np.vectorize(lambda x: (x%1) != 0)(database)
    is_float = np.where(np.all(integers_comparison, axis=0))[0]
    is_integer = np.where(~np.all(integers_comparison, axis=0))[0]

    # Define the column-specific transformations.
    column_transform = sklearn.compose.ColumnTransformer(
        [('categorical', sklearn.preprocessing.OneHotEncoder(sparse_output=False), is_integer),
        ('continuous', sklearn.preprocessing.StandardScaler(), is_float)]
    )

    # Define the polynomial feature generation step.
    polynomial_transform = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)

    # Chain the column transformations and polynomial feature generation together.
    pipe = sklearn.pipeline.Pipeline(
        steps=[
            ('process_columns', column_transform),
            ('add_polynomial', polynomial_transform)
        ]
    )

    # Fit and transform the data using the defined pipeline.
    result = pipe.fit_transform(database)
    return result

train_data = create_features(x_train)
test_data = create_features(x_test)

print(train_data[:5])
print(test_data[:5])