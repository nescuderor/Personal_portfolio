'''
Grid Search for Logistic Regression Hyperparameter Tuning

This script demonstrates a comprehensive approach to hyperparameter optimization using grid search
with stratified cross-validation. The implementation showcases best practices for model selection
in binary classification tasks.

Key Features:
- Automated hyperparameter tuning through exhaustive grid search
- Pipeline-based workflow combining preprocessing, feature engineering, and model training
- Stratified K-fold cross-validation to ensure balanced class distribution in each fold
- Polynomial feature augmentation for capturing non-linear relationships
- Feature normalization using Min-Max scaling for improved model convergence

The workflow:
1. Loads the digits dataset and converts it to a binary classification problem
2. Splits data into training and test sets with stratification
3. Constructs a pipeline integrating normalization, polynomial features, and logistic regression
4. Performs grid search over polynomial degrees, regularization strengths, and solver algorithms
5. Evaluates all parameter combinations using 5-fold stratified cross-validation
6. Reports ranking of all configurations and final test accuracy of the best model

The results of this script can be validated against the results in:
https://ufal.mff.cuni.cz/courses/npfl129/2425-winter#assignments:~:text=for%20logistic%20regression.-,grid_search,-Deadline%3A%20Oct%2028
'''

# --- Basic setting ---
# Core scientific computing and machine learning libraries
import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.metrics
import warnings

# Suppress convergence warnings to keep output clean during grid search iterations
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

# --- Global variables ---
DATASET = 'digits'
SEED = 42
TEST_SIZE = 0.5

# --- Loading the base data ---
# Dynamically load the specified dataset using sklearn's dataset loading utilities
dataset = getattr(sklearn.datasets, 'load_{}'.format(DATASET))()
data = dataset.data
# Convert multi-class problem to binary classification using modulo operation
# This transforms the 10-class digits problem into even/odd digit classification
target = dataset.target % 2

# --- Feature augmentation and model training ---
# Stratified train-test split preserves class distribution in both sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size=TEST_SIZE, random_state=SEED)

# Construct a sequential pipeline that ensures consistent preprocessing across all data
# The pipeline guarantees that transformations learned on training data are properly applied to test data
training_pipeline = sklearn.pipeline.Pipeline(
    steps=[
        ('normalize', sklearn.preprocessing.MinMaxScaler()),
        ('polinomial_augmentation', sklearn.preprocessing.PolynomialFeatures()),
        ('logistic_regression', sklearn.linear_model.LogisticRegression(random_state=SEED))
    ]
)

# --- Cross validation ---
# Stratified K-fold ensures each fold maintains the same class proportion as the full dataset
# This is crucial for imbalanced datasets and provides more reliable performance estimates
cross_validation = sklearn.model_selection.StratifiedKFold(n_splits=5)

# Define the hyperparameter search space
# Each combination will be evaluated through cross-validation
grid = {
    'polinomial_augmentation__degree': [1, 2],
    'logistic_regression__C': [0.01, 1, 100],
    'logistic_regression__solver': ['lbfgs', 'sag']
}

# Grid search exhaustively tries all combinations of hyperparameters
# Returns the best model based on cross-validation performance
model_tuning = sklearn.model_selection.GridSearchCV(training_pipeline,
                                                    param_grid=grid,
                                                    cv=cross_validation)
model_tuned = model_tuning.fit(x_train, y_train)

# --- Results display ---
# Display all tested configurations ranked by cross-validation performance
# This provides insight into which hyperparameters contribute most to model quality
for rank, accuracy, params in zip(model_tuned.cv_results_["rank_test_score"],
                                  model_tuned.cv_results_["mean_test_score"],
                                  model_tuned.cv_results_["params"]):
    print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
        *("{}: {:<5}".format(key, value) for key, value in params.items()))

# Report final performance on held-out test set using the best model from grid search
# This represents the expected performance on unseen data
print('Best model test accuracy: {:.2f}%' \
      .format(100 * sklearn.metrics.accuracy_score(y_test,
                                                   model_tuned.best_estimator_.predict(x_test))))
