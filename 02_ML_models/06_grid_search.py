'''
The goal of this script is to demonstrate the application of grid search and the application of a base logistic regression algorithm.

The results of this script can be validated against the results in https://ufal.mff.cuni.cz/courses/npfl129/2425-winter#assignments:~:text=for%20logistic%20regression.-,grid_search,-Deadline%3A%20Oct%2028
'''

# --- Basic setting ---
# Base libraries
import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.metrics
import warnings

# Controlling for warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

# --- Global variables ---
DATASET = 'digits'
SEED = 42
TEST_SIZE = 0.7

# --- Loading the base data ---
dataset = getattr(sklearn.datasets, 'load_{}'.format(DATASET))()
data = dataset.data
target = dataset.target % 2 # Manipulation for the purposes of the exercise

# --- Feature augmentation and model training ---
# Train-test split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size=TEST_SIZE, random_state=SEED)

# Basic processing piepline for feature engineering and model training
training_pipeline = sklearn.pipeline.Pipeline(
    steps=[
        ('normalize', sklearn.preprocessing.MinMaxScaler()),
        ('polinomial_augmentation', sklearn.preprocessing.PolynomialFeatures()),
        ('logistic_regression', sklearn.linear_model.LogisticRegression(random_state=SEED))
    ]
)

# --- Cross validation ---
# Cross validation component for grid search
cross_validation = sklearn.model_selection.StratifiedKFold(n_splits=5)

# Grid to search
grid = {
    'polinomial_augmentation__degree': [1, 2],
    'logistic_regression__C': [0.01, 1, 100],
    'logistic_regression__solver': ['lbfgs', 'sag']
}

# Grid search with stratified cross validation
model_tuning = sklearn.model_selection.GridSearchCV(training_pipeline,
                                                    param_grid=grid,
                                                    cv=cross_validation)
model_tuned = model_tuning.fit(x_train, y_train)

# --- Results display ---
for rank, accuracy, params in zip(model_tuned.cv_results_["rank_test_score"],
                                  model_tuned.cv_results_["mean_test_score"],
                                  model_tuned.cv_results_["params"]):
    print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
        *("{}: {:<5}".format(key, value) for key, value in params.items()))

print('Best model test accuracy: {:.2f}%' \
      .format(100 * sklearn.metrics.accuracy_score(y_test,
                                                   model_tuned.best_estimator_.predict(x_test))))
