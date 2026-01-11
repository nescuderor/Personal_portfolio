# Creating classes for developing customized transformers for sklearn.

One of the ways to develop a customized transformer compatible with sklearn framework is via the construction of customized classes. Within them, it is necessary to guarantee the methods `.fit()`, `.transform()`, and `.fit_transform()`. In addition, it is recommended to also create the method `.inverse_transform()` to restore the original values, if needed.


```python
from sklearn.base import BaseEstimator, TransformerMixin
# BaseEstimator serves as a base class for avoiding the need to define the .fit_transform method.
# TransformerMixin allows the extraction of other methods such as .get_feature_names_out().
from sklearn.utils.validation import check_array, check_is_fitted
# These two methods are for validating the structure of the constructed class.

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
```