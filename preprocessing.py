from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

class AuthorEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def fit(self, X, y=None):
        self.encoder.fit(X[["author_id"]])
        return self

    def transform(self, X):
        X = X.copy()
        X["author_id"] = self.encoder.transform(X[["author_id"]]).flatten()
        return X