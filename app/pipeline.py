import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

class logar(BaseEstimator, TransformerMixin):
    '''
    Escalado min max de las variables menos el objetivo.
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.log(X)
        return X
        
    def inverse_transform(self,X):
        X = np.exp(X)

        return X


pipe = Pipeline([
    ("log",logar()),
    ("Scaler",MinMaxScaler())
])        