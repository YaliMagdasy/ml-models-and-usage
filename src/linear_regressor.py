import numpy as np

class LinearRegressor:
    def __init__(self):
        self.weights = None

    # add bias
    def __extend_input(self, X):
        Xext = np.ones((X.shape[0], X.shape[1] + 1))
        Xext[:, :-1] = X
        return Xext

    # training the model
    def fit(self, X, y):
        Xext = self.__extend_input(X)
        
        inv_XtX = np.linalg.pinv(Xext.T @ Xext)
        Xty = Xext.T @ y
        self.weights = inv_XtX @ Xty

    # predicting
    def predict(self, X):
        Xext = self.__extend_input(X)
        return Xext @ self.weights