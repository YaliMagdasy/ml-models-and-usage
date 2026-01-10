import numpy as np

# This is a simple linear regressor. it is built to predict simple linear relationships.

class LinearRegressor:
    def __init__(self):
        self.weights = None

    # add bias
    def __extend_input(self, X):
        Xext = np.ones((X.shape[0], X.shape[1] + 1))
        Xext[:, :-1] = X
        return Xext

    # training the model using Newton's Method (the Normal equation)
    def fit(self, X, y):
        self.Xext = self.__extend_input(X)
        inv_XtX = np.linalg.pinv(self.Xext.T @ self.Xext)
        Xty = self.Xext.T @ y

        self.weights = inv_XtX @ Xty

    # predicting
    def predict(self, X):
        Xext = self.__extend_input(X)
        return self.weights @ Xext.T