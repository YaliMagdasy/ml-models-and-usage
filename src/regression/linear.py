import numpy as np

class LinearRegressor:
    def __init__(self):
        self.weights = None

    # Adding bias
    def __extend_input(self, X):
        Xext = np.ones((X.shape[0], X.shape[1] + 1))
        Xext[:, :-1] = X
        return Xext

    # Training the model using the Normal Equation
    def fit(self, X, y):
        Xext = self.__extend_input(X)
        inv_XtX = np.linalg.pinv(Xext.T @ Xext)
        Xty = Xext.T @ y

        self.weights = inv_XtX @ Xty

    # Predicting
    def predict(self, X):
        Xext = self.__extend_input(X)
        return Xext @ self.weights