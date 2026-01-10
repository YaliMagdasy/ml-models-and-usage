import numpy as np

class LogisticRegressor:
    def __init__(self):
        self.weights = None

    # add bias
    def __extend_input(self, X):
        return np.c_[X, np.ones(X.shape[0])]

    # sigmoid function
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    # training the model
    def fit(self, X, y, show_progress=False, learning_rate=0.1, n_epochs=1000):
        Xext = self.__extend_input(X)
        n_samples, n_features = Xext.shape
        
        if self.weights is None:
            self.weights = np.random.randn(n_features) * 0.01

        for epoch in range(n_epochs):
            z = np.dot(Xext, self.weights)
            y_pred = self.__sigmoid(z)
            
            gradient = np.dot(Xext.T, (y_pred - y)) / n_samples
            self.weights -= learning_rate * gradient

            if show_progress and epoch % 50 == 0:
                loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
                print(f"Epoch {epoch} - Loss: {loss:.4f}")

    # predicting
    def predict(self, X):
        Xext = self.__extend_input(X)
        z = np.dot(Xext, self.weights)
        return self.__sigmoid(z)