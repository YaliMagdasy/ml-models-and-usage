import numpy as np

class Perceptron:
    def __init__(self, n_features, learning_rate=0.01, random_seed=42):
        np.random.seed(random_seed)
        self.weights = np.random.randn(n_features + 1)
        self.learning_rate = learning_rate

    # add bias
    def _extend_input(self, X):
        self.Xext = np.ones((X.shape[0], X.shape[1] + 1))

        self.Xext[:, :-1] = X
        return self.Xext

    # training the model using Gradient Descent
    def fit(self, X, y, n_epochs=10, show_progress=False):
        n_samples = X.shape[0]
        accuracy_history = []

        for epoch in range(n_epochs):
            n_errors = 0

            for i in range(n_samples):
                xi = X[i:i+1]

                prediction = self.predict(xi)[0]

                if prediction != y[i]:
                    n_errors += 1
                    self.weights += self.learning_rate * y[i] * self.Xext[0]

            accuracy = (n_samples - n_errors) / n_samples
            accuracy_history.append(accuracy)

            if (show_progress):
                print(f"Epoch {epoch + 1}/{n_epochs} - Accuracy: {accuracy:.4f}")

        return accuracy_history
    
    # predicting
    def predict(self, X):
        self.Xext = self._extend_input(X)
        z = self.weights @ self.Xext.T
        return np.where(z >= 0, 1, -1)