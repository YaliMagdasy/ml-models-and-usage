import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = None

    # Adding bias
    def __extend_input(self, X): 
        Xext = np.ones((X.shape[0], X.shape[1] + 1))
        Xext[:, :-1] = X
        return Xext

    # Training the model using Gradient Descent
    def fit(self, X, y, learning_rate: float=0.1, n_epochs: int=100, show_progress: bool=False):
        Xext = self.__extend_input(X)
        n_samples, n_features = Xext.shape
        
        if self.weights is None:
            self.weights = np.zeros(n_features)

        accuracy_history = []

        for epoch in range(n_epochs):
            n_errors = 0
            for i in range(n_samples):
                xi = Xext[i]
                
                activation = np.dot(self.weights, xi)
                prediction = 1 if activation >= 0 else -1

                if prediction != y[i]:
                    n_errors += 1
                    self.weights += learning_rate * y[i] * xi

            accuracy = (n_samples - n_errors) / n_samples
            accuracy_history.append(accuracy)

            if show_progress and epoch % max(1, n_epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} - Accuracy: {accuracy:.4f}")

        return accuracy_history
    
    # Predicting
    def predict(self, X):
        Xext = self.__extend_input(X)
        z = np.dot(Xext, self.weights)
        return np.where(z >= 0, 1, -1)