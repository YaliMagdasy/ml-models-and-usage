import numpy as np

class KNNRegressor:
    # Training 
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y
        
    # Predicting
    def predict(self, X, k: int=5):
        predictions = []
        for x_pred in X:
            distances = np.linalg.norm(self.X_train - x_pred, axis=1)
            
            k_indices = np.argpartition(distances, k)[:k]
            
            k_nearest_labels = self.y_train[k_indices]
            
            average = np.mean(k_nearest_labels)
            predictions.append(average)
            
        return np.array(predictions)