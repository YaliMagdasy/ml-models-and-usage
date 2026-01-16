import numpy as np

class KNNClassifier:
    # Training 
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y
        
    # Predicting
    def predict(self, X, k: int=5):
        predictions = []
        for x_pred in X:
            distances = np.linalg.norm(self.X_train - x_pred, axis=1)
            
            k_indices = np.argpartition(distances, k)
            
            k_nearest_labels = self.y_train[k_indices]
            
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
            
        return np.array(predictions)