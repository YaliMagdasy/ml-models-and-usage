import numpy as np

class KNNClassifier:
    # Training 
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y
        
    # Predicting
    def predict(self, X, k: int=5, weighted: bool=False):
        predictions = []
        for x_pred in X:
            distances = np.linalg.norm(self.X_train - x_pred, axis=1)
            k_indices = np.argpartition(distances, k)[:k]
            
            k_nearest_labels = self.y_train[k_indices]
            
            if weighted:
                k_nearest_dists = distances[k_indices]
                weights = 1 / (k_nearest_dists + 1e-10)
                
                classes = np.unique(k_nearest_labels)
                class_scores = {}
                
                for c in classes:
                    is_class_c = (k_nearest_labels == c)
                    class_scores[c] = np.sum(weights[is_class_c])

                predictions.append(max(class_scores, key=class_scores.get))
            else:
                counts = np.bincount(k_nearest_labels.astype(int))

                predictions.append(np.argmax(counts))
                
        return np.array(predictions)