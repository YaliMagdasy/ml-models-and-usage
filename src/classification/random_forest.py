import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from src.classification.decision_tree import DecisionTreeClassifier

def _fit_single_tree(args):
    tree_params, X, y = args
    tree = DecisionTreeClassifier(**tree_params)
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace=True)
    tree.fit(X[indices], y[indices])
    return tree

class RandomForestClassifier:
    def __init__(self, n_trees=50, max_depth=10, min_samples_split=2, n_features=None, n_jobs=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()
        self.trees = []

    def fit(self, X, y):
        tree_params = {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'n_features': self.n_features
        }
        
        args_list = [(tree_params, X, y) for _ in range(self.n_trees)]
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            self.trees = list(executor.map(_fit_single_tree, args_list))

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        def _get_mode(arr):
            vals, counts = np.unique(arr, return_counts=True)
            return vals[np.argmax(counts)]
            
        return np.apply_along_axis(_get_mode, axis=0, arr=tree_preds)