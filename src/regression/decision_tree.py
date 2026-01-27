import numpy as np

class __Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.__build_tree(X, y)

    def __build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self.__calculate_leaf_value(y)
            return __Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self.__best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self.__split(X[:, best_feat], best_thresh)
        left = self.__build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.__build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return __Node(best_feat, best_thresh, left, right)

    def __best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thresh in thresholds:
                gain = self.__variance_reduction(y, X_column, thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def __variance_reduction(self, y, X_column, thresh):
        parent_variance = self.__variance(y)
        left_idxs, right_idxs = self.__split(X_column, thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        v_l, v_r = self.__variance(y[left_idxs]), self.__variance(y[right_idxs])
        child_variance = (n_l / n) * v_l + (n_r / n) * v_r

        return parent_variance - child_variance

    def __split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def __variance(self, y):
        return np.var(y)

    def __calculate_leaf_value(self, y):
        return np.mean(y)

    def predict(self, X):
        return np.array([self.__traverse_tree(x, self.root) for x in X])

    def __traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.__traverse_tree(x, node.left)
        return self.__traverse_tree(x, node.right)