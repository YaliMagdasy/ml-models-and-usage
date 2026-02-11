import numpy as np

class _Node:
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
            return _Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self.__best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self.__split(X[:, best_feat], best_thresh)
        left = self.__build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.__build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return _Node(best_feat, best_thresh, left, right)

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
    

    def print_tree(self, figsize=(12, 8)):
            import matplotlib.pyplot as plt

            def get_depth(node):
                if not node: return 0
                if node.is_leaf_node(): return 1
                return max(get_depth(node.left), get_depth(node.right)) + 1

            def get_width(node):
                if not node: return 0
                if node.is_leaf_node(): return 1
                return get_width(node.left) + get_width(node.right)

            depth = get_depth(self.root)
            width = get_width(self.root)

            dynamic_width = max(figsize[0], width * 0.8)
            dynamic_height = max(figsize[1], depth * 1.0)

            fig, ax = plt.subplots(figsize=(dynamic_width, dynamic_height))
            ax.axis("off")
            
            self.leaf_count = 0
            self._plot_node(self.root, depth, 0, ax)
            plt.tight_layout()
            plt.show()

    def _plot_node(self, node, total_depth, current_depth, ax):
        if node is None:
            return None

        left_x = self._plot_node(node.left, total_depth, current_depth + 1, ax)
        right_x = self._plot_node(node.right, total_depth, current_depth + 1, ax)

        y = total_depth - current_depth
        
        if node.is_leaf_node():
            x = self.leaf_count
            self.leaf_count += 1
            
            if isinstance(node.value, (int, str)):
                 text = f"{node.value}"
            else:
                 text = f"{node.value:.2f}"

            bbox = dict(boxstyle="circle,pad=0.3", fc="lightgreen", ec="black")
        else:
            x = (left_x + right_x) / 2.0
            text = f"X[{node.feature}]\n<= {node.threshold:.2f}"
            bbox = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black")

        ax.text(x, y, text, ha="center", va="center", bbox=bbox, fontsize=10, zorder=10)

        if node.left:
            ax.plot([x, left_x], [y, y - 1], "k-", lw=1, zorder=1)
        if node.right:
            ax.plot([x, right_x], [y, y - 1], "k-", lw=1, zorder=1)

        return x