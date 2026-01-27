import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    
    test_idxs = indices[:n_test]
    train_idxs = indices[n_test:]
    
    return X[train_idxs], X[test_idxs], y[train_idxs], y[test_idxs]

def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def calculate_precision(y_true, y_pred):
    classes = np.unique(y_true)
    precision_scores = []
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        
        if (tp + fp) > 0:
            precision_scores.append(tp / (tp + fp))
        else:
            precision_scores.append(0.0)
            
    return np.mean(precision_scores)

def calculate_recall(y_true, y_pred):
    classes = np.unique(y_true)
    recall_scores = []
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        if (tp + fn) > 0:
            recall_scores.append(tp / (tp + fn))
        else:
            recall_scores.append(0.0)
            
    return np.mean(recall_scores)

def calculate_f1(y_true, y_pred):
    p = calculate_precision(y_true, y_pred)
    r = calculate_recall(y_true, y_pred)
    
    if (p + r) == 0:
        return 0.0
        
    return 2 * (p * r) / (p + r)

def calculate_confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
            
    return matrix

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0
        
    return 1 - (ss_res / ss_tot)