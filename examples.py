"""Runnable learning examples; training loops are over steps, never samples."""
import inspect
import numpy as np


def linear_regression(A, B, learning_rate=0.05, steps=1000):
    y = B.reshape(-1, 1)
    w = np.zeros((A.shape[1], 1))
    for _ in range(steps):
        residual = A @ w - y
        gradient = A.T @ residual / len(A)
        w -= learning_rate * gradient
    return w


def logistic_regression(A, B, learning_rate=0.1, steps=1000, positive_weight=2.0):
    # Custom class-weighted binary cross-entropy, computed from logits.
    y = B.reshape(-1, 1)
    weights = np.where(y == 1, positive_weight, 1.0)
    w = np.zeros((A.shape[1], 1))
    for _ in range(steps):
        logits = A @ w
        probabilities = np.exp(-np.logaddexp(0, -logits))
        gradient = A.T @ (weights * (probabilities - y)) / weights.sum()
        w -= learning_rate * gradient
    logits = A @ w
    losses = np.where(y == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits))
    cost = np.sum(weights * losses) / weights.sum()
    return w, cost


def knn(A, B, labels, k=3):
    # A: queries; B: training rows. Sorted classes break vote ties.
    distances = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    neighbors = np.argsort(distances, axis=1, kind='stable')[:, :k]
    classes = np.unique(labels)
    votes = np.sum(labels[neighbors, None] == classes, axis=1)
    return classes[np.argmax(votes, axis=1)]


def gini(A):
    _, counts = np.unique(A, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)


def decision_stump(A, B):
    # All feature/threshold candidates in one tensor. A depth-one tree.
    classes = np.unique(B)
    ordered = np.sort(A, axis=0)
    thresholds = (ordered[:-1] + ordered[1:]) / 2
    left = A[:, None, :] <= thresholds[None, :, :]
    one_hot = (B[:, None] == classes).astype(float)
    lc = np.einsum('ntd,nc->tdc', left, one_hot)
    rc = one_hot.sum(axis=0) - lc
    nl, nr = lc.sum(-1), rc.sum(-1)
    gl = 1 - np.sum((lc / np.maximum(nl[..., None], 1)) ** 2, axis=-1)
    gr = 1 - np.sum((rc / np.maximum(nr[..., None], 1)) ** 2, axis=-1)
    score = np.where((nl > 0) & (nr > 0), (nl * gl + nr * gr) / len(A), np.inf)
    if not np.isfinite(score).any():
        raise ValueError('No valid split: every feature is constant.')
    row, feature = np.unravel_index(np.argmin(score), score.shape)
    return np.array([feature, thresholds[row, feature], score[row, feature]])


def cross_entropy(A, B):
    # B: one integer class ID per row. Log-sum-exp avoids log(0).
    shifted = A - np.max(A, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return -np.mean(log_probs[np.arange(len(A)), B.astype(int).reshape(-1)])


def tensor_puzzle(A, B):
    # (N, 1, D) - (1, K, D) → (N, K, D) → (N, K)
    return np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)


EXAMPLES = [
 dict(id='linear_gd',title='Linear Regression · gradient descent',function=linear_regression,A=[[1,0],[1,1],[1,2],[1,3]],B=[1,3,5,7],note='1000 vectorized updates. Result: intercept and slope. Compare with least squares; check loss convergence.'),
 dict(id='logistic',title='Logistic Regression · custom cost',function=logistic_regression,A=[[1,-2],[1,-1],[1,1],[1,2]],B=[0,0,1,1],note='Positive examples get 2× weight. Stable logaddexp cost. Hold out validation data to detect overfitting.'),
 dict(id='knn',title='KNN · vectorized majority vote',function=knn,A=[[0,0],[9,9]],B=[[0,0],[1,0],[0,1],[8,9],[9,8],[9,9]],note='k=3; training labels = [0,0,0,1,1,1]. Squared Euclidean distance. First class wins vote ties.'),
 dict(id='gini',title='Decision Tree · Gini impurity',function=gini,A=[0,0,1,1],B=[0],note='Pure node → 0. Balanced binary node → 0.5. Weight child impurities to choose a split.'),
 dict(id='stump',title='Decision Tree · best Gini split',function=decision_stump,A=[[0],[1],[3],[4]],B=[0,0,1,1],note='Returns [feature index, threshold, weighted impurity]. Depth-one tree; limiting depth helps curb overfitting.'),
 dict(id='cross_entropy',title='Softmax + Cross-Entropy',function=cross_entropy,A=[[1000,1001,1002],[-1000,-1000,-1000]],B=[2,0],note='Subtract max → exp → sum → log → select labels → mean. Stable even when the true-class probability underflows.'),
 dict(id='tensor_puzzle',title='Tensor puzzle · all pairs',function=tensor_puzzle,A=[[0,0],[2,2],[8,9]],B=[[0,1],[9,9]],note='Predict (3, 2). Expand both arrays, subtract, square, then reduce the feature dimension.'),
]


def public_examples():
    return [{**{k:v for k,v in e.items() if k != 'function'}, 'code':'import numpy as np\n\n'+inspect.getsource(e['function'])} for e in EXAMPLES]
