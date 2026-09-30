"""Five vectorized reference solutions. Only NumPy is required."""
import inspect
import numpy as np


def standardize(A, axis=0):
    mean = np.mean(A, axis=axis, keepdims=True)
    std = np.std(A, axis=axis, keepdims=True)
    scale = np.where(std == 0, 1, std)
    return (A - mean) / scale


def softmax(A, axis=-1):
    shifted = A - np.max(A, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def nearest_center(A, B):
    delta = np.expand_dims(A, 1) - np.expand_dims(B, 0)
    distances = np.sum(np.square(delta), axis=-1)
    return np.argmin(distances, axis=1)


def least_squares(A, B):
    # Handles rank deficiency without inverting X.T @ X.
    return np.linalg.lstsq(A, B, rcond=None)[0]


def convolution(A, B, padding=0, stride=1):
    # ML convention: cross-correlation (the kernel is not flipped).
    padded = np.pad(A, padding)
    windows = np.lib.stride_tricks.sliding_window_view(padded, B.shape)
    windows = windows[::stride, ::stride]
    return np.sum(windows * B, axis=(-2, -1))


DRILLS = [
    dict(id='standardize', title='Standardize feature columns', description='Zero mean. Unit variance. Features on speaking terms.', tags=['axis=0', 'keepdims', 'broadcasting'], A=[[1,10,5],[2,20,5],[3,30,5]], B=[0], params=dict(axis='0'), function=standardize, steps=['mean → (1, 3)', 'std → (1, 3)', 'safe scale → subtract → divide'], note='Constant columns become zeros. Fit statistics on training data only.'),
    dict(id='softmax', title='Stable row-wise softmax', description='Big logits, small probabilities. No exp-losions.', tags=['numerical stability', 'axis=-1', 'keepdims'], A=[[1000,1001,1002],[-1000,-1000,-1000]], B=[0], params=dict(axis='-1'), function=softmax, steps=['max → (2, 1)', 'subtract → exp → (2, 3)', 'sum → divide → (2, 3)'], note='Subtract the row maximum before exp. Every output row sums to one.'),
    dict(id='nearest_center', title='Nearest center / KNN distance', description='Find your closest neighbor. Skip the small talk.', tags=['broadcasting','square','argmin'], A=[[0,0],[2,2],[8,9]], B=[[0,1],[9,9]], params={}, function=nearest_center, steps=['expand dims → (3, 1, 2) and (1, 2, 2)', 'subtract → square → sum → (3, 2)', 'argmin → (3,)'], note='Squared Euclidean distance preserves ranking. Ties pick the first center; memory is O(NKD).'),
    dict(id='least_squares', title='Least squares Linear Regression', description='Fit a line. Give the guesswork a day off.', tags=['matmul','transpose','lstsq'], A=[[1,0],[1,1],[1,2],[1,3]], B=[1,3,5,7], params={}, function=least_squares, steps=['A = [ones, feature] → (4, 2)', 'least squares → weights (2,)', 'A @ weights → predictions (4,)'], note='Leading ones fit an intercept. lstsq is safer than inv(A.T @ A) on collinear features.'),
    dict(id='convolution', title='Convolution 2D from scratch', description='A tiny sliding window. A surprisingly big idea.', tags=['padding','strides','multiply + sum'], A=[[1,2,3],[4,5,6],[7,8,9]], B=[[1,0],[0,-1]], params=dict(padding=1,stride=2), function=convolution, steps=['pad → (5, 5)', 'windows + stride → (2, 2, 2, 2)', 'multiply → sum last two axes → (2, 2)'], note='ML cross-correlation, no kernel flip. Output = floor((input + 2 × padding − kernel) / stride) + 1.'),
]


def public_drills():
    return [{**{k:v for k,v in d.items() if k != 'function'}, 'code':'import numpy as np\n\n'+inspect.getsource(d['function'])} for d in DRILLS]
