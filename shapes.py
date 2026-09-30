"""Small shape rules for previews, explanations, and friendly alignment errors."""
import math
import numpy as np


def shape_text(shape):
    return '(' + ','.join(map(str, shape)) + (',' if len(shape) == 1 else '') + ')'


def broadcast_shape(a, b):
    try:
        return np.broadcast_shapes(a, b)
    except ValueError as exc:
        raise ValueError(
            f"Shapes {shape_text(a)} and {shape_text(b)} don't align. "
            'Try transpose or reshape. Broadcasting compares axes from the right; '
            'each pair must match or one must be 1.'
        ) from exc


def matmul_shape(a, b):
    if not 1 <= len(a) <= 2 or not 1 <= len(b) <= 2:
        raise ValueError('Matmul needs vectors or 2D matrices. Try reshape first.')
    if a[-1] != (b[-2] if len(b) == 2 else b[0]):
        raise ValueError(f"Shapes {shape_text(a)} and {shape_text(b)} don't align for matmul. "
                         'The last axis of A must match the first axis of B. Try transpose or reshape.')
    return (a[:-1] if len(a) == 2 else ()) + (b[1:] if len(b) == 2 else ())


def predict_shape(op, A, B=None, params=None):
    # Import lazily to share the deck's parameter rules without a module cycle.
    from engine import OPERATIONS, axis_value, bounded_int, validate_windows, MAX_ELEMENTS
    if op not in OPERATIONS:
        raise ValueError('Choose an operation from the deck.')
    params = params or {}
    spec = OPERATIONS[op]
    a = A.shape
    b = B.shape if B is not None else None
    if spec['binary'] and B is None:
        raise ValueError('This block needs array B, too.')
    axis = axis_value(params) if 'axis' in spec['params'] else None
    if isinstance(axis, tuple) and op not in ('sum', 'mean', 'std', 'max', 'norm'):
        raise ValueError('Choose a single axis for this block.')
    out = a
    why = 'This block changes values, keeping every input axis.'
    left = f'A{shape_text(a)} · {spec["label"]}'
    if op in ('add', 'subtract', 'multiply', 'divide'):
        out = broadcast_shape(a, b)
        symbol = dict(add='+', subtract='−', multiply='×', divide='/')[op]
        left = f'A{shape_text(a)} {symbol} B{shape_text(b)}'
        why = 'Via broadcasting: align axes from the right; size-1 axes stretch to match.'
    elif op == 'matmul':
        out = matmul_shape(a, b)
        left = f'A{shape_text(a)} @ B{shape_text(b)}'
        why = 'The matching inner axis is summed away; the outer axes remain.'
    elif op == 'reshape':
        shape = tuple(int(x.strip()) for x in str(params.get('reshape_to', '1,9')).split(','))
        if not 1 <= len(shape) <= 6 or any(x < -1 or x == 0 or x > MAX_ELEMENTS for x in shape):
            raise ValueError('Use 1–6 positive dimensions; one dimension may be -1.')
        try:
            out = A.reshape(shape).shape
        except ValueError as exc:
            raise ValueError(f'Reshape must keep all {A.size} values. Try a shape whose dimensions multiply to {A.size}.') from exc
        why = f'The same {A.size} values are arranged into new axes; -1 infers the missing size.'
    elif op == 'transpose':
        out = a[::-1]
        why = 'Transpose reverses the axis order. A 1D vector stays 1D.'
    elif op == 'flatten':
        out = (A.size,)
        why = 'All axes collapse into one axis, with one slot per input value.'
    elif op == 'expand_dims':
        if axis is None:
            raise ValueError('Expand dims needs an axis: try 0 or -1.')
        if len(a) >= 6:
            raise ValueError('Bench limit: 6 dimensions.')
        out = np.expand_dims(A, axis).shape
        why = f'A new size-1 axis is inserted at axis {axis}; no values are added.'
    elif op == 'squeeze':
        out = np.squeeze(A, axis=axis).shape
        why = 'Only size-1 axes are removed. All values are kept.'
    elif op == 'diagonal':
        if len(a) not in (1, 2):
            raise ValueError('Diagonal expects a vector or matrix.')
        out = (a[0], a[0]) if len(a) == 1 else (min(a),)
        why = 'A vector builds a square diagonal matrix; a matrix yields its main diagonal.'
    elif op in ('sum', 'mean', 'std', 'max', 'argmax', 'argmin', 'norm'):
        axes = tuple(range(len(a))) if axis is None else axis if isinstance(axis, tuple) else (axis,)
        if a:
            if any(not -len(a) <= x < len(a) for x in axes):
                raise ValueError(f'Axis {axis} does not exist in shape {shape_text(a)}. Try None or -1.')
            axes = tuple(x % len(a) for x in axes)
            if len(set(axes)) != len(axes):
                raise ValueError('Choose distinct axes to reduce.')
        elif axis is not None and axis not in (0, -1):
            raise ValueError('A scalar has no axes. Choose None.')
        keep = params.get('keepdims', False) is True
        out = tuple(1 if i in axes else n for i, n in enumerate(a)) if keep else tuple(n for i, n in enumerate(a) if i not in axes)
        why = f'Reduced axes {axis if axis is not None else "(all)"} ' + ('stay as size 1 for broadcasting.' if keep else 'disappear; the other axes keep their sizes.')
    elif op == 'unique':
        out = None
        why = 'The output is 1D. Its length is the number of distinct values, known after running.'
    elif op in ('det', 'inverse'):
        if len(a) != 2 or a[0] != a[1]:
            raise ValueError('Use a square 2D matrix for det / inverse.')
        out = () if op == 'det' else a
        why = 'A square matrix has one scalar determinant.' if op == 'det' else 'An inverse has the same square shape as its input.'
    elif op in ('standardize', 'softmax'):
        if axis is not None and (not a or not -len(a) <= axis < len(a)):
            raise ValueError(f'Axis {axis} does not exist in shape {shape_text(a)}. Try None.')
        why = 'Statistics keep size-1 axes, then broadcast back across A. The output keeps A’s shape.'
    elif op == 'sort':
        if axis is None:
            out = (A.size,)
        elif not -len(a) <= axis < len(a):
            raise ValueError('Choose an existing axis or None to flatten before sorting.')
        why = 'Sorting preserves axes; axis=None flattens all values first.'
    elif op == 'nearest_center':
        if len(a) != 2 or len(b) != 2 or a[1] != b[1]:
            raise ValueError('Points and centers need the same number of feature columns.')
        out = (a[0],)
        why = f'Broadcast to {shape_text((a[0], b[0], a[1]))}, sum features, then argmin gives one center index per point.'
    elif op == 'least_squares':
        if len(a) != 2 or len(b) not in (1, 2) or a[0] != b[0]:
            raise ValueError('Use A feature rows and one matching target row in B.')
        out = (a[1],) + b[1:]
        why = 'There is one fitted weight per feature column, for each target in B.'
    elif op in ('convolution', 'windows'):
        padding = bounded_int(params, 'padding', 0, 0, 16) if op == 'convolution' else 0
        stride = bounded_int(params, 'stride', 1, 1, 32)
        validate_windows(A, B, padding, stride)
        out = tuple((x + 2 * padding - k) // stride + 1 for x, k in zip(a, b))
        if op == 'windows':
            out += b
        why = 'Each spatial axis is floor((input + 2 × padding − kernel) / stride) + 1. ' + ('Window axes keep the kernel shape.' if op == 'windows' else 'Summing the kernel axes leaves one value per window.')
    elif op == 'pad':
        padding = bounded_int(params, 'padding', 0, 0, 16)
        out = tuple(n + 2 * padding for n in a)
        why = f'Add {padding} zeros at both ends of every axis.'
    if out is not None and math.prod(out) > MAX_ELEMENTS:
        raise ValueError('Predicted output exceeds 4096 values. Try smaller arrays.')
    return dict(shape=None if out is None else list(out),
                hint=f'{left} → {shape_text(out) if out is not None else "(unique count,)"}' + (' via broadcasting' if op in ('add','subtract','multiply','divide') else ''),
                explanation=why)
