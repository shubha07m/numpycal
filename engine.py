"""Bounded, literal-only input parsing and a fixed operation deck."""
import ast
import inspect
import json
import math
import re
import numpy as np
import drills
from examples import EXAMPLES

MAX_ELEMENTS = 4096
MAX_WORK = 1_000_000

# id, label, hint, icon, parameter names, uses B
GROUPS = [
 ('SHAPE', [
 ('reshape','reshape','target shape','↪',['reshape_to'],False),
 ('transpose','transpose','swap axes','Aᵀ',[],False),
 ('flatten','flatten','copy to 1D','—',[],False),
 ('expand_dims','expand dims','uses axis','+□',['axis'],False),
 ('squeeze','squeeze','size-1 axes','−□',['axis'],False),
 ('diagonal','diagonal','extract / build','↘',[],False)]),
 ('REDUCE',[(x,x,h,i,['axis','keepdims'] if x != 'unique' else [],False) for x,h,i in [
 ('sum','uses axis','Σ'),('mean','uses axis','x̄'),('std','uses axis','σ'),('max','uses axis','↑'),('argmax','index of max','i*'),('unique','sorted values','≠')]]),
 ('LINALG + ML',[
 ('norm','norm','magnitude','‖A‖',['axis','keepdims'],False),
 ('det','det','square matrix','|A|',[],False),('inverse','inverse','square matrix','A⁻¹',[],False),
 ('standardize','standardize','per axis','z',['axis'],False),('softmax','softmax','stable','pᵢ',['axis'],False),
 ('clip','clip','min / max','[a,b]',['clip_min','clip_max'],False),('sort','sort','uses axis','⇅',['axis'],False)]),
 ('COMBINE + LINALG',[
 ('add','add','A + B','+ ',[],True),('multiply','multiply','A × B','×',[],True),('matmul','matmul','A @ B','@',[],True),
 ('nearest_center','nearest center','A points · B centers','μ*',[],True),('least_squares','least squares','A features · B y','β',[],True),
 ('convolution','convolution','A image · B kernel','⊞',['padding','stride'],True)]),
 ('EXTRA LEGO BLOCKS',[
 ('subtract','subtract','A − B','−',[],True),('divide','divide','A / B','÷',[],True),
 ('square','square','elementwise','²',[],False),('exp','exp','elementwise','eˣ',[],False),('log','log','natural logarithm','ln',[],False),
 ('argmin','argmin','index of min','i↓',['axis','keepdims'],False),
 ('safe_scale','safe scale','replace zeros with 1','0→1',[],False),
 ('pad','pad','zero padding','□',['padding'],False),('windows','windows','B shape · slice stride','▦',['stride'],True)])]
OPERATIONS = {row[0]:dict(id=row[0],label=row[1],hint=row[2],icon=row[3],params=row[4],binary=row[5]) for _,rows in GROUPS for row in rows}


def parse_array(text, mode='auto'):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Enter a few numbers first.')
    if len(text) > 100_000:
        raise ValueError('Keep the bench small: at most 4096 numbers.')
    try:
        if text.strip().startswith(('[','(')) or (mode == 'auto' and not re.search(r'[,;\s]', text.strip())):
            data = ast.literal_eval(text.strip())
        else:
            rows = [r.strip() for r in re.split(r'[;\n]', text.strip()) if r.strip()]
            data = [[ast.literal_eval(v) for v in re.split(r'[,\s]+', r) if v] for r in rows]
            if len(data) == 1:
                data = data[0]
        result = np.asarray(data)
    except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError('Use numbers separated by commas; each matrix row needs the same length.') from exc
    if result.dtype.kind not in 'iuf' or result.size == 0 or not np.isfinite(result).all():
        raise ValueError('Use finite real numbers. No strings, NaN, infinity, or empty arrays.')
    if result.size > MAX_ELEMENTS or result.ndim > 6:
        raise ValueError('Bench limit: 4096 numbers and 6 dimensions.')
    if mode == '1':
        result = result.reshape(-1)
    elif mode == '2':
        result = np.atleast_2d(result)
        if result.ndim != 2:
            raise ValueError('This tensor has more than two axes. Choose 1D to flatten it.')
    elif mode != 'auto':
        raise ValueError('Unknown array mode.')
    # Never silently overflow integer arithmetic in the interactive bench.
    if result.dtype.kind in 'iu' and np.any(np.abs(result.astype(float)) > 2**53):
        raise ValueError('Keep integer inputs within ±2**53 for safe floating-point ML operations.')
    return result


def packed(value):
    arr = np.asarray(value)
    if arr.size > MAX_ELEMENTS:
        raise ValueError('Result exceeds the 4096-element bench limit. Try smaller arrays.')
    if not np.isfinite(arr).all():
        raise ValueError('Non-finite result. Scale the input or check division/log domains.')
    return dict(result=np.array2string(arr, precision=7, threshold=MAX_ELEMENTS, max_line_width=88),
                shape=str(arr.shape), dtype=str(arr.dtype), data=arr.tolist(),
                literal=json.dumps(arr.tolist(), allow_nan=False), ndim=arr.ndim)


def axis_value(params):
    value = str(params.get('axis', 'None'))
    if value == 'None': return None
    if value in ('0','1','-1'): return int(value)
    if value == '-2,-1': return (-2, -1)
    raise ValueError('Choose None, 0, 1, -1, or the last two axes.')


def bounded_int(params, name, default, minimum, maximum):
    value = int(params.get(name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}.')
    return value


def source_call(fn, args):
    return 'import numpy as np\n\n' + inspect.getsource(fn) + '\nresult = ' + fn.__name__ + '(' + args + ')'


def calculate(op, A, B=None, params=None):
    if op not in OPERATIONS: raise ValueError('Choose an operation from the deck.')
    params = params or {}
    spec = OPERATIONS[op]
    axis = axis_value(params) if 'axis' in spec['params'] else None
    keep = params.get('keepdims', False) is True
    if isinstance(axis, tuple) and op not in ('sum','mean','std','max','norm'):
        raise ValueError('This operation takes one axis; select None, 0, 1, or -1.')
    # Float conversion is explicit in expressions; shape-only ops preserve dtype.
    numeric = op in {'sum','mean','std','norm','det','inverse','standardize','softmax','add','subtract','multiply','divide','matmul','square','exp','log','nearest_center','least_squares','convolution'}
    prefix = ''
    if numeric:
        A = A.astype(np.float64)
        prefix = 'A = A.astype(np.float64)\n'
        if B is not None:
            B = B.astype(np.float64)
            prefix += 'B = B.astype(np.float64)\n'
    if spec['binary'] and B is None: raise ValueError('This operation needs B.')
    note = ''
    steps = []
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        if op in ('sum','mean','std','max','argmax','argmin'):
            result = getattr(np, op)(A, axis=axis, keepdims=keep)
            expression = f'np.{op}(A, axis={axis!r}, keepdims={keep})'
        elif op == 'norm':
            result = np.sqrt(np.sum(A * A, axis=axis, keepdims=keep))
            expression = f'np.sqrt(np.sum(A * A, axis={axis!r}, keepdims={keep}))'
        elif op == 'reshape':
            shape = tuple(int(x.strip()) for x in str(params.get('reshape_to','1,9')).split(','))
            if not 1 <= len(shape) <= 6 or any(x < -1 or x == 0 or x > MAX_ELEMENTS for x in shape):
                raise ValueError('Use 1–6 positive dimensions; one dimension may be -1.')
            result = A.reshape(shape); expression = f'A.reshape({shape!r})'
        elif op == 'transpose': result = A.T; expression = 'A.T'
        elif op == 'flatten': result = A.flatten(); expression = 'A.flatten()'
        elif op == 'expand_dims':
            if axis is None: raise ValueError('Expand dims needs an axis: try 0 or -1.')
            if A.ndim >= 6: raise ValueError('Bench limit: 6 dimensions.')
            result = np.expand_dims(A,axis); expression = f'np.expand_dims(A, axis={axis})'
        elif op == 'squeeze': result = np.squeeze(A,axis=axis); expression = f'np.squeeze(A, axis={axis!r})'
        elif op == 'diagonal':
            if A.ndim not in (1,2): raise ValueError('Diagonal expects a vector or matrix.')
            if A.ndim == 1 and A.size**2 > MAX_ELEMENTS: raise ValueError('Diagonal would exceed the bench limit.')
            result = np.diag(A); expression = 'np.diag(A)'
        elif op == 'unique': result = np.unique(A); expression = 'np.unique(A)'
        elif op in ('det','inverse'):
            if A.ndim != 2 or A.shape[0] != A.shape[1]: raise ValueError('Use a square 2D matrix for det / inverse.')
            fn = np.linalg.det if op == 'det' else np.linalg.inv
            result = fn(A); expression = f'np.linalg.{"det" if op == "det" else "inv"}(A)'
            if np.linalg.cond(A) > 1e10: note = 'Ill-conditioned matrix: small input changes can cause large answer changes.'
        elif op in ('standardize','softmax'):
            fn = getattr(drills,op); result = fn(A,axis=axis)
            expression = source_call(fn,f'A, axis={axis!r}')
            note = 'Constant columns become zero.' if op == 'standardize' else 'Subtract max before exp; the selected axis sums to one.'
        elif op == 'clip':
            lo,hi = float(params.get('clip_min',2)),float(params.get('clip_max',8))
            if not np.isfinite([lo,hi]).all() or lo > hi: raise ValueError('Clip min must be finite and no greater than clip max.')
            result = np.minimum(np.maximum(A,lo),hi); expression = f'np.minimum(np.maximum(A, {lo!r}), {hi!r})'
        elif op == 'sort': result = np.sort(A,axis=axis); expression = f'np.sort(A, axis={axis!r})'
        elif op in ('add','subtract','multiply','divide'):
            if math.prod(np.broadcast_shapes(A.shape,B.shape)) > MAX_ELEMENTS: raise ValueError('Broadcast result exceeds 4096 elements.')
            result = getattr(np,op)(A,B)
            expression = f'A {dict(add="+",subtract="-",multiply="*",divide="/")[op]} B'
        elif op == 'matmul':
            if A.ndim > 2 or B.ndim > 2: raise ValueError('Matmul on this bench takes vectors or 2D matrices.')
            if A.size * B.size > MAX_WORK: raise ValueError('Try smaller matrices for this bench.')
            result = A @ B; expression = 'A @ B'
        elif op in ('nearest_center','least_squares','convolution'):
            if A.ndim != 2: raise ValueError('A must be a 2D matrix.')
            if op == 'nearest_center':
                if B.ndim != 2 or A.shape[1] != B.shape[1]: raise ValueError('Points and centers need the same number of feature columns.')
                if len(A)*B.size > MAX_WORK: raise ValueError('Pairwise distance tensor too large. Try fewer points.')
                args='A, B'; result=drills.nearest_center(A,B)
                steps=[f'expand → {A[:,None,:].shape} and {B[None,:,:].shape}',f'distance matrix → {(len(A),len(B))}',f'argmin → {result.shape}']
            elif op == 'least_squares':
                if B.ndim not in (1,2) or len(B) != len(A): raise ValueError('B must have one target row per feature row in A.')
                if A.shape[1] * (B.shape[1] if B.ndim == 2 else 1) > MAX_ELEMENTS:
                    raise ValueError('Weight matrix would exceed the 4096-element bench limit.')
                args='A, B'; result=drills.least_squares(A,B)
                residual=np.mean((A @ result-B)**2)
                note=f'Training MSE: {residual:.6g}. Add a column of ones for an intercept. Rank: {np.linalg.matrix_rank(A)}/{A.shape[1]}.'
                steps=[f'features → {A.shape}',f'weights → {result.shape}',f'predictions → {(A @ result).shape}']
            else:
                padding=bounded_int(params,'padding',0,0,16); stride=bounded_int(params,'stride',1,1,32)
                validate_windows(A,B,padding,stride)
                args=f'A, B, padding={padding}, stride={stride}'
                result=drills.convolution(A,B,padding,stride)
                steps=[f'pad → {tuple(n+2*padding for n in A.shape)}',f'windows → {result.shape+B.shape}',f'multiply → sum → {result.shape}']
                note='ML cross-correlation: kernel is not flipped. Sum the last two axes of each window.'
            expression=source_call(getattr(drills,op),args)
        elif op in ('square','exp','log'):
            result=getattr(np,op)(A); expression=f'np.{op}(A)'
        elif op == 'safe_scale': result=np.where(A==0,1,A); expression='np.where(A == 0, 1, A)'
        elif op == 'pad':
            padding=bounded_int(params,'padding',0,0,16)
            if math.prod(n+2*padding for n in A.shape)>MAX_ELEMENTS: raise ValueError('Padded result too large.')
            result=np.pad(A,padding); expression=f'np.pad(A, {padding})'
        elif op == 'windows':
            stride=bounded_int(params,'stride',1,1,32); validate_windows(A,B,0,stride)
            result=np.lib.stride_tricks.sliding_window_view(A,B.shape)[::stride,::stride]
            expression=f'np.lib.stride_tricks.sliding_window_view(A, B.shape)[::{stride}, ::{stride}]'
    expression = 'import numpy as np\n\n' + prefix + expression.removeprefix('import numpy as np\n\n')
    return {**packed(result), 'expression':expression,'note':note,'steps':steps,'operation':op}


def validate_windows(A,B,padding,stride):
    if A.ndim != 2 or B.ndim != 2: raise ValueError('Image A and kernel B must both be 2D.')
    dims=tuple((a+2*padding-b)//stride+1 for a,b in zip(A.shape,B.shape))
    if min(dims)<1: raise ValueError('Kernel is larger than the padded image.')
    if math.prod(dims)*B.size>MAX_WORK: raise ValueError('Window tensor too large. Use a smaller image/kernel.')


def run_example(example_id):
    e=next((e for e in EXAMPLES if e['id']==example_id),None)
    if e is None: raise ValueError('Unknown example.')
    A,B=np.array(e['A'],dtype=float),np.array(e['B'],dtype=float)
    fn=e['function']; args='A, B'; note=e['note']
    if example_id=='gini': result=fn(A); args='A'
    elif example_id=='knn':
        labels=np.array([0,0,0,1,1,1]); result=fn(A,B,labels)
        args='A, B, labels=np.array([0, 0, 0, 1, 1, 1]), k=3'
    elif example_id=='logistic':
        result,cost=fn(A,B); note+=f' Final weighted cost: {cost:.7f}.'
    else: result=fn(A,B)
    expression = source_call(fn,args)
    if example_id == 'logistic':
        expression = expression.replace('result = logistic_regression(', 'weights, cost = logistic_regression(') + '\nresult = weights'
    return {**packed(result),'expression':expression,'note':note,'steps':[], 'operation':example_id,'fixture':dict(A=e['A'],B=e['B'])}
