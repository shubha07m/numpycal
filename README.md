# numpycal

An offline NumPy calculator that makes shapes obvious. Pastel cards, real NumPy, zero cloud.

## Setup and run

On an Apple laptop with Miniconda or Miniforge installed:

```sh
cd /Users/shubh/numpycal
source ./setup.sh
python app.py
```

The setup script creates the **numpycal** Conda environment from `environment.yml` and activates it. It supports Bash and macOS Zsh. Running `bash setup.sh` also installs it, but activation in a subprocess cannot change your parent shell: run `conda activate numpycal` afterward. An already-working environment is reused.

Equivalent manual setup:

```sh
conda env create -f environment.yml
conda activate numpycal
python app.py
```

Python 3.11, NumPy, and Flask are the only direct dependencies. Conda resolves Flask's small transitive dependencies. Setup may download packages; **after setup, turn off Wi-Fi and run normally**. No model weights, accounts, CDN assets, fonts, external APIs, database, or npm build are involved.

`python app.py` opens **http://localhost:8765**. The single Python process binds only to `127.0.0.1:8765`. Stop it with Ctrl+C. For manual browser opening:

```sh
python app.py --no-browser
```

If port 8765 is occupied, stop the previous instance before starting another. Nothing needs hosting or deployment. All CSS and JavaScript are inline in `ui.html`; the browser sends small JSON requests via XMLHttpRequest to the loopback Flask process. There are no `fetch()` calls or remote application requests. No calculations leave the laptop. Reloading resets the bench; there is no persistence.

## Try it

1. Edit A and B with comma-separated rows, semicolons, or nested Python/JSON numeric lists.
2. Tap transpose, matmul, or softmax. The result shows its shape, dtype, and the exact executed NumPy expression.
3. Relevant parameters appear above the deck. Changing a parameter recalculates; **Run again** repeats the selected operation. Reductions default to `keepdims=False`; enable it to retain broadcastable axes. Standardize defaults to axis 0; softmax to -1.
4. **Copy** copies expression and output. **Use as A/B** chains the result into another block, preserving scalar and higher-dimensional shapes. 1D flattens; 2D wraps a vector into a row. For 3D+ inputs, both toggles are unselected until you explicitly flatten.
5. Open **5 ML drills**. **Reveal** shows the actual reference function; **Load + run** loads the fixture, calculates it, and displays the composition steps. Inside Reveal, **Load fixture only** lets you predict first.

The five drills are standardization, stable softmax, nearest center, least squares, and convolution. Extra lego buttons expose subtract, divide, square, exp, log, argmin, safe scale, zero padding, and sliding windows, so the drill internals can be assembled on the bench. For convolution, pad A → use as A → windows (B supplies kernel shape) → use as A → multiply B → use as A → sum over the last two axes. Set the stride on windows and leave keepdims unchecked.

The **Pattern Library** has runnable reference fixtures for learning with small models: linear regression gradient descent, logistic regression with a custom class-weighted cost, vectorized KNN voting, Gini impurity, a decision stump choosing the best Gini split, stable softmax cross-entropy, and all-pairs broadcasting. These examples deliberately run their labeled fixtures; the calculator operations run your edited bench inputs. The tree example is a depth-one classifier split, not a full recursive tree.

## Numerical choices and limits

- Standardization uses population standard deviation (`ddof=0`). Constant columns map to zero. Learn preprocessing statistics on training data, then reuse them on validation/test data.
- Softmax subtracts the maximum before exponentiating. Cross-entropy uses log-sum-exp, so incorrect predictions with extremely small probabilities still have finite loss.
- Least squares uses `np.linalg.lstsq`; explicitly inverting `X.T @ X` is unstable for collinear features. Include a column of ones to fit an intercept. The output includes training MSE and rank; low training loss alone does not establish generalization.
- Nearest center uses squared Euclidean distances. KNN votes across three nearest fixture neighbors. First center/sorted class wins ties. Pairwise tensors use O(NKD) memory.
- Convolution follows the ML **cross-correlation** convention: the kernel is not flipped. It uses zero padding, sliding window views, stride slicing, multiplication, and summation; no loops over pixels. For mathematical convolution, reverse the kernel along both axes first.
- Gradient descent loops over training steps; each sample/feature computation is vectorized. Logistic regression uses positive-class weight 2. Inspect held-out performance to catch overfitting.
- Shape-only operations preserve dtype. Arithmetic and ML operations explicitly cast to float64, shown in the expression, to avoid silent integer overflow. Display rounds to seven decimal places; result reuse retains full numeric precision.
- Finite real numeric literals only; input is never executed as Python code. Up to 4096 values and 6 dimensions per array/result. Larger pairwise/window intermediates are bounded. Matmul accepts vectors and 2D matrices. Invalid shapes, singular inverse, overflow, log-domain errors, and division by zero produce readable errors without stopping the app.


## Verify offline

```sh
conda activate numpycal
python -m unittest -v
```

The stdlib suite covers every deck operation and runnable fixture, stable probabilities and loss, gradient convergence, rank-deficient least squares, scalar/tensor round trips, rejected inputs, and convolution against an independent loop oracle for several paddings and strides. Loops in that test provide an independent correctness check; the app implementation remains vectorized.

An optional browser check uses an already-installed Google Chrome and Node 22+ (neither is an app dependency; no downloads or npm packages):

```sh
# Leave python app.py running in another terminal.
node scripts/browser_smoke.mjs
```

It exercises desktop/mobile UI flows and checks for remote page requests. Screenshots are written under `.cache/browser/`.

## Files

- `app.py`: loopback Flask entry point, routes, error handling.
- `ui.html`: the entire inline vanilla HTML/CSS/JS interface.
- `engine.py`: literal parser, bounded operation deck, NumPy execution.
- `drills.py`: the five reference solutions and fixtures.
- `examples.py`: additional learning examples.
- `environment.yml`, `setup.sh`: isolated Python environment setup.
- `test_numpycal.py`, `scripts/browser_smoke.mjs`: offline verification.
- `static/`: reserved; all current assets are inline.
