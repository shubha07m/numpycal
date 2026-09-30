# numpycal

An offline NumPy calculator that makes shapes obvious. Pastel cards, real NumPy, zero cloud.

A tiny playground for anyone learning NumPy shapes, broadcasting, and vectorization. Edit arrays, snap together small operations, and see the result beside the code that made it.

## Setup

With Miniconda or Miniforge installed on your Mac:

```sh
cd /Users/shubh/numpycal
source ./setup.sh
python app.py
```

The script creates and activates the `numpycal` environment: Python 3.11, NumPy, and Flask. It reuses an existing working environment. Source it from Bash or Zsh to keep activation in your current shell; if you run `bash setup.sh`, follow it with `conda activate numpycal`.

Manual setup works too:

```sh
conda env create -f environment.yml
conda activate numpycal
python app.py
```

The app opens **http://localhost:8765**. Stop it with Ctrl+C. Use `python app.py --no-browser` to open the page yourself. If the port is occupied, stop the previous instance first.

Setup may download packages. After that, the app works with Wi-Fi off. One local Python process serves the interface and computes with NumPy. All CSS and JavaScript are inline, with no CDN, external API, models, database, tracking, or accounts. Browser requests stay on the loopback connection. No new dependencies or JavaScript build step are needed.

## Try it

1. Edit A and B using commas and newlines, semicolons, or nested numeric lists. The shape badges update as you type. Use 1D to flatten or 2D to wrap a vector into a row.
2. Hover or keyboard-focus an operation to preview its shape. For example, `A(3,3) + B(3,1) → (3,3) via broadcasting`. Tap to run; relevant parameters appear above the deck. Editing a parameter previews it; committing the change recalculates.
3. Read the output shape, dtype, and exact expression. Open **Explain shape** for a short explanation of the axes that stay, stretch, or disappear.
4. **Copy result** copies a full-precision numeric literal. **Copy as Python** copies a runnable snippet with the original inputs, operation, and shape/dtype comment. **Use as A/B** feeds the result into the next block, preserving scalar and higher-dimensional shapes.
5. Click one of the **last five runs** above Output to restore its saved inputs and parameters and run it again. History includes successful calculator, pattern, and library runs. It stays in memory and resets on reload.
6. Open **5 Patterns** for standardization, stable softmax, nearest centers, least squares, and 2D convolution. **Reveal** shows the reference function; **Load + run** loads the fixture and its composition steps. Inside Reveal, **Load fixture only** leaves the calculation to you.

The **Pattern Library** adds seven runnable examples: line fitting by gradient descent, binary classification with a custom weighted cost, nearest-neighbor voting, Gini impurity, a decision stump, cross-entropy, and all-pairs distances. Library examples run their labeled fixtures; calculator buttons always use the current bench inputs.

Keyboard shortcuts work in the Calculator tab when you are outside a text editor, parameter field, or dropdown:

| Key | Action |
| --- | --- |
| `1`–`6` | Reshape, transpose, flatten, expand dims, squeeze, diagonal |
| `7`–`9` | Sum, mean, std |
| `c` | Copy as Python |
| `u` | Use result as A |

**More lego blocks** exposes subtract, divide, square, exp, log, argmin, safe scale, padding, and sliding windows. Try composing convolution: pad A → use as A → windows (B supplies the kernel shape) → use as A → multiply B → use as A → sum over the last two axes. Set stride on windows and leave keepdims unchecked.

## Numerical notes

- Standardization uses population standard deviation (`ddof=0`); constant columns become zero. Fit statistics on training data, then reuse them on validation/test data.
- Softmax subtracts the maximum before exponentiating. Cross-entropy uses log-sum-exp so very small true-class probabilities do not cause `log(0)`.
- Least squares uses `np.linalg.lstsq` for rank-deficient inputs. Add a column of ones for an intercept. Output notes include rank and training MSE; check held-out data to assess generalization.
- Nearest-center distances are squared Euclidean distances. The KNN fixture votes over three neighbors; ties choose the first center or sorted class. Pairwise tensors require O(NKD) memory.
- Convolution follows the ML cross-correlation convention: no kernel flip. It combines zero padding, window views, stride slicing, multiplication, and summation. Reverse both kernel axes first for mathematical convolution.
- Gradient-descent loops iterate over training steps; sample and feature computations are vectorized. The binary-classification example assigns positive examples twice the weight. The decision stump selects a single Gini split, not a full recursive tree.
- Shape operations preserve dtype; arithmetic and ML operations explicitly use float64 to avoid silent integer overflow. Displayed values are rounded to seven decimal places. Copy and reuse retain full precision.
- Shape previews use shape rules, not the numerical operation. They cannot tell whether an inverse is singular or determine the number of unique values before running. Actual calculation still validates numeric constraints.
- Inputs are finite real numeric literals and are never executed as code. Each array/result is limited to 4096 values and six dimensions, with additional bounds on larger intermediates. Matmul accepts vectors and 2D matrices. Invalid inputs and numerical errors appear inline without stopping the app.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Local Flask entry point, routes, and errors |
| `ui.html` | Inline HTML, CSS, and vanilla JavaScript |
| `engine.py` | Input parsing, operation deck, NumPy execution, Python export |
| `shapes.py` | Shape previews, explanations, and alignment guidance |
| `drills.py` | Five pattern reference functions and fixtures |
| `examples.py` | Seven Pattern Library examples |
| `environment.yml`, `setup.sh` | Conda environment setup |
| `test_numpycal.py` | Offline numerical and API checks |
| `scripts/browser_smoke.mjs` | Optional desktop/mobile browser checks |

The reserved `static/` directory is unused; the interface needs no external assets.

## Verify offline

```sh
conda activate numpycal
python -m unittest -v
```

The suite covers all operations and fixtures, stable probabilities and losses, gradient convergence, convolution against an independent oracle, invalid inputs, scalar/tensor reuse, shape predictions, and runnable Python exports.

An optional browser check uses already-installed Google Chrome and Node 22+; neither is an app dependency, and no npm packages are needed. Leave the app running in another terminal:

```sh
node scripts/browser_smoke.mjs
```

It checks desktop/mobile layouts, pattern loading, history replay, live badges, shape hints, copy contents, shortcuts, and errors. Clipboard writes are captured inside the test so your clipboard is untouched. It also checks that the page makes no remote requests. Screenshots are saved in `.cache/browser/`.
