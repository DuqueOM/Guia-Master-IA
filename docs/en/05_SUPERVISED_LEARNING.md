# Module 05 - Supervised Learning

> **Goal:** master Linear/Logistic Regression, evaluation metrics, and model validation.
> **Phase:** 2 - ML Core | **Weeks 9–12**
> **Pathway course:** Introduction to Machine Learning: Supervised Learning

---

<a id="m05-0"></a>

## How to use this module (0→100 mode)

**Purpose:** build an exam-grade supervised pipeline:

- train (linear / logistic regression)
- evaluate (metrics)
- validate (train/test + K-fold)
- control overfitting (regularization)

### Learning outcomes (measurable)

By the end of this module you can:

- Implement Linear Regression and Logistic Regression from scratch.
- Derive MSE and Cross-Entropy gradients (including the `Xᵀ(ŷ - y)` form).
- Choose metrics based on FP/FN cost.
- Apply validation correctly and avoid leakage.
- Use **Shadow Mode (sklearn)** as a ground truth reference.
- Explain **Entropy vs Gini**, **Information Gain**, and the difference **Bagging vs Boosting** (concept-level).

### Capsule (required): Extreme vectorization (no Python loops)

Practical rule for the whole module:

- **Forbidden**: `for` loops over samples (`N`) or features (`D`) to compute predictions, losses, or gradients.
- **Allowed**: loops over training steps / epochs (`for step in range(...)`).

Target mental model: the ML core should look like:

- `logits = X @ W`
- `grad = X.T @ something`

Canonical NumPy patterns (with strict **shape discipline**, no loops):

```python
import numpy as np  # NumPy: linear algebra and vectorized operations


# ============================================================
# 1) Multiclass forward: logits = X @ W
# ============================================================
N = 5  # N: number of samples
D = 4  # D: number of features
K = 3  # K: number of classes

X = np.random.randn(N, D).astype(float)  # X:(N,D) input batch
assert X.shape == (N, D)  # Shape contract for X

W = np.random.randn(D, K).astype(float)  # W:(D,K) weights per class
assert W.shape == (D, K)  # Shape contract for W

logits = X @ W  # logits:(N,K) because (N,D)@(D,K)=(N,K)
assert logits.shape == (N, K)  # Contract: logits must be 2D (batch x classes)


# ============================================================
# 2) Binary logistic regression: vectorized gradient ∇w = (1/N) X^T(ŷ - y)
# ============================================================
w = np.random.randn(D).astype(float)  # w:(D,) binary weights
assert w.shape == (D,)  # Shape contract for w

y = (np.random.rand(N) > 0.5).astype(float)  # y:(N,) binary labels in {0,1}
assert y.shape == (N,)  # Shape contract for y

z = X @ w  # z:(N,) logits
assert z.shape == (N,)  # Shape contract for z

y_hat = 1.0 / (1.0 + np.exp(-z))  # sigmoid(z) vectorized (no loops)
assert y_hat.shape == (N,)  # Shape contract for ŷ

grad_w = (X.T @ (y_hat - y)) / N  # (D,N)@(N,)=(D,) (exam form)
assert grad_w.shape == (D,)  # Contract: gradient must match w shape
# ============================================================
# 3) Pairwise distances without loops (kNN / clustering):
#    dist2[i,j] = ||X_query[i] - X_train[j]||^2
# ============================================================
M = 6  # M: number of queries
X_train = np.random.randn(N, D).astype(float)  # X_train:(N,D)
X_query = np.random.randn(M, D).astype(float)  # X_query:(M,D)
assert X_train.shape == (N, D)  # Shape contract: X_train must be (N,D) to ensure correct broadcasting
assert X_query.shape == (M, D)  # Shape contract: X_query must be (M,D) to ensure correct broadcasting

# Algebra trick: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
q_norm2 = np.sum(X_query ** 2, axis=1, keepdims=True)  # (M,1) ||q_i||^2, sum of squares along each query vector
t_norm2 = np.sum(X_train ** 2, axis=1, keepdims=True).T  # (1,N) ||t_j||^2, sum of squares along each training vector
cross = X_query @ X_train.T  # (M,N) dot products between each (q_i, t_j), measures similarity between query and training vectors

dist2 = q_norm2 + t_norm2 - 2.0 * cross  # (M,N) squared distances, applying the algebra trick
dist2 = np.maximum(dist2, 0.0)  # Guard against negative zeros due to float error, ensuring distances are non-negative
assert dist2.shape == (M, N)  # Shape contract: distance matrix must be (M,N) to represent pairwise distances
```

Quick references (Spanish source):

- [Module 04 (Probability → Cross-Entropy)](04_PROBABILIDAD_ML.md)
- [Glossary](GLOSARIO.md)
- [Resources](RECURSOS.md)
- [Plan v4](PLAN_V4_ESTRATEGICO.md)
- [Plan v5](PLAN_V5_ESTRATEGICO.md)
- Rubric: `study_tools/RUBRICA_v1.md` (scope `M05` in `rubrica.csv`)

---

## What matters most (high-signal core)

### Linear Regression

- Model: `ŷ = Xθ`
- Loss (MSE): `J(θ) = (1/2m) ||Xθ - y||²`
- Gradient: `∇J = (1/m) Xᵀ(Xθ - y)`

### Logistic Regression

- Probabilistic view: `P(y=1|x) = σ(Xθ)`
- Loss (Binary Cross-Entropy / NLL):
  - `L = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)]`
- Gradient (must be derived and understood):
  - `∇L = (1/m) Xᵀ(ŷ - y)`

### Weights interpretation (deliverable)

Goal: connect the learned weight vector to “what the model is looking at”.

- Recommended dataset: MNIST (28×28) binary (e.g., 0 vs 1) via `sklearn.datasets.fetch_openml("mnist_784", as_frame=False)`.
- Train your logistic regression on flattened images (`784` features).
- Visualize:
  - take `theta[1:]` (no bias), reshape to `(28, 28)`, plot with `imshow`.
  - use a diverging colormap centered at 0 and save the figure.
- Write 5–10 lines interpreting positive vs negative regions.

### Metrics

- Confusion matrix: TP/TN/FP/FN.
- Accuracy vs Precision vs Recall vs F1.
- For imbalanced datasets: accuracy can lie.

### Validation (anti-leakage)

- Split *before* any target-dependent transformations.
- Fit preprocessing only on train data.
- Use K-fold to estimate performance more reliably.

---

## Recursion warning (before trees)

Decision trees are built **recursively**. If you don’t define and test stopping conditions, you’ll hit infinite recursion or unbounded depth.

- Minimum stopping conditions: `max_depth`, purity, `min_samples_split`, “no split improves”.
- Recommended resource: https://realpython.com/python-recursion/

### Micro-sprint (15 minutes): recursion you need for trees

Two rules you must internalize:

- **Base case:** the smallest case you can answer immediately (this is where recursion stops).
- **Recursive step:** reduce the problem to a smaller version of itself.

If you cannot state the base case in 1 line, your tree implementation will likely recurse forever.

#### Example: recursive sum (practice the mental model)

```python
from typing import Sequence

def sum_recursive(xs: Sequence[float]) -> float:
    # Base case: the sum of an empty list is 0
    if len(xs) == 0:
        return 0.0

    # Recursive step: reduce the problem size by removing the first element
    return float(xs[0]) + sum_recursive(xs[1:])


assert sum_recursive([]) == 0.0
assert sum_recursive([3.0]) == 3.0
assert sum_recursive([3.0, 2.0, 5.0]) == 10.0
```

#### Call stack (what Python is doing)

```text
sum_recursive([3, 2, 5])
= 3 + sum_recursive([2, 5])
    = 2 + sum_recursive([5])
        = 5 + sum_recursive([])
            = 0
```

#### Connection to Decision Trees: stopping conditions = base cases

When building a tree node, your base case should trigger when:

- `depth >= max_depth`
- the node is **pure** (all labels are the same)
- `n_samples < min_samples_split`
- no candidate split improves impurity (information gain <= 0)

Minimal debug during development: print `depth`, `n_samples`, and the chosen split criterion per node.

## Tree-Based Models (Week 12 add-on)

### Impurity and split quality

Two standard impurity measures:

- **Gini:** fast and common.
- **Entropy:** information-theoretic.

Split quality is typically measured via **Information Gain** (impurity decrease).

### Ensembles (intro)

- **Bagging (Random Forest):** independent trees trained on bootstrap samples; reduces variance.
- **Boosting (Gradient Boosting):** sequential trees trained to correct errors; reduces bias (but can overfit).

Deliverable in this repo:

- `scripts/decision_tree_from_scratch.py`

---

## 🎯 Topic-based progressive exercises + solutions

Rules:

- **Try first** without looking at solutions.
- **Suggested timebox:** 20–45 min per exercise.
- **Minimum success:** your solution must pass the `assert` checks.

---

### Exercise 5.1: Linear regression (Normal Equation) + recover weights

#### Prompt

1) **Basic**

- Generate a synthetic dataset: `y = Xw + noise`.

2) **Intermediate**

- Estimate `w_hat` with the normal equation using `np.linalg.solve`.

3) **Advanced**

- Verify `w_hat` is close to `w_true` and the MSE is small.

#### Solution

```python
import numpy as np

np.random.seed(0)
n, d = 500, 3
X = np.random.randn(n, d)
w_true = np.array([0.7, -1.5, 2.0])
noise = 0.05 * np.random.randn(n)
y = X @ w_true + noise

XtX = X.T @ X
Xty = X.T @ y
w_hat = np.linalg.solve(XtX, Xty)

mse = np.mean((X @ w_hat - y) ** 2)

assert w_hat.shape == (d,)
assert np.linalg.norm(w_hat - w_true) < 0.15
assert mse < 0.01
```

---

### Exercise 5.2: Linear regression (Gradient Descent) + compare to Normal Equation

#### Prompt

1) **Basic**

- Implement GD for MSE: `w <- w - α (1/n) X^T (Xw - y)`.

2) **Intermediate**

- Compare `w_gd` to `w_ne` (normal equation).

3) **Advanced**

- Verify the loss decreases (final loss <= initial loss).

#### Solution

```python
import numpy as np

np.random.seed(1)
n, d = 400, 4
X = np.random.randn(n, d)
w_true = np.array([1.0, -2.0, 0.5, 3.0])
y = X @ w_true + 0.1 * np.random.randn(n)

XtX = X.T @ X
Xty = X.T @ y
w_ne = np.linalg.solve(XtX, Xty)

w = np.zeros(d)
alpha = 0.05
losses = []
for _ in range(3000):
    r = X @ w - y
    grad = (X.T @ r) / n
    w = w - alpha * grad
    losses.append(float(np.mean(r**2)))

w_gd = w

assert losses[-1] <= losses[0]
assert np.linalg.norm(w_gd - w_ne) < 0.2
```

---

### Exercise 5.3: Metrics from confusion matrix (TP/TN/FP/FN)

#### Prompt

1) **Basic**

- Implement TP/TN/FP/FN for binary classification.

2) **Intermediate**

- Implement accuracy, precision, recall, F1.

3) **Advanced**

- Validate with a known case using `assert`.

#### Solution

```python
import numpy as np

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return float(acc), float(prec), float(rec), float(f1)


y_true = np.array([1, 1, 1, 0, 0, 0])
y_pred = np.array([1, 0, 1, 0, 1, 0])
tp, tn, fp, fn = confusion_counts(y_true, y_pred)

assert (tp, tn, fp, fn) == (2, 2, 1, 1)

acc, prec, rec, f1 = precision_recall_f1(y_true, y_pred)
assert np.isclose(acc, 4/6)
assert np.isclose(prec, 2/3)
assert np.isclose(rec, 2/3)
assert np.isclose(f1, 2/3)
```

---

### Exercise 5.4: Logistic regression - sigmoid + stable BCE

#### Prompt

1) **Basic**

- Implement `sigmoid(z)` using `np.clip` to avoid overflow.

2) **Intermediate**

- Implement numerically-stable Binary Cross-Entropy (use `clip`).

3) **Advanced**

- Verify:
  - BCE near 0 for near-perfect predictions.
  - BCE ≈ `-log(0.9)` when `y=1` and `p=0.9`.

#### Solution

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


y_true = np.array([1.0, 0.0, 1.0, 0.0])
y_pred_good = np.array([0.999, 0.001, 0.999, 0.001])
assert bce(y_true, y_pred_good) < 0.01
assert np.isclose(bce(np.array([1.0]), np.array([0.9])), -np.log(0.9), atol=1e-12)
```

---

### Exercise 5.5: Logistic regression gradient (numerical check)

#### Prompt

1) **Basic**

- Implement BCE gradient for logistic regression:
  - `ŷ = sigmoid(Xw)`
  - `∇w = (1/n) X^T (ŷ - y)`

2) **Intermediate**

- Implement the loss `L(w)`.

3) **Advanced**

- Check one gradient coordinate using central differences.

#### Solution

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce_from_logits(X: np.ndarray, y: np.ndarray, w: np.ndarray, eps: float = 1e-15) -> float:
    logits = X @ w
    y_hat = sigmoid(logits)
    y_hat = np.clip(y_hat, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))


def grad_bce(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    y_hat = sigmoid(X @ w)
    return (X.T @ (y_hat - y)) / X.shape[0]


np.random.seed(2)
n, d = 200, 3
X = np.random.randn(n, d)
w0 = np.array([0.3, -0.7, 1.2])
probs = sigmoid(X @ w0)
y = (np.random.rand(n) < probs).astype(float)

w = np.random.randn(d)
g = grad_bce(X, y, w)

idx = 1
h = 1e-6
e = np.zeros(d)
e[idx] = 1.0
L_plus = bce_from_logits(X, y, w + h * e)
L_minus = bce_from_logits(X, y, w - h * e)
g_num = (L_plus - L_minus) / (2.0 * h)

assert np.isclose(g[idx], g_num, rtol=1e-4, atol=1e-6)
```

---

### Exercise 5.6: Threshold and precision/recall trade-off

#### Prompt

1) **Basic**

- Given probabilities `p` and labels `y`, build predictions with threshold `t`.

2) **Intermediate**

- Compute precision/recall for `t=0.5` and `t=0.3`.

3) **Advanced**

- Verify that lowering the threshold typically increases recall (on the same dataset).

#### Solution

```python
import numpy as np

def predict_threshold(p: np.ndarray, t: float) -> np.ndarray:
    return (np.asarray(p) >= t).astype(int)


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    eps = 1e-12
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return float(prec), float(rec)


np.random.seed(3)
y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
p = np.array([0.9, 0.6, 0.55, 0.52, 0.4, 0.35, 0.2, 0.1])

pred_05 = predict_threshold(p, 0.5)
pred_03 = predict_threshold(p, 0.3)

prec05, rec05 = precision_recall(y_true, pred_05)
prec03, rec03 = precision_recall(y_true, pred_03)

assert rec03 >= rec05
```

---

### Exercise 5.7: L2 regularization (Ridge) + weight norm

#### Prompt

1) **Basic**

- Implement Ridge Regression: `(X^T X + λI) w = X^T y`.

2) **Intermediate**

- Compare `||w_ridge||` vs `||w_ols||`.

3) **Advanced**

- Verify that for `λ>0`, typically `||w_ridge|| <= ||w_ols||`.

#### Solution

```python
import numpy as np

np.random.seed(4)
n, d = 300, 5
X = np.random.randn(n, d)
w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
y = X @ w_true + 0.2 * np.random.randn(n)

XtX = X.T @ X
Xty = X.T @ y
w_ols = np.linalg.solve(XtX, Xty)

lam = 10.0
w_ridge = np.linalg.solve(XtX + lam * np.eye(d), Xty)

assert np.linalg.norm(w_ridge) <= np.linalg.norm(w_ols) + 1e-8
```

---

### Exercise 5.8: Reproducible train/test split (seed)

#### Prompt

1) **Basic**

- Implement `train_test_split(X,y,test_size,seed)`.

2) **Intermediate**

- Verify that with the same seed you get identical splits.

3) **Advanced**

- Verify shapes and that no samples are lost.

#### Solution

```python
import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 0):
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


np.random.seed(0)
X = np.random.randn(100, 2)
y = (np.random.rand(100) < 0.5).astype(int)

Xtr1, Xte1, ytr1, yte1 = train_test_split(X, y, test_size=0.25, seed=42)
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, test_size=0.25, seed=42)

assert np.allclose(Xtr1, Xtr2)
assert np.allclose(Xte1, Xte2)
assert np.all(ytr1 == ytr2)
assert np.all(yte1 == yte2)
assert Xtr1.shape[0] + Xte1.shape[0] == 100
```

---

### Exercise 5.9: K-Fold cross-validation (correct partition)

#### Prompt

1) **Basic**

- Implement a fold generator (train/val indices).

2) **Intermediate**

- Verify each index appears exactly once in validation.

3) **Advanced**

- Verify train/val do not overlap.

#### Solution

```python
import numpy as np

def kfold_indices(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


n = 23
k = 5
seen = np.zeros(n, dtype=int)
for tr, va in kfold_indices(n, k, seed=123):
    assert len(np.intersect1d(tr, va)) == 0
    seen[va] += 1
assert np.all(seen == 1)
```

---

### Exercise 5.10: Trees - Gini and Information Gain (1D split)

#### Prompt

1) **Basic**

- Implement Gini impurity for binary labels.

2) **Intermediate**

- For a 1D feature and threshold `t`, compute Information Gain.

3) **Advanced**

- Find the best threshold among candidates and verify the result.

#### Solution

```python
import numpy as np

def gini(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.size == 0:
        return 0.0
    p1 = np.mean(y == 1)
    p0 = 1.0 - p1
    return float(1.0 - (p0**2 + p1**2))


def info_gain_gini(x: np.ndarray, y: np.ndarray, t: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    parent = gini(y)
    left = y[x <= t]
    right = y[x > t]
    w_left = left.size / y.size
    w_right = right.size / y.size
    child = w_left * gini(left) + w_right * gini(right)
    return float(parent - child)


x = np.array([0.1, 0.2, 0.25, 0.8, 0.85, 0.9])
y = np.array([0, 0, 0, 1, 1, 1])

candidates = [0.2, 0.25, 0.8]
gains = [info_gain_gini(x, y, t) for t in candidates]
best_t = candidates[int(np.argmax(gains))]

assert best_t in [0.25, 0.8]
assert max(gains) > 0.0
```

---

### (Bonus) Exercise 5.11: Shadow Mode - compare GD vs closed-form on a mini dataset

#### Prompt

- Fit linear regression with GD and compare predictions against the closed-form solution on a tiny dataset.

#### Solution

```python
import numpy as np

np.random.seed(5)
n, d = 30, 2
X = np.random.randn(n, d)
w_true = np.array([1.2, -0.4])
y = X @ w_true + 0.01 * np.random.randn(n)

w_ne = np.linalg.solve(X.T @ X, X.T @ y)

w = np.zeros(d)
alpha = 0.1
for _ in range(2000):
    grad = (X.T @ (X @ w - y)) / n
    w = w - alpha * grad

y_ne = X @ w_ne
y_gd = X @ w

assert np.mean((y_ne - y_gd) ** 2) < 1e-4
```

## Deliverables

- `supervised_learning.py`:
  - linear regression (normal equation + gradient descent)
  - logistic regression (+ optional L1/L2)
  - core metrics
  - validation helpers (split, K-fold)

- `artifacts/m05_logreg_weights.png`:
  - 28×28 visualization of learned logistic regression weights + short interpretation

- `scripts/decision_tree_from_scratch.py`:
  - simple decision tree from scratch
  - runnable training + reporting on a toy dataset

Minimum acceptance criteria:

- You can run your implementations end-to-end.
- You can explain the gradients / split logic in plain language.
- Shadow Mode checks are reasonably close.

---

## Completion checklist (v3.3)

### Knowledge

- [ ] Implemented linear regression (Normal Equation + GD).
- [ ] Understand MSE and its gradient.
- [ ] Implemented logistic regression from scratch.
- [ ] Understand sigmoid and binary cross-entropy.
- [ ] Can compute TP/TN/FP/FN from a confusion matrix.
- [ ] Implemented accuracy, precision, recall, F1.
- [ ] Implemented train/test split.
- [ ] Implemented K-fold cross validation.
- [ ] Understand L1 vs L2 regularization.
- [ ] Understand impurity and Information Gain (trees).

### Shadow Mode (required)

- [ ] **Linear Regression:** my MSE ≈ sklearn (ratio < 1.1).
- [ ] **Logistic Regression:** my accuracy ≈ sklearn (diff < 5%).

### Code deliverables

- [ ] `supervised_learning.py` with tests passing.
- [ ] `mypy` passes.
- [ ] `pytest` passes.
- [ ] `scripts/decision_tree_from_scratch.py` runs and reports train/test accuracy.
- [ ] `artifacts/m05_logreg_weights.png` saved and interpreted (5–10 lines).

### Analytical derivation (required)

- [ ] Derived cross-entropy gradient by hand.
- [ ] Can explain why `∇L = Xᵀ(ŷ - y)`.

### Feynman

- [ ] Explain sigmoid in 5 lines.
- [ ] Explain Cross-Entropy vs MSE in 5 lines.
- [ ] Explain Bagging vs Boosting in 5 lines.

---

## Navigation

| Previous | Index | Next |
|---|---|---|
| [04_PROBABILIDAD_ML](04_PROBABILIDAD_ML.md) | [00_INDICE](00_INDICE.md) | [06_UNSUPERVISED_LEARNING](06_UNSUPERVISED_LEARNING.md) |
