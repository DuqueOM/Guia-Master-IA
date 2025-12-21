# Module 08 - Final Project: MNIST Analyst

> **Goal:** build an end-to-end pipeline that proves competence across the 3 Line-1 courses.
> **Phase:** 3 - Capstone | **Weeks 21–24** (4 weeks)
> **Dataset:** **Fashion-MNIST** (primary, 28×28, 10 classes) / MNIST (fallback, same format)

---

 <a id="m08-0"></a>

 ## How to use this module (0→100 mode)

 **Purpose:** demonstrate end-to-end mastery by integrating:

 - unsupervised learning (PCA + K-Means)
 - supervised learning (Logistic Regression OvA)
 - deep learning (MLP with manual backprop)

 **Execution rule:** keep everything reproducible and runnable end-to-end (one command, one report).

## What you are building

A 4-week, exam-grade project that demonstrates:

- Unsupervised learning (PCA + K-Means)
- Supervised learning (Logistic Regression One-vs-All)
- Deep learning (MLP with manual backprop)

Plus a final benchmark report and minimal deployment workflow.

---

## Schedule (4 weeks)

| Week | Focus | Course signal | Output |
|---|---|---|---|
| 21 | EDA + PCA + K-Means | Unsupervised Algorithms | PCA + K-Means working + plots |
| 22 | Classical classification | Supervised Learning | Logistic Regression OvA |
| 23 | Deep Learning | Intro to Deep Learning | MLP with manual backprop |
| 24 | Benchmark + report | Integration | `MODEL_COMPARISON.md` + minimal deployment |

Rubric:

- `study_tools/RUBRICA_v1.md` (scope `M08` in `rubrica.csv`)
- Hard condition: **PB-23 ≥ 80/100** (if PB-23 < 80 ⇒ “not ready” even if the global score is high)

---

## Practical notes (Week 24)

- **Fashion-MNIST (main run):** benchmark on Fashion-MNIST for a more realistic gap between LR vs MLP/CNN.
- **Dirty Data Check:** generate a corrupted dataset (noise/NaNs/inversion) with `scripts/corrupt_mnist.py` and document cleaning.
- **Minimal deployment:** train + save a CNN with `scripts/train_cnn_pytorch.py`, then predict a single 28×28 image with `scripts/predict.py`.

---

## 🎯 Phase-based progressive exercises + solutions

Rules:

- **Try first** without looking at solutions.
- **Suggested timebox:** 30–90 min per exercise.
- **Minimum success:** your solution must pass the `assert` checks.

Note: these exercises use **synthetic data** so they are reproducible without downloading MNIST. The goal is to validate pipeline *invariants* (shapes, numerical stability, metrics, convergence, reproducibility).

---

### Exercise 8.1: Reproducibility (seed) + deterministic split

#### Prompt

1) **Basic**

- Implement a reproducible train/test split using a seed.

2) **Intermediate**

- Verify the same seed produces the same split.

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
X = np.random.randn(100, 784)
y = np.random.randint(0, 10, size=(100,))

Xtr1, Xte1, ytr1, yte1 = train_test_split(X, y, test_size=0.25, seed=123)
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, test_size=0.25, seed=123)

assert np.allclose(Xtr1, Xtr2)
assert np.allclose(Xte1, Xte2)
assert np.all(ytr1 == ytr2)
assert np.all(yte1 == yte2)

assert Xtr1.shape[0] + Xte1.shape[0] == X.shape[0]
```

<details open>
<summary><strong>Pedagogical add-on — Exercise 8.1: Reproducibility + deterministic split</strong></summary>

#### 1) Metadata
- **ID (optional):** `M08-E08_1`
- **Estimated duration:** 20–35 min
- **Level:** Basic → Intermediate

#### 2) Key idea
- Reproducibility is a *pipeline invariant*: with the same seed, you must get the same split.
- A deterministic split is the foundation for fair model comparison in Week 24.

#### 3) Common mistakes
- Shuffling `X` and `y` independently (breaks alignment).
- Using a global RNG implicitly and then changing it elsewhere.
- Forgetting to check that `n_train + n_test == n`.

#### 4) Teaching note
- Ask the student to print the first 5 indices of `train_idx` and show they repeat across runs.
</details>

---

### Exercise 8.2: MNIST-like data invariants (shapes + ranges)

#### Prompt

1) **Basic**

- Simulate `uint8` images in `[0,255]` with shape `(n, 784)`.

2) **Intermediate**

- Normalize to float in `[0,1]`.

3) **Advanced**

- Verify no `NaN/inf` and dtype is float.

#### Solution

```python
import numpy as np

rng = np.random.default_rng(1)
n = 256
X_uint8 = rng.integers(0, 256, size=(n, 784), dtype=np.uint8)

X = X_uint8.astype(np.float32) / 255.0

assert X.shape == (n, 784)
assert X.dtype in (np.float32, np.float64)
assert np.isfinite(X).all()
assert X.min() >= 0.0
assert X.max() <= 1.0
```

<details open>
<summary><strong>Pedagogical add-on — Exercise 8.2: Data invariants (shape, dtype, range)</strong></summary>

#### 1) Metadata
- **ID (optional):** `M08-E08_2`
- **Estimated duration:** 15–30 min
- **Level:** Basic

#### 2) Key idea
- Many “training bugs” are actually *data bugs*.
- Lock these invariants early:
  - `X.shape == (n, 784)`
  - `X.dtype` is float
  - values are in `[0,1]`
  - finite (`isfinite`) everywhere

#### 3) Common mistakes
- Normalizing with integer division (old Python or accidental casting).
- Forgetting to cast to float before dividing.
- Assuming min/max without checking.

#### 4) Teaching note
- Ask the student to deliberately inject a `NaN` and confirm the assert catches it.
</details>

---

### Exercise 8.3: One-hot encoding (multiclass)

#### Prompt

1) **Basic**

- Implement `one_hot(y, num_classes=10)`.

2) **Intermediate**

- Verify each row sums to 1.

3) **Advanced**

- Verify `argmax(one_hot(y)) == y`.

#### Solution

```python
import numpy as np

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    Y = np.zeros((y.size, num_classes), dtype=float)
    Y[np.arange(y.size), y] = 1.0
    return Y


y = np.array([0, 2, 9, 2, 1])
Y = one_hot(y, num_classes=10)

assert Y.shape == (y.size, 10)
assert np.allclose(np.sum(Y, axis=1), 1.0)
assert np.all(np.argmax(Y, axis=1) == y)
```

<details open>
<summary><strong>Pedagogical add-on — Exercise 8.3: One-hot encoding</strong></summary>

#### 1) Metadata
- **ID (optional):** `M08-E08_3`
- **Estimated duration:** 15–25 min
- **Level:** Basic

#### 2) Key idea
- One-hot converts `y:(n,)` into `Y:(n,k)` so that cross-entropy can be expressed with vectorized operations.
- The invariant is: `argmax(Y[i]) == y[i]`.

#### 3) Common mistakes
- Forgetting to cast labels to `int` (breaks indexing).
- Passing labels out of range `[0, k-1]`.
- Using shape `(n,1)` labels and mixing it with `(n,)` without being explicit.

#### 4) Teaching note
- Ask the student to test a label set that includes `0` and `k-1` to cover boundaries.
</details>

---

### Exercise 8.4: PCA via SVD (explained variance + reconstruction)

#### Prompt

1) **Basic**

- Implement PCA via SVD to reduce to `k` components.

2) **Intermediate**

- Compute explained variance ratio and verify it is sorted descending.

3) **Advanced**

- Reconstruct with `k=10` vs `k=50` and verify reconstruction error decreases.

#### Solution

```python
import numpy as np

def pca_svd_fit_transform(X: np.ndarray, k: int):
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k].T
    Z = Xc @ Vk
    var = (S ** 2) / (Xc.shape[0] - 1)
    ratio = var / np.sum(var)
    return Z, Vk, mu, ratio


def pca_reconstruct(Z: np.ndarray, Vk: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return Z @ Vk.T + mu


rng = np.random.default_rng(1)
X = rng.normal(size=(300, 784)).astype(np.float64)

Z10, V10, mu, ratio = pca_svd_fit_transform(X, k=10)
Z50, V50, mu2, ratio2 = pca_svd_fit_transform(X, k=50)

assert np.allclose(mu, mu2)
assert ratio[0] >= ratio[1]
assert ratio2[0] >= ratio2[1]

X10 = pca_reconstruct(Z10, V10, mu)
X50 = pca_reconstruct(Z50, V50, mu)

err10 = np.linalg.norm(X - X10)
err50 = np.linalg.norm(X - X50)
assert err50 <= err10 + 1e-12
```

<details open>
<summary><strong>Pedagogical add-on — Exercise 8.4: PCA (SVD) and explained variance</strong></summary>

#### 1) Metadata
- **ID (optional):** `M08-E08_4`
- **Estimated duration:** 30–60 min
- **Level:** Intermediate

#### 2) Key idea
- PCA requires centering: `Xc = X - mean(X)`.
- SVD gives principal directions via `Vt`; the top-`k` rows of `Vt` define the subspace.
- Reconstruction error should decrease as `k` increases.

#### 3) Common mistakes
- Forgetting to center, then misinterpreting components.
- Confusing `U` and `V` roles in SVD.
- Computing explained variance ratios without dividing by total variance.

#### 4) Teaching note
- Ask the student to explain why `k=784` reconstructs perfectly (up to numerical error).
</details>

---

### Exercise 8.5: K-Means (inertia) - one iteration should not increase J

#### Prompt

1) **Basic**

- Implement nearest-centroid assignment.

2) **Intermediate**

- Implement centroid update as means (handle empty clusters).

3) **Advanced**

- Verify inertia `J` does not increase after one iteration.

#### Solution

```python
import numpy as np

def assign_labels(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    D2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
    return np.argmin(D2, axis=1)


def update_centroids(X: np.ndarray, labels: np.ndarray, C: np.ndarray) -> np.ndarray:
    C_new = C.copy()
    for j in range(C.shape[0]):
        mask = labels == j
        if np.any(mask):
            C_new[j] = np.mean(X[mask], axis=0)
    return C_new


def inertia(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    diffs = X - C[labels]
    return float(np.sum(diffs ** 2))


rng = np.random.default_rng(2)
X = np.vstack([
    rng.normal(loc=-1.0, scale=0.5, size=(100, 2)),
    rng.normal(loc=+1.0, scale=0.5, size=(100, 2)),
])
C0 = np.array([[-1.0, 1.0], [1.0, -1.0]])

labels0 = assign_labels(X, C0)
J0 = inertia(X, C0, labels0)

C1 = update_centroids(X, labels0, C0)
labels1 = assign_labels(X, C1)
J1 = inertia(X, C1, labels1)

assert J1 <= J0 + 1e-12
```

<details open>
<summary><strong>Pedagogical add-on — Exercise 8.5: K-Means inertia monotonicity</strong></summary>

#### 1) Metadata
- **ID (optional):** `M08-E08_5`
- **Estimated duration:** 30–60 min
- **Level:** Intermediate

#### 2) Key idea
- Lloyd’s algorithm alternates:
  - assignment (closest centroid)
  - update (centroid = mean of assigned points)
- Each step should not increase inertia `J` (with the usual definitions).

#### 3) Common mistakes
- Not handling empty clusters (mean of empty set).
- Computing distances incorrectly due to broadcasting mistakes.
- Measuring `J` with mismatched labels/centroids.

#### 4) Teaching note
- Ask the student to force an empty cluster and explain the chosen fallback strategy.
</details>

---

### Exercise 8.6: Logistic Regression OvA - gradient check (single class)

#### Prompt

1) **Basic**

- Implement sigmoid and BCE for a binary class target `y_c`.

2) **Intermediate**

- Implement gradient `∇w = (1/n) X^T (p - y_c)`.

3) **Advanced**

- Check one gradient coordinate using central differences.

#### Solution

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce_from_logits(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, eps: float = 1e-15) -> float:
    p = sigmoid(X @ w + b)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def grad_w(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    p = sigmoid(X @ w + b)
    return (X.T @ (p - y)) / X.shape[0]


rng = np.random.default_rng(3)
n, d = 120, 50
X = rng.normal(size=(n, d))
y = (rng.random(size=(n, 1)) < 0.4).astype(float)

w = rng.normal(size=(d, 1)) * 0.1
b = 0.0

g = grad_w(X, y, w, b)

idx = 7
h = 1e-6
E = np.zeros_like(w)
E[idx, 0] = 1.0
L_plus = bce_from_logits(X, y, w + h * E, b)
L_minus = bce_from_logits(X, y, w - h * E, b)
g_num = (L_plus - L_minus) / (2.0 * h)

assert np.isclose(g[idx, 0], g_num, rtol=1e-4, atol=1e-6)
```

<details open>
<summary><strong>Pedagogical add-on — Exercise 8.6: Logistic Regression gradient check</strong></summary>

#### 1) Metadata
- **ID (optional):** `M08-E08_6`
- **Estimated duration:** 40–80 min
- **Level:** Advanced

#### 2) Key idea
- Gradient checking validates your analytic gradient against a numerical approximation on a few coordinates.
- For OvA, you can validate one classifier (one class) before scaling to 10.

#### 3) Common mistakes
- Mixing `y:(n,)` with `p:(n,1)` and getting silent broadcasting bugs.
- Forgetting normalization by `n`.
- Using `h` too large (bias) or too small (floating point noise).

#### 4) Teaching note
- Ask the student to check 2 random coordinates and compare the relative error.
</details>

---

### Exercise 8.7: MLP sanity - overfit a mini-batch

#### Prompt

1) **Basic**

- Implement a minimal MLP `784→32→10` (ReLU + softmax) and cross-entropy.

2) **Intermediate**

- Train on a tiny set (e.g. 64 samples) and verify loss decreases.

3) **Advanced**

- Verify training accuracy is reasonably high (overfit signal).

#### Solution

```python
import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


def logsumexp(z: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    m = np.max(z, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(z - logsumexp(z))


def cross_entropy(y_onehot: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-np.mean(np.sum(y_onehot * np.log(p), axis=1)))


rng = np.random.default_rng(4)
n, d_in, d_h, d_out = 64, 784, 32, 10
X = rng.normal(size=(n, d_in))
y = rng.integers(0, d_out, size=(n,))
Y = np.zeros((n, d_out), dtype=float)
Y[np.arange(n), y] = 1.0

W1 = rng.normal(size=(d_in, d_h)) * 0.01
b1 = np.zeros(d_h)
W2 = rng.normal(size=(d_h, d_out)) * 0.01
b2 = np.zeros(d_out)

lr = 1.0
loss0 = None
for _ in range(200):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    P = softmax(Z2)
    loss = cross_entropy(Y, P)
    if loss0 is None:
        loss0 = loss

    dZ2 = (P - Y) / n
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

loss_end = cross_entropy(Y, softmax(relu(X @ W1 + b1) @ W2 + b2))
pred = np.argmax(softmax(relu(X @ W1 + b1) @ W2 + b2), axis=1)
acc = float(np.mean(pred == y))

assert loss_end <= loss0
assert acc > 0.6
```

<details open>
<summary><strong>Pedagogical add-on — Exercise 8.7: MLP overfit sanity check</strong></summary>

#### 1) Metadata
- **ID (optional):** `M08-E08_7`
- **Estimated duration:** 45–90 min
- **Level:** Advanced

#### 2) Key idea
- Overfitting a tiny batch is a *mandatory* debug protocol: if it cannot fit 64 samples, assume a bug.
- For stability, softmax should be implemented with `logsumexp`.

#### 3) Common mistakes
- Too small initialization or too small learning rate → no progress.
- Unstable softmax (overflow) → `NaN` losses.
- Shape mismatches in gradients (especially biases with broadcasting).

#### 4) Teaching note
- Ask the student to log `loss` every 20 steps and explain the trend.
</details>

---

### Exercise 8.8: Metrics (confusion matrix + macro F1)

#### Prompt

1) **Basic**

- Implement `confusion_matrix(y_true, y_pred, k)`.

2) **Intermediate**

- Implement per-class precision/recall/F1.

3) **Advanced**

- Implement macro F1 and verify it is in `[0,1]`.

#### Solution

```python
import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def prf_from_cm(cm: np.ndarray):
    k = cm.shape[0]
    eps = 1e-12
    precision = np.zeros(k)
    recall = np.zeros(k)
    f1 = np.zeros(k)
    for c in range(k):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        precision[c] = tp / (tp + fp + eps)
        recall[c] = tp / (tp + fn + eps)
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + eps)
    return precision, recall, f1


y_true = np.array([0, 1, 2, 2, 2, 1])
y_pred = np.array([0, 2, 2, 2, 1, 1])
cm = confusion_matrix(y_true, y_pred, k=3)
prec, rec, f1 = prf_from_cm(cm)
f1_macro = float(np.mean(f1))

assert cm.shape == (3, 3)
assert 0.0 <= f1_macro <= 1.0
```

<details open>
<summary><strong>Pedagogical add-on — Exercise 8.8: Confusion matrix and macro F1</strong></summary>

#### 1) Metadata
- **ID (optional):** `M08-E08_8`
- **Estimated duration:** 30–60 min
- **Level:** Intermediate

#### 2) Key idea
- Accuracy can hide class imbalance.
- Macro-F1 averages per-class F1, weighting all classes equally.

#### 3) Common mistakes
- Dividing by zero when a class has no predictions / no true samples (use `eps`).
- Using micro-F1 when the goal is macro-F1.
- Building `cm` with swapped indices (`cm[p,t]` vs `cm[t,p]`).

#### 4) Teaching note
- Ask the student to create a case where one class is never predicted and interpret the metrics.
</details>

---

### Exercise 8.9: Model comparison table (consistency checks)

#### Prompt

1) **Basic**

- Given a dict `{model: accuracy}`, sort models best→worst.

2) **Intermediate**

- Verify the best accuracy is first.

3) **Advanced**

- Verify all accuracies are in `[0,1]`.

#### Solution

```python
results = {
    "K-Means": 0.00,
    "Logistic Regression": 0.88,
    "MLP": 0.94,
}

items = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
assert items[0][1] == max(results.values())
for _, acc in items:
    assert 0.0 <= acc <= 1.0
```

<details open>
<summary><strong>Pedagogical add-on — Exercise 8.9: Model comparison consistency</strong></summary>

#### 1) Metadata
- **ID (optional):** `M08-E08_9`
- **Estimated duration:** 15–25 min
- **Level:** Basic

#### 2) Key idea
- Comparisons must use a consistent metric and the same data split.
- Sorting is trivial, but the *invariants* matter: values in `[0,1]`, best-first order, stable naming.

#### 3) Common mistakes
- Mixing train accuracy for one model and test accuracy for another.
- Comparing models trained with different seeds/splits.
- Forgetting to validate the range of metrics.

#### 4) Teaching note
- Ask the student to extend the dict with a new model and confirm all checks still pass.
</details>

---

## Deliverables (high-signal)

- Reproducible training/evaluation pipeline.
- Clear plots:
  - PCA 2D visualization
  - learning curves (bias/variance diagnosis)
  - error analysis grid
- `MODEL_COMPARISON.md` report (methods, results, discussion).
- Minimal deployment proof:
  - saved checkpoint
  - single-image inference output

---

## Completion checklist (v3.3)

### Week 21: EDA + Unsupervised

- [ ] PCA reduces MNIST to 2D/50D with visualization.
- [ ] Analyzed explained variance per component.
- [ ] K-Means clusters digits without labels.
- [ ] Visualized centroids as 28×28 images.

### Week 22: Supervised classification

- [ ] Logistic Regression One-vs-All working.
- [ ] Accuracy >85% on test set.
- [ ] Computed per-class Precision/Recall/F1.
- [ ] Analyzed confusion matrix.

### Week 23: Deep Learning

- [ ] MLP architecture 784→128→64→10.
- [ ] Forward and backward passes implemented.
- [ ] Mini-batch SGD works.
- [ ] Accuracy >90% on test set.

### Week 24: Benchmark + report

- [ ] `MODEL_COMPARISON.md` completed.
- [ ] `README.md` professional in English.
- [ ] Optional benchmark: ran **Fashion-MNIST** (or justified why not).
- [ ] Dirty Data Check: created corrupted dataset with `scripts/corrupt_mnist.py` and documented cleaning.
- [ ] Minimal deployment: trained CNN with `scripts/train_cnn_pytorch.py` and saved checkpoint.
- [ ] Minimal deployment: ran `scripts/predict.py` on a 28×28 image and reported prediction.

### v3.3 requirements

- [ ] Bias–variance analysis with a practical experiment.
- [ ] Paper-style notebook/report (Abstract, Methods, Results, Discussion).
- [ ] Error analysis with visualization.
- [ ] Learning curves with diagnosis.
- [ ] “Error Analysis” section in `MODEL_COMPARISON.md`.
- [ ] `mypy` passes.
- [ ] `pytest` passes.

### Feynman

- [ ] Explain why MLP beats Logistic in 5 lines.
- [ ] Explain bias vs variance in 5 lines.
- [ ] Explain why 4↔9 confusions happen in 5 lines.

---

## Navigation

| Previous | Index |
|---|---|
| [07_DEEP_LEARNING](07_DEEP_LEARNING.md) | [00_INDICE](00_INDICE.md) |
