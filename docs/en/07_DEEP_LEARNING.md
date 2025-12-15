# Module 07 - Deep Learning

> **Goal:** implement an MLP with manual backprop + CNN forward (NumPy) + CNN training with PyTorch.
> **Phase:** 2 - ML Core | **Weeks 17–20**
> **Pathway course:** Introduction to Deep Learning

---

<a id="m07-0"></a>

## How to use this module (0→100 mode)

**Purpose:** build and debug a neural network from scratch:

- forward pass
- backpropagation
- optimization (SGD / Momentum / Adam)
- sanity checks (overfit test)

### Learning outcomes (measurable)

By the end of this module you can:

- Implement an MLP that solves XOR.
- Explain backprop as Chain Rule applied to a computation graph.
- Debug training using an overfit test (if it cannot memorize, there is a bug).
- Implement the forward pass of a simple CNN (convolution + pooling) in NumPy to master shapes.
- Train an equivalent CNN using PyTorch (`torch.nn`) **without** implementing a manual CNN backward pass.

Quick references:

- [Module 03: Calculus (Chain Rule)](03_CALCULO_MULTIVARIANTE.md)
- [Glossary](GLOSARIO.md)
- [Resources](RECURSOS.md)
- [Plan v4](PLAN_V4_ESTRATEGICO.md)
- [Plan v5](PLAN_V5_ESTRATEGICO.md)
- Rubric: `study_tools/RUBRICA_v1.md` (scope `M07` in `rubrica.csv`; closes Week 20)

---

## Module structure (Weeks 17–20)

| Week | Focus | Output |
|---|---|---|
| 17 | Perceptron + activations + MLP forward pass | `activations.py` + forward utilities |
| 18 | Backpropagation (manual) | `backward()` + gradients |
| 19 | **CNNs: theory + forward (NumPy)** | conv/pooling forward + shape quiz |
| 20 | **PyTorch for CNN training** | `../scripts/train_cnn_pytorch.py` |

---

## What matters most (high-signal core)

### MLP correctness

- Your MLP must pass basic sanity checks:
  - gradients have correct shapes
  - training decreases loss
  - it can overfit a tiny dataset

### CNN split approach (v3.3)

- **NumPy:** forward pass only (shape mastery, not a full framework).
- **PyTorch:** full CNN training loop (realistic workflow).

This preserves the “from scratch” learning goal while avoiding a large, error-prone manual CNN backward.

---

## 🎯 Topic-based progressive exercises + solutions

Rules:

- **Try first** without looking at solutions.
- **Suggested timebox:** 30–75 min per exercise.
- **Minimum success:** your solution must pass the `assert` checks.

---

### Exercise 7.1: Activations + derivatives (numerical check)

#### Prompt

1) **Basic**

- Implement `sigmoid(z)` and `relu(z)`.

2) **Intermediate**

- Implement derivatives `sigmoid'(z)` and `relu'(z)`.

3) **Advanced**

- Verify `sigmoid'(z)` using central finite differences.

#### Solution

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    a = sigmoid(z)
    return a * (1.0 - a)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.asarray(z, dtype=float))


def relu_deriv(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return (z > 0.0).astype(float)


def num_derivative(f, z: np.ndarray, h: float = 1e-6) -> np.ndarray:
    return (f(z + h) - f(z - h)) / (2.0 * h)


np.random.seed(0)
z = np.random.randn(10)
g_num = num_derivative(sigmoid, z)
g_ana = sigmoid_deriv(z)
assert np.allclose(g_num, g_ana, rtol=1e-5, atol=1e-6)
```

---

### Exercise 7.2: Dense layer forward (batch) + shape reasoning

#### Prompt

1) **Basic**

- Implement `dense_forward(X, W, b)` with `X:(n,d_in)`, `W:(d_in,d_out)`, `b:(d_out,)`.

2) **Intermediate**

- Verify output shape `Z:(n,d_out)`.

3) **Advanced**

- Verify it matches a loop implementation on a tiny case.

#### Solution

```python
import numpy as np

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b


np.random.seed(1)
n, d_in, d_out = 5, 3, 4
X = np.random.randn(n, d_in)
W = np.random.randn(d_in, d_out)
b = np.random.randn(d_out)

Z = dense_forward(X, W, b)
assert Z.shape == (n, d_out)

Z_loop = np.zeros_like(Z)
for i in range(n):
    Z_loop[i] = X[i] @ W + b

assert np.allclose(Z, Z_loop)
```

---

### Exercise 7.3: Stable softmax + categorical cross-entropy

#### Prompt

1) **Basic**

- Implement stable `logsumexp` and `softmax`.

2) **Intermediate**

- Implement categorical cross-entropy for one-hot `y_true`.

3) **Advanced**

- Verify:
  - `softmax` sums to 1.
  - loss decreases when the correct-class probability increases.

#### Solution

```python
import numpy as np

def logsumexp(z: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    m = np.max(z, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    lse = logsumexp(z, axis=axis, keepdims=True)
    return np.exp(z - lse)


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))


z = np.array([[10.0, 0.0, -10.0]])
p = softmax(z)
assert np.isclose(np.sum(p), 1.0)
assert np.argmax(p) == 0

y_true = np.array([[1.0, 0.0, 0.0]])
loss_good = categorical_cross_entropy(y_true, np.array([[0.9, 0.05, 0.05]]))
loss_bad = categorical_cross_entropy(y_true, np.array([[0.4, 0.3, 0.3]]))
assert loss_good < loss_bad
```

---

### Exercise 7.4: Two-layer backprop + gradient checking

#### Prompt

Network (batch):

- `Z1 = XW1 + b1`, `A1 = relu(Z1)`
- `Z2 = A1W2 + b2`, `P = sigmoid(Z2)`
- BCE loss: `L = -mean(y log(P) + (1-y) log(1-P))`

1) **Basic**

- Implement forward + loss.

2) **Intermediate**

- Implement backward gradients: `dW1, db1, dW2, db2`.

3) **Advanced**

- Check one coordinate of `dW2` via central differences.

#### Solution

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    P = sigmoid(Z2)
    cache = (X, Z1, A1, Z2, P)
    return P, cache


def loss_fn(X, y, W1, b1, W2, b2):
    P, _ = forward(X, W1, b1, W2, b2)
    return bce(y, P)


def backward(y, cache, W2):
    X, Z1, A1, Z2, P = cache
    n = X.shape[0]
    dZ2 = (P - y) / n
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    return dW1, db1, dW2, db2


np.random.seed(0)
n, d_in, d_h = 8, 3, 5
X = np.random.randn(n, d_in)
y = (np.random.rand(n, 1) < 0.5).astype(float)
W1 = np.random.randn(d_in, d_h) * 0.1
b1 = np.zeros(d_h)
W2 = np.random.randn(d_h, 1) * 0.1
b2 = np.zeros(1)

P, cache = forward(X, W1, b1, W2, b2)
dW1, db1, dW2, db2 = backward(y, cache, W2)

i, j = 2, 0
h = 1e-6
E = np.zeros_like(W2)
E[i, j] = 1.0
L_plus = loss_fn(X, y, W1, b1, W2 + h * E, b2)
L_minus = loss_fn(X, y, W1, b1, W2 - h * E, b2)
g_num = (L_plus - L_minus) / (2.0 * h)
assert np.isclose(dW2[i, j], g_num, rtol=1e-4, atol=1e-6)
```

---

### Exercise 7.5: Overfit test (mandatory sanity check)

#### Prompt

1) **Basic**

- Build a tiny (8–16 samples) linearly-separable dataset.

2) **Intermediate**

- Train logistic regression with GD and verify loss decreases.

3) **Advanced**

- Verify high training accuracy (e.g. > 95%).

#### Solution

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


np.random.seed(1)
n = 16
X_pos = np.random.randn(n // 2, 2) + np.array([2.0, 2.0])
X_neg = np.random.randn(n // 2, 2) + np.array([-2.0, -2.0])
X = np.vstack([X_pos, X_neg])
y = np.vstack([np.ones((n // 2, 1)), np.zeros((n // 2, 1))])

w = np.zeros((2, 1))
b = 0.0
lr = 0.2

loss0 = None
for _ in range(400):
    logits = X @ w + b
    p = sigmoid(logits)
    loss = bce(y, p)
    if loss0 is None:
        loss0 = loss
    dz = (p - y) / n
    dw = X.T @ dz
    db = float(np.sum(dz))
    w -= lr * dw
    b -= lr * db

loss_end = bce(y, sigmoid(X @ w + b))
pred = (sigmoid(X @ w + b) >= 0.5).astype(int)
acc = float(np.mean(pred == y.astype(int)))

assert loss_end <= loss0
assert acc > 0.95
```

---

### Exercise 7.6: Optimizers on a quadratic (SGD vs Adam)

#### Prompt

Minimize `f(w) = (w - 3)^2`.

1) **Basic**

- Implement SGD.

2) **Intermediate**

- Implement Adam.

3) **Advanced**

- Verify both approach `w≈3`.

#### Solution

```python
import numpy as np

def grad_f(w: float) -> float:
    return 2.0 * (w - 3.0)


def sgd(w0: float, lr: float, steps: int) -> float:
    w = float(w0)
    for _ in range(steps):
        w -= lr * grad_f(w)
    return w


def adam(w0: float, lr: float, steps: int, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> float:
    w = float(w0)
    m = 0.0
    v = 0.0
    t = 0
    for _ in range(steps):
        t += 1
        g = grad_f(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return w


w_sgd = sgd(w0=10.0, lr=0.1, steps=50)
w_adam = adam(w0=10.0, lr=0.2, steps=50)

assert abs(w_sgd - 3.0) < 1e-2
assert abs(w_adam - 3.0) < 1e-2
```

---

### Exercise 7.7: Gradient clipping (prevent exploding gradients)

#### Prompt

1) **Basic**

- Implement norm clipping: if `||g|| > max_norm`, set `g <- g * (max_norm/||g||)`.

2) **Intermediate**

- Verify clipped gradient has norm `<= max_norm`.

3) **Advanced**

- Verify gradients below threshold are unchanged.

#### Solution

```python
import numpy as np

def clip_by_norm(g: np.ndarray, max_norm: float) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    n = np.linalg.norm(g)
    if n == 0.0:
        return g
    if n <= max_norm:
        return g
    return g * (max_norm / n)


g_big = np.array([3.0, 4.0])
g_clip = clip_by_norm(g_big, max_norm=1.0)
assert np.linalg.norm(g_clip) <= 1.0 + 1e-12

g_small = np.array([0.3, 0.4])
g_keep = clip_by_norm(g_small, max_norm=1.0)
assert np.allclose(g_small, g_keep)
```

---

### Exercise 7.8: Convolution output shape (padding/stride)

#### Prompt

1) **Basic**

- Implement `conv2d_out(H, W, KH, KW, stride, padding)` (no dilation).

2) **Intermediate**

- Verify MNIST case: `28x28`, `5x5`, `stride=1`, `padding=0` -> `24x24`.

3) **Advanced**

- Verify padding case: `28x28`, `5x5`, `stride=1`, `padding=2` -> `28x28`.

#### Solution

```python
def conv2d_out(H: int, W: int, KH: int, KW: int, stride: int = 1, padding: int = 0):
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1
    return int(H_out), int(W_out)


assert conv2d_out(28, 28, 5, 5, stride=1, padding=0) == (24, 24)
assert conv2d_out(28, 28, 5, 5, stride=1, padding=2) == (28, 28)
```

---

## Deliverables

- `neural_network.py`:
  - MLP forward + backward
  - SGD / Momentum / Adam
  - XOR training demo

- `overfit_test.py` (required):
  - memorization test on a tiny dataset (XOR or small batch)

- CNN practical training:
  - run: `scripts/train_cnn_pytorch.py`

Minimum acceptance criteria:

- XOR is solved reliably.
- Overfit test reaches near-zero loss.
- You can explain each piece in plain language.

---

## Completion checklist (v3.3)

### Knowledge

- [ ] Understand biological neuron → artificial neuron analogy.
- [ ] Implemented sigmoid, ReLU, tanh, softmax and derivatives.
- [ ] Understand why XOR is not linearly separable.
- [ ] Implemented MLP forward pass.
- [ ] Understand Chain Rule applied to backprop.
- [ ] Implemented backward pass (gradients).
- [ ] Implemented SGD, SGD+Momentum, Adam.
- [ ] Network solves XOR.

### CNNs (theory)

- [ ] Understand convolution, stride, padding, pooling.
- [ ] Can compute CNN output dimensions.
- [ ] Know LeNet-5 at a concept level.

### CNNs (practice)

- [ ] Implemented NumPy forward pass (conv + pooling) for a LeNet-like architecture.
- [ ] Trained an equivalent CNN with PyTorch using `scripts/train_cnn_pytorch.py`.

### Code deliverables

- [ ] `neural_network.py` tests passing.
- [ ] `mypy` passes.
- [ ] `pytest` passes.

### Overfit test (required v3.3)

- [ ] `overfit_test.py` implemented.
- [ ] Network overfits XOR (loss < 0.01).
- [ ] If it failed, I debugged using gradient checks.

### Analytical derivation (required)

- [ ] Derived backprop equations by hand.
- [ ] Documented a computation graph.

### Feynman

- [ ] Explain backprop in 5 lines.
- [ ] Explain ReLU vs sigmoid in 5 lines.
- [ ] Explain convolution in 5 lines.
- [ ] Explain pooling in 5 lines.

---

## Navigation

| Previous | Index | Next |
|---|---|---|
| [06_UNSUPERVISED_LEARNING](06_UNSUPERVISED_LEARNING.md) | [00_INDICE](00_INDICE.md) | [08_PROYECTO_MNIST](08_PROYECTO_MNIST.md) |
