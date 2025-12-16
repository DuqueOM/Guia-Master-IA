# Module 04 - Essential Probability for Machine Learning

> **Week 8 | Prerequisite for Loss Functions, Softmax/Cross-Entropy, and later GMM intuition**
> **Philosophy:** only the probability you need for Line 1 (Machine Learning)

---

<a id="m04-0"></a>

## How to use this module (0→100 mode)

**Purpose:** connect probability to what you will actually use later:

- cross-entropy as *negative log-likelihood*
- probabilistic classification (logistic/softmax)
- Gaussians as the atom of generative models (GMM intuition)
- numerical stability (avoid `NaN`)

### Learning outcomes (measurable)

By the end of this module you can:

- Explain conditional probability and Bayes’ theorem with a classification example.
- Use the MLE perspective: “choose parameters that make the data most likely”.
- Derive why minimizing cross-entropy = maximizing log-likelihood (binary + multiclass).
- Implement softmax/log-softmax stably (log-sum-exp).
- Diagnose typical numerical failures: `log(0)`, overflow/underflow, probabilities not summing to 1.

Quick references (Spanish source, stable anchors):

- [Resources](RECURSOS.md)
- [Glossary: Binary Cross-Entropy](GLOSARIO.md#binary-cross-entropy)
- [Glossary: Softmax](GLOSARIO.md#softmax)
- [Glossary: Chain Rule](GLOSARIO.md#chain-rule)

### Integration with v4/v5

- Protocols:
  - [Plan v4](PLAN_V4_ESTRATEGICO.md)
  - [Plan v5](PLAN_V5_ESTRATEGICO.md)
- Error log: `study_tools/DIARIO_ERRORES.md`
- Rubric: `study_tools/RUBRICA_v1.md` (scope `M04` in `rubrica.csv`; includes PB-8)

---

## What you should cover (compact scope)

- Probability basics, conditional probability, independence.
- Bayes theorem as a classifier lens.
- Gaussian (univariate + multivariate) as a modeling primitive.
- Maximum Likelihood Estimation (MLE).
- Softmax + log-sum-exp for stability.
- Cross-entropy (binary + categorical).
- Optional “seed” concept: Markov chains (notation + transition matrices).

---

## 🎯 Topic-based progressive exercises + solutions

Rules:

- **Try first** without looking at solutions.
- **Suggested timebox:** 15–30 min per exercise.
- **Minimum success:** your solution must pass the `assert` checks.

---

### Exercise 4.1: Conditional probability (P(A|B)) + consistency

#### Prompt

1) **Basic**

- Given event counts, compute `P(A)`, `P(B)`, and `P(A ∩ B)`.

2) **Intermediate**

- Compute `P(A|B) = P(A∩B)/P(B)` and verify it is in `[0,1]`.

3) **Advanced**

- Verify `P(A∩B) = P(A|B)·P(B)`.

#### Solution

```python
import numpy as np

n = 100
count_A = 40
count_B = 50
count_A_and_B = 20

P_A = count_A / n
P_B = count_B / n
P_A_and_B = count_A_and_B / n

P_A_given_B = P_A_and_B / P_B

assert 0.0 <= P_A <= 1.0
assert 0.0 <= P_B <= 1.0
assert 0.0 <= P_A_given_B <= 1.0
assert np.isclose(P_A_and_B, P_A_given_B * P_B)
```

---

### Exercise 4.2: Bayes as a classifier lens (unnormalized posterior)

#### Prompt

1) **Basic**

- Compute unnormalized scores:
  - `score_spam = P(x|spam)·P(spam)`
  - `score_ham = P(x|ham)·P(ham)`

2) **Intermediate**

- Normalize to get `P(spam|x)` and `P(ham|x)`.

3) **Advanced**

- Verify the normalized probabilities sum to 1.

#### Solution

```python
import numpy as np

P_spam = 0.3
P_ham = 1.0 - P_spam

P_x_given_spam = 0.8
P_x_given_ham = 0.1

score_spam = P_x_given_spam * P_spam
score_ham = P_x_given_ham * P_ham

Z = score_spam + score_ham
P_spam_given_x = score_spam / Z
P_ham_given_x = score_ham / Z

assert np.isclose(P_spam_given_x + P_ham_given_x, 1.0)
assert P_spam_given_x > P_ham_given_x
```

---

### Exercise 4.3: Independence (empirical test)

#### Prompt

1) **Basic**

- Simulate two independent Bernoulli variables `A` and `B`.

2) **Intermediate**

- Estimate `P(A)`, `P(B)`, `P(A∩B)` and verify `P(A∩B) ≈ P(A)P(B)`.

3) **Advanced**

- Simulate a dependent case and verify the equality breaks.

#### Solution

```python
import numpy as np

np.random.seed(0)
n = 20000

A = (np.random.rand(n) < 0.4)
B = (np.random.rand(n) < 0.5)

P_A = A.mean()
P_B = B.mean()
P_A_and_B = (A & B).mean()

assert abs(P_A_and_B - (P_A * P_B)) < 0.01

# Dependent: B is almost A
B_dep = (A | (np.random.rand(n) < 0.05))
P_B_dep = B_dep.mean()
P_A_and_B_dep = (A & B_dep).mean()

assert abs(P_A_and_B_dep - (P_A * P_B_dep)) > 0.02
```

---

### Exercise 4.4: Bernoulli MLE ("fraction of heads")

#### Prompt

1) **Basic**

- Generate Bernoulli samples with `p_true`.

2) **Intermediate**

- Implement the MLE `p_hat = mean(x)`.

3) **Advanced**

- Verify `p_hat` approaches `p_true` with enough samples.

#### Solution

```python
import numpy as np

np.random.seed(1)
p_true = 0.7
n = 5000
x = (np.random.rand(n) < p_true).astype(float)

p_hat = float(np.mean(x))
assert abs(p_hat - p_true) < 0.02
```

---

### Exercise 4.5: Univariate Gaussian PDF (sanity check)

#### Prompt

1) **Basic**

- Implement the PDF of `N(μ,σ²)`.

2) **Intermediate**

- Verify that for `N(0,1)` at `x=0` the density is ≈ `0.39894228`.

3) **Advanced**

- Verify symmetry: `pdf(a) == pdf(-a)` when `μ=0`.

#### Solution

```python
import numpy as np  # NumPy: arrays + vectorized math + exp/sqrt

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:  # PDF of N(μ, σ²)
    x = np.asarray(x, dtype=float)  # Ensure float dtype (exp/log stability)
    sigma = float(sigma)  # Normalize sigma type (scalar float)
    assert sigma > 0  # Sanity: standard deviation must be positive
    z = (x - mu) / sigma  # Standardize: convert to z-scores
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * z**2)  # Closed-form Gaussian PDF


val0 = gaussian_pdf(np.array([0.0]), mu=0.0, sigma=1.0)[0]  # Standard normal at 0
assert np.isclose(val0, 0.39894228, atol=1e-4)  # ≈ 1/sqrt(2π)

a = 1.7  # Arbitrary positive point to test symmetry
assert np.isclose(
    gaussian_pdf(np.array([a]), 0.0, 1.0)[0],  # pdf(a)
    gaussian_pdf(np.array([-a]), 0.0, 1.0)[0],  # pdf(-a)
    rtol=1e-12,  # Relative tolerance for float comparisons
    atol=1e-12,  # Absolute tolerance for float comparisons
)
```

---

### Exercise 4.6: Multivariate Gaussian (2D) + valid covariance

#### Prompt

1) **Basic**

- Implement the density of `N(μ, Σ)` in 2D.

2) **Intermediate**

- For `μ=0` and `Σ=I`, verify `pdf(0) = 1/(2π)`.

3) **Advanced**

- Verify `Σ` is positive definite (eigenvalues > 0) before inverting.

4) **Bonus (covariance ellipse)**

- For a non-diagonal covariance matrix, generate points on the 2D covariance ellipse for a chosen scale `k` (e.g., `k=2`) using eigendecomposition, and verify they satisfy `(x-μ)^T Σ^{-1} (x-μ) ≈ k^2`.

#### Solution

```python
import numpy as np

def multivariate_gaussian_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)  # x:(d,) ensure float dtype
    mu = np.asarray(mu, dtype=float)  # mu:(d,) ensure float dtype
    cov = np.asarray(cov, dtype=float)  # cov:(d,d) ensure float dtype
    d = x.shape[0]  # d: dimension of the feature space

    assert mu.shape == (d,)  # Shape contract: mean vector must match x
    assert cov.shape == (d, d)  # Shape contract: covariance must be square
    assert np.allclose(cov, cov.T)  # Covariance must be symmetric
    eigvals = np.linalg.eigvals(cov)  # Eigenvalues (used to verify positive-definiteness)
    assert np.all(eigvals > 0)  # Positive-definite => valid covariance and invertible

    diff = x - mu  # Centered coordinates: (x-μ)
    inv = np.linalg.inv(cov)  # Σ^{-1} for the quadratic form
    det = np.linalg.det(cov)  # |Σ| for normalization
    norm = 1.0 / (np.sqrt(((2.0 * np.pi) ** d) * det))  # 1/sqrt((2π)^d |Σ|)
    expo = -0.5 * float(diff.T @ inv @ diff)  # -(1/2) (x-μ)^T Σ^{-1} (x-μ)
    return float(norm * np.exp(expo))  # Density value (scalar)


mu = np.array([0.0, 0.0])
cov = np.eye(2)
pdf0 = multivariate_gaussian_pdf(np.array([0.0, 0.0]), mu, cov)
assert np.isclose(pdf0, 1.0 / (2.0 * np.pi), atol=1e-6)
assert pdf0 > 0.0

def covariance_ellipse_points(mu: np.ndarray, cov: np.ndarray, k: float = 2.0, n: int = 200) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)  # mu:(2,) ensure float
    cov = np.asarray(cov, dtype=float)  # cov:(2,2) ensure float
    assert mu.shape == (2,)  # This helper is 2D-only
    assert cov.shape == (2, 2)  # 2x2 covariance
    assert np.allclose(cov, cov.T)  # Symmetry check

    eigvals, eigvecs = np.linalg.eigh(cov)  # cov = Q Λ Q^T for symmetric matrices
    assert np.all(eigvals > 0)  # Positive-definite => ellipse exists

    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)  # Angles around the unit circle
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)  # circle:(2,n)

    transform = eigvecs @ np.diag(np.sqrt(eigvals))  # Unit circle -> covariance ellipse (k=1)
    pts = (mu.reshape(2, 1) + (k * transform @ circle)).T  # pts:(n,2) translate by μ, scale by k
    return pts  # Points ready for plotting


mu2 = np.array([0.0, 0.0])
cov2 = np.array([
    [2.0, 1.2],
    [1.2, 1.0],
], dtype=float)
pts = covariance_ellipse_points(mu2, cov2, k=2.0, n=180)
inv2 = np.linalg.inv(cov2)

q = np.einsum('...i,ij,...j->...', pts - mu2, inv2, pts - mu2)
assert np.allclose(q, 4.0, atol=1e-6)
```

---

### Exercise 4.6B: Visualization (2D Gaussian varying covariance) (REQUIRED)

#### Prompt

Make covariance **visible**:

1) **Basic**

- Build a 2D grid and plot Gaussian density contours for `N(μ, Σ)`.

2) **Intermediate**

- Compare at least 3 covariance matrices:
  - isotropic (`Σ = I`)
  - anisotropic (different variances per axis)
  - correlated (non-zero off-diagonal)

3) **Advanced**

- On each plot, draw the **covariance ellipse** for `k=2` and verify points satisfy `(x-μ)^T Σ^{-1} (x-μ) ≈ k^2`.

#### Solution

```python
import numpy as np  # NumPy: grid building + linear algebra + vectorized evaluation
import matplotlib.pyplot as plt  # Matplotlib: 2D contour plots + ellipse overlay


def multivariate_gaussian_pdf_grid(xx: np.ndarray, yy: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    # xx, yy: 2D grids (H,W), typically from np.meshgrid
    xx = np.asarray(xx, dtype=float)  # Ensure float dtype for stable exp/log math
    yy = np.asarray(yy, dtype=float)  # Same contract: shape (H,W)
    mu = np.asarray(mu, dtype=float)  # mu:(2,) mean vector
    cov = np.asarray(cov, dtype=float)  # cov:(2,2) covariance matrix

    assert mu.shape == (2,)  # Sanity: this visualization is 2D
    assert cov.shape == (2, 2)  # Sanity: 2x2 covariance
    assert np.allclose(cov, cov.T)  # Covariance must be symmetric

    eigvals = np.linalg.eigvalsh(cov)  # Eigenvalues for symmetric matrix (real + stable)
    assert np.all(eigvals > 0.0)  # Positive-definite => invertible covariance

    inv = np.linalg.inv(cov)  # Σ^{-1} for quadratic form
    det = np.linalg.det(cov)  # |Σ| for normalization

    pos = np.dstack([xx, yy])  # pos:(H,W,2) stacks coordinates into the last axis
    diff = pos - mu.reshape(1, 1, 2)  # diff:(H,W,2) subtract mean via broadcasting

    quad = np.einsum('...i,ij,...j->...', diff, inv, diff)  # (x-μ)^T Σ^{-1} (x-μ) on the whole grid
    expo = -0.5 * quad  # Gaussian exponent

    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))  # 2D normalization constant
    pdf = norm * np.exp(expo)  # pdf:(H,W) density over the grid

    return pdf  # Ready for contour/contourf


def covariance_ellipse_points(mu: np.ndarray, cov: np.ndarray, k: float = 2.0, n: int = 200) -> np.ndarray:
    # Generates points on: (x-μ)^T Σ^{-1} (x-μ) = k^2
    mu = np.asarray(mu, dtype=float)  # Ensure float mean
    cov = np.asarray(cov, dtype=float)  # Ensure float covariance

    assert mu.shape == (2,)  # This helper is 2D-only
    assert cov.shape == (2, 2)  # 2x2 covariance
    assert np.allclose(cov, cov.T)  # Symmetry check

    eigvals, eigvecs = np.linalg.eigh(cov)  # Symmetric eigendecomposition: cov = Q Λ Q^T
    assert np.all(eigvals > 0.0)  # Positive-definite eigenvalues

    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)  # Angles around the unit circle
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)  # circle:(2,n) unit circle

    transform = eigvecs @ np.diag(np.sqrt(eigvals))  # Maps unit circle -> k=1 covariance ellipse
    pts = (mu.reshape(2, 1) + (k * transform @ circle)).T  # pts:(n,2) translate by μ and scale by k

    return pts  # Points for plotting


mu = np.array([0.0, 0.0], dtype=float)  # Use μ=0 to isolate the effect of Σ

covs = [
    np.eye(2, dtype=float),  # Σ1: isotropic (circle)
    np.array([[3.0, 0.0], [0.0, 1.0]], dtype=float),  # Σ2: anisotropic (axis-aligned ellipse)
    np.array([[2.0, 1.2], [1.2, 1.0]], dtype=float),  # Σ3: correlated (rotated ellipse)
]  # Covariance candidates

labels = [
    "Σ = I (isotropic)",  # Label for subplot 1
    "Σ = diag(3,1) (anisotropic)",  # Label for subplot 2
    "Σ with correlation (rotated ellipse)",  # Label for subplot 3
]  # Titles

grid = np.linspace(-4.0, 4.0, 250)  # 1D axis to build the 2D grid
xx, yy = np.meshgrid(grid, grid)  # xx,yy:(H,W) grid coordinates

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)  # Layout: 1x3

for ax, cov, title in zip(axes, covs, labels):  # Iterate over covariance scenarios
    Z = multivariate_gaussian_pdf_grid(xx, yy, mu, cov)  # Z:(H,W) density values
    ax.contour(xx, yy, Z, levels=10)  # Contour lines: equal density

    pts = covariance_ellipse_points(mu, cov, k=2.0, n=240)  # pts:(n,2) k=2 ellipse
    ax.plot(pts[:, 0], pts[:, 1])  # Overlay ellipse

    inv = np.linalg.inv(cov)  # Σ^{-1} for verification
    q = np.einsum('...i,ij,...j->...', pts - mu, inv, pts - mu)  # Quadratic form values
    assert np.allclose(q, 4.0, atol=1e-6)  # Must equal k^2=4 (within tolerance)

    ax.set_title(title)  # Set subplot title
    ax.set_aspect('equal', 'box')  # Preserve aspect ratio so ellipse is not distorted
    ax.set_xlabel('x1')  # Horizontal axis label
    ax.set_ylabel('x2')  # Vertical axis label

plt.savefig('gaussian_covariance_contours.png', dpi=160)  # Save figure for reports
```

---

### Exercise 4.7: Log-Sum-Exp and stable log-softmax (REQUIRED)

#### Prompt

1) **Basic**

- Implement a stable `logsumexp(z)` (subtract `max(z)`).

2) **Intermediate**

- Implement `log_softmax(z) = z - logsumexp(z)`.

3) **Advanced**

- Verify `sum(exp(log_softmax(z))) == 1` and no `inf` for large logits.

#### Solution

```python
import numpy as np  # NumPy: stable log-sum-exp and basic numeric checks

def logsumexp(z: np.ndarray) -> float:  # log(sum(exp(z))) computed in a stable way
    z = np.asarray(z, dtype=float)  # Ensure float dtype for exp/log stability
    m = np.max(z)  # m = max(z) used as a numerical "anchor"
    return float(m + np.log(np.sum(np.exp(z - m))))  # logsumexp trick: m + log(sum(exp(z-m)))


def log_softmax(z: np.ndarray) -> np.ndarray:  # log-softmax computed from logsumexp
    z = np.asarray(z, dtype=float)  # Ensure float dtype
    return z - logsumexp(z)  # log_softmax(z) = z - log(sum(exp(z)))


z = np.array([1000.0, 0.0, -1000.0])  # Large-magnitude logits to stress-test stability
lsm = log_softmax(z)  # lsm:(3,) stable log-probabilities
probs = np.exp(lsm)  # Convert back to probabilities (should be finite)
assert np.isfinite(lsm).all()  # No NaN/inf in log-probabilities
assert np.isfinite(probs).all()  # No NaN/inf in probabilities
assert np.isclose(np.sum(probs), 1.0)  # Softmax probabilities must sum to 1
```

#### Solution (NaN trap: naive vs stable + verification) (REQUIRED)

```python
import numpy as np  # NumPy: exp/log + numerical validation
import warnings  # warnings: silence expected overflow warnings in the naive demo


def softmax_naive(z: np.ndarray) -> np.ndarray:  # Naive softmax (prone to overflow/underflow)
    z = np.asarray(z, dtype=float)  # Ensure float dtype
    expz = np.exp(z)  # Danger: exp(1000) -> inf
    return expz / np.sum(expz)  # If expz contains inf, normalization can become NaN


def softmax_stable(z: np.ndarray) -> np.ndarray:  # Stable softmax via max-shift
    z = np.asarray(z, dtype=float)  # Ensure float dtype
    z_shift = z - np.max(z)  # Subtracting max(z) keeps softmax invariant but prevents overflow
    expz = np.exp(z_shift)  # Now exponentials are <= 1
    return expz / np.sum(expz)  # Normalize to a valid probability distribution


z_big = np.array([1000.0, 1001.0, 1002.0])  # Extremely large logits

with warnings.catch_warnings():  # Avoid noisy RuntimeWarnings in output
    warnings.simplefilter("ignore")  # The overflow warning is expected in the naive version
    p_naive = softmax_naive(z_big)  # This typically contains NaN/inf

naive_ok = np.isfinite(p_naive).all() and np.isclose(np.sum(p_naive), 1.0)  # "valid distribution" predicate
assert not naive_ok  # We EXPECT this to fail: this is the whole point of the trap

p_stable = softmax_stable(z_big)  # Stable version should work
assert np.isfinite(p_stable).all()  # Must not contain NaN/inf
assert np.isclose(np.sum(p_stable), 1.0)  # Must sum to 1
assert np.argmax(p_stable) == np.argmax(z_big)  # Must preserve ordering
```

---

### Exercise 4.8: Stable softmax (invariance to constants)

#### Prompt

1) **Basic**

- Implement stable softmax: `exp(z-max)/sum(exp(z-max))`.

2) **Intermediate**

- Verify it sums to 1.

3) **Advanced**

- Verify invariance: `softmax(z) == softmax(z + c)`.

#### Solution

```python
import numpy as np  # NumPy: arrays, exp/max, and numerical checks

def softmax(z: np.ndarray) -> np.ndarray:  # Stable softmax for 1D logits
    z = np.asarray(z, dtype=float)  # Ensure float dtype
    z_shift = z - np.max(z)  # Max-shift: softmax(z) == softmax(z - max(z)) (prevents overflow)
    expz = np.exp(z_shift)  # Safe exponentials (values <= 0 => exp <= 1)
    return expz / np.sum(expz)  # Normalize into a valid probability distribution


z = np.array([2.0, 1.0, 0.0])  # Example logits
p = softmax(z)  # p:(3,) probabilities
assert np.isclose(np.sum(p), 1.0)  # Softmax must sum to 1

c = 100.0  # Add a constant offset
p2 = softmax(z + c)  # Invariance check
assert np.allclose(p, p2)  # Softmax is invariant to constant shifts
assert np.argmax(p) == np.argmax(z)  # Ordering preserved: argmax(prob) == argmax(logits)
```

---

### Exercise 4.9: Stable Binary Cross-Entropy (avoid log(0))

#### Prompt

1) **Basic**

- Implement BCE: `-mean(y log(p) + (1-y) log(1-p))`.

2) **Intermediate**

- Use `clip`/`epsilon` to avoid `log(0)`.

3) **Advanced**

- Verify:
  - BCE is near 0 for near-perfect predictions.
  - BCE ≈ `-log(0.9)` when `y=1` and `p=0.9`.

#### Solution

```python
import numpy as np  # NumPy: arrays, clipping, log, and mean

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:  # Stable BCE
    y_true = np.asarray(y_true, dtype=float)  # y_true:(N,) with values in {0,1}
    y_pred = np.asarray(y_pred, dtype=float)  # y_pred:(N,) predicted probabilities in [0,1]
    y_pred = np.clip(y_pred, eps, 1.0 - eps)  # Avoid log(0) and log(1-1)=log(0)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))  # Mean NLL


y_true = np.array([1.0, 0.0, 1.0, 0.0])  # Ground-truth labels
y_pred_good = np.array([0.999, 0.001, 0.999, 0.001])  # Near-perfect predictions
assert binary_cross_entropy(y_true, y_pred_good) < 0.01  # Loss should be near 0

assert np.isclose(binary_cross_entropy(np.array([1.0]), np.array([0.9])), -np.log(0.9), atol=1e-12)  # Single-sample sanity
```

---

### Exercise 4.10: Categorical Cross-Entropy (multiclass) + one-hot

#### Prompt

1) **Basic**

- Implement CCE: `-mean(sum(y_true * log(y_pred)))`.

2) **Intermediate**

- Make sure `y_pred` has no zeros (epsilon).

3) **Advanced**

- Verify the loss decreases when the correct-class probability increases.

#### Solution

```python
import numpy as np  # NumPy: arrays, clipping, log, and mean

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:  # Stable CCE
    y_true = np.asarray(y_true, dtype=float)  # y_true:(N,K) one-hot labels
    y_pred = np.asarray(y_pred, dtype=float)  # y_pred:(N,K) probabilities (rows sum to 1)
    y_pred = np.clip(y_pred, eps, 1.0)  # Avoid log(0) for predicted probabilities
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))  # Mean NLL over the batch


y_true = np.array([[0, 1, 0], [1, 0, 0]], dtype=float)  # Two samples, 3 classes (one-hot)
y_pred_bad = np.array([[0.34, 0.33, 0.33], [0.34, 0.33, 0.33]], dtype=float)  # Uncertain predictions
y_pred_good = np.array([[0.05, 0.90, 0.05], [0.90, 0.05, 0.05]], dtype=float)  # Confident on correct class

loss_bad = categorical_cross_entropy(y_true, y_pred_bad)  # Higher loss expected
loss_good = categorical_cross_entropy(y_true, y_pred_good)  # Lower loss expected
assert loss_good < loss_bad  # Sanity: better predictions => lower cross-entropy
```

---

### (Bonus) Exercise 4.11: Markov chain (transition matrix)

#### Prompt

1) **Basic**

- Define a transition matrix `P` (rows sum to 1).

2) **Intermediate**

- Propagate a distribution `π_{t+1} = π_t P` and verify it remains a distribution.

3) **Advanced**

- Approximate a stationary distribution by iterating and verify `π ≈ πP`.

4) **Bonus (matrix powers)**

- Verify that iterating `π_{t+1} = π_t P` for `k` steps matches `π_t P^k` via `np.linalg.matrix_power`.

#### Solution

```python
import numpy as np

P = np.array([
    [0.9, 0.1],
    [0.2, 0.8],
], dtype=float)
assert np.allclose(P.sum(axis=1), 1.0)

k = 50
pi0 = np.array([1.0, 0.0])
pi = pi0.copy()
for _ in range(k):
    pi = pi @ P
    assert np.isclose(np.sum(pi), 1.0)
    assert np.all(pi >= 0)

pi_power = pi0 @ np.linalg.matrix_power(P, k)
assert np.allclose(pi, pi_power, atol=1e-12)

pi_star = pi.copy()
assert np.allclose(pi_star, pi_star @ P, atol=1e-6)
```

## Deliverables

### E1: `probability.py`

Implement a minimal probability toolkit with NumPy:

- `gaussian_pdf`
- `multivariate_gaussian_pdf`
- `mle_gaussian`
- stable `softmax`
- `cross_entropy` (binary)
- `categorical_cross_entropy` (multiclass)

### E2: Tests

- Add `tests/test_probability.py`.
- Tests should cover:
  - Gaussian PDF sanity (standard normal at 0 ≈ 0.3989)
  - softmax sums to 1 and preserves ordering
  - MLE recovers parameters with enough data
  - cross-entropy near 0 for near-perfect predictions

---

## Consolidation (common errors + debugging)

Common failure patterns:

- Confusing PDF value with probability (continuous vs discrete).
- `log(0)` in cross-entropy (always use `epsilon` / `clip`).
- overflow in `exp` (use log-sum-exp / log-softmax).
- “magic MLE”: if you cannot explain why the mean appears, redo the Bernoulli worked example.

Debugging checklist:

- If something becomes `nan/inf`, check:
  - `np.log` on zeros
  - `np.exp` on large logits
  - probability normalization
- Record cases and fixes in `study_tools/DIARIO_ERRORES.md`.

---

## Feynman challenge (whiteboard)

Explain in 5 lines or less:

1. Why is maximizing likelihood equivalent to minimizing negative log-likelihood?
2. Why is the Bernoulli MLE just “fraction of heads”?
3. What does `π_{t+1} = π_t P` mean and why is it linear algebra?

---

## Completion checklist

- [ ] I can explain Bayes’ theorem with an example.
- [ ] I can compute a Gaussian PDF by hand.
- [ ] I understand why MLE leads to cross-entropy as the loss.
- [ ] I implemented numerically-stable softmax.
- [ ] I can derive the Bernoulli MLE and explain it.
- [ ] I can explain what a Markov chain is and what a transition matrix represents.
- [ ] All tests for `probability.py` pass.

---

## Navigation

**[← Module 03: Calculus](03_CALCULO_MULTIVARIANTE.md)** | **[Module 05: Supervised Learning →](05_SUPERVISED_LEARNING.md)**
