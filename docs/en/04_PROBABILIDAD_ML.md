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
import numpy as np

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sigma = float(sigma)
    assert sigma > 0
    z = (x - mu) / sigma
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * z**2)


val0 = gaussian_pdf(np.array([0.0]), mu=0.0, sigma=1.0)[0]
assert np.isclose(val0, 0.39894228, atol=1e-4)

a = 1.7
assert np.isclose(
    gaussian_pdf(np.array([a]), 0.0, 1.0)[0],
    gaussian_pdf(np.array([-a]), 0.0, 1.0)[0],
    rtol=1e-12,
    atol=1e-12,
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

#### Solution

```python
import numpy as np

def multivariate_gaussian_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    d = x.shape[0]

    assert mu.shape == (d,)
    assert cov.shape == (d, d)
    assert np.allclose(cov, cov.T)
    eigvals = np.linalg.eigvals(cov)
    assert np.all(eigvals > 0)

    diff = x - mu
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    norm = 1.0 / (np.sqrt(((2.0 * np.pi) ** d) * det))
    expo = -0.5 * float(diff.T @ inv @ diff)
    return float(norm * np.exp(expo))


mu = np.array([0.0, 0.0])
cov = np.eye(2)
pdf0 = multivariate_gaussian_pdf(np.array([0.0, 0.0]), mu, cov)
assert np.isclose(pdf0, 1.0 / (2.0 * np.pi), atol=1e-6)
assert pdf0 > 0.0
```

---

### Exercise 4.7: Log-Sum-Exp and stable log-softmax

#### Prompt

1) **Basic**

- Implement a stable `logsumexp(z)` (subtract `max(z)`).

2) **Intermediate**

- Implement `log_softmax(z) = z - logsumexp(z)`.

3) **Advanced**

- Verify `sum(exp(log_softmax(z))) == 1` and no `inf` for large logits.

#### Solution

```python
import numpy as np

def logsumexp(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    m = np.max(z)
    return float(m + np.log(np.sum(np.exp(z - m))))


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return z - logsumexp(z)


z = np.array([1000.0, 0.0, -1000.0])
lsm = log_softmax(z)
probs = np.exp(lsm)
assert np.isfinite(lsm).all()
assert np.isfinite(probs).all()
assert np.isclose(np.sum(probs), 1.0)
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
import numpy as np

def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z_shift = z - np.max(z)
    expz = np.exp(z_shift)
    return expz / np.sum(expz)


z = np.array([2.0, 1.0, 0.0])
p = softmax(z)
assert np.isclose(np.sum(p), 1.0)

c = 100.0
p2 = softmax(z + c)
assert np.allclose(p, p2)
assert np.argmax(p) == np.argmax(z)
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
import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


y_true = np.array([1.0, 0.0, 1.0, 0.0])
y_pred_good = np.array([0.999, 0.001, 0.999, 0.001])
assert binary_cross_entropy(y_true, y_pred_good) < 0.01

assert np.isclose(binary_cross_entropy(np.array([1.0]), np.array([0.9])), -np.log(0.9), atol=1e-12)
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
import numpy as np

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))


y_true = np.array([[0, 1, 0], [1, 0, 0]], dtype=float)
y_pred_bad = np.array([[0.34, 0.33, 0.33], [0.34, 0.33, 0.33]], dtype=float)
y_pred_good = np.array([[0.05, 0.90, 0.05], [0.90, 0.05, 0.05]], dtype=float)

loss_bad = categorical_cross_entropy(y_true, y_pred_bad)
loss_good = categorical_cross_entropy(y_true, y_pred_good)
assert loss_good < loss_bad
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

#### Solution

```python
import numpy as np

P = np.array([
    [0.9, 0.1],
    [0.2, 0.8],
], dtype=float)
assert np.allclose(P.sum(axis=1), 1.0)

pi = np.array([1.0, 0.0])
for _ in range(50):
    pi = pi @ P
    assert np.isclose(np.sum(pi), 1.0)
    assert np.all(pi >= 0)

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
