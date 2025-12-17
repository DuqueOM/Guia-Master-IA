# Module 06 - Unsupervised Learning

> **Goal:** master K-Means and PCA (with SVD) for structure discovery and dimensionality reduction.
> **Phase:** 2 - ML Core | **Weeks 13–16**
> **Pathway course:** Unsupervised Algorithms in Machine Learning

---

<a id="m06-0"></a>

## How to use this module (0→100 mode)

**Purpose:** be able to:

- find structure without labels (clustering)
- reduce dimensionality with rigor (PCA)
- decide when *not* to use these methods

### Learning outcomes (measurable)

By the end of this module you can:

- Implement K-Means (Lloyd) + K-Means++ initialization.
- Evaluate clustering with inertia/elbow and silhouette (and explain limitations).
- Implement PCA via SVD and use explained variance to pick `n_components`.
- Diagnose failure modes and propose alternatives.

Quick references (Spanish source):

- [Glossary](GLOSARIO.md)
- [Resources](RECURSOS.md)
- [Plan v4](PLAN_V4_ESTRATEGICO.md)
- [Plan v5](PLAN_V5_ESTRATEGICO.md)
- Rubric: `study_tools/RUBRICA_v1.md` (scope `M06` in `rubrica.csv`; includes PB-16)

---

## What matters most (high-signal core)

### K-Means

- Objective (inertia):
  - `J = Σᵢ Σ_{x∈Cᵢ} ||x - μᵢ||²`
- Lloyd algorithm:
  - assign to nearest centroid
  - recompute centroids as mean
  - repeat until convergence
- K-Means++ initialization reduces bad local minima risk.

Failure modes to recognize:

- scale sensitivity (needs normalization)
- empty clusters
- non-spherical clusters (K-Means is not a general clustering tool)

### PCA

- Center the data.
- PCA in practice is best implemented via **SVD**:
  - `X_c = U S Vᵀ`
  - principal components are in `V`
- Explained variance ratio tells you how much structure you retain.

---

## 🎯 Topic-based progressive exercises + solutions

Rules:

- **Try first** without looking at solutions.
- **Suggested timebox:** 25–60 min per exercise.
- **Minimum success:** your solution must pass the `assert` checks.

---

### Exercise 6.1: Vectorized distances (K-Means) - shapes and argmin

#### Prompt

1) **Basic**

- Given `X` shape `(n,d)` and centroids `C` shape `(k,d)`, build `D2` shape `(n,k)` where `D2[i,j] = ||X_i - C_j||^2`.

2) **Intermediate**

- Compute assignments `labels = argmin_j D2[i,j]`.

3) **Advanced**

- Verify with an `assert` that `D2` matches a manual computation for one point.

#### Solution

```python
import numpy as np

X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
C = np.array([[0.0, 0.0], [2.0, 2.0]])

diff = X[:, None, :] - C[None, :, :]
D2 = np.sum(diff ** 2, axis=2)

assert D2.shape == (X.shape[0], C.shape[0])

labels = np.argmin(D2, axis=1)
assert labels.shape == (X.shape[0],)
assert labels.min() >= 0 and labels.max() < C.shape[0]

i = 2
manual0 = np.sum((X[i] - C[0]) ** 2)
manual1 = np.sum((X[i] - C[1]) ** 2)
assert np.isclose(D2[i, 0], manual0)
assert np.isclose(D2[i, 1], manual1)
assert labels[i] == int(np.argmin([manual0, manual1]))
```

<details open>
<summary><strong>📌 Pedagogical add-on — Exercise 6.1: Vectorized distances (shapes + broadcasting + argmin)</strong></summary>

#### 1) Metadata
- **Title:** From `||x-c||²` to an `(n,k)` distance matrix with no loops
- **ID (optional):** `M06-E06_1`
- **Estimated time:** 30–60 min
- **Level:** Intermediate
- **Dependencies:** Broadcasting + `axis` (Module 01), L2 norm (Module 02)

#### 2) Goals
- Build `D2:(n,k)` without Python loops over `n` or `k`.
- Pick the correct `axis` in `sum` and `argmin`.
- Debug shapes with a tiny example you can verify by hand.

#### 3) Common mistakes
- Reducing the wrong axis in `np.sum(..., axis=...)` (you must reduce over features `d`).
- Computing `sqrt` unnecessarily (for `argmin`, dist and dist² rank pairs the same).
- Using `argmin(axis=0)` (answers a different question).

#### 4) Teacher note
- Ask the student what each axis of `D2` represents.
</details>

---

### Exercise 6.2: Update step (centroids as means) + empty cluster case

#### Prompt

1) **Basic**

- Given `X` and `labels`, compute `C_new[j] = mean(X[labels==j])`.

2) **Intermediate**

- Verify shapes and no `NaN`.

3) **Advanced**

- Handle empty clusters: if a cluster has no points, keep the previous centroid.

#### Solution

```python
import numpy as np

X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
C = np.array([[0.0, 0.0], [10.0, 10.0]])

labels = np.argmin(np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2), axis=1)

C_new = C.copy()
for j in range(C.shape[0]):
    mask = labels == j
    if np.any(mask):
        C_new[j] = np.mean(X[mask], axis=0)

assert C_new.shape == C.shape
assert np.isfinite(C_new).all()
```

<details open>
<summary><strong>📌 Pedagogical add-on — Exercise 6.2: Centroid update (means + empty clusters)</strong></summary>

#### 1) Metadata
- **Title:** Why the centroid is the mean (and what to do if a cluster is empty)
- **ID (optional):** `M06-E06_2`
- **Estimated time:** 30–60 min
- **Level:** Intermediate

#### 2) Key ideas
- With labels fixed, the mean minimizes `Σ ||x-μ||²`.
- If `labels==j` selects no points, `mean` on an empty slice produces `NaN`.

#### 3) Empty-cluster strategies
- Keep the previous centroid (simple and stable).
- Reinitialize to a random point from `X`.
- Reinitialize to the highest-error point (advanced).

#### 4) Common mistakes
- Averaging with the wrong axis (you want a `(d,)` vector, so use `axis=0`).
- Skipping `np.isfinite` checks and letting NaNs propagate.

#### 5) Teacher note
- Ask the student to force an empty cluster and explain what breaks.
</details>

---

### Exercise 6.3: Inertia (objective) + Lloyd monotonicity

#### Prompt

1) **Basic**

- Implement `inertia(X, C, labels) = sum_i ||X_i - C_{labels_i}||^2`.

2) **Intermediate**

- Run one Lloyd iteration (assign → update → assign) and compare inertias.

3) **Advanced**

- Verify inertia does **not increase**.

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


np.random.seed(0)
X = np.vstack([
    np.random.randn(50, 2) + np.array([0.0, 0.0]),
    np.random.randn(50, 2) + np.array([5.0, 5.0]),
])

C0 = np.array([[0.0, 5.0], [5.0, 0.0]])
labels0 = assign_labels(X, C0)
J0 = inertia(X, C0, labels0)

C1 = update_centroids(X, labels0, C0)
labels1 = assign_labels(X, C1)
J1 = inertia(X, C1, labels1)

assert J1 <= J0 + 1e-12
assert J0 >= 0.0 and J1 >= 0.0
```

<details open>
<summary><strong>📌 Pedagogical add-on — Exercise 6.3: Inertia + Lloyd monotonicity (convergence ≠ global optimum)</strong></summary>

#### 1) Metadata
- **Title:** What inertia measures and why Lloyd decreases it
- **ID (optional):** `M06-E06_3`
- **Estimated time:** 30–75 min
- **Level:** Intermediate

#### 2) Core idea
- Assignment: with `C` fixed, choosing the nearest center minimizes `J` w.r.t. labels.
- Update: with labels fixed, setting each centroid to the mean minimizes `J` w.r.t. `C`.
- Alternating both steps ⇒ `J` decreases or stays the same.

#### 3) Convergence ≠ global optimum
- Lloyd converges, but initialization matters and local minima exist.
- That is why K-Means++ and multiple restarts are standard.

#### 4) Debugging
- If `J` increases, it is usually a wrong `axis`, wrong indexing (`C[labels]`), or `NaN`.

#### 5) Teacher note
- Ask the student to explain: “converges” vs “finds the best clustering”.
</details>

---

### Exercise 6.4: K-Means++ (correct probabilities)

#### Prompt

1) **Basic**

- Implement K-Means++ to select `k` centroids from `X`.

2) **Intermediate**

- Verify selected centroids are points from `X`.

3) **Advanced**

- Verify sampling probabilities sum to 1 at each step.

#### Solution

```python
import numpy as np

def kmeans_plus_plus(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centroids = [X[rng.integers(n)]]

    for _ in range(1, k):
        C = np.array(centroids)
        d2 = np.min(np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2), axis=1)
        probs = d2 / np.sum(d2)
        assert np.isclose(np.sum(probs), 1.0)
        centroids.append(X[rng.choice(n, p=probs)])

    return np.array(centroids)


np.random.seed(1)
X = np.random.randn(30, 2)
C = kmeans_plus_plus(X, k=3, seed=123)
assert C.shape == (3, 2)
for j in range(C.shape[0]):
    assert np.any(np.all(np.isclose(X, C[j]), axis=1))
```

<details open>
<summary><strong>📌 Pedagogical add-on — Exercise 6.4: K-Means++ (correct probabilities)</strong></summary>

#### 1) Metadata
- **Title:** Initialization that reduces bad local minima
- **ID (optional):** `M06-E06_4`
- **Estimated time:** 30–60 min
- **Level:** Intermediate

#### 2) Key idea
- K-Means++ samples new centroids with probability proportional to the squared distance to the nearest existing centroid.
- Intuition: centroids start spread out, covering the space better.

#### 3) Important checks
- `probs` must sum to 1.
- Selected centroids must be actual points from `X`.

#### 4) Edge case
- If `np.sum(d2) == 0`, there is no signal to sample a new centroid (all points are already at distance 0 from some centroid). In practice you can break or fallback to random.

#### 5) Teacher note
- Ask the student to compare random init vs K-Means++ on a clearly separated dataset.
</details>

---

### Exercise 6.5: Scale sensitivity (why normalization matters)

#### Prompt

1) **Basic**

- Build an example where scaling one feature changes the nearest-centroid assignment.

2) **Intermediate**

- Compute labels for scale `s=0.1` vs `s=10`.

3) **Advanced**

- Verify at least one label changes.

#### Solution

```python
import numpy as np

def assign_labels(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    D2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
    return np.argmin(D2, axis=1)


X = np.array([[2.0, 0.0]], dtype=float)
C = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=float)

labels_small = assign_labels(X * np.array([1.0, 0.1]), C * np.array([1.0, 0.1]))
labels_big = assign_labels(X * np.array([1.0, 10.0]), C * np.array([1.0, 10.0]))

assert labels_small.shape == (1,)
assert labels_big.shape == (1,)
assert labels_small[0] != labels_big[0]
```

<details open>
<summary><strong>📌 Pedagogical add-on — Exercise 6.5: Scale sensitivity (normalization)</strong></summary>

#### 1) Metadata
- **Title:** Why K-Means needs comparable feature scales
- **ID (optional):** `M06-E06_5`
- **Estimated time:** 20–45 min
- **Level:** Intermediate

#### 2) Key idea
- K-Means optimizes Euclidean distances: if one feature has a much larger scale, it dominates the distance.

#### 3) Practical rule
- Before K-Means/PCA, it is often mandatory to:
  - standardize (mean 0, std 1), or
  - normalize by range, depending on the domain.

#### 4) Teacher note
- Ask the student why normalization changes what “close” means.
</details>

---

### Exercise 6.6: PCA via SVD (shapes + sorted explained variance)

#### Prompt

1) **Basic**

- Center `X` and compute `U,S,Vt = svd(Xc)`.

2) **Intermediate**

- Project to `k=2` components and verify shapes.

3) **Advanced**

- Compute explained variance ratio and verify it is sorted descending.

#### Solution

```python
import numpy as np

def pca_svd(X: np.ndarray, k: int):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k].T
    Xk = Xc @ comps
    var = (S ** 2) / (Xc.shape[0] - 1)
    ratio = var / np.sum(var)
    return Xk, comps, ratio[:k]


np.random.seed(0)
n = 300
z = np.random.randn(n)
X = np.stack([z, 2.0 * z + 0.1 * np.random.randn(n), -z + 0.1 * np.random.randn(n)], axis=1)

X2, comps, r = pca_svd(X, k=2)
assert X2.shape == (n, 2)
assert comps.shape == (3, 2)
assert r.shape == (2,)
assert r[0] >= r[1]
assert 0.0 <= r.sum() <= 1.0
```

<details open>
<summary><strong>📌 Pedagogical add-on — Exercise 6.6: PCA via SVD (shapes + explained variance)</strong></summary>

#### 1) Metadata
- **Title:** Numerically stable PCA (SVD) without forming the covariance matrix
- **ID (optional):** `M06-E06_6`
- **Estimated time:** 45–90 min
- **Level:** Intermediate/Advanced

#### 2) Shapes you must justify
- `X:(n,d)` → `Xc:(n,d)` after centering
- `Vt` contains directions in feature space; `comps = Vt[:k].T` has shape `(d,k)`
- `Xk = Xc @ comps` has shape `(n,k)`

#### 3) Explained variance
- With SVD, singular values `S` relate to variance: `var = S^2/(n-1)`.
- `var/sum(var)` gives the explained variance ratio.

#### 4) Teacher note
- Ask why centering is required for PCA.
</details>

---

### Exercise 6.7: PCA reconstruction (error decreases with more components)

#### Prompt

1) **Basic**

- Reconstruct `X` from `k` components: `X_rec = Xc @ V_k @ V_k^T + mean`.

2) **Intermediate**

- Compare reconstruction error for `k=1` vs `k=2`.

3) **Advanced**

- Verify the error with `k=2` is smaller or equal.

#### Solution

```python
import numpy as np

def pca_reconstruct(X: np.ndarray, k: int) -> np.ndarray:
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k].T
    Xk = Xc @ Vk
    X_rec = Xk @ Vk.T + mu
    return X_rec


np.random.seed(1)
n = 200
z = np.random.randn(n)
X = np.stack([z, 2.0 * z + 0.2 * np.random.randn(n), -z + 0.2 * np.random.randn(n)], axis=1)

X1 = pca_reconstruct(X, k=1)
X2 = pca_reconstruct(X, k=2)
err1 = np.linalg.norm(X - X1)
err2 = np.linalg.norm(X - X2)
assert err2 <= err1 + 1e-12
```

<details open>
<summary><strong>📌 Pedagogical add-on — Exercise 6.7: PCA reconstruction (bias vs compression)</strong></summary>

#### 1) Metadata
- **Title:** More components ⇒ lower reconstruction error (but less compression)
- **ID (optional):** `M06-E06_7`
- **Estimated time:** 30–60 min
- **Level:** Intermediate

#### 2) Key idea
- `Vk Vk^T` is the projector onto the `k`-dimensional subspace.
- Increasing `k` enlarges the subspace, so projection error cannot increase.

#### 3) Teacher note
- Ask the student to connect reconstruction error to cumulative explained variance.
</details>

---

### (Bonus) Exercise 6.8: Silhouette (minimal implementation for a tiny dataset)

#### Prompt

- Implement silhouette for a tiny dataset.
- Verify the mean silhouette score is in `[-1, 1]`.

#### Solution

```python
import numpy as np

def pairwise_dist(X: np.ndarray) -> np.ndarray:
    D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.maximum(D2, 0.0))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    D = pairwise_dist(X)
    n = X.shape[0]
    uniq = np.unique(labels)
    s = np.zeros(n, dtype=float)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = np.mean(D[i, same]) if np.any(same) else 0.0

        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            mask = labels == c
            if np.any(mask):
                b = min(b, float(np.mean(D[i, mask])))

        if b == np.inf:
            s[i] = 0.0
        else:
            denom = max(a, b)
            s[i] = 0.0 if denom == 0.0 else (b - a) / denom
    return float(np.mean(s))


X = np.array([[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.1, 4.9]])
labels = np.array([0, 0, 1, 1])
score = silhouette_score(X, labels)
assert -1.0 <= score <= 1.0
```

<details open>
<summary><strong>📌 Pedagogical add-on — Exercise 6.8: Silhouette (intuition + limits)</strong></summary>

#### 1) Metadata
- **Title:** Internal clustering metric (no labels)
- **ID (optional):** `M06-E06_8`
- **Estimated time:** 30–75 min
- **Level:** Advanced

#### 2) Intuition
- For each point:
  - `a` = mean distance to its own cluster
  - `b` = best (smallest) mean distance to another cluster
- `s = (b-a)/max(a,b)` lies in `[-1, 1]`.

#### 3) Limitations
- Requires pairwise distances: O(n²) cost (so we keep it tiny).
- Depends on the distance metric.

#### 4) Teacher note
- Ask the student to interpret: `s≈1`, `s≈0`, and `s<0`.
</details>

---

## Deliverable

### `unsupervised_learning.py`

Implement from scratch:

- K-Means + K-Means++
- inertia + silhouette
- PCA via SVD
- optional: reconstruction from PCA

Integration note (Capstone Week 21):

- Use PCA for 2D visualization and for fast downstream workflows.
- Use K-Means (`k=10`) to cluster MNIST and compare clusters vs labels.

---

## Completion checklist

- [ ] Implemented K-Means with K-Means++.
- [ ] Understand Lloyd algorithm.
- [ ] Can compute inertia and use elbow method.
- [ ] Implemented silhouette score.
- [ ] Implemented PCA using SVD.
- [ ] Understand explained variance and can pick `n_components`.
- [ ] Can reconstruct data from PCA.
- [ ] Applied PCA for 2D visualization.
- [ ] All module tests pass.

---

## Navigation

| Previous | Index | Next |
|---|---|---|
| [05_SUPERVISED_LEARNING](05_SUPERVISED_LEARNING.md) | [00_INDICE](00_INDICE.md) | [07_DEEP_LEARNING](07_DEEP_LEARNING.md) |
