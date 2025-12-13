# Module 06 - Unsupervised Learning

> **Goal:** master K-Means and PCA (with SVD) for structure discovery and dimensionality reduction.
> **Phase:** 2 - ML Core | **Weeks 13–16**
> **Pathway course:** Unsupervised Algorithms in Machine Learning

**Language:** English | [Español →](../06_UNSUPERVISED_LEARNING.md)

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
