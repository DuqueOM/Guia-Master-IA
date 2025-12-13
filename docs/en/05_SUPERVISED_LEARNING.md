# Module 05 - Supervised Learning

> **Goal:** master Linear/Logistic Regression, evaluation metrics, and model validation.
> **Phase:** 2 - ML Core | **Weeks 9–12**
> **Pathway course:** Introduction to Machine Learning: Supervised Learning

**Language:** English | [Español →](../05_SUPERVISED_LEARNING.md)

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

### Metrics

- Confusion matrix: TP/TN/FP/FN.
- Accuracy vs Precision vs Recall vs F1.
- For imbalanced datasets: accuracy can lie.

### Validation (anti-leakage)

- Split *before* any target-dependent transformations.
- Fit preprocessing only on train data.
- Use K-fold to estimate performance more reliably.

---

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

## Deliverables

- `supervised_learning.py`:
  - linear regression (normal equation + gradient descent)
  - logistic regression (+ optional L1/L2)
  - core metrics
  - validation helpers (split, K-fold)

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
