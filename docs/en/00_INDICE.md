# Master Guide Index – ML Specialist (v3.3)

> 24-week roadmap for the CU Boulder MS in AI Pathway (Line 1: Machine Learning)

**Language:** [Español →](../00_INDICE.md)

---

## What this guide optimizes for

- **Correctness**: derivations + gradient checking where applicable
- **Implementation skill**: build core algorithms from scratch
- **Exam readiness**: weekly cadence + PB-style checkpoints
- **Professional habits**: typing (`mypy`), linting (`ruff`), tests (`pytest`)

---

## Roadmap (3 phases)

1) **Phase 1 – Foundations (Weeks 1–8)**

- Python for data + linear algebra + multivariate calculus + probability.

2) **Phase 2 – ML Core (Weeks 9–20)**

- Supervised + Unsupervised + Deep Learning.
- Includes **Tree-Based Models** (Decision Tree from scratch).
- CNN track is split into:
  - NumPy forward pass (shape mastery)
  - PyTorch CNN training (practical, no manual CNN backward)

3) **Phase 3 – Capstone (Weeks 21–24)**

- End-to-end MNIST pipeline + benchmark report + minimal deployment.
- Optional benchmark: **Fashion-MNIST**.

---

## Quick navigation (0→100 blocks)

These links go to the English modules (anchors are stable):

- M01: [How to use](01_PYTHON_CIENTIFICO.md#m01-0)
- M02: [How to use](02_ALGEBRA_LINEAL_ML.md#m02-0)
- M03: [How to use](03_CALCULO_MULTIVARIANTE.md#m03-0)
- M04: [How to use](04_PROBABILIDAD_ML.md#m04-0)
- M05: [How to use](05_SUPERVISED_LEARNING.md#m05-0)
- M06: [How to use](06_UNSUPERVISED_LEARNING.md#m06-0)
- M07: [How to use](07_DEEP_LEARNING.md#m07-0)
- M08: [How to use](08_PROYECTO_MNIST.md#m08-0)

---

## Deliverables by phase

### Phase 1

- Data loading/cleaning pipeline (Pandas → NumPy)
- Linear algebra utilities
- Gradient Descent + gradient intuition
- Probability primitives used later (softmax, MLE intuition)
- Visualization protocol (run provided scripts)

### Phase 2

- Logistic Regression + metrics + cross-validation
- Unsupervised stack: K-Means + PCA
- Deep Learning: MLP with manual backprop
- **Tree-Based Models**: `../scripts/decision_tree_from_scratch.py`
- **CNN practical training**: `../scripts/train_cnn_pytorch.py`

### Phase 3

- Benchmark + report (MODEL_COMPARISON)
- Optional: **Fashion-MNIST** benchmark
- Dirty data check (generate corrupted dataset): `../scripts/corrupt_mnist.py`
- Minimal deployment: checkpoint + single-image inference: `../scripts/predict.py`

---

## Next

- Study plan: **[PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md)**
- Checklist: **[CHECKLIST.md](CHECKLIST.md)**
