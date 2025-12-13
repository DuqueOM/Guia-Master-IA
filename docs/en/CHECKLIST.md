# Final Checklist – ML Specialist (v3.3)

> High-signal verification list for a 24-week program.

**Language:** [Español →](../CHECKLIST.md)

---

## Evaluation (Rubric)

- `study_tools/RUBRICA_v1.md`
- `rubrica.csv`

Recommended cadence:

- Weekly: light scoring
- Module closes (Weeks 12/16/20): full scoring
- PB checkpoints: PB-8 / PB-16 / PB-23

---

## Phase 1 (Weeks 1–8)

- Python + NumPy basics (arrays, broadcasting, axis ops)
- Linear algebra basics (norms, projections, SVD intuition)
- Calculus:
  - gradients
  - Gradient Descent working and explained
  - chain rule applied correctly
- Probability:
  - Bayes intuition
  - softmax (stable) + cross-entropy intuition

---

## Phase 2 (Weeks 9–20)

Supervised:

- Logistic Regression from scratch + metrics + CV
- **Tree-Based Models:** you can explain impurity + information gain
- Decision Tree deliverable runnable: `scripts/decision_tree_from_scratch.py`

Unsupervised:

- K-Means + PCA from scratch

Deep Learning:

- MLP with manual backprop (passes gradient sanity checks)
- CNNs:
  - NumPy forward pass implemented
  - PyTorch CNN training runnable: `scripts/train_cnn_pytorch.py`

---

## Phase 3 (Weeks 21–24) – MNIST Analyst

- Benchmark report (MODEL_COMPARISON)
- Optional: Fashion-MNIST benchmark
- Dirty data check:
  - generate corrupted dataset: `scripts/corrupt_mnist.py`
  - document cleaning decisions
- Minimal deployment:
  - saved checkpoint
  - single-image inference: `scripts/predict.py`
