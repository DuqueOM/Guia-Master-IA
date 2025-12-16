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

## Hard gate (setup + reproducibility)

- [ ] You work inside a virtual environment (venv/conda), not system Python.
- [ ] Dependencies installed reproducibly:
  - `bash setup_env.sh` **or** `pip install -r requirements.txt`
- [ ] `python --version` satisfies `pyproject.toml` (`>=3.10`).
- [ ] `python -c "import numpy, pandas, matplotlib"` works.
- [ ] `pre-commit install` done.

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
- **Weights interpretation (LogReg):** reshape weights into a 28×28 image (MNIST-like) and write a short interpretation.
- **Tree-Based Models:** you can explain impurity + information gain
- **Recursion hard gate (trees):** stopping conditions tested (`max_depth`, purity, `min_samples_split`, “no split improves”).
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

- Benchmark report: `docs/en/MODEL_COMPARISON.md`
- Main benchmark: Fashion-MNIST
- Dirty data check:
  - generate corrupted dataset: `scripts/corrupt_mnist.py`
  - document cleaning decisions
- Minimal deployment:
  - saved checkpoint
  - single-image inference: `scripts/predict.py`
