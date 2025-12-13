# Study Plan – ML Specialist (v3.3)

> 24 weeks | ~6h/day | Mon–Sat | Pathway Line 1: Machine Learning

**Language:** [Español →](../PLAN_ESTUDIOS.md)

---

## Hard gate (non-negotiable): virtual environment + reproducibility

If your environment is not isolated, **do not start Week 1**.

- From the repo root:
  - create/activate venv + install deps: `bash setup_env.sh`
  - or install reproducibly: `pip install -r requirements.txt`
- Success criteria:
  - `which python` points to the env (not system Python)
  - `python --version` satisfies `pyproject.toml` (`>=3.10`)
  - `python -c "import numpy, pandas, matplotlib"` works
  - `pre-commit install` done
  - PyTorch (Week 20) is installed **inside** the env only

## Weekly overview

- **Weeks 1–8 (Foundations)**: Python, Linear Algebra, Calculus, Probability
- **Weeks 9–20 (ML Core)**: Supervised, Unsupervised, Deep Learning
- **Weeks 21–24 (Capstone)**: MNIST Analyst project

---

## Default daily structure (recommended)

- **Theory (2–2.5h)**: notation + derivations + mental models
- **Implementation (2–2.5h)**: NumPy/Python from scratch
- **Practice (1h)**: exercises + visualization + error log

---

## Milestones (the only thing that matters)

### Phase 1 milestones

- You can implement **Gradient Descent** and explain learning rate stability.
- You can compute and verify gradients for common losses.
- You can explain chain rule as “local derivatives composed”.

Practical visualization (run, don’t build from scratch):

- `visualizations/viz_gradient_3d.py` (exports an interactive HTML)

### Phase 2 milestones

- Supervised:
  - Logistic Regression + metrics
  - Cross-validation
  - **Decision Tree from scratch** (Tree-Based Models)

Week 11 → prep for Week 12:

- **Recursion hard gate:** you must be able to define and test stopping conditions (`max_depth`, purity, `min_samples_split`, “no split improves”) before implementing a decision tree.

- Unsupervised:
  - K-Means
  - PCA

- Deep Learning:
  - MLP with manual backprop (correctness-focused)
  - CNNs:
    - NumPy forward pass (shape mastery)
    - PyTorch CNN training (real-world workflow)

### Phase 3 milestones (Capstone)

- End-to-end benchmark report (MODEL_COMPARISON)
- Optional benchmark: **Fashion-MNIST**
- Dirty data robustness:
  - create corrupted dataset: `scripts/corrupt_mnist.py`
  - document cleaning choices
- Minimal deployment:
  - train + save checkpoint: `scripts/train_cnn_pytorch.py`
  - run inference on one image: `scripts/predict.py`

---

## How to use this English plan

- Use this file as your **weekly compass**.
- For module-level details, use the English modules in `docs/en/` (and the Spanish originals only if you want extra depth).
