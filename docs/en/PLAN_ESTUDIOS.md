# Study Plan – ML Specialist (v3.3)

> 24 weeks | ~6h/day | Mon–Sat | Pathway Line 1: Machine Learning

**Language:** [Español →](../PLAN_ESTUDIOS.md)

---

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
- For full detail and exercises, follow the linked Spanish modules until each module gets an English adaptation.
