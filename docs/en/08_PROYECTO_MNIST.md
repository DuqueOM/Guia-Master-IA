# Module 08 - Final Project: MNIST Analyst

> **Goal:** build an end-to-end pipeline that proves competence across the 3 Line-1 courses.
> **Phase:** 3 - Capstone | **Weeks 21–24** (4 weeks)
> **Dataset:** MNIST (digits, 28×28) / **Fashion-MNIST** (optional, same format)

**Language:** English | [Español →](../08_PROYECTO_MNIST.md)

---

 <a id="m08-0"></a>

 ## How to use this module (0→100 mode)

 **Purpose:** demonstrate end-to-end mastery by integrating:

 - unsupervised learning (PCA + K-Means)
 - supervised learning (Logistic Regression OvA)
 - deep learning (MLP with manual backprop)

 **Execution rule:** keep everything reproducible and runnable end-to-end (one command, one report).

## What you are building

A 4-week, exam-grade project that demonstrates:

- Unsupervised learning (PCA + K-Means)
- Supervised learning (Logistic Regression One-vs-All)
- Deep learning (MLP with manual backprop)

Plus a final benchmark report and minimal deployment workflow.

---

## Schedule (4 weeks)

| Week | Focus | Course signal | Output |
|---|---|---|---|
| 21 | EDA + PCA + K-Means | Unsupervised Algorithms | PCA + K-Means working + plots |
| 22 | Classical classification | Supervised Learning | Logistic Regression OvA |
| 23 | Deep Learning | Intro to Deep Learning | MLP with manual backprop |
| 24 | Benchmark + report | Integration | `MODEL_COMPARISON.md` + minimal deployment |

Rubric:

- `study_tools/RUBRICA_v1.md` (scope `M08` in `rubrica.csv`)
- Hard condition: **PB-23 ≥ 80/100** (if PB-23 < 80 ⇒ “not ready” even if the global score is high)

---

## Practical notes (Week 24)

- **Fashion-MNIST (optional):** run the benchmark on Fashion-MNIST for a more realistic gap between LR vs MLP/CNN.
- **Dirty Data Check:** generate a corrupted dataset (noise/NaNs/inversion) with `scripts/corrupt_mnist.py` and document cleaning.
- **Minimal deployment:** train + save a CNN with `scripts/train_cnn_pytorch.py`, then predict a single 28×28 image with `scripts/predict.py`.

---

## Deliverables (high-signal)

- Reproducible training/evaluation pipeline.
- Clear plots:
  - PCA 2D visualization
  - learning curves (bias/variance diagnosis)
  - error analysis grid
- `MODEL_COMPARISON.md` report (methods, results, discussion).
- Minimal deployment proof:
  - saved checkpoint
  - single-image inference output

---

## Completion checklist (v3.3)

### Week 21: EDA + Unsupervised

- [ ] PCA reduces MNIST to 2D/50D with visualization.
- [ ] Analyzed explained variance per component.
- [ ] K-Means clusters digits without labels.
- [ ] Visualized centroids as 28×28 images.

### Week 22: Supervised classification

- [ ] Logistic Regression One-vs-All working.
- [ ] Accuracy >85% on test set.
- [ ] Computed per-class Precision/Recall/F1.
- [ ] Analyzed confusion matrix.

### Week 23: Deep Learning

- [ ] MLP architecture 784→128→64→10.
- [ ] Forward and backward passes implemented.
- [ ] Mini-batch SGD works.
- [ ] Accuracy >90% on test set.

### Week 24: Benchmark + report

- [ ] `MODEL_COMPARISON.md` completed.
- [ ] `README.md` professional in English.
- [ ] Optional benchmark: ran **Fashion-MNIST** (or justified why not).
- [ ] Dirty Data Check: created corrupted dataset with `scripts/corrupt_mnist.py` and documented cleaning.
- [ ] Minimal deployment: trained CNN with `scripts/train_cnn_pytorch.py` and saved checkpoint.
- [ ] Minimal deployment: ran `scripts/predict.py` on a 28×28 image and reported prediction.

### v3.3 requirements

- [ ] Bias–variance analysis with a practical experiment.
- [ ] Paper-style notebook/report (Abstract, Methods, Results, Discussion).
- [ ] Error analysis with visualization.
- [ ] Learning curves with diagnosis.
- [ ] “Error Analysis” section in `MODEL_COMPARISON.md`.
- [ ] `mypy` passes.
- [ ] `pytest` passes.

### Feynman

- [ ] Explain why MLP beats Logistic in 5 lines.
- [ ] Explain bias vs variance in 5 lines.
- [ ] Explain why 4↔9 confusions happen in 5 lines.

---

## Navigation

| Previous | Index |
|---|---|
| [07_DEEP_LEARNING](07_DEEP_LEARNING.md) | [00_INDICE](00_INDICE.md) |
