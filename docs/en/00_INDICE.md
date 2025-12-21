# 📚 MASTER GUIDE: MS AI PATHWAY - ML SPECIALIST (v3.3)

> **From Basic Python to CU Boulder MS in AI Candidate**
> **24 Weeks (exactly 6 months) | Focus: Line 1 - Machine Learning**
> **Philosophy: “Applied Math to Code”**

**Language:** English | [Español →](../00_INDICE.md)

---

## 🎯 Goal of this guide

**Absolute mastery of the 3 courses in the Machine Learning line** of the Performance-Based Admission Pathway:

### ⭐ LINE 1: Machine Learning (3 credits) - PRIMARY FOCUS

| Pathway course | Module in this guide |
|-------------------|---------------------|
| Introduction to Machine Learning: Supervised Learning | **Module 05** |
| Unsupervised Algorithms in Machine Learning | **Module 06** |
| Introduction to Deep Learning | **Module 07** |

### 📖 LINE 2: Probability and Statistics (optional reading)

| Pathway course | Status |
|-------------------|--------|
| Probability Foundations for Data Science and AI | Optional reading |
| Discrete-Time Markov Chains and Monte Carlo Methods | Optional reading |
| Statistical Estimation for Data Science and AI | Optional reading |

> **Note:** Line 2 belongs to the Statistics specialization. This guide includes only the probability essential for ML (**Module 04**).

---

## 🗺️ The roadmap: 3 critical phases

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: FOUNDATIONS (Weeks 1-8)                                             │
│  Goal: scientific Python + math to read ML papers                             │
├───────────────────────────────────────────────────────────────────────────────┤
│  Module 01  Python + Pandas + NumPy   Data loading, vectorization     [2 wk]  │
│  Module 02  Linear Algebra for ML     Matrices, norms, SVD, eigen      [3 wk] │
│  Module 03  Multivariate Calculus     Gradients, Chain Rule            [2 wk] │
│  Module 04  Probability for ML        Bayes, Gaussian, MLE             [1 wk] │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: MACHINE LEARNING CORE (Weeks 9-20)                                 │
│  ⭐ PATHWAY SIMULATION - LINE 1                                              │
│  Goal: implement from scratch the algorithms of the 3 courses                │
├──────────────────────────────────────────────────────────────────────────────┤
│  Module 05  Supervised Learning       Regression, Logistic, CV        [4 wk] │
│  Module 06  Unsupervised Learning     K-Means, PCA, GMM               [4 wk] │
│  Module 07  Deep Learning             MLP, Backprop, CNNs             [4 wk] │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: CAPSTONE PROJECT “MNIST ANALYST” (Weeks 21-24)                     │
│  Goal: a project that demonstrates competence across the 3 areas             │
├──────────────────────────────────────────────────────────────────────────────┤
│  Module 08  MNIST End-to-End Pipeline                                 [4 wk] │
│             PCA + K-Means + Logistic Regression + MLP from scratch           │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Total: 8 mandatory modules | 24 weeks | ~864 hours**

---

## 👤 Entry profile

```
┌──────────────────────────────────────────────────────────────────┐
│  IDEAL ENTRY PROFILE                                             │
├──────────────────────────────────────────────────────────────────┤
│  ✅ Basic Python (variables, functions, lists, dictionaries)     │
│  ✅ Programming logic (if/else, loops)                           │
│  ✅ High-school math (basic algebra)                             │
│  ✅ Desire to understand “how it works inside”                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Mandatory modules

### Interactive Labs (See to Understand)

- [INTERACTIVE_LABS.md](INTERACTIVE_LABS.md)

### Quick links (0→100 blocks)

These shortcuts take you directly to the **“How to use this module (0→100 mode)”** section inside each module:

| Module | Shortcut |
|--------|-------|
| 01 | [M01 → How to use](01_PYTHON_CIENTIFICO.md#m01-0) |
| 02 | [M02 → How to use](02_ALGEBRA_LINEAL_ML.md#m02-0) |
| 03 | [M03 → How to use](03_CALCULO_MULTIVARIANTE.md#m03-0) |
| 04 | [M04 → How to use](04_PROBABILIDAD_ML.md#m04-0) |
| 05 | [M05 → How to use](05_SUPERVISED_LEARNING.md#m05-0) |
| 06 | [M06 → How to use](06_UNSUPERVISED_LEARNING.md#m06-0) |
| 07 | [M07 → How to use](07_DEEP_LEARNING.md#m07-0) |

### PHASE 1: Foundations (Weeks 1-8)

*Scientific Python with Pandas, essential math, and basic probability for ML.*

| # | Module | Description | Time | File |
|---|--------|-------------|--------|---------|
| 01 | **Python + Pandas + NumPy** | Data loading, cleaning, vectorization | 2 wk | [01_PYTHON_CIENTIFICO.md](01_PYTHON_CIENTIFICO.md) |
| 02 | **Linear Algebra for ML** | Vectors, matrices, norms, SVD, eigenvalues | 3 wk | [02_ALGEBRA_LINEAL_ML.md](02_ALGEBRA_LINEAL_ML.md) |
| 03 | **Multivariate Calculus** | Partial derivatives, gradient, Chain Rule | 2 wk | [03_CALCULO_MULTIVARIANTE.md](03_CALCULO_MULTIVARIANTE.md) |
| 04 | **Probability for ML** | Bayes theorem, Gaussian, MLE | 1 wk | [04_PROBABILIDAD_ML.md](04_PROBABILIDAD_ML.md) |

**Phase 1 deliverables:**

- CSV loading and cleaning script with Pandas
- `linear_algebra.py` library with projections and distances
- Manual Gradient Descent to minimize functions
- MLE implementation to estimate Gaussian parameters
- Generative visualizations (Protocol D): linear transforms and interactive gradient descent
- Cognitive rescue and transfer (Protocol E): weekly closing, metacognition diary, theory↔code bridge, and PB-8 simulation

---

### PHASE 2: Machine Learning Core (Weeks 9-20) ⭐ PATHWAY LINE 1

*The 3 Pathway courses implemented from scratch.*

| # | Module | Pathway course | Time | File |
|---|--------|-------------------|--------|---------|
| 05 | **Supervised Learning** | Introduction to ML: Supervised Learning | 4 wk | [05_SUPERVISED_LEARNING.md](05_SUPERVISED_LEARNING.md) |
| 06 | **Unsupervised Learning** | Unsupervised Algorithms in ML | 4 wk | [06_UNSUPERVISED_LEARNING.md](06_UNSUPERVISED_LEARNING.md) |
| 07 | **Deep Learning** | Introduction to Deep Learning | 4 wk | [07_DEEP_LEARNING.md](07_DEEP_LEARNING.md) |

**Phase 2 deliverables:**

- `logistic_regression.py` with L2 regularization
- `scripts/decision_tree_from_scratch.py` (Tree-Based Models: decision tree from scratch)
- `kmeans.py` and `pca.py` working
- `neural_network.py` with manual backprop (MLP)
- CNNs: theory + forward pass (NumPy) + CNN training with PyTorch (`scripts/train_cnn_pytorch.py`)
- Cognitive rescue and transfer (Protocol E): weekly theory↔code bridge, module badges, and PB-16 simulation

---

### PHASE 3: Final Project – MNIST Analyst (Weeks 21-24)

*A full pipeline in 4 weeks. MNIST is simple; you don’t need more.*

| # | Module | Description | Time | File |
|---|--------|-------------|--------|---------|
| 08 | **MNIST Analyst** | End-to-end handwritten digit classification pipeline | 4 wk | [08_PROYECTO_MNIST.md](08_PROYECTO_MNIST.md) |

**Project: “End-to-End Handwritten Digit Analysis Pipeline”**

| Week | Component | Demonstrated area |
|--------|------------|-------------------|
| 21 | EDA + PCA + K-Means | Unsupervised Algorithms |
| 22 | Logistic Regression One-vs-All | Supervised Learning |
| 23 | MLP with backprop from scratch | Deep Learning |
| 24 | Report + comparison + minimal deployment | Integration |

Protocol E extension (motivation + simulation):

- Module badges: `study_tools/BADGES_CHECKPOINTS.md`
- Performance-based simulations: `study_tools/SIMULACRO_PERFORMANCE_BASED.md` (PB-8, PB-16, PB-23)

---

## Final project structure

```
mnist-analyst/
├── src/
│   ├── __init__.py
│   │
│   ├── # PHASE 1: FOUNDATIONS
│   ├── data_loader.py         # Pandas loading, cleaning (Module 01)
│   ├── linear_algebra.py      # Vectors, matrices, norms (Module 02)
│   ├── calculus.py            # Gradients, derivatives (Module 03)
│   ├── probability.py         # Bayes, Gaussian, MLE (Module 04)
│   │
│   ├── # PHASE 2: ML CORE
│   ├── logistic_regression.py # Binary/multiclass classification (Module 05)
│   ├── metrics.py             # Accuracy, Precision, Recall, F1 (Module 05)
│   ├── kmeans.py              # K-Means++ clustering (Module 06)
│   ├── pca.py                 # Dimensionality reduction via SVD (Module 06)
│   ├── neural_network.py      # MLP with backprop (Module 07)
│   ├── activations.py         # Sigmoid, ReLU, Softmax (Module 07)
│   ├── optimizers.py          # SGD, Adam (Module 07)
│   │
│   └── # INTEGRATION
│   └── mnist_pipeline.py      # Full pipeline (Module 08)
│
├── tests/
│   ├── test_linear_algebra.py
│   ├── test_logistic_regression.py
│   ├── test_kmeans.py
│   ├── test_pca.py
│   ├── test_neural_network.py
│   └── test_pipeline.py
│
├── data/
│   └── mnist/                 # MNIST dataset (28x28 images)
│
├── notebooks/
│   ├── 01_eda_visualization.ipynb
│   ├── 02_pca_kmeans.ipynb
│   ├── 03_logistic_ova.ipynb
│   └── 04_mlp_benchmark.ipynb
│
├── docs/
│   ├── MATHEMATICAL_FOUNDATIONS.md
│   └── MODEL_COMPARISON.md
│
├── README.md                  # Documentation (English)
├── pyproject.toml
└── requirements.txt           # working stack (numpy/pandas/matplotlib/plotly/ipywidgets/jupyterlab + tooling)
```

---

## ⏱️ Total time

| Phase | Weeks | Hours (~36h/week) | Focus |
|------|---------|------------------|---------|
| Foundations (01-04) | 8 | ~288h | Python + Math + Probability |
| ML Core (05-07) | 12 | ~432h | Pathway algorithms |
| MNIST project (08) | 4 | ~144h | Integration and demo |
| **TOTAL** | **24** | **~864h** | |

**Duration:** exactly 6 months with 6h/day (Mon–Sat)

---

## 📦 Reference material

| Document | Description | Use |
|-----------|-------------|-----|
| [GLOSARIO.md](GLOSARIO.md) | ML technical definitions | Reference |
| [RECURSOS.md](RECURSOS.md) | External courses and books | Go deeper |
| [CHECKLIST.md](CHECKLIST.md) | Deliverable verification | Tracking |
| [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md) | Enhanced Action Plan v4.0 (execution strategy and daily study) | Plan execution |
| [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md) | Refined Action Plan v5.0 (data rigor, external validation, and mock exam) | Validation and certification |

---

## 🚀 Start here

**[→ Module 01: Python + Pandas + NumPy](01_PYTHON_CIENTIFICO.md)**

### ⚡ Quick links (0→100)

- **M01 (Scientific Python) — 0→100:** [How to use this module](01_PYTHON_CIENTIFICO.md#m01-0)
- **M02 (Linear Algebra) — 0→100:** [How to use this module](02_ALGEBRA_LINEAL_ML.md#m02-0)
- **M03 (Multivariate Calculus) — 0→100:** [How to use this module](03_CALCULO_MULTIVARIANTE.md#m03-0)
- **M04 (Probability for ML) — 0→100:** [How to use this module](04_PROBABILIDAD_ML.md#m04-0)
- **M05 (Supervised Learning) — 0→100:** [How to use this module](05_SUPERVISED_LEARNING.md#m05-0)
- **M06 (Unsupervised Learning) — 0→100:** [How to use this module](06_UNSUPERVISED_LEARNING.md#m06-0)
- **M07 (Deep Learning) — 0→100:** [How to use this module](07_DEEP_LEARNING.md#m07-0)

---

## 📌 Project constraints

- ✅ **NumPy + Pandas allowed** - real ML tools
- ❌ **No sklearn/tensorflow/pytorch** - algorithms from scratch
- ✅ **100% local** - everything runs on your machine
- ✅ **Math first** - understand before implementing
- ✅ **MNIST as benchmark** - industry standard dataset

---

## 🎯 Pathway competency verification

| Pathway course | Covered? | Evidence in the project |
|-------------------|------------|--------------------------|
| **ML: Supervised Learning** | ✅ | Logistic Regression OvA, metrics, CV |
| **ML: Unsupervised Algorithms** | ✅ | K-Means++, PCA via SVD from scratch |
| **ML: Deep Learning** | ✅ | MLP with backprop + CNN theory |

---
