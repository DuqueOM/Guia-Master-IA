# Interactive Labs (See to Understand)

This guide includes **executable labs** to build visual intuition before doing the exercises.

Root folder:

- [`interactive_labs/`](../../interactive_labs)

## Installation

Base project:

```bash
pip install -r requirements.txt
```

Visual labs:

```bash
pip install -r requirements-visual.txt
```

PyTorch (Module 07):

```bash
pip install -e ".[pytorch]"
```

## Run commands

Important: run these commands **from the repository root**.

### M02 — Linear Algebra

- App: 2×2 linear transformation (grid + eigenvectors)
  - `streamlit run interactive_labs/m02_linear_algebra/transformacion_lineal_app.py`

- Manim animation: linear transformation (shear)
  - `manim -pqh interactive_labs/m02_linear_algebra/animacion_matriz.py AnimacionMatriz`

### M05 — Supervised Learning

- App: Linear regression intuition
  - `streamlit run interactive_labs/m05_supervised/visualizacion_regresion.py`

- App: Overfitting / Bias–Variance with polynomial degree
  - `streamlit run interactive_labs/m05_supervised/overfitting_bias_variance_app.py`

### M06 — Unsupervised Learning

- App: PCA intuition (manual 3D rotation → 2D projection) + SVD reference
  - `streamlit run interactive_labs/m06_unsupervised/pca_rotation_plotly_app.py`

### M07 — Deep Learning

- App: PyTorch playground (MLP training on XOR + decision boundary)
  - `streamlit run interactive_labs/m07_deep_learning/pytorch_training_playground_app.py`
