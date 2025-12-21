# Laboratorios Interactivos (Ver para Entender)

Esta guía no se limita a texto: incluye **laboratorios ejecutables** para construir intuición visual antes de resolver ejercicios.

Directorio raíz:

- [`interactive_labs/`](../interactive_labs)

## Instalación

Base del proyecto:

```bash
pip install -r requirements.txt
```

Laboratorios (visual):

```bash
pip install -r requirements-visual.txt
```

PyTorch (para M07):

```bash
pip install -e ".[pytorch]"
```

## Ejecución (comandos)

Importante: ejecuta estos comandos **desde la raíz del repo**.

### M02 — Álgebra Lineal

- App: Transformación lineal 2×2 (rejilla + eigenvectors)
  - `streamlit run interactive_labs/m02_linear_algebra/transformacion_lineal_app.py`

- Animación (Manim): transformación lineal (shear)
  - `manim -pqh interactive_labs/m02_linear_algebra/animacion_matriz.py AnimacionMatriz`

### M05 — Supervised Learning

- App: Intuición de regresión lineal
  - `streamlit run interactive_labs/m05_supervised/visualizacion_regresion.py`

- App: Overfitting / Bias–Variance con polinomios
  - `streamlit run interactive_labs/m05_supervised/overfitting_bias_variance_app.py`

### M06 — Unsupervised Learning

- App: Intuición de PCA (rotación manual 3D → proyección 2D) + referencia SVD
  - `streamlit run interactive_labs/m06_unsupervised/pca_rotation_plotly_app.py`

- Animación (Manim): proyección 2D sobre el eje principal (PC1)
  - `manim -pqh interactive_labs/m06_unsupervised/animacion_pca_proyeccion.py AnimacionPCAProyeccion`

### M07 — Deep Learning

- App: Playground PyTorch (entrenamiento MLP en XOR + frontera de decisión)
  - `streamlit run interactive_labs/m07_deep_learning/pytorch_training_playground_app.py`
