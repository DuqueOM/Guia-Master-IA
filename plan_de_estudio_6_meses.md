# 📅 Plan de Estudio: 6 Meses para el MS-AI Pathway

> **Duración Total:** 24 semanas (~864 horas)
> **Ritmo:** 6 horas/día, Lunes a Sábado
> **Filosofía:** Matemáticas Aplicadas a Código

---

## 🗓️ Cronograma General

| Fase | Semanas | Módulos | Enfoque |
|------|---------|---------|---------|
| **FUNDAMENTOS** | 1-8 | M01-M04 | Python + Matemáticas |
| **ML CORE** | 9-20 | M05-M07 | Algoritmos del Pathway ⭐ |
| **INTEGRACIÓN** | 21-24 | M08 | Proyecto MNIST |

---

## 📘 FASE 1: FUNDAMENTOS (Semanas 1-8)

### Semanas 1-2: M01 - Python Científico

| Día | Actividad | Duración | Entregable |
|-----|-----------|----------|------------|
| L-M | Teoría NumPy/Pandas | 12h | Notas en papel |
| X-J | Notebooks prácticos | 12h | Scripts funcionando |
| V | Romper cosas (edge cases) | 6h | Diario de errores |
| S | Simulacro + Cierre | 6h | Checklist completado |

**Laboratorios Interactivos:**
- `M01_Fundamentos_Python/Laboratorios_Interactivos/`

---

### Semanas 3-5: M02 - Álgebra Lineal para ML

| Semana | Tema | Conceptos Clave |
|--------|------|-----------------|
| 3 | Vectores y Matrices | Dot product, normas, proyecciones |
| 4 | Transformaciones Lineales | Eigenvalues, determinantes |
| 5 | SVD y Aplicaciones | Compresión, PCA numérico |

**Laboratorios Interactivos:**
```bash
streamlit run M02_Algebra_Lineal/Laboratorios_Interactivos/transformacion_lineal_app.py
manim -pqh M02_Algebra_Lineal/Laboratorios_Interactivos/animacion_matriz.py AnimacionMatriz
```

---

### Semanas 6-7: M03 - Cálculo y Optimización

| Semana | Tema | Conceptos Clave |
|--------|------|-----------------|
| 6 | Derivadas y Gradientes | Parciales, Chain Rule |
| 7 | Gradient Descent | Learning rate, convergencia |

**Laboratorios Interactivos:**
```bash
streamlit run M03_Calculo_Optimizacion/Laboratorios_Interactivos/viz_gradient_3d.py
```

---

### Semana 8: M04 - Probabilidad y Estadística

| Día | Tema | Conceptos Clave |
|-----|------|-----------------|
| L-M | Teorema de Bayes | Prior, Likelihood, Posterior |
| X-J | Distribuciones | Gaussiana, Bernoulli |
| V-S | MLE y Cross-Entropy | Conexión con Loss Functions |

**Laboratorios Interactivos:**
```bash
python M04_Probabilidad_Estadistica/Laboratorios_Interactivos/gmm_3_gaussians_contours.py
```

---

## ⭐ FASE 2: ML CORE - PATHWAY (Semanas 9-20)

### Semanas 9-12: M05 - Aprendizaje Supervisado

| Semana | Tema | Implementación |
|--------|------|----------------|
| 9 | Regresión Lineal | Normal Equation + GD |
| 10 | Regresión Logística | Cross-Entropy, Sigmoid |
| 11 | Regularización | L1/L2, Bias-Variance |
| 12 | Árboles y Ensembles | Decision Tree from scratch |

**Laboratorios Interactivos:**
```bash
streamlit run M05_Aprendizaje_Supervisado/Laboratorios_Interactivos/overfitting_bias_variance_app.py
streamlit run M05_Aprendizaje_Supervisado/Laboratorios_Interactivos/visualizacion_regresion.py
```

**Entregables:**
- [ ] `logistic_regression.py` con tests
- [ ] Derivación analítica del gradiente

---

### Semanas 13-16: M06 - Aprendizaje No Supervisado

| Semana | Tema | Implementación |
|--------|------|----------------|
| 13 | K-Means | Lloyd's algorithm, K-Means++ |
| 14 | PCA | SVD, varianza explicada |
| 15 | GMM | Algoritmo EM |
| 16 | t-SNE/UMAP | Visualización de embeddings |

**Laboratorios Interactivos:**
```bash
streamlit run M06_Aprendizaje_No_Supervisado/Laboratorios_Interactivos/pca_rotation_plotly_app.py
```

**Entregables:**
- [ ] `kmeans.py` y `pca.py` con tests
- [ ] Visualización 2D de MNIST

---

### Semanas 17-20: M07 - Deep Learning

| Semana | Tema | Implementación |
|--------|------|----------------|
| 17 | Perceptrón y MLP | Forward pass |
| 18 | Backpropagation | Gradientes manuales |
| 19 | CNNs | Convolución, pooling |
| 20 | RNNs/LSTM | Secuencias (teoría) |

**Laboratorios Interactivos:**
```bash
streamlit run M07_Deep_Learning/Laboratorios_Interactivos/pytorch_training_playground_app.py
```

**Entregables:**
- [ ] `neural_network.py` con backprop manual
- [ ] Overfit test en XOR
- [ ] CNN entrenada con PyTorch

---

## 🎯 FASE 3: INTEGRACIÓN (Semanas 21-24)

### Semanas 21-24: M08 - Proyecto MNIST

| Semana | Componente | Demuestra |
|--------|------------|-----------|
| 21 | EDA + PCA + K-Means | Unsupervised |
| 22 | Logistic Regression OvA | Supervised |
| 23 | MLP desde cero | Deep Learning |
| 24 | Informe + Deployment | Integración |

**Entregables Finales:**
- [ ] Pipeline end-to-end funcional
- [ ] MODEL_COMPARISON.md con benchmarks
- [ ] README profesional en inglés
- [ ] Deployment mínimo con `predict.py`

---

## 📊 Ritmo Semanal Recomendado

```
┌──────────────────────────────────────────────────────────────┐
│  LUNES - MARTES (Días de Concepto)                           │
│  • Leer teoría en Teoria/                                    │
│  • Dibujar en papel (método Feynman)                         │
│  • NO escribir código nuevo                                  │
├──────────────────────────────────────────────────────────────┤
│  MIÉRCOLES - JUEVES (Días de Implementación)                 │
│  • Ejecutar notebooks en Notebooks/                          │
│  • Implementar algoritmos                                    │
│  • Validar con asserts                                       │
├──────────────────────────────────────────────────────────────┤
│  VIERNES (Día de "Romper Cosas")                             │
│  • Cambiar learning_rate de 0.01 a 10.0                      │
│  • Inicializar pesos en cero                                 │
│  • Documentar síntomas y causas                              │
├──────────────────────────────────────────────────────────────┤
│  SÁBADO (Día de Consolidación)                               │
│  • Simulacro de examen (1 hora)                              │
│  • Cierre semanal                                            │
│  • Ejecutar laboratorios interactivos                        │
└──────────────────────────────────────────────────────────────┘
```

---

## ✅ Checkpoints de Evaluación

| Semana | Checkpoint | Criterio de Éxito |
|--------|------------|-------------------|
| 8 | PB-8 | Fundamentos matemáticos sólidos |
| 16 | PB-16 | ML Supervisado + No Supervisado |
| 23 | PB-23 | Deep Learning + Proyecto 80% |
| 24 | FINAL | Portafolio completo |

---

## 📚 Recursos por Fase

### Fase 1 (Fundamentos)
- Mathematics for Machine Learning (Deisenroth)
- 3Blue1Brown - Essence of Linear Algebra

### Fase 2 (ML Core)
- Pattern Recognition and ML (Bishop)
- Elements of Statistical Learning (Hastie)

### Fase 3 (Integración)
- Deep Learning (Goodfellow)
- Papers originales de algoritmos

---

*Plan alineado con el MS-AI Pathway de la University of Colorado Boulder*
