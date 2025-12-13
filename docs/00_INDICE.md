# 📚 GUÍA MAESTRA: MS AI PATHWAY - ML SPECIALIST (v3.3)

> **De Python Básico a Candidato del MS in AI de CU Boulder**
> **24 Semanas (6 Meses Exactos) | Enfoque: Línea 1 - Machine Learning**
> **Filosofía: "Matemáticas Aplicadas a Código"**

---

## 🎯 Objetivo de Esta Guía

**Dominio absoluto de las 3 materias de la Línea de Machine Learning** del Performance-Based Admission Pathway:

### ⭐ LÍNEA 1: Machine Learning (3 créditos) - FOCO PRINCIPAL

| Curso del Pathway | Módulo de Esta Guía |
|-------------------|---------------------|
| Introduction to Machine Learning: Supervised Learning | **Módulo 05** |
| Unsupervised Algorithms in Machine Learning | **Módulo 06** |
| Introduction to Deep Learning | **Módulo 07** |

### 📖 LÍNEA 2: Probabilidad y Estadística (Lectura Opcional)

| Curso del Pathway | Estado |
|-------------------|--------|
| Probability Foundations for Data Science and AI | Lectura opcional |
| Discrete-Time Markov Chains and Monte Carlo Methods | Lectura opcional |
| Statistical Estimation for Data Science and AI | Lectura opcional |

> **Nota:** La Línea 2 pertenece a la especialización de Estadística. Esta guía incluye solo la probabilidad esencial para ML (Módulo 04).

---

## 🗺️ El Mapa de Ruta: 3 Fases Críticas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FASE 1: FUNDAMENTOS (Semanas 1-8)                                          │
│  Objetivo: Python científico + matemáticas para leer papers de ML           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Módulo 01  Python + Pandas + NumPy   Carga de datos, vectorización [2 sem] │
│  Módulo 02  Álgebra Lineal para ML    Matrices, normas, SVD, eigen  [3 sem] │
│  Módulo 03  Cálculo Multivariante     Gradientes, Chain Rule        [2 sem] │
│  Módulo 04  Probabilidad para ML      Bayes, Gaussiana, MLE         [1 sem] │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FASE 2: NÚCLEO DE MACHINE LEARNING (Semanas 9-20)                          │
│  ⭐ SIMULACIÓN DEL PATHWAY - LÍNEA 1                                        │
│  Objetivo: Implementar desde cero los algoritmos de los 3 cursos            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Módulo 05  Supervised Learning       Regresión, Logística, CV      [4 sem] │
│  Módulo 06  Unsupervised Learning     K-Means, PCA, GMM             [4 sem] │
│  Módulo 07  Deep Learning             MLP, Backprop, CNNs           [4 sem] │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FASE 3: PROYECTO INTEGRADOR "MNIST ANALYST" (Semanas 21-24)                │
│  Objetivo: Un proyecto que demuestra competencia en las 3 áreas             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Módulo 08  MNIST End-to-End Pipeline                               [4 sem] │
│             PCA + K-Means + Logistic Regression + MLP desde cero            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Total: 8 módulos obligatorios | 24 semanas | ~864 horas**

---

## 👤 Perfil de Entrada

```
┌─────────────────────────────────────────────────────────────────┐
│  PERFIL IDEAL DE ENTRADA                                        │
├─────────────────────────────────────────────────────────────────┤
│  ✅ Python básico (variables, funciones, listas, diccionarios) │
│  ✅ Lógica de programación (if/else, loops)                    │
│  ✅ Matemáticas de bachillerato (álgebra básica)               │
│  ✅ Ganas de entender "cómo funciona por dentro"               │
│  ⚠️  NO se requiere: numpy, pandas, sklearn, ML previo         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📖 Módulos Obligatorios

### ⚡ Enlaces rápidos (bloques 0→100)

Estos atajos te llevan directo a la sección **"Cómo usar este módulo (modo 0→100)"** dentro de cada módulo:

| Módulo | Atajo |
|--------|-------|
| 01 | [M01 → Cómo usar](01_PYTHON_CIENTIFICO.md#m01-0) |
| 02 | [M02 → Cómo usar](02_ALGEBRA_LINEAL_ML.md#m02-0) |
| 03 | [M03 → Cómo usar](03_CALCULO_MULTIVARIANTE.md#m03-0) |
| 04 | [M04 → Cómo usar](04_PROBABILIDAD_ML.md#m04-0) |
| 05 | [M05 → Cómo usar](05_SUPERVISED_LEARNING.md#m05-0) |
| 06 | [M06 → Cómo usar](06_UNSUPERVISED_LEARNING.md#m06-0) |
| 07 | [M07 → Cómo usar](07_DEEP_LEARNING.md#m07-0) |

### FASE 1: Fundamentos (Semanas 1-8)

*Python científico con Pandas, matemáticas esenciales y probabilidad básica para ML.*

| # | Módulo | Descripción | Tiempo | Archivo |
|---|--------|-------------|--------|---------|
| 01 | **Python + Pandas + NumPy** | Carga de datos, limpieza, vectorización | 2 sem | [01_PYTHON_CIENTIFICO.md](01_PYTHON_CIENTIFICO.md) |
| 02 | **Álgebra Lineal para ML** | Vectores, matrices, normas, SVD, eigenvalues | 3 sem | [02_ALGEBRA_LINEAL_ML.md](02_ALGEBRA_LINEAL_ML.md) |
| 03 | **Cálculo Multivariante** | Derivadas parciales, gradiente, Chain Rule | 2 sem | [03_CALCULO_MULTIVARIANTE.md](03_CALCULO_MULTIVARIANTE.md) |
| 04 | **Probabilidad para ML** | Teorema de Bayes, Gaussiana, MLE | 1 sem | [04_PROBABILIDAD_ML.md](04_PROBABILIDAD_ML.md) |

**Entregables Fase 1:**
- Script de carga y limpieza de CSV con Pandas
- Librería `linear_algebra.py` con proyecciones y distancias
- Gradient Descent manual para minimizar funciones
- Implementación de MLE para estimar parámetros de Gaussiana
- Visualizaciones generativas (Protocolo D): transformaciones lineales y gradient descent interactivo
- Rescate cognitivo y transferencia (Protocolo E): cierre semanal, diario metacognitivo, puente teoría↔código y simulacro PB-8

---

### FASE 2: Núcleo de Machine Learning (Semanas 9-20) ⭐ PATHWAY LÍNEA 1

*Los 3 cursos del Pathway implementados desde cero.*

| # | Módulo | Curso del Pathway | Tiempo | Archivo |
|---|--------|-------------------|--------|---------|
| 05 | **Supervised Learning** | Introduction to ML: Supervised Learning | 4 sem | [05_SUPERVISED_LEARNING.md](05_SUPERVISED_LEARNING.md) |
| 06 | **Unsupervised Learning** | Unsupervised Algorithms in ML | 4 sem | [06_UNSUPERVISED_LEARNING.md](06_UNSUPERVISED_LEARNING.md) |
| 07 | **Deep Learning** | Introduction to Deep Learning | 4 sem | [07_DEEP_LEARNING.md](07_DEEP_LEARNING.md) |

**Entregables Fase 2:**
- `logistic_regression.py` con regularización L2
- `kmeans.py` y `pca.py` funcionales
- `neural_network.py` con backprop manual (MLP)
- Teoría de CNNs (convolución, pooling, stride)
- Rescate cognitivo y transferencia (Protocolo E): puente teoría↔código semanal, badges por módulo y simulacro PB-16

---

### FASE 3: Proyecto Final MNIST Analyst (Semanas 21-24)

*Pipeline completo en 4 semanas. MNIST es simple, no necesita más.*

| # | Módulo | Descripción | Tiempo | Archivo |
|---|--------|-------------|--------|---------|
| 08 | **MNIST Analyst** | Pipeline end-to-end de clasificación de dígitos | 4 sem | [08_PROYECTO_MNIST.md](08_PROYECTO_MNIST.md) |

**Proyecto: "End-to-End Handwritten Digit Analysis Pipeline"**

| Semana | Componente | Materia Demostrada |
|--------|------------|-------------------|
| 21 | EDA + PCA + K-Means | Unsupervised Algorithms |
| 22 | Regresión Logística One-vs-All | Supervised Learning |
| 23 | MLP con Backprop desde cero | Deep Learning |
| 24 | Informe + Comparación de Modelos | Integración |

Extensión Protocolo E (motivación + simulacro):

- Badges por módulo: `study_tools/BADGES_CHECKPOINTS.md`
- Simulacros performance-based: `study_tools/SIMULACRO_PERFORMANCE_BASED.md` (PB-8, PB-16, PB-23)

---

## 🔨 Estructura del Proyecto Final

```
mnist-analyst/
├── src/
│   ├── __init__.py
│   │
│   ├── # FASE 1: FUNDAMENTOS
│   ├── data_loader.py         # Carga con Pandas, limpieza (Módulo 01)
│   ├── linear_algebra.py      # Vectores, matrices, normas (Módulo 02)
│   ├── calculus.py            # Gradientes, derivadas (Módulo 03)
│   ├── probability.py         # Bayes, Gaussiana, MLE (Módulo 04)
│   │
│   ├── # FASE 2: ML CORE
│   ├── logistic_regression.py # Clasificación binaria/multiclase (Módulo 05)
│   ├── metrics.py             # Accuracy, Precision, Recall, F1 (Módulo 05)
│   ├── kmeans.py              # Clustering K-Means++ (Módulo 06)
│   ├── pca.py                 # Reducción dimensional SVD (Módulo 06)
│   ├── neural_network.py      # MLP con backprop (Módulo 07)
│   ├── activations.py         # Sigmoid, ReLU, Softmax (Módulo 07)
│   ├── optimizers.py          # SGD, Adam (Módulo 07)
│   │
│   └── # INTEGRACIÓN
│   └── mnist_pipeline.py      # Pipeline completo (Módulo 08)
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
│   └── mnist/                 # Dataset MNIST (28x28 imágenes)
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
├── README.md                  # Documentación (inglés)
├── pyproject.toml
└── requirements.txt           # stack de trabajo (numpy/pandas/matplotlib/plotly/ipywidgets/jupyterlab + tooling)
```

---

## ⏱️ Tiempo Total

| Fase | Semanas | Horas (~36h/sem) | Enfoque |
|------|---------|------------------|---------|
| Fundamentos (01-04) | 8 | ~288h | Python + Matemáticas + Probabilidad |
| ML Core (05-07) | 12 | ~432h | Algoritmos del Pathway |
| Proyecto MNIST (08) | 4 | ~144h | Integración y demo |
| **TOTAL** | **24** | **~864h** | |

**Duración:** 6 meses exactos con 6h/día (L-S)

---

## 📦 Material de Referencia

| Documento | Descripción | Uso |
|-----------|-------------|-----|
| [GLOSARIO.md](GLOSARIO.md) | Definiciones técnicas de ML | Consulta |
| [RECURSOS.md](RECURSOS.md) | Cursos y libros externos | Profundizar |
| [CHECKLIST.md](CHECKLIST.md) | Verificación de entregables | Seguimiento |

| [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md) | Plan de Acción Mejorado v4.0 (estrategia de ejecución y estudio diario) | Implementación del plan |
| [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md) | Plan de Acción Perfeccionado v5.0 (data rigor, validación externa y examen simulado) | Validación y certificación |

---

## 🚀 Comenzar

**[→ Módulo 01: Python + Pandas + NumPy](01_PYTHON_CIENTIFICO.md)**

### ⚡ Links rápidos (0→100)

- **M01 (Python Científico) — 0→100:** [Cómo usar este módulo](01_PYTHON_CIENTIFICO.md#m01-0)
- **M02 (Álgebra Lineal) — 0→100:** [Cómo usar este módulo](02_ALGEBRA_LINEAL_ML.md#m02-0)
- **M03 (Cálculo Multivariante) — 0→100:** [Cómo usar este módulo](03_CALCULO_MULTIVARIANTE.md#m03-0)
- **M04 (Probabilidad para ML) — 0→100:** [Cómo usar este módulo](04_PROBABILIDAD_ML.md#m04-0)
- **M05 (Supervised Learning) — 0→100:** [Cómo usar este módulo](05_SUPERVISED_LEARNING.md#m05-0)
- **M06 (Unsupervised Learning) — 0→100:** [Cómo usar este módulo](06_UNSUPERVISED_LEARNING.md#m06-0)
- **M07 (Deep Learning) — 0→100:** [Cómo usar este módulo](07_DEEP_LEARNING.md#m07-0)

---

## 📌 Restricciones del Proyecto

- ✅ **NumPy + Pandas permitidos** - Herramientas reales de ML
- ❌ **Sin sklearn/tensorflow/pytorch** - Algoritmos desde cero
- ✅ **100% local** - Todo se ejecuta en tu máquina
- ✅ **Matemáticas primero** - Entender antes de implementar
- ✅ **MNIST como benchmark** - Dataset estándar de la industria

---

## 🎯 Verificación de Competencias del Pathway

| Curso del Pathway | ¿Cubierto? | Evidencia en el Proyecto |
|-------------------|------------|--------------------------|
| **ML: Supervised Learning** | ✅ | Logistic Regression OvA, métricas, CV |
| **ML: Unsupervised Algorithms** | ✅ | K-Means++, PCA con SVD desde cero |
| **ML: Deep Learning** | ✅ | MLP con Backprop + teoría CNNs |

---
