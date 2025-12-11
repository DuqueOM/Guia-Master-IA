# 📚 GUÍA MAESTRA: MS AI PATHWAY - ML SPECIALIST (v3.1)

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
└── requirements.txt           # numpy, pandas, matplotlib, pytest
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

## ❌ Qué Se Eliminó del Plan Original

Para que esto quepa en 6 meses y sea efectivo para la **Línea 1 de ML**:

| Eliminado | Razón |
|-----------|-------|
| Linked Lists, Stacks, Queues | Irrelevante para matemáticas del Pathway |
| Binary Trees, BST | No se usa en los 3 cursos de ML |
| Grafos (BFS/DFS) | No es parte del currículo |
| QuickSort, MergeSort | En ML usas `numpy.sort()` |
| Inverted Index, TF-IDF | Proyecto de IR, no de CV/ML |
| Cadenas de Markov | Pertenece a Línea 2 (Estadística) |
| Motor de Búsqueda | Reemplazado por MNIST Pipeline |

---

## 📦 Material de Referencia

| Documento | Descripción | Uso |
|-----------|-------------|-----|
| [GLOSARIO.md](GLOSARIO.md) | Definiciones técnicas de ML | Consulta |
| [RECURSOS.md](RECURSOS.md) | Cursos y libros externos | Profundizar |
| [CHECKLIST.md](CHECKLIST.md) | Verificación de entregables | Seguimiento |

---

## 🚀 Comenzar

**[→ Módulo 01: Python + Pandas + NumPy](01_PYTHON_CIENTIFICO.md)**

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

## ✨ Cambios en v3.1 (vs v3.0)

| Cambio | Razón |
|--------|-------|
| **24 semanas** (antes 26) | Proyecto MNIST reducido a 4 sem (es dataset simple) |
| **Pandas en Módulo 01** | Necesario para cargar y limpiar datos reales |
| **Probabilidad para ML (Módulo 04)** | Bayes y MLE son esenciales para entender loss functions |
| **CNNs en Módulo 07** | El curso de Deep Learning de CU Boulder las cubre |

---

## ✨ Cambios en v3.2 (vs v3.1)

| Cambio | Razón |
|--------|-------|
| **Debugging NumPy (M01)** | 5 errores comunes que causan horas de frustración |
| **Estándares Profesionales** | `mypy`, `ruff`, `pytest` obligatorios desde Semana 2 |
| **Metodología Feynman** | "Reto del Tablero Blanco" en cada módulo |
| **Derivación Analítica (M05, M07)** | Simula exámenes de posgrado: derivar gradientes a mano |
| **Análisis Bias-Variance (M08)** | Concepto central de ML para diseño de modelos |
| **Formato Paper (M08)** | Notebook final con estructura académica |

---

## ✨ Cambios en v3.3 (vs v3.2)

| Cambio | Razón |
|--------|-------|
| **Gradient Checking (M03)** | Validación matemática de derivadas (técnica CS231n Stanford) |
| **Log-Sum-Exp Trick (M04)** | Softmax numéricamente estable (evita NaN) |
| **Shadow Mode (M05)** | Validar implementaciones vs sklearn |
| **Overfit Test (M07)** | Si no hace overfit en 10 ejemplos, tiene bug |
| **Análisis de Errores (M08)** | Visualizar y explicar fallos (nivel senior) |
| **Curvas de Aprendizaje (M08)** | Diagnóstico gráfico de Bias-Variance |

### Nuevos Entregables v3.3

| Módulo | Nuevo Entregable |
|--------|------------------|
| 03 | `grad_check.py` - validación numérica de derivadas |
| 04 | `softmax` con log-sum-exp trick |
| 05 | Comparativa Shadow Mode vs sklearn |
| 07 | `overfit_test.py` - debugging de redes |
| 08 | Sección "Error Analysis" + Learning Curves |

---

> 💡 **Filosofía v3.3:** Esta guía incluye **validación matemática rigurosa** en cada paso. No confíes en que tu código "parece funcionar"—valídalo con gradient checking, shadow mode y overfit tests. Si completas v3.3, tu código es **matemáticamente correcto y profesionalmente validado**.
