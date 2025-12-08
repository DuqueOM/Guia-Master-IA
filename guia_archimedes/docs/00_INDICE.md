# 📚 Guía 0→100: MS in AI Pathway

> **De Python Básico a Candidato del MS in AI de CU Boulder**  
> **6 meses | 6h/día | 100% enfocado en las 6 materias del Pathway**

---

## 🎯 Objetivo Único de Esta Guía

Prepararte para aprobar las **6 materias obligatorias** del Performance-Based Admission Pathway:

### ⭐ Línea 1: Aprendizaje Automático (3 créditos)
| Curso del Pathway | Módulo de Esta Guía |
|-------------------|---------------------|
| Introduction to Machine Learning: Supervised Learning | **07_ML_SUPERVISADO** |
| Unsupervised Algorithms in Machine Learning | **08_ML_NO_SUPERVISADO** |
| Introduction to Deep Learning | **09_INTRO_DEEP_LEARNING** |

### ⭐ Línea 2: Probabilidad y Estadística (3 créditos)
| Curso del Pathway | Módulo de Esta Guía |
|-------------------|---------------------|
| Probability Fundamentals for Data Science and AI | **04_PROBABILIDAD** |
| Discrete-Time Markov Chains and Monte Carlo Methods | **06_MARKOV_MONTECARLO** |
| Statistical Estimation for Data Science and AI | **05_ESTADISTICA** |

---

## 📋 Estructura del Programa (TODO OBLIGATORIO)

```
┌─────────────────────────────────────────────────────────────────┐
│  FASE 1: FUNDAMENTOS (Semanas 1-6)                              │
│  Objetivo: Python profesional + base matemática para ML         │
├─────────────────────────────────────────────────────────────────┤
│  01_PYTHON_PROFESIONAL    Type hints, funciones puras, PEP8     │
│  02_OOP_DESDE_CERO        Clases, herencia, composición         │
│  03_ALGEBRA_LINEAL        Vectores, matrices, operaciones       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FASE 2: PROBABILIDAD Y ESTADÍSTICA (Semanas 7-14)              │
│  ⭐ PATHWAY LÍNEA 2 - 3 CRÉDITOS                                │
├─────────────────────────────────────────────────────────────────┤
│  04_PROBABILIDAD          Bayes, distribuciones, esperanza      │
│  05_ESTADISTICA           MLE, MAP, intervalos, hipótesis       │
│  06_MARKOV_MONTECARLO     Cadenas Markov, MCMC, PageRank        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FASE 3: MACHINE LEARNING (Semanas 15-22)                       │
│  ⭐ PATHWAY LÍNEA 1 - 3 CRÉDITOS                                │
├─────────────────────────────────────────────────────────────────┤
│  07_ML_SUPERVISADO        Regresión, clasificación, árboles     │
│  08_ML_NO_SUPERVISADO     K-Means, PCA, clustering              │
│  09_INTRO_DEEP_LEARNING   MLP, backprop, CNN/RNN conceptos      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FASE 4: PROYECTO FINAL (Semanas 23-26)                         │
│  Integración de todo el Pathway                                 │
├─────────────────────────────────────────────────────────────────┤
│  10_PROYECTO_FINAL        Pipeline ML completo desde cero       │
└─────────────────────────────────────────────────────────────────┘
```

**Total: 10 módulos obligatorios | 26 semanas | ~6 meses**

---

## 👤 Perfil de Entrada

```
┌─────────────────────────────────────────────────────────────────┐
│  PERFIL IDEAL DE ENTRADA                                        │
├─────────────────────────────────────────────────────────────────┤
│  ✅ Python básico (variables, funciones, listas, diccionarios) │
│  ✅ Lógica de programación (if/else, loops)                    │
│  ✅ Ganas de entender "cómo funciona por dentro"               │
│  ✅ Matemáticas de bachillerato (álgebra básica)               │
│  ⚠️  NO se requiere: numpy, pandas, sklearn, ML previo         │
└─────────────────────────────────────────────────────────────────┘
```

---

---

## 📖 Módulos Obligatorios

### FASE 1: Fundamentos (Semanas 1-6)
*Base de programación profesional necesaria para implementar ML*

| # | Módulo | Descripción | Tiempo | Archivo |
|---|--------|-------------|--------|---------|
| 01 | Python Profesional | Type hints, funciones puras, PEP8 | 2 sem | [01_PYTHON_PROFESIONAL.md](01_PYTHON_PROFESIONAL.md) |
| 02 | OOP desde Cero | Clases, herencia, composición | 2 sem | [02_OOP_DESDE_CERO.md](02_OOP_DESDE_CERO.md) |
| 03 | Álgebra Lineal para ML | Vectores, matrices, operaciones | 2 sem | [10_ALGEBRA_LINEAL.md](10_ALGEBRA_LINEAL.md) |

**Entregable:** Clase `Vector` y `Matrix` con operaciones básicas desde cero.

---

### FASE 2: Probabilidad y Estadística (Semanas 7-14) ⭐ PATHWAY LÍNEA 2
*Preparación directa para los 3 cursos de Probability & Statistics*

| # | Módulo | Curso del Pathway | Tiempo | Archivo |
|---|--------|-------------------|--------|---------|
| 04 | Fundamentos de Probabilidad | Probability Fundamentals for DS and AI | 3 sem | [19_PROBABILIDAD_FUNDAMENTOS.md](19_PROBABILIDAD_FUNDAMENTOS.md) |
| 05 | Estadística Inferencial | Statistical Estimation for DS and AI | 3 sem | [20_ESTADISTICA_INFERENCIAL.md](20_ESTADISTICA_INFERENCIAL.md) |
| 06 | Markov y Monte Carlo | Discrete-Time Markov Chains and Monte Carlo | 2 sem | [21_CADENAS_MARKOV_MONTECARLO.md](21_CADENAS_MARKOV_MONTECARLO.md) |

**Entregable:** Implementación de Bayes, MLE, MCMC, PageRank desde cero.

---

### FASE 3: Machine Learning (Semanas 15-22) ⭐ PATHWAY LÍNEA 1
*Preparación directa para los 3 cursos de Machine Learning*

| # | Módulo | Curso del Pathway | Tiempo | Archivo |
|---|--------|-------------------|--------|---------|
| 07 | ML Supervisado | Introduction to ML: Supervised Learning | 3 sem | [22_ML_SUPERVISADO.md](22_ML_SUPERVISADO.md) |
| 08 | ML No Supervisado | Unsupervised Algorithms in ML | 2 sem | [23_ML_NO_SUPERVISADO.md](23_ML_NO_SUPERVISADO.md) |
| 09 | Deep Learning | Introduction to Deep Learning | 3 sem | [24_INTRO_DEEP_LEARNING.md](24_INTRO_DEEP_LEARNING.md) |

**Entregable:** Regresión, Naive Bayes, K-Means, MLP con backprop desde cero.

---

### FASE 4: Proyecto Final (Semanas 23-26)
*Integración de todo lo aprendido en un pipeline ejecutable*

| # | Módulo | Descripción | Tiempo | Archivo |
|---|--------|-------------|--------|---------|
| 10 | Proyecto Integrador | Pipeline ML completo | 4 sem | [12_PROYECTO_INTEGRADOR.md](12_PROYECTO_INTEGRADOR.md) |

**Entregable:** Sistema que clasifica texto usando NB, KMeans, MLP y genera texto con Markov.

---

## 🔨 Proyecto Final: ML Pipeline

```
ml-pathway-project/
├── src/
│   ├── __init__.py
│   ├── vector.py              # Álgebra lineal (Módulo 03)
│   ├── probability.py         # Bayes, distribuciones (Módulo 04)
│   ├── statistics.py          # MLE, intervalos (Módulo 05)
│   ├── markov.py              # Cadenas Markov, MCMC (Módulo 06)
│   ├── naive_bayes.py         # Clasificador NB (Módulo 07)
│   ├── linear_regression.py   # Regresión (Módulo 07)
│   ├── kmeans.py              # Clustering (Módulo 08)
│   ├── pca.py                 # Reducción dim (Módulo 08)
│   ├── neural_network.py      # MLP + backprop (Módulo 09)
│   ├── activations.py         # Funciones activación (Módulo 09)
│   └── pipeline.py            # Integración (Módulo 10)
├── tests/
│   └── test_*.py              # Tests para cada módulo
├── data/
│   └── sample_texts/          # Datos de prueba
├── notebooks/
│   └── demo.ipynb             # Demo interactivo
├── README.md
└── requirements.txt           # Solo pytest (sin numpy/sklearn)
```

---

## ⏱️ Tiempo Total

| Fase | Semanas | Horas (~36h/sem) |
|------|---------|------------------|
| Fundamentos (01-03) | 6 | ~216h |
| Probabilidad (04-06) | 8 | ~288h |
| Machine Learning (07-09) | 8 | ~288h |
| Proyecto Final (10) | 4 | ~144h |
| **TOTAL** | **26** | **~936h** |

**Duración:** 6 meses con 6h/día (L-S)

---

## 📦 Material Complementario (Opcional)

| Documento | Descripción | Obligatorio |
|-----------|-------------|-------------|
| [EJERCICIOS.md](EJERCICIOS.md) | Práctica adicional por módulo | Recomendado |
| [GLOSARIO.md](GLOSARIO.md) | Definiciones técnicas | Consulta |
| [SIMULACRO_ENTREVISTA.md](SIMULACRO_ENTREVISTA.md) | Preguntas tipo Pathway | Recomendado |
| [RECURSOS.md](RECURSOS.md) | Cursos y libros externos | Consulta |

### DSA Avanzado (Solo si necesitas para entrevistas técnicas)

Estos módulos **NO son necesarios para el Pathway**, pero pueden ser útiles para entrevistas de trabajo:

| Documento | Tema |
|-----------|------|
| [04_ARRAYS_STRINGS.md](04_ARRAYS_STRINGS.md) | Arrays y manipulación |
| [05_HASHMAPS_SETS.md](05_HASHMAPS_SETS.md) | Hash tables |
| [07_RECURSION.md](07_RECURSION.md) | Recursión |
| [08_SORTING.md](08_SORTING.md) | Ordenamiento |
| [14_TREES.md](14_TREES.md) | Árboles y BST |
| [15_GRAPHS.md](15_GRAPHS.md) | Grafos, BFS, DFS |
| [16_DYNAMIC_PROGRAMMING.md](16_DYNAMIC_PROGRAMMING.md) | DP |

---

## 🚀 Comenzar

**[→ Módulo 01: Python Profesional](01_PYTHON_PROFESIONAL.md)**

---

## 📌 Restricciones del Proyecto

- ✅ **Python puro** - Sin numpy, pandas, sklearn, tensorflow
- ✅ **100% local** - Todo se ejecuta en tu máquina
- ✅ **Desde cero** - Cada algoritmo implementado manualmente
- ✅ **Enfocado** - Solo lo necesario para el Pathway

---

> 💡 **Filosofía:** Si puedes implementar Naive Bayes, K-Means, MLP y Markov desde cero, estás listo para los cursos del Pathway. DSA avanzado es útil para entrevistas, pero **no es el objetivo de esta guía**.
