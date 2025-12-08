# 📋 Syllabus - MS in AI Pathway

> **10 Módulos Obligatorios | 6 Meses | 100% Enfocado en el Pathway**

---

## 🎯 Objetivo Único

Prepararte para aprobar las **6 materias** del Performance-Based Admission Pathway de CU Boulder.

---

## 📊 Estructura: 10 Módulos Obligatorios

| Módulo | Nombre | Semanas | Fase | Curso del Pathway |
|--------|--------|---------|------|-------------------|
| **01** | Python Profesional | 2 | Fundamentos | - |
| **02** | OOP desde Cero | 2 | Fundamentos | - |
| **03** | Álgebra Lineal para ML | 2 | Fundamentos | - |
| **04** | Fundamentos de Probabilidad | 3 | ⭐ Pathway L2 | Probability Fundamentals |
| **05** | Estadística Inferencial | 3 | ⭐ Pathway L2 | Statistical Estimation |
| **06** | Markov y Monte Carlo | 2 | ⭐ Pathway L2 | Markov Chains & Monte Carlo |
| **07** | ML Supervisado | 3 | ⭐ Pathway L1 | Intro to ML: Supervised |
| **08** | ML No Supervisado | 2 | ⭐ Pathway L1 | Unsupervised Algorithms |
| **09** | Deep Learning | 3 | ⭐ Pathway L1 | Intro to Deep Learning |
| **10** | Proyecto Final | 4 | Integración | - |

**Total: 26 semanas = 6 meses** (6h/día, L-S)

---

## 📚 Mapeo Módulos → Código → Cursos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FASE 1: FUNDAMENTOS (Semanas 1-6)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Módulos: 01, 02, 03                                                         │
│ Código:  src/vector.py, src/matrix.py                                       │
│ Entregable: Clases Vector y Matrix con operaciones desde cero               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FASE 2: PROBABILIDAD Y ESTADÍSTICA ⭐ PATHWAY LÍNEA 2 (Semanas 7-14)        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Módulos: 04, 05, 06                                                         │
│ Código:  src/probability.py, src/statistics.py, src/markov.py               │
│ Cursos:  Probability, Statistical Estimation, Markov Chains                 │
│ Entregable: Bayes, MLE, MCMC, PageRank desde cero                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FASE 3: MACHINE LEARNING ⭐ PATHWAY LÍNEA 1 (Semanas 15-22)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Módulos: 07, 08, 09                                                         │
│ Código:  src/naive_bayes.py, src/kmeans.py, src/neural_network.py           │
│ Cursos:  Supervised Learning, Unsupervised, Deep Learning                   │
│ Entregable: Regresión, NB, K-Means, MLP con backprop desde cero             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FASE 4: PROYECTO FINAL (Semanas 23-26)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Módulo: 10                                                                  │
│ Código:  src/pipeline.py (integra todo)                                     │
│ Entregable: Pipeline ML completo + comparación estadística de modelos       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📖 Detalle por Módulo

### Módulo 01: Python Profesional

| Contenido | Entregable |
|-----------|------------|
| Type hints y anotaciones | Código tipado con `mypy` pasando |
| Funciones puras vs impuras | Funciones sin side effects |
| PEP8 y estilo consistente | Código que pasa `ruff` o `flake8` |
| Docstrings y documentación | Cada función documentada |

**Mini-proyecto:** Función `clean_text(text: str) -> str` tipada y documentada.

**Validación:** `mypy src/ && ruff check src/`

---

### Módulo 02: OOP desde Cero

| Contenido | Entregable |
|-----------|------------|
| Clases y objetos | Clase `Document` con atributos |
| `__init__`, `__repr__`, `__str__` | Métodos mágicos implementados |
| Encapsulamiento | Properties y validación |
| Composición vs Herencia | Clase `Corpus` que contiene `Document`s |
| Principios SOLID básicos | Single Responsibility aplicado |

**Mini-proyecto:** Clases `Document` y `Corpus` funcionales.

**Validación:** `python -m pytest tests/test_document.py -v`

---

### Módulo 03: Lógica y Matemáticas Discretas

| Contenido | Entregable |
|-----------|------------|
| Teoría de conjuntos | Uso correcto de `set` en Python |
| Lógica proposicional | Expresiones booleanas complejas |
| Notación Big O (introducción) | Explicar O(1), O(n), O(n²) |
| Demostraciones simples | Documentar "por qué funciona" |

**Mini-proyecto:** Lista de stop words como `set` con análisis de complejidad.

**Validación:** Documento explicando complejidad de operaciones `in` en `list` vs `set`.

---

### Módulo 04: Arrays, Strings y Memoria

| Contenido | Entregable |
|-----------|------------|
| Listas en Python (bajo nivel) | Entender slicing y copia |
| Manipulación de strings | Tokenización básica |
| Complejidad de operaciones | Tabla de O() para list |
| Inmutabilidad vs mutabilidad | Evitar bugs de referencia |

**Mini-proyecto:** Tokenizador básico que separa texto en palabras.

**Validación:** `python -m pytest tests/test_tokenizer.py -v`

---

### Módulo 05: Hash Maps y Sets

| Contenido | Entregable |
|-----------|------------|
| Cómo funciona un diccionario | Entender hashing |
| Colisiones y resolución | Saber que existen, no implementar |
| Complejidad O(1) amortizada | Explicar cuándo y por qué |
| Sets para búsqueda rápida | Stop words como `frozenset` |

**Mini-proyecto:** Diccionario de frecuencia de palabras.

**Validación:** Benchmark `list` vs `set` para búsqueda (script incluido).

---

### Módulo 06: Índice Invertido

| Contenido | Entregable |
|-----------|------------|
| Qué es un índice invertido | Diagrama y explicación |
| Estructura `{palabra: [doc_ids]}` | Clase `InvertedIndex` |
| Agregar documentos al índice | Método `add_document()` |
| Buscar documentos por palabra | Método `search(query)` |

**Mini-proyecto:** `InvertedIndex` que indexa y busca en corpus de prueba.

**Validación:** `python -m pytest tests/test_index.py -v`

**Análisis requerido:** ¿Cuál es la complejidad de `add_document()`? ¿Y de `search()`?

---

### Módulo 07: Recursión y Divide & Conquer

| Contenido | Entregable |
|-----------|------------|
| Pensamiento recursivo | Funciones recursivas simples |
| Caso base y caso recursivo | Identificar en ejemplos |
| Call stack y límites | Entender `RecursionError` |
| Divide & Conquer pattern | Factorial, Fibonacci, suma de lista |

**Mini-proyecto:** `factorial()`, `fibonacci()`, `sum_list()` recursivos.

**Validación:** Tests que verifican casos base y casos grandes.

---

### Módulo 08: Algoritmos de Ordenamiento

| Contenido | Entregable |
|-----------|------------|
| QuickSort desde cero | Implementación funcional |
| Pivot selection | Random pivot para evitar O(n²) |
| MergeSort (opcional) | Implementación alternativa |
| Análisis de complejidad | O(n log n) promedio, O(n²) peor |

**Mini-proyecto:** `quicksort()` y `mergesort()` en `sorting.py`.

**Validación:** `python -m pytest tests/test_sorting.py -v`

**Análisis requerido:** Documento explicando cuándo QuickSort es O(n²).

---

### Módulo 09: Búsqueda Binaria

| Contenido | Entregable |
|-----------|------------|
| Binary Search clásica | Implementación sin errores |
| Off-by-one errors | Cómo evitarlos sistemáticamente |
| Variantes | Buscar primer/último elemento |
| Cuándo aplicar | Lista ordenada, O(log n) |

**Mini-proyecto:** `binary_search()` con variantes en `searching.py`.

**Validación:** `python -m pytest tests/test_searching.py -v`

---

### Módulo 10: Álgebra Lineal sin NumPy

| Contenido | Entregable |
|-----------|------------|
| Vectores como listas | Representación básica |
| Suma de vectores | `add_vectors(v1, v2)` |
| Producto punto | `dot_product(v1, v2)` |
| Norma de un vector | `magnitude(v)` |
| Matrices como listas de listas | Representación 2D |

**Mini-proyecto:** Módulo `linear_algebra.py` con operaciones básicas.

**Validación:** Tests que verifican matemáticamente cada operación.

---

### Módulo 11: TF-IDF y Similitud de Coseno

| Contenido | Entregable |
|-----------|------------|
| Term Frequency (TF) | Función `compute_tf()` |
| Inverse Document Frequency (IDF) | Función `compute_idf()` |
| TF-IDF combinado | Función `compute_tfidf()` |
| Similitud de coseno | Función `cosine_similarity()` |
| Vectorización de documentos | Cada doc como vector TF-IDF |

**Mini-proyecto:** Sistema de ranking por relevancia.

**Validación:** Tests + comparación manual con resultados conocidos.

---

### Módulo 12: Proyecto Integrador

| Contenido | Entregable |
|-----------|------------|
| Ensamblaje de componentes | `SearchEngine` que usa todo |
| API de búsqueda | Método `search(query, top_k)` |
| Análisis Big O completo | `COMPLEXITY_ANALYSIS.md` |
| README profesional | Documentación de uso |
| Tests de integración | `test_engine.py` |

**Entregable final:**
1. Motor de búsqueda funcional
2. Análisis de complejidad de cada operación
3. README en inglés
4. Suite de tests con >80% coverage

**Validación:** Demo en vivo + defensa del análisis Big O.

---

## 📊 Rúbrica General (100 puntos)

| Dimensión | Puntos | Criterio |
|-----------|--------|----------|
| **Funcionalidad** | 30 | El motor busca y rankea correctamente |
| **Código limpio** | 20 | PEP8, type hints, docstrings |
| **Tests** | 20 | Cobertura >80%, casos edge |
| **Análisis Big O** | 20 | Documento completo y correcto |
| **Documentación** | 10 | README claro, en inglés |

### Niveles

| Puntuación | Nivel |
|------------|-------|
| 90-100 | Listo para Pathway + entrevistas técnicas |
| 75-89 | Buen nivel, reforzar áreas débiles |
| 60-74 | Necesita más práctica antes de Pathway |
| <60 | Revisar módulos fundamentales |

---

## 🎯 Preparación para Pathway - CURSOS EXACTOS

El Pathway tiene **2 líneas con 6 cursos específicos**:

### LÍNEA 1: Machine Learning (3 créditos)

| Curso del Pathway | Módulo Preparación | Temas Cubiertos |
|-------------------|-------------------|-----------------|
| **Introduction to ML: Supervised Learning** | 22 | Regresión, clasificación, árboles, SVM, evaluación |
| **Unsupervised Algorithms in ML** | 23 | K-Means, clustering jerárquico, PCA, anomalías |
| **Introduction to Deep Learning** | 24 | Perceptrón, MLP, backprop, CNN/RNN conceptos |

### LÍNEA 2: Probability & Statistics (3 créditos)

| Curso del Pathway | Módulo Preparación | Temas Cubiertos |
|-------------------|-------------------|-----------------|
| **Probability Theory: Foundation** | 19 | Bayes, distribuciones, esperanza, varianza |
| **Discrete-Time Markov Chains** | 21 | Cadenas de Markov, PageRank, MCMC |
| **Statistical Inference** | 20 | MLE, MAP, intervalos, hipótesis |

### Cobertura de esta Guía

| Componente del Pathway | ¿Cubierto? | Evidencia |
|------------------------|------------|-----------|
| Naive Bayes | ✅ | Módulo 19 + 22 |
| Regresión Lineal/Logística | ✅ | Módulo 22 |
| Árboles de Decisión | ✅ | Módulo 22 |
| K-Means Clustering | ✅ | Módulo 23 |
| PCA | ✅ | Módulo 23 |
| Redes Neuronales | ✅ | Módulo 24 |
| Backpropagation | ✅ | Módulo 24 |
| Teorema de Bayes | ✅ | Módulo 19 |
| Cadenas de Markov | ✅ | Módulo 21 |
| MLE/MAP | ✅ | Módulo 20 |
| Intervalos de Confianza | ✅ | Módulo 20 |

---

## 📅 Cronograma Sugerido

Ver [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md) para el cronograma día a día.

---

## ✅ Checklist de Finalización del Programa

### Prerrequisitos (Módulos 01-18)
- [ ] Python profesional con type hints
- [ ] OOP y diseño SOLID
- [ ] Estructuras de datos implementadas
- [ ] Algoritmos clásicos dominados

### Línea 2: Probabilidad (Módulos 19-21)
- [ ] Teorema de Bayes explicado y aplicado
- [ ] MLE y MAP implementados
- [ ] Cadenas de Markov y MCMC entendidos
- [ ] Intervalos de confianza calculados

### Línea 1: Machine Learning (Módulos 22-24)
- [ ] Regresión lineal/logística desde cero
- [ ] K-Means y PCA implementados
- [ ] Red neuronal con backpropagation
- [ ] Métricas de evaluación dominadas

### Proyecto Integrador (Módulo 12)
- [ ] Pipeline ML completo funcional
- [ ] Comparación estadística de modelos
- [ ] README en inglés
- [ ] Demo presentable

### Preparación Final
- [ ] Simulacro de entrevista completado (100+ preguntas)
- [ ] Capaz de explicar cada modelo en inglés
- [ ] Cursos del Pathway auditados en Coursera

---

> 💡 **Recuerda:** El objetivo es aprobar los 6 cursos del Pathway. Esta guía te prepara para todos ellos. ¡No uses sklearn hasta dominar las implementaciones desde cero!
