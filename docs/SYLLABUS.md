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

### Módulo 03: Álgebra Lineal para ML

| Contenido | Entregable |
|-----------|------------|
| Vectores y matrices en Python puro | Representaciones en `list` |
| Suma, producto punto y norma | `vector.py` con operaciones básicas |
| Producto matriz-vector | `matrix_vector_mult()` |
| Intuición geométrica | Explicar similitud como ángulo |

**Mini-proyecto:** Módulo `vector.py` y `matrix.py` usados en los módulos de ML.

---

### Módulo 04: Fundamentos de Probabilidad ⭐

| Contenido | Entregable |
|-----------|------------|
| Espacio muestral y eventos | Funciones para experimentos simples |
| Probabilidad condicional y Bayes | Implementar Teorema de Bayes |
| Variables aleatorias y distribuciones | Simulaciones de Bernoulli, Binomial, Normal |
| Esperanza y varianza | Funciones `expected_value()` y `variance()` |

**Mini-proyecto:** Simulador de lanzamiento de monedas/dados con comparación teórica vs empírica.

---

### Módulo 05: Estadística Inferencial ⭐

| Contenido | Entregable |
|-----------|------------|
| Estimadores puntuales | Funciones para media, varianza, proporciones |
| MLE y MAP (introducción) | Implementar estimación de parámetros sencillos |
| Intervalos de confianza | Cálculo para media y proporción |
| Tests de hipótesis básicos | Z-test / t-test simplificados |

**Mini-proyecto:** Pequeño experimento (por ejemplo, conversión en A/B test) con estimación de intervalo y decisión estadística.

---

### Módulo 06: Cadenas de Markov y Monte Carlo ⭐

| Contenido | Entregable |
|-----------|------------|
| Cadenas de Markov de tiempo discreto | Matriz de transición e iteración |
| Distribución estacionaria | Cálculo numérico vía potencia |
| PageRank simplificado | Implementación desde cero |
| Monte Carlo y MCMC | Simulación y muestreo básico |

**Mini-proyecto:** Implementación de un PageRank simple y un ejemplo de Monte Carlo para integrar funciones.

---

### Módulo 07: Machine Learning Supervisado ⭐

| Contenido | Entregable |
|-----------|------------|
| Pipeline de ML (train/val/test) | Script de entrenamiento básico |
| Regresión lineal | Implementación con Gradient Descent |
| Regresión logística | Clasificador binario desde cero |
| Árboles de decisión (visión simplificada) | Árbol pequeño implementado a mano |
| Métricas de evaluación | Accuracy, precision, recall, F1 |

**Mini-proyecto:** Clasificador binario desde cero (por ejemplo, spam/no spam) con regresión logística.

---

### Módulo 08: Machine Learning No Supervisado ⭐

| Contenido | Entregable |
|-----------|------------|
| K-Means clustering | Implementación de K-Means |
| PCA | Reducción de dimensionalidad con autovectores |
| Detección de anomalías (simple) | Umbrales sobre distancia al centroide |

**Mini-proyecto:** Segmentación de clientes o agrupamiento de textos usando K-Means + PCA.

---

### Módulo 09: Introducción al Deep Learning ⭐

| Contenido | Entregable |
|-----------|------------|
| Perceptrón y neurona artificial | Implementación de una neurona |
| MLP (Multilayer Perceptron) | Red de 2–3 capas con backpropagation |
| Funciones de activación | ReLU, Sigmoid, Tanh |
| Entrenamiento con Gradient Descent | Entrenar en un dataset pequeño |

**Mini-proyecto:** MLP sencillo para clasificar puntos 2D o dígitos muy simples.

---

### Módulo 10: Proyecto Final

| Contenido | Entregable |
|-----------|------------|
| Diseño del pipeline ML completo | `pipeline.py` integrando todos los módulos |
| Integración de probabilidad y estadística | Comparación de modelos con métricas e intervalos |
| Comparación de modelos | Tabla de resultados y análisis |
| Documentación y presentación | README y defensa técnica |

**Entregable final:**
1. Pipeline ML completo funcional (scripts + módulos `src/`).
2. Informe comparando al menos 2–3 modelos.
3. README en inglés explicando decisiones.

---

## 📊 Rúbrica General (100 puntos)

| Dimensión | Puntos | Criterio |
|-----------|--------|----------|
| **Funcionalidad** | 30 | El pipeline entrena, evalúa y compara modelos correctamente |
| **Código limpio** | 20 | PEP8, type hints, docstrings |
| **Tests** | 20 | Cobertura razonable de funciones críticas |
| **Análisis estadístico** | 20 | Uso correcto de métricas, intervalos e interpretación |
| **Documentación** | 10 | README claro, en inglés |

### Niveles

| Puntuación | Nivel |
|------------|-------|
| 90-100 | Listo para Pathway y entrevistas técnicas de ML |
| 75-89 | Buen nivel, reforzar áreas concretas |
| 60-74 | Necesita más práctica antes del Pathway |
| <60 | Revisar módulos fundamentales |

---

## 🎯 Preparación para Pathway - CURSOS EXACTOS

El Pathway tiene **2 líneas con 6 cursos específicos**:

### LÍNEA 1: Machine Learning (3 créditos)

| Curso del Pathway | Módulo Preparación | Temas Cubiertos |
|-------------------|-------------------|-----------------|
| **Introduction to ML: Supervised Learning** | 07 | Regresión, clasificación, métricas |
| **Unsupervised Algorithms in ML** | 08 | K-Means, clustering, PCA, anomalías |
| **Introduction to Deep Learning** | 09 | Perceptrón, MLP, backprop, conceptos CNN/RNN |

### LÍNEA 2: Probability & Statistics (3 créditos)

| Curso del Pathway | Módulo Preparación | Temas Cubiertos |
|-------------------|-------------------|-----------------|
| **Probability Fundamentals for DS and AI** | 04 | Bayes, distribuciones, esperanza, varianza |
| **Discrete-Time Markov Chains and Monte Carlo Methods** | 06 | Cadenas de Markov, PageRank, MCMC |
| **Statistical Estimation for DS and AI** | 05 | MLE, MAP, intervalos, hipótesis |

### Cobertura de esta Guía

| Componente del Pathway | ¿Cubierto? | Evidencia |
|------------------------|------------|-----------|
| Naive Bayes | ✅ | Módulos 04 + 07 |
| Regresión Lineal/Logística | ✅ | Módulo 07 |
| Árboles de Decisión (básico) | ✅ | Módulo 07 |
| K-Means Clustering | ✅ | Módulo 08 |
| PCA | ✅ | Módulo 08 |
| Redes Neuronales | ✅ | Módulo 09 |
| Backpropagation | ✅ | Módulo 09 |
| Teorema de Bayes | ✅ | Módulo 04 |
| Cadenas de Markov | ✅ | Módulo 06 |
| MLE/MAP | ✅ | Módulo 05 |
| Intervalos de Confianza | ✅ | Módulo 05 |

---

## 📅 Cronograma Sugerido

Ver [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md) para el cronograma resumido de 26 semanas.

---

## ✅ Checklist de Finalización del Programa

### Fundamentos (Módulos 01-03)
- [ ] Python profesional con type hints
- [ ] OOP y diseño SOLID básico
- [ ] Álgebra lineal implementada en Python puro

### Línea 2: Probabilidad (Módulos 04-06)
- [ ] Teorema de Bayes explicado y aplicado
- [ ] MLE y MAP implementados en ejemplos sencillos
- [ ] Cadenas de Markov y MCMC entendidos y simulados
- [ ] Intervalos de confianza calculados

### Línea 1: Machine Learning (Módulos 07-09)
- [ ] Regresión lineal/logística desde cero
- [ ] K-Means y PCA implementados
- [ ] Red neuronal con backpropagation
- [ ] Métricas de evaluación dominadas

### Proyecto Integrador (Módulo 10)
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
