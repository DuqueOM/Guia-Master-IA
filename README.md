# 🎓 Guía Master IA - Preparación MS in AI CU Boulder

> **Programa de 6 meses (24 semanas) para dominar los fundamentos del MS in Artificial Intelligence**
>
> 🎯 **Objetivo**: Aprobar CSCA 5622, CSCA 5632 y CSCA 5642 con confianza
>
> 📚 **Metodología**: Teoría (.md) → Práctica (.py) → Visualización (Streamlit)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Study Time](https://img.shields.io/badge/Study%20Time-6%20Months-brightgreen.svg)]()
[![Pathway](https://img.shields.io/badge/Pathway-CU%20Boulder%20MS--AI-purple.svg)](https://www.colorado.edu/cs/academics/graduate-programs/professional-master-science-artificial-intelligence)

---

## 🚀 Cómo Empezar (Quick Start)

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/DuqueOM/Guia-Master-IA.git
cd Guia-Master-IA
```

### Paso 2: Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
```

### Paso 3: Instalar Dependencias

```bash
# Instalación básica (CPU)
pip install -r requirements.txt

# Con soporte GPU (NVIDIA CUDA)
pip install -r requirements.txt
pip install tensorflow[and-cuda]
```

### Paso 4: Verificar Instalación

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import sklearn; print(f'Scikit-learn {sklearn.__version__}')"
```

### Paso 5: Comenzar el Estudio

```bash
# Abrir JupyterLab
jupyter lab

# O ejecutar un laboratorio interactivo
streamlit run M02_Algebra_Lineal/Laboratorios_Interactivos/transformacion_lineal_app.py
```

---

## 📅 Plan de Estudio: 24 Semanas (6 Meses)

### 🗓️ Mes 1-2: Fundamentos Matemáticos (Semanas 1-8)

| Semana | Módulo | Tema | Horas/Sem | Entregable |
|--------|--------|------|-----------|------------|
| **1** | M01 | Python Científico: NumPy Avanzado | 10h | Quiz NumPy |
| **2** | M01 | Pandas: Manipulación de DataFrames | 10h | Mini-proyecto EDA |
| **3** | M02 | Vectores, Matrices, Operaciones Básicas | 12h | Ejercicios escritos |
| **4** | M02 | Eigenvalues, Eigenvectors, Diagonalización | 12h | Implementación from scratch |
| **5** | M02 | SVD y Aplicaciones (PCA preview) | 12h | Lab interactivo SVD |
| **6** | M03 | Derivadas, Gradientes, Regla de la Cadena | 10h | Derivación manual backprop |
| **7** | M03 | Optimización: Gradiente Descendente | 10h | Implementación GD |
| **8** | M04 | Probabilidad, Bayes, Distribuciones | 10h | Ejercicios MLE/MAP |

**🎯 Checkpoint Mes 2**: Simulacro de examen teórico (M01-M04)

---

### 🗓️ Mes 3: Aprendizaje Supervisado + Ética (Semanas 9-11)

| Semana | Módulo | Tema | Curso Alineado | Entregable |
|--------|--------|------|----------------|------------|
| **9** | M05 | Regresión Lineal/Logística from scratch | CSCA 5622 | Notebook validado |
| **10** | M05 | Árboles de Decisión, Random Forest, SVM | CSCA 5622 | Comparativa modelos |
| **11** | M05 | **Ética IA + XAI**: SHAP, LIME | CSCA 5622 | Reporte interpretabilidad |

**🎯 Checkpoint Mes 3**: Proyecto mini - Clasificación con explicabilidad

---

### 🗓️ Mes 4: Aprendizaje No Supervisado + Recomendación (Semanas 12-15)

| Semana | Módulo | Tema | Curso Alineado | Entregable |
|--------|--------|------|----------------|------------|
| **12** | M06 | K-Means, Clustering Jerárquico | CSCA 5632 | Implementación from scratch |
| **13** | M06 | PCA, Reducción de Dimensionalidad | CSCA 5632 | Visualización t-SNE |
| **14** | M06 | GMM, Algoritmo EM | CSCA 5632 | Derivación matemática |
| **15** | M06 | **Sistemas de Recomendación** (SVD, MovieLens) | CSCA 5632 | Recomendador funcional |

**🎯 Checkpoint Mes 4**: Proyecto - Sistema de recomendación end-to-end

---

### 🗓️ Mes 5: Deep Learning con Keras (Semanas 16-20)

| Semana | Módulo | Tema | Curso Alineado | Entregable |
|--------|--------|------|----------------|------------|
| **16** | M07 | Perceptrón, MLP from scratch | CSCA 5642 | Backprop manual |
| **17** | M07 | **Keras**: Sequential + Functional API | CSCA 5642 | Modelo híbrido |
| **18** | M07 | CNNs: Convoluciones, Pooling, Arquitecturas | CSCA 5642 | Clasificador CIFAR-10 |
| **19** | M07 | RNNs, LSTMs, GRUs | CSCA 5642 | Predicción secuencias |
| **20** | M07 | Regularización, Callbacks, Transfer Learning | CSCA 5642 | Fine-tuning VGG/ResNet |

**🎯 Checkpoint Mes 5**: Proyecto - CNN para clasificación de imágenes

---

### 🗓️ Mes 6: Proyecto Capstone NLP (Semanas 21-24)

| Semana | Módulo | Tema | Entregable |
|--------|--------|------|------------|
| **21** | M08 | EDA + Preprocessing (Disaster Tweets) | Notebook 01 limpio |
| **22** | M08 | Baseline Models (TF-IDF, LogReg, NB) | Notebook 02 + métricas |
| **23** | M08 | Deep Learning (BiLSTM + GloVe) | Notebook 03 + curvas |
| **24** | M08 | Transfer Learning (BERT) + **REPORTE FINAL** | Notebook 04 + REPORT.md |

**🎯 Entrega Final**: Proyecto completo evaluado con [RUBRIC.md](M08_Proyecto_Integrador/RUBRIC.md)

---

## 📊 Resumen Visual del Programa

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    PROGRAMA DE 24 SEMANAS                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  MES 1-2        │  MES 3          │  MES 4          │  MES 5    │ MES 6  ║
║  FUNDAMENTOS    │  SUPERVISADO    │  NO SUPERVISADO │  DEEP     │CAPSTONE║
║  ────────────── │  ────────────── │  ────────────── │  LEARNING │  ───── ║
║  M01: Python    │  M05: ML Core   │  M06: Clustering│  M07:     │  M08:  ║
║  M02: Álgebra   │  + Ética/XAI    │  + PCA + GMM    │  Keras    │  NLP   ║
║  M03: Cálculo   │                 │  + RecSys       │  CNN/RNN  │ Tweets ║
║  M04: Prob/Est  │                 │                 │           │        ║
║  ────────────── │  ────────────── │  ────────────── │  ──────── │  ───── ║
║  Semanas 1-8    │  Semanas 9-11   │  Semanas 12-15  │  16-20    │  21-24 ║
║                 │  CSCA 5622 ⭐   │  CSCA 5632 ⭐   │ CSCA 5642⭐│        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 🎯 Cursos Alineados (MS-AI Pathway)

| Curso | Código | Módulo | Descripción |
|-------|--------|--------|-------------|
| **Supervised Learning** | CSCA 5622 | M05 | Regresión, Clasificación, Árboles, SVM, XAI |
| **Unsupervised Learning** | CSCA 5632 | M06 | Clustering, PCA, GMM, Sistemas de Recomendación |
| **Deep Learning** | CSCA 5642 | M07 | MLPs, CNNs, RNNs, Transfer Learning con Keras |

---

## 🗺️ Estructura del Repositorio

```
Guia-Master-IA/
├── README.md                          # Este archivo
├── plan_de_estudio_6_meses.md         # Cronograma semana a semana
├── requirements.txt                   # Dependencias base
│
├── M01_Fundamentos_Python/            # Semanas 1-2
│   ├── Teoria/
│   ├── Notebooks/
│   └── Laboratorios_Interactivos/
│
├── M02_Algebra_Lineal/                # Semanas 3-5
│   ├── Teoria/
│   ├── Notebooks/
│   └── Laboratorios_Interactivos/
│
├── M03_Calculo_Optimizacion/          # Semanas 6-7
│   ├── Teoria/
│   ├── Notebooks/
│   └── Laboratorios_Interactivos/
│
├── M04_Probabilidad_Estadistica/      # Semana 8
│   ├── Teoria/
│   ├── Notebooks/
│   └── Laboratorios_Interactivos/
│
├── M05_Aprendizaje_Supervisado/       # Semanas 9-11 ⭐ CSCA 5622
│   ├── Teoria/
│   ├── Notebooks/                     # Incluye paridad Scikit-Learn
│   ├── Laboratorios_Interactivos/
│   └── 📌 NUEVO: Ética/XAI (SHAP, LIME)
│
├── M06_Aprendizaje_No_Supervisado/    # Semanas 12-15 ⭐ CSCA 5632
│   ├── Teoria/
│   ├── Notebooks/
│   ├── Laboratorios_Interactivos/
│   └── 📌 NUEVO: Sistemas de Recomendación (SVD/MovieLens)
│
├── M07_Deep_Learning/                 # Semanas 16-20 ⭐ CSCA 5642
│   ├── Teoria/
│   ├── Notebooks_Keras/               # RUTA PRINCIPAL (tf.keras)
│   ├── Advanced_Track_PyTorch/        # Opcional
│   └── Laboratorios_Interactivos/
│
├── M08_Proyecto_Integrador/           # Semanas 21-24 🎯 CAPSTONE
│   ├── 📌 NUEVO: NLP Disaster Tweets Pipeline
│   ├── notebooks/
│   │   ├── 01_EDA_Preprocessing.ipynb
│   │   ├── 02_Baseline_Models.ipynb
│   │   ├── 03_Deep_Learning_LSTM.ipynb
│   │   └── 04_Transfer_Learning_BERT.ipynb
│   ├── reports/
│   │   └── REPORT.md                  # Reporte académico
│   └── Archive_MNIST/                 # MNIST archivado como intro
│
├── Recursos_Adicionales/
│   ├── Glosarios/
│   ├── Planes_Estrategicos/
│   └── Cheat_Sheets/
│
└── Herramientas_Estudio/
    ├── DIARIO_ERRORES.md
    └── SIMULACRO_EXAMEN_TEORICO.md
```

---

## 📅 Cronograma de 24 Semanas

| Fase | Semanas | Módulo | Temas Clave |
|------|---------|--------|-------------|
| **FUNDAMENTOS** | 1-2 | M01 | Python Científico (NumPy, Pandas) |
| | 3-5 | M02 | Álgebra Lineal (SVD, Eigenvalues) |
| | 6-7 | M03 | Cálculo y Optimización (Gradientes) |
| | 8 | M04 | Probabilidad (Bayes, MLE) |
| **ML CORE** | 9-10 | M05 | Regresión, Árboles + **Paridad Sklearn** |
| | 11 | M05 | **Ética IA & XAI** (SHAP, LIME) 🆕 |
| | 12 | M06 | Clustering (K-Means) |
| | 13 | M06 | PCA / Reducción Dimensionalidad |
| | 14 | M06 | GMM / Algoritmo EM |
| | 15 | M06 | **Sistemas de Recomendación** (SVD) 🆕 |
| | 16 | M07 | Perceptrón, MLP desde cero |
| | 17 | M07 | **Keras APIs** (Sequential + Funcional) |
| | 18 | M07 | CNNs en Keras |
| | 19 | M07 | RNNs / LSTMs en Keras |
| | 20 | M07 | Regularización, Transfer Learning |
| **CAPSTONE** | 21 | M08 | EDA & Preprocessing (Disaster Tweets) |
| | 22 | M08 | Baseline Models (TF-IDF, LogReg, NB) |
| | 23 | M08 | Deep Learning (Bi-LSTM + GloVe) |
| | 24 | M08 | Transfer Learning (BERT) + **REPORT.md** |

---

## 🔄 Metodología de Aprendizaje

### El Ciclo de 3 Fases

```
┌─────────────────────────────────────────────────────────────────┐
│  FASE 1: TEORÍA (.md)                                           │
│  • Lee el contenido en Teoria/                                  │
│  • Estudia definiciones, fórmulas y analogías                   │
│  • Dibuja conceptos en papel (método Feynman)                   │
├─────────────────────────────────────────────────────────────────┤
│  FASE 2: PRÁCTICA (.ipynb / .py)                                │
│  • Ejecuta notebooks en Notebooks/                              │
│  • Implementa desde cero + valida con Scikit-Learn              │
│  • Valida con asserts y tests                                   │
├─────────────────────────────────────────────────────────────────┤
│  FASE 3: VISUALIZACIÓN (Streamlit/Manim)                        │
│  • Ejecuta apps en Laboratorios_Interactivos/                   │
│  • Manipula parámetros en tiempo real                           │
│  • Conecta intuición visual con matemáticas                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Inicio Rápido

### 1. Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/Guia-Master-IA.git
cd Guia-Master-IA

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar un laboratorio interactivo

```bash
# Ejemplo: Visualización de transformaciones lineales
streamlit run M02_Algebra_Lineal/Laboratorios_Interactivos/transformacion_lineal_app.py
```

### 3. Seguir el plan de estudio

Ver [plan_de_estudio_6_meses.md](plan_de_estudio_6_meses.md) para el cronograma detallado.

---

## 📊 Progreso por Módulo

| Módulo | Semanas | Curso Alineado | Descripción | Estado |
|--------|---------|----------------|-------------|--------|
| M01 | 1-2 | — | Python Científico (NumPy, Pandas) | 📚 |
| M02 | 3-5 | — | Álgebra Lineal (SVD, Eigenvalues) | 📚 |
| M03 | 6-7 | — | Cálculo y Optimización (Gradientes) | 📚 |
| M04 | 8 | — | Probabilidad (Bayes, MLE) | 📚 |
| M05 | 9-11 | **CSCA 5622** | Supervised + Ética/XAI | ⭐ |
| M06 | 12-15 | **CSCA 5632** | Unsupervised + Recomendadores | ⭐ |
| M07 | 16-20 | **CSCA 5642** | Deep Learning (Keras Principal) | ⭐ |
| M08 | 21-24 | — | **Capstone NLP: Disaster Tweets** | 🎯 |

---

## 🛠️ Stack Tecnológico

| Categoría | Herramienta | Uso |
|-----------|-------------|-----|
| **Core** | Python 3.10+ | Lenguaje base |
| **Científico** | NumPy, Pandas | Computación y datos |
| **ML Clásico** | Scikit-Learn | Paridad con implementaciones |
| **Deep Learning** | **Keras/TensorFlow** | Framework principal (alineado CSCA 5642) |
| **DL Avanzado** | PyTorch | Track opcional |
| **NLP** | NLTK, SpaCy, HuggingFace | Proyecto Capstone |
| **Visualización** | Matplotlib, Plotly, Streamlit | Gráficas e interactividad |
| **XAI** | SHAP, LIME | Interpretabilidad |

---

## 📖 Recursos Adicionales

- [Glosario Matemático](Recursos_Adicionales/Glosarios/GLOSARIO.md)
- [Planes Estratégicos](Recursos_Adicionales/Planes_Estrategicos/)
- [Herramientas de Estudio](Herramientas_Estudio/README.md)

---

## 🎯 Perfil de Salida

Al completar este programa podrás:

1. ✅ Implementar algoritmos de ML desde cero Y replicarlos con Scikit-Learn
2. ✅ Construir modelos de Deep Learning con la API Funcional de Keras
3. ✅ Explicar modelos de caja negra con SHAP/LIME
4. ✅ Construir sistemas de recomendación con factorización de matrices
5. ✅ Procesar texto no estructurado (NLP) con técnicas modernas
6. ✅ **Aprobar los 3 cursos del MS-AI Pathway (CSCA 5622, 5632, 5642)**
7. ✅ Producir reportes académicos de calidad publicable

---

## 💡 Cambios Clave vs. Versión Anterior

| Área | Antes | Ahora |
|------|-------|-------|
| **M05** | Solo from scratch | + Paridad Sklearn + Ética/XAI |
| **M06** | Solo clustering | + Sistemas de Recomendación (SVD) |
| **M07** | PyTorch | **Keras principal**, PyTorch opcional |
| **M08** | MNIST básico | **NLP Disaster Tweets** (nivel maestría) |

---

*Desarrollado como preparación para el MS in AI de la University of Colorado Boulder*
*Currículo alineado con CSCA 5622, CSCA 5632, CSCA 5642*
