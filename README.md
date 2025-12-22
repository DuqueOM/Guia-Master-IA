# 🎓 Guía Master IA - Ecosistema Educativo Híbrido

> **Preparación de 6 meses para el MS in AI de CU Boulder**
> Metodología: **Teoría (.md) → Práctica (.ipynb/.py) → Visualización (Streamlit/Manim)**

---

## 🎯 Objetivo

Dominio absoluto de los **3 cursos clave del MS-AI Pathway**:

| Track | Curso (Código) | Módulo | Semanas |
|-------|----------------|--------|---------|
| **Supervised Learning** | CSCA 5622 | M05 | 9-11 |
| **Unsupervised Learning** | CSCA 5632 | M06 | 12-15 |
| **Deep Learning** | CSCA 5642 | M07 | 16-20 |

---

## 🗺️ Estructura del Ecosistema (24 Semanas)

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
