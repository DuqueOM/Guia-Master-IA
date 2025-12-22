# Módulo 08: Proyecto Capstone - NLP Disaster Analysis Pipeline

> **Semanas:** 21-24 | **Fase:** Integración 🎯 | **Nivel:** Maestría

---

## 🎯 Descripción del Proyecto

### "Natural Language Processing with Disaster Tweets"

**Dataset:** [Kaggle - Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)

Este proyecto final integra **los 3 cursos del MS-AI Pathway** en un pipeline completo:
- **CSCA 5622 (Supervisado):** Regresión Logística, Naive Bayes, métricas de evaluación
- **CSCA 5632 (No Supervisado):** Word Embeddings como representaciones latentes
- **CSCA 5642 (Deep Learning):** LSTMs bidireccionales, Transfer Learning con BERT

### ¿Por qué este proyecto?

| Aspecto | MNIST (Anterior) | Disaster Tweets (Nuevo) |
|---------|------------------|-------------------------|
| Datos | Limpios, estructurados | Sucios, texto no estructurado |
| Preprocesamiento | Mínimo | Regex, tokenización, lematización |
| Complejidad | Introductorio | Nivel de maestría |
| Evaluación | Accuracy simple | F1-Score, datos desbalanceados |
| Comunicación | Opcional | REPORT.md obligatorio |

---

## 📁 Estructura del Proyecto

```
M08_Proyecto_Integrador/
├── README.md                              # Este archivo
├── data/
│   ├── raw/                              # Datos originales de Kaggle
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/                        # Datos limpios
│   │   └── train_clean.csv
│   └── README_data.md                    # Instrucciones de descarga
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb        # Semana 21
│   ├── 02_Baseline_Models.ipynb          # Semana 22
│   ├── 03_Deep_Learning_LSTM.ipynb       # Semana 23
│   └── 04_Transfer_Learning_BERT.ipynb   # Semana 24
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                  # Funciones de limpieza
│   ├── features.py                       # TF-IDF, embeddings
│   ├── models.py                         # Clases de modelos
│   └── evaluation.py                     # Métricas y visualización
├── models/
│   ├── baseline_logreg.pkl
│   ├── lstm_best.h5
│   └── bert_finetuned/
├── reports/
│   ├── REPORT.md                         # Reporte final académico
│   ├── figures/                          # Gráficas para el reporte
│   └── MODEL_COMPARISON.md               # Benchmarks de modelos
├── Archive_MNIST/                        # Proyecto MNIST archivado
│   └── README.md                         # Referencia como tarea introductoria
└── requirements_capstone.txt             # Dependencias específicas
```

---

## 📓 Notebooks del Proyecto

### Notebook 1: EDA & Preprocessing (Semana 21)

**Archivo:** `notebooks/01_EDA_Preprocessing.ipynb`

| Tarea | Técnica | Librería |
|-------|---------|----------|
| Carga y exploración | `df.info()`, `df.describe()` | Pandas |
| Limpieza de texto | Regex para URLs, HTML tags, menciones | `re` |
| Tokenización | Word tokenization | NLTK / SpaCy |
| Lematización | Reducir a raíz | NLTK WordNetLemmatizer |
| Visualización | WordClouds comparativas | `wordcloud`, Matplotlib |
| Análisis de desbalance | Proporción de clases | Pandas |

**Entregables:**
- [ ] `train_clean.csv` generado
- [ ] WordCloud de tweets reales vs falsos
- [ ] Análisis estadístico de longitud de tweets

---

### Notebook 2: Baseline Models (Semana 22)

**Archivo:** `notebooks/02_Baseline_Models.ipynb`

| Modelo | Vectorización | Hiperparámetros |
|--------|---------------|-----------------|
| Logistic Regression | TF-IDF | C, max_iter |
| Multinomial Naive Bayes | Bag of Words | alpha |
| SVM | TF-IDF | kernel, C |

**Métricas de Evaluación (CRÍTICO):**

```python
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ¿Por qué NO usar accuracy en datos desbalanceados?
# Si 70% son clase 0, un modelo que prediga siempre 0 tiene 70% accuracy
# pero es completamente inútil. F1-Score balancea Precision y Recall.

print(classification_report(y_true, y_pred))
print(f"F1-Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
```

**Entregables:**
- [ ] Pipeline de vectorización + modelo
- [ ] Matriz de confusión visualizada
- [ ] Comparación F1-Score de baselines
- [ ] Análisis de errores (falsos positivos/negativos)

---

### Notebook 3: Deep Learning - LSTM (Semana 23)

**Archivo:** `notebooks/03_Deep_Learning_LSTM.ipynb`

**Arquitectura:**

```python
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model

# Arquitectura Bidirectional LSTM
inputs = Input(shape=(max_length,))
x = Embedding(vocab_size, embedding_dim)(inputs)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Bidirectional(LSTM(32))(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
```

**Embeddings:**
- Opción A: Entrenar desde cero
- Opción B: Usar GloVe preentrenados (recomendado)

**Regularización (CRÍTICO para calificación):**
- `Dropout` entre capas
- `EarlyStopping` con `patience=5`
- `ModelCheckpoint` para guardar mejor modelo

**Entregables:**
- [ ] LSTM bidireccional funcionando
- [ ] Curvas de learning (loss, accuracy)
- [ ] Comparación con/sin GloVe embeddings
- [ ] Análisis de overfitting

---

### Notebook 4: Transfer Learning & Reporte (Semana 24)

**Archivo:** `notebooks/04_Transfer_Learning_BERT.ipynb`

**Bonus Track - BERT con HuggingFace:**

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Cargar modelo preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenizar datos
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='tf')

# Fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

**Entregables:**
- [ ] BERT fine-tuned en disaster tweets
- [ ] Comparación BERT vs LSTM vs Baselines
- [ ] **REPORT.md** completo

---

## 📄 REPORT.md - Estructura Académica

El reporte final debe seguir esta estructura:

```markdown
# NLP Disaster Tweet Classification - Final Report

## Abstract (150 palabras)
Resumen ejecutivo del proyecto, metodología y resultados principales.

## 1. Introduction
- Contexto del problema
- Relevancia en aplicaciones reales (detección de emergencias)
- Objetivos del proyecto

## 2. Dataset Description
- Estadísticas descriptivas
- Análisis de desbalance de clases
- Ejemplos de tweets difíciles

## 3. Methodology
### 3.1 Preprocessing Pipeline
### 3.2 Feature Engineering
### 3.3 Model Architectures

## 4. Experiments
### 4.1 Baseline Models
### 4.2 LSTM Results
### 4.3 BERT Results

## 5. Results & Discussion
- Tabla comparativa de todos los modelos
- Análisis de errores
- Limitaciones del estudio

## 6. Conclusion
- Resumen de hallazgos
- Modelo recomendado para producción
- Trabajo futuro

## References
- Papers citados
- Documentación de librerías
```

---

## ⚡ Inicio Rápido

```bash
# 1. Descargar datos de Kaggle
# Ir a https://www.kaggle.com/c/nlp-getting-started/data
# Descargar train.csv y test.csv → data/raw/

# 2. Instalar dependencias adicionales
pip install -r requirements_capstone.txt

# 3. Ejecutar notebooks en orden
jupyter notebook notebooks/01_EDA_Preprocessing.ipynb
jupyter notebook notebooks/02_Baseline_Models.ipynb
jupyter notebook notebooks/03_Deep_Learning_LSTM.ipynb
jupyter notebook notebooks/04_Transfer_Learning_BERT.ipynb
```

---

## 📦 Dependencias Específicas

```txt
# requirements_capstone.txt
nltk>=3.8
spacy>=3.5
wordcloud>=1.9
transformers>=4.30
datasets>=2.12
scikit-learn>=1.2
tensorflow>=2.12
```

---

## ✅ Checklist Final de Entrega

### Semana 21
- [ ] Dataset descargado y explorado
- [ ] Pipeline de preprocesamiento completo
- [ ] WordClouds generados

### Semana 22
- [ ] Baselines entrenados (LogReg, NB)
- [ ] F1-Score documentado
- [ ] Matriz de confusión analizada

### Semana 23
- [ ] LSTM bidireccional entrenado
- [ ] Embeddings GloVe integrados
- [ ] Regularización implementada (Dropout, EarlyStopping)

### Semana 24
- [ ] BERT fine-tuned (bonus)
- [ ] MODEL_COMPARISON.md completo
- [ ] **REPORT.md entregado**
- [ ] Código limpio y documentado

---

## 🏆 Criterios de Éxito

| Criterio | Mínimo | Excelente |
|----------|--------|-----------|
| F1-Score Baseline | > 0.70 | > 0.78 |
| F1-Score LSTM | > 0.75 | > 0.80 |
| F1-Score BERT | > 0.80 | > 0.85 |
| REPORT.md | Completo | Publicable |
| Código | Funcional | Modular y testeado |

---

## 📚 Recursos

### Dataset
- [Kaggle Competition](https://www.kaggle.com/c/nlp-getting-started)
- [Dataset Paper](https://arxiv.org/abs/1907.11692)

### NLP
- [NLTK Book](https://www.nltk.org/book/)
- [SpaCy Course](https://course.spacy.io/)

### Deep Learning NLP
- [CS224n Stanford](https://web.stanford.edu/class/cs224n/)
- [HuggingFace Course](https://huggingface.co/course)

### GloVe Embeddings
- [Download GloVe](https://nlp.stanford.edu/projects/glove/)
- Usar `glove.6B.100d.txt` (100 dimensiones)

---

## 🔗 Navegación

| Anterior | Índice | Final |
|----------|--------|-------|
| [M07 Deep Learning](../M07_Deep_Learning/) | [README](../README.md) | 🎓 Completado | |
