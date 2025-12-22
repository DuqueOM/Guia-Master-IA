# 📅 Plan de Estudio: 6 Meses para el MS-AI Pathway

> **Duración Total:** 24 semanas (~864 horas)
> **Ritmo:** 6 horas/día, Lunes a Sábado
> **Filosofía:** From Scratch → Production Ready → Comunicación Científica

---

## 🗓️ Cronograma General

| Fase | Semanas | Módulos | Enfoque | Cursos Alineados |
|------|---------|---------|---------|------------------|
| **FUNDAMENTOS** | 1-8 | M01-M04 | Python + Matemáticas | — |
| **ML CORE** | 9-20 | M05-M07 | Algoritmos del Pathway ⭐ | CSCA 5622, 5632, 5642 |
| **CAPSTONE** | 21-24 | M08 | NLP Disaster Tweets 🎯 | Integración total |

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

### Semanas 9-11: M05 - Aprendizaje Supervisado (CSCA 5622)

| Semana | Tema | Implementación | Novedad |
|--------|------|----------------|---------|
| 9 | Regresión Lineal | Normal Equation + GD | **+ Paridad Sklearn** |
| 10 | Regresión Logística + Árboles | Cross-Entropy, Decision Trees | **+ `sklearn.tree`** |
| 11 | **Ética IA & XAI** 🆕 | SHAP, LIME, Bias/Fairness | Interpretabilidad |

**Laboratorios Interactivos:**
```bash
streamlit run M05_Aprendizaje_Supervisado/Laboratorios_Interactivos/overfitting_bias_variance_app.py
streamlit run M05_Aprendizaje_Supervisado/Laboratorios_Interactivos/shap_explainer_app.py  # NUEVO
```

**Entregables:**
- [ ] `logistic_regression.py` con tests (from scratch)
- [ ] Notebook de paridad: resultados manuales == sklearn
- [ ] Análisis SHAP de un modelo Random Forest
- [ ] Documento de reflexión ética (500 palabras)

---

### Semanas 12-15: M06 - Aprendizaje No Supervisado (CSCA 5632)

| Semana | Tema | Implementación | Novedad |
|--------|------|----------------|---------|
| 12 | K-Means | Lloyd's algorithm, K-Means++ | Silhouette Score |
| 13 | PCA | SVD, varianza explicada | t-SNE/UMAP |
| 14 | GMM | Algoritmo EM | Latent variables |
| 15 | **Sistemas de Recomendación** 🆕 | SVD, Factorización Matrices | **MovieLens** |

**Laboratorios Interactivos:**
```bash
streamlit run M06_Aprendizaje_No_Supervisado/Laboratorios_Interactivos/pca_rotation_plotly_app.py
streamlit run M06_Aprendizaje_No_Supervisado/Laboratorios_Interactivos/movie_recommender_app.py  # NUEVO
```

**Entregables:**
- [ ] `kmeans.py` y `pca.py` con tests
- [ ] `gmm.py` con algoritmo EM
- [ ] **`movie_recommender.py` usando SVD** (CRÍTICO para CSCA 5632)
- [ ] Análisis completo MovieLens con métricas (RMSE, Precision@K)

---

### Semanas 16-20: M07 - Deep Learning (CSCA 5642)

> ⚠️ **Stack Principal: Keras/TensorFlow** (alineado con curso oficial)
> PyTorch disponible en `Advanced_Track_PyTorch/` como track opcional.

| Semana | Tema | Implementación | Framework |
|--------|------|----------------|-----------|
| 16 | Perceptrón y MLP | Forward pass, Backprop manual | NumPy |
| 17 | **Keras APIs** | Sequential + **Funcional** 🔑 | tf.keras |
| 18 | CNNs | Conv2D, MaxPooling2D, Flatten | Keras |
| 19 | RNNs/LSTM | LSTM, GRU, Bidirectional, Embedding | Keras |
| 20 | Regularización | Dropout, EarlyStopping, Transfer Learning | Keras |

**Laboratorios Interactivos:**
```bash
streamlit run M07_Deep_Learning/Laboratorios_Interactivos/keras_training_playground_app.py
streamlit run M07_Deep_Learning/Laboratorios_Interactivos/cnn_filter_visualization_app.py
```

**Código Crítico - API Funcional de Keras:**
```python
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

inputs = Input(shape=(784,))
x = Dense(256, activation='relu')(inputs)
x = Dropout(0.3)(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
```

**Entregables:**
- [ ] `neural_network.py` con backprop manual
- [ ] MLP en Keras usando **API Funcional**
- [ ] CNN para MNIST con >98% accuracy (Keras)
- [ ] LSTM para clasificación de texto (Keras)
- [ ] Modelo con EarlyStopping y ModelCheckpoint

---

## 🎯 FASE 3: CAPSTONE - NLP Disaster Tweets (Semanas 21-24)

> **Dataset:** [Kaggle - Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
> Este proyecto integra **CSCA 5622 + 5632 + 5642** en un pipeline completo.

### Semana 21: EDA & Preprocessing

| Tarea | Técnica | Librería |
|-------|---------|----------|
| Limpieza de texto | Regex (URLs, HTML, menciones) | `re` |
| Tokenización | Word tokenization | NLTK / SpaCy |
| Lematización | Reducir a raíz | WordNetLemmatizer |
| Visualización | WordClouds comparativas | `wordcloud` |

**Entregables:**
- [ ] `train_clean.csv` generado
- [ ] WordCloud de tweets reales vs falsos
- [ ] Análisis de desbalance de clases

---

### Semana 22: Baseline Models (Supervisado)

| Modelo | Vectorización | Evaluación |
|--------|---------------|------------|
| Logistic Regression | TF-IDF | F1-Score |
| Naive Bayes | Bag of Words | Matriz Confusión |
| SVM | TF-IDF | Precision/Recall |

**Punto Crítico:** ¿Por qué NO usar Accuracy?
```python
# En datos desbalanceados (70% clase 0), un modelo trivial tiene 70% accuracy
# F1-Score balancea Precision y Recall → métrica correcta
from sklearn.metrics import f1_score
print(f"F1-Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
```

**Entregables:**
- [ ] Pipeline de vectorización + modelo
- [ ] Matriz de confusión analizada
- [ ] Comparación F1-Score de baselines

---

### Semana 23: Deep Learning - LSTM (Deep Learning)

**Arquitectura Bidirectional LSTM:**
```python
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model

inputs = Input(shape=(max_length,))
x = Embedding(vocab_size, embedding_dim)(inputs)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Bidirectional(LSTM(32))(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
```

**Opciones de Embeddings:**
- Entrenar desde cero
- Usar GloVe preentrenados (recomendado)

**Entregables:**
- [ ] LSTM bidireccional funcionando
- [ ] Curvas de learning (loss, accuracy)
- [ ] Comparación con/sin GloVe
- [ ] Regularización: Dropout + EarlyStopping

---

### Semana 24: Transfer Learning + Reporte Final

**Bonus Track - BERT:**
```python
from transformers import BertTokenizer, TFBertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

**REPORT.md - Estructura Académica:**
1. Abstract (150 palabras)
2. Introduction
3. Dataset Description
4. Methodology
5. Experiments & Results
6. Discussion
7. Conclusion
8. References

**Entregables Finales:**
- [ ] BERT fine-tuned (bonus)
- [ ] MODEL_COMPARISON.md con benchmarks
- [ ] **REPORT.md académico**
- [ ] Código limpio y documentado

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
│  • Ejecutar notebooks                                        │
│  • Implementar from scratch + validar con Sklearn            │
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
| 11 | PB-11 | Supervisado + Paridad Sklearn + XAI |
| 15 | PB-15 | No Supervisado + Recomendadores |
| 20 | PB-20 | Deep Learning en Keras |
| 24 | **FINAL** | Capstone NLP + REPORT.md entregado |

---

## 🏆 Criterios de Éxito del Capstone

| Criterio | Mínimo | Excelente |
|----------|--------|-----------|
| F1-Score Baseline | > 0.70 | > 0.78 |
| F1-Score LSTM | > 0.75 | > 0.80 |
| F1-Score BERT | > 0.80 | > 0.85 |
| REPORT.md | Completo | Publicable |
| Código | Funcional | Modular y testeado |

---

## 📚 Recursos por Fase

### Fase 1 (Fundamentos)
- Mathematics for Machine Learning (Deisenroth)
- 3Blue1Brown - Essence of Linear Algebra

### Fase 2 (ML Core)
- Pattern Recognition and ML (Bishop)
- **Deep Learning with Python** (Chollet) - Para Keras
- Documentación SHAP: https://shap.readthedocs.io/
- Surprise Library: https://surprise.readthedocs.io/

### Fase 3 (Capstone)
- CS224n Stanford - NLP with Deep Learning
- HuggingFace Course: https://huggingface.co/course
- NLTK Book: https://www.nltk.org/book/

---

## 💡 Cambios Clave vs. Plan Anterior

| Semana | Antes | Ahora |
|--------|-------|-------|
| 11 | Regularización | **Ética IA & XAI** (SHAP, LIME) |
| 15 | t-SNE/UMAP | **Sistemas de Recomendación** (SVD) |
| 17-20 | PyTorch | **Keras/TensorFlow** (principal) |
| 21-24 | Proyecto MNIST | **NLP Disaster Tweets** (nivel maestría) |

---

*Plan alineado con el MS-AI Pathway de la University of Colorado Boulder*
*Cursos: CSCA 5622 (Supervised), CSCA 5632 (Unsupervised), CSCA 5642 (Deep Learning)*
