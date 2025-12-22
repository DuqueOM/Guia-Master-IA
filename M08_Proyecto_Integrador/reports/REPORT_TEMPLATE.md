# NLP Disaster Tweet Classification - Final Report

**Autor:** [Tu Nombre]
**Fecha:** [Fecha de entrega]
**Curso:** MS-AI Pathway Capstone Project

---

## Abstract

[Escribir un resumen de 150 palabras máximo que incluya: problema abordado, metodología principal, resultados clave, y conclusión principal.]

---

## 1. Introduction

### 1.1 Problem Context

La detección temprana de desastres a través de redes sociales se ha convertido en una herramienta crítica para servicios de emergencia. Twitter, con más de 500 millones de tweets diarios, ofrece información en tiempo real que puede salvar vidas.

Sin embargo, el lenguaje humano presenta un desafío fundamental: las mismas palabras pueden usarse de forma literal ("There's a fire in the building!") o figurativa ("This song is fire! 🔥").

### 1.2 Problem Statement

**Objetivo:** Desarrollar un clasificador binario que distinga tweets sobre desastres reales de aquellos que usan lenguaje metafórico o no relacionado con emergencias.

### 1.3 Contributions

Este proyecto presenta:
1. Un pipeline completo de preprocesamiento de texto para tweets
2. Comparación sistemática de modelos baseline (TF-IDF + ML clásico)
3. Implementación de arquitectura LSTM bidireccional con embeddings pre-entrenados
4. Fine-tuning de BERT para clasificación de texto
5. Análisis detallado de errores y recomendaciones para producción

---

## 2. Dataset Description

### 2.1 Data Source

- **Fuente:** Kaggle Competition "Natural Language Processing with Disaster Tweets"
- **URL:** https://www.kaggle.com/c/nlp-getting-started

### 2.2 Dataset Statistics

| Característica | Valor |
|----------------|-------|
| Total de muestras (train) | 7,613 |
| Clase 0 (No desastre) | 4,342 (57.0%) |
| Clase 1 (Desastre real) | 3,271 (43.0%) |
| Longitud promedio de tweet | XX palabras |
| Tweets con keyword | 7,552 (99.2%) |
| Tweets con location | 5,080 (66.7%) |

### 2.3 Exploratory Data Analysis

[Incluir visualizaciones:]
- Distribución de clases (gráfico de barras)
- Distribución de longitud de tweets por clase
- WordClouds comparativos (desastre vs no-desastre)
- Top 20 palabras más frecuentes por clase

### 2.4 Data Challenges

1. **Desbalance moderado:** Ratio 57:43, manejable pero requiere métricas apropiadas
2. **Ruido en texto:** URLs, menciones, hashtags, emojis, errores tipográficos
3. **Ambigüedad semántica:** "fire", "crash", "explosion" usados metafóricamente
4. **Valores faltantes:** ~33% de tweets sin ubicación

---

## 3. Methodology

### 3.1 Preprocessing Pipeline

```python
# Descripción del pipeline implementado
1. Conversión a minúsculas
2. Eliminación de URLs (regex: http\S+)
3. Eliminación de menciones (@usuario)
4. Procesamiento de hashtags (conservar palabra)
5. Eliminación de HTML tags
6. Eliminación de caracteres especiales
7. Tokenización (NLTK word_tokenize)
8. Lematización (WordNetLemmatizer)
9. Eliminación de stopwords (opcional)
```

**Decisiones de diseño:**
- Se optó por lematización sobre stemming para preservar palabras válidas
- Se conservó el contenido de hashtags (#earthquake → earthquake)
- No se eliminaron stopwords en modelos deep learning (LSTM captura contexto)

### 3.2 Feature Engineering

#### TF-IDF Vectorization

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

**Justificación de parámetros:**
- `max_features=5000`: Balance entre información y dimensionalidad
- `ngram_range=(1,2)`: Captura frases como "breaking news", "stay safe"
- `sublinear_tf=True`: Reduce impacto de términos muy frecuentes

#### Word Embeddings

- **GloVe 100d:** Pre-entrenados en 6B tokens de Wikipedia + Gigaword
- **Cobertura de vocabulario:** XX% de palabras del dataset

### 3.3 Model Architectures

#### Baseline Models

1. **Logistic Regression**
   - Regularización L2 (C=1.0)
   - class_weight='balanced'

2. **Multinomial Naive Bayes**
   - Laplace smoothing (alpha=1.0)

#### Deep Learning: Bidirectional LSTM

```
Input (max_length=100)
    ↓
Embedding (100d, GloVe pre-trained, frozen)
    ↓
Bidirectional LSTM (64 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dropout (0.3)
    ↓
Dense (64, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (1, Sigmoid)
```

**Regularización:** Dropout + EarlyStopping (patience=5)

#### Transfer Learning: BERT

- **Modelo base:** bert-base-uncased
- **Fine-tuning:** Learning rate 2e-5, 3 epochs
- **Max sequence length:** 128 tokens

---

## 4. Experiments

### 4.1 Experimental Setup

- **Split:** 80% train, 20% test (stratified)
- **Validation:** 20% of training set for early stopping
- **Random seed:** 42 (reproducibilidad)
- **Hardware:** [Especificar GPU/CPU]

### 4.2 Evaluation Metrics

Dado el desbalance moderado de clases, se priorizó **F1-Score** sobre Accuracy.

- **Precision:** Proporción de predicciones positivas correctas
- **Recall:** Proporción de positivos reales detectados
- **F1-Score:** Media armónica de Precision y Recall

### 4.3 Results

| Model | Precision | Recall | F1-Score | Accuracy | Training Time |
|-------|-----------|--------|----------|----------|---------------|
| Logistic Regression + TF-IDF | X.XX | X.XX | X.XX | X.XX | Xs |
| Naive Bayes + BoW | X.XX | X.XX | X.XX | X.XX | Xs |
| Bi-LSTM + GloVe | X.XX | X.XX | X.XX | X.XX | Xm |
| BERT fine-tuned | X.XX | X.XX | X.XX | X.XX | Xm |

[Incluir gráficos:]
- Matrices de confusión para cada modelo
- Curvas de aprendizaje (loss/accuracy vs epochs) para modelos DL
- Comparación de F1-Score (gráfico de barras)

---

## 5. Results & Discussion

### 5.1 Model Performance Analysis

[Analizar resultados de la tabla anterior]

**Observaciones clave:**
1. [Insight 1]
2. [Insight 2]
3. [Insight 3]

### 5.2 Error Analysis

#### Falsos Positivos (Predijo desastre, era metáfora)

| Tweet | Predicción | Análisis |
|-------|------------|----------|
| "My heart is on fire for you" | 1 (Desastre) | Uso metafórico de "fire" |
| [Más ejemplos] | | |

#### Falsos Negativos (No detectó desastre real)

| Tweet | Predicción | Análisis |
|-------|------------|----------|
| "Prayers for the victims" | 0 (No desastre) | No menciona desastre explícitamente |
| [Más ejemplos] | | |

### 5.3 Feature Importance (Modelos Lineales)

Top 10 features más predictivas para clase "Desastre":
1. [feature 1]: coeficiente X.XX
2. [feature 2]: coeficiente X.XX
...

### 5.4 Limitations

1. **Dataset:** Solo tweets en inglés, puede no generalizar a otros idiomas
2. **Temporalidad:** Entrenado en datos de 2015-2019, nuevos tipos de desastres no cubiertos
3. **Contexto:** No se usa información de keywords o location en modelos finales
4. **Sarcasmo:** Difícil de detectar sin contexto adicional

---

## 6. Conclusion

### 6.1 Summary

[Resumir hallazgos principales en 2-3 párrafos]

### 6.2 Recommended Model for Production

**Recomendación:** [Modelo recomendado]

**Justificación:**
- Performance: F1-Score de X.XX
- Latencia: Xms por predicción
- Complejidad: [Fácil/Media/Alta] de deployar

### 6.3 Future Work

1. **Ensemble:** Combinar predicciones de múltiples modelos
2. **Data augmentation:** Back-translation, synonym replacement
3. **Multimodal:** Incorporar imágenes adjuntas a tweets
4. **Real-time:** Streaming pipeline con Kafka/Spark
5. **Multilingual:** Extender a español, francés, etc.

---

## References

1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.

2. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. EMNLP.

3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.

4. [Agregar más referencias utilizadas]

---

## Appendix

### A. Code Repository Structure

```
M08_Proyecto_Integrador/
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Baseline_Models.ipynb
│   ├── 03_Deep_Learning_LSTM.ipynb
│   └── 04_Transfer_Learning_BERT.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
├── models/
│   ├── baseline_logreg.pkl
│   └── lstm_best.h5
└── reports/
    └── REPORT.md
```

### B. Hyperparameter Tuning Results

[Tabla con resultados de grid search si se realizó]

### C. Additional Visualizations

[Gráficos adicionales que no cupieron en el cuerpo principal]
