# 📋 Rúbrica de Evaluación: Proyecto Disaster Tweets

> **Curso**: CSCA 5642 - Deep Learning (Capstone Project)
> **Proyecto**: Clasificación de Tweets de Desastres usando NLP
> **Puntuación Total**: 100 puntos

---

## 🎯 Objetivo del Proyecto

Construir un pipeline completo de NLP para clasificar tweets como relacionados con desastres reales o no, demostrando dominio de:
- Preprocesamiento de texto
- Modelos baseline de ML clásico
- Arquitecturas de Deep Learning (LSTM/GRU)
- Transfer Learning (BERT)

---

## 📊 Distribución de Puntos

| Categoría | Puntos | Peso |
|-----------|--------|------|
| 1. Limpieza de Datos | 20 | 20% |
| 2. Modelado Base | 20 | 20% |
| 3. Deep Learning | 30 | 30% |
| 4. Reporte y Comunicación | 30 | 30% |
| **TOTAL** | **100** | **100%** |

---

## 1️⃣ Limpieza de Datos (20 puntos)

### Criterios de Evaluación

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **1.1 Manejo de URLs** | 4 | Detecta y remueve/reemplaza URLs con regex apropiado |
| **1.2 Manejo de HTML** | 4 | Elimina tags HTML (`<br>`, `&amp;`, etc.) correctamente |
| **1.3 Stopwords** | 4 | Implementa remoción de stopwords (NLTK o custom) |
| **1.4 Tokenización** | 4 | Usa tokenizador apropiado para tweets (maneja @mentions, #hashtags) |
| **1.5 Normalización** | 4 | Aplica lowercasing, lemmatization/stemming según corresponda |

### Niveles de Desempeño

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| **Excelente** | 18-20 | Pipeline robusto que maneja todos los casos edge. Código modular y reutilizable. |
| **Competente** | 14-17 | Cubre los 5 criterios pero puede faltar manejo de casos especiales. |
| **En Desarrollo** | 10-13 | Implementa 3-4 criterios. Código funcional pero no robusto. |
| **Insuficiente** | 0-9 | Menos de 3 criterios implementados o errores críticos. |

### Checklist de Auto-evaluación

```
[ ] ¿Mi regex para URLs captura http, https, y www?
[ ] ¿Manejo correctamente emojis y caracteres especiales?
[ ] ¿Preservo información útil de hashtags (#earthquake → earthquake)?
[ ] ¿Mi pipeline es reproducible (misma entrada → misma salida)?
[ ] ¿Documenté las decisiones de preprocesamiento?
```

---

## 2️⃣ Modelado Base (20 puntos)

### Criterios de Evaluación

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **2.1 Vectorización TF-IDF** | 5 | Implementa TF-IDF con parámetros justificados (ngram_range, max_features) |
| **2.2 Modelo Naive Bayes** | 5 | Entrena MultinomialNB y reporta métricas |
| **2.3 Modelo Logistic Regression** | 5 | Entrena LogReg con regularización y reporta métricas |
| **2.4 Comparación de Baselines** | 5 | Tabla comparativa con Accuracy, Precision, Recall, F1 |

### Niveles de Desempeño

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| **Excelente** | 18-20 | Ambos modelos implementados. Justifica elección de hiperparámetros. F1 > 0.75. |
| **Competente** | 14-17 | Modelos funcionales. Métricas reportadas correctamente. |
| **En Desarrollo** | 10-13 | Solo un modelo o métricas incompletas. |
| **Insuficiente** | 0-9 | Modelos no funcionales o ausencia de métricas. |

### Checklist de Auto-evaluación

```
[ ] ¿Usé train_test_split ANTES de fit TF-IDF? (evitar data leakage)
[ ] ¿Reporté F1-Score además de Accuracy? (dataset desbalanceado)
[ ] ¿Probé diferentes valores de ngram_range?
[ ] ¿Comparé al menos 2 modelos baseline?
```

---

## 3️⃣ Deep Learning (30 puntos)

### Criterios de Evaluación

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **3.1 Word Embeddings** | 8 | Usa embeddings (trainable o pre-trained como GloVe) |
| **3.2 Arquitectura LSTM/GRU** | 8 | Implementa red recurrente bidireccional |
| **3.3 Regularización** | 7 | Aplica Dropout, Early Stopping, o L2 para evitar overfitting |
| **3.4 Curvas de Aprendizaje** | 7 | Grafica loss/accuracy en train vs validation |

### Niveles de Desempeño

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| **Excelente** | 27-30 | BiLSTM con GloVe, múltiples técnicas de regularización, curvas claras. F1 > 0.78. |
| **Competente** | 21-26 | LSTM funcional con al menos una técnica de regularización. |
| **En Desarrollo** | 15-20 | Modelo entrena pero hay overfitting evidente o arquitectura básica. |
| **Insuficiente** | 0-14 | Modelo no entrena o errores fundamentales en arquitectura. |

### Checklist de Auto-evaluación

```
[ ] ¿Mi modelo usa Bidirectional LSTM/GRU?
[ ] ¿Implementé al menos 2 técnicas anti-overfitting?
[ ] ¿Las curvas de aprendizaje muestran convergencia sin overfitting severo?
[ ] ¿Puedo explicar por qué elegí esa arquitectura específica?
[ ] ¿Probé diferentes valores de embedding_dim y lstm_units?
```

### Arquitectura Mínima Esperada

```python
# Ejemplo de arquitectura que cumple los criterios
model = Sequential([
    Embedding(vocab_size, 100, weights=[glove_matrix], trainable=False),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(64, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

---

## 4️⃣ Reporte y Comunicación (30 puntos)

### Criterios de Evaluación

| Criterio | Puntos | Descripción |
|----------|--------|-------------|
| **4.1 Justificación de Arquitectura** | 10 | Explica POR QUÉ se eligió cada componente del modelo |
| **4.2 Análisis de Matriz de Confusión** | 8 | Interpreta FP, FN y sus implicaciones en el contexto de desastres |
| **4.3 Comparación de Modelos** | 7 | Tabla final comparando Baseline vs LSTM vs BERT (si aplica) |
| **4.4 Conclusiones y Limitaciones** | 5 | Discute limitaciones y posibles mejoras futuras |

### Niveles de Desempeño

| Nivel | Puntos | Descripción |
|-------|--------|-------------|
| **Excelente** | 27-30 | Reporte publicable. Narrativa clara. Visualizaciones profesionales. |
| **Competente** | 21-26 | Cubre todos los criterios. Explicaciones correctas pero pueden ser más profundas. |
| **En Desarrollo** | 15-20 | Reporte incompleto o análisis superficial. |
| **Insuficiente** | 0-14 | Sin reporte o sin análisis de resultados. |

### Checklist de Auto-evaluación

```
[ ] ¿Explico por qué BiLSTM es mejor que LSTM unidireccional para este problema?
[ ] ¿Discuto qué significa un False Positive en el contexto de alertas de desastre?
[ ] ¿Mi tabla comparativa incluye al menos 3 modelos?
[ ] ¿Menciono al menos 2 limitaciones de mi enfoque?
[ ] ¿Propongo mejoras concretas para trabajo futuro?
```

### Preguntas Guía para el Análisis

1. **Falsos Positivos (FP)**: "El modelo predijo desastre pero no lo era"
   - ¿Qué tan grave es esto en un sistema de alertas real?

2. **Falsos Negativos (FN)**: "El modelo NO predijo desastre pero SÍ lo era"
   - ¿Cuál es el costo de no alertar sobre un desastre real?

3. **Trade-off Precision vs Recall**:
   - ¿Prefiero más FP o más FN en este contexto?

---

## ⛔ FATAL FLAWS (Errores Fatales)

> **IMPORTANTE**: Los siguientes errores causan **REPROBACIÓN AUTOMÁTICA** independientemente de la puntuación en otras secciones.

### 1. Data Leakage (Fuga de Datos)

```python
# ❌ INCORRECTO - Causa reprobación
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['text'])  # fit en TODO el dataset
X_train, X_test = train_test_split(X_tfidf, ...)

# ✅ CORRECTO
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], ...)
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)  # fit SOLO en train
X_test_tfidf = tfidf.transform(X_test)        # transform en test
```

**¿Por qué es fatal?** El modelo "ve" información del test set durante el entrenamiento, inflando artificialmente las métricas.

### 2. No Reportar Métricas Apropiadas

```python
# ❌ INCORRECTO - Solo reportar Accuracy en dataset desbalanceado
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# ✅ CORRECTO - Reportar F1, Precision, Recall
print(classification_report(y_test, y_pred))
```

**¿Por qué es fatal?** Con clases desbalanceadas, un modelo que predice siempre la clase mayoritaria puede tener alta accuracy pero ser inútil.

### 3. Modelo No Reproducible

- No fijar `random_state` en train_test_split
- No fijar seeds de NumPy/TensorFlow
- Resultados varían significativamente entre ejecuciones

### 4. Plagio o Código Copiado sin Atribución

- Copiar código de Kaggle/GitHub sin citar fuente
- Usar soluciones de otros estudiantes

### 5. Modelo No Entrena o Errores de Ejecución

- Notebooks con celdas que fallan
- Modelo con accuracy ~50% (random guessing)
- No se puede reproducir el entrenamiento

---

## 📈 Escala de Calificación Final

| Puntuación | Letra | Descripción |
|------------|-------|-------------|
| 90-100 | A | Excelente. Listo para portfolio profesional. |
| 80-89 | B | Competente. Cumple todos los objetivos con calidad. |
| 70-79 | C | Satisfactorio. Cumple requisitos mínimos. |
| 60-69 | D | En desarrollo. Necesita mejoras significativas. |
| <60 | F | Insuficiente. No cumple requisitos mínimos. |

---

## 🔄 Proceso de Peer Review Simulado

### Paso 1: Auto-evaluación (Antes de entregar)
1. Completa TODOS los checklists de esta rúbrica
2. Verifica que no tienes ningún Fatal Flaw
3. Asigna puntos a cada categoría honestamente

### Paso 2: Revisión Cruzada (Si trabajas en equipo)
1. Intercambia notebooks con un compañero
2. Cada uno evalúa el trabajo del otro usando esta rúbrica
3. Discutan discrepancias en las puntuaciones

### Paso 3: Reflexión Final
Responde en tu reporte:
- ¿Qué fue lo más difícil del proyecto?
- ¿Qué harías diferente con más tiempo?
- ¿Qué aprendiste que no sabías antes?

---

## 📚 Recursos de Referencia

- [Kaggle Competition: Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
- [NLTK Documentation](https://www.nltk.org/)
- [Keras Text Classification Tutorial](https://keras.io/examples/nlp/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

---

## ✅ Entregables Finales

1. **Notebooks** (4 archivos .ipynb):
   - `01_EDA_Preprocessing.ipynb`
   - `02_Baseline_Models.ipynb`
   - `03_Deep_Learning_LSTM.ipynb`
   - `04_Transfer_Learning_BERT.ipynb` (opcional para puntos extra)

2. **Reporte** (`REPORT.md`):
   - Máximo 2000 palabras
   - Incluir visualizaciones clave

3. **Código fuente** (`src/`):
   - Módulos reutilizables de preprocessing y evaluation

---

*Rúbrica diseñada para el MS in AI - CU Boulder*
*Alineada con estándares de CSCA 5642 - Deep Learning*
