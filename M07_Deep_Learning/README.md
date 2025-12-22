# Módulo 07: Deep Learning

> **Semanas:** 16-20 | **Fase:** ML Core ⭐ | **Curso Alineado:** CSCA 5642

---

## ⚠️ Stack Tecnológico: Keras/TensorFlow (Principal)

> **IMPORTANTE:** El curso CSCA 5642 utiliza **Keras/TensorFlow** como framework principal.
> Este módulo prioriza Keras para máxima alineación con el pathway.
> PyTorch se ofrece como track avanzado opcional.

---

## 📁 Estructura

```
M07_Deep_Learning/
├── Teoria/
│   ├── 01_perceptron_mlp.md
│   ├── 02_backpropagation.md
│   ├── 03_cnns.md
│   ├── 04_rnns_lstm.md
│   └── 05_regularizacion_dl.md
├── Notebooks_Keras/                        # RUTA PRINCIPAL (tf.keras)
│   ├── 01_perceptron_scratch.ipynb        # Implementación matemática
│   ├── 02_mlp_keras_sequential.ipynb      # API Sequential
│   ├── 03_mlp_keras_functional.ipynb      # API Funcional (CRÍTICO)
│   ├── 04_backprop_manual.ipynb           # Gradientes a mano
│   ├── 05_cnn_keras.ipynb                 # Conv2D, MaxPooling2D
│   ├── 06_rnn_lstm_keras.ipynb            # LSTM, GRU
│   ├── 07_regularizacion_callbacks.ipynb  # Dropout, EarlyStopping, ModelCheckpoint
│   └── 08_transfer_learning_keras.ipynb   # Fine-tuning modelos preentrenados
├── Advanced_Track_PyTorch/                 # OPCIONAL
│   ├── README.md
│   ├── 01_tensors_autograd.ipynb
│   ├── 02_mlp_pytorch.ipynb
│   ├── 03_cnn_pytorch.ipynb
│   └── 04_rnn_pytorch.ipynb
├── Laboratorios_Interactivos/
│   ├── keras_training_playground_app.py
│   ├── cnn_filter_visualization_app.py
│   └── lstm_sequence_app.py
└── assets/
```

---

## 🎯 Objetivos de Aprendizaje

### Semana 16: Fundamentos de Redes Neuronales

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Implementar perceptrón desde cero | Forward pass + update rule funcionando |
| Implementar MLP desde cero | Backpropagation manual con derivadas |
| Overfit en problema XOR | Demostrar no-linealidad aprendida |

### Semana 17: Keras - APIs Sequential y Funcional

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Dominar `tf.keras.Sequential` | Construir MLP para clasificación |
| **Dominar API Funcional de Keras** | `inputs = Input(...)`, `x = Dense(...)(x)` |
| Compilar y entrenar modelos | `model.compile()`, `model.fit()`, `model.evaluate()` |
| Visualizar entrenamiento | `history` plots, TensorBoard básico |

### Semana 18: CNNs (Redes Convolucionales)

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Entender convolución y pooling | Implementar Conv2D desde cero (conceptual) |
| Construir CNN en Keras | `Conv2D`, `MaxPooling2D`, `Flatten` |
| Clasificar MNIST/CIFAR-10 | >98% accuracy en MNIST, >70% en CIFAR-10 |
| Visualizar filtros aprendidos | Feature maps de capas convolucionales |

### Semana 19: RNNs y LSTMs

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Entender secuencias y estados | Vanishing gradient problem |
| Implementar LSTM/GRU en Keras | `LSTM`, `GRU`, `Bidirectional` |
| Procesamiento de texto básico | Clasificación de sentimientos |
| Entender embeddings | `Embedding` layer en Keras |

### Semana 20: Regularización y Buenas Prácticas

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Implementar Dropout | Prevenir overfitting |
| Usar Callbacks de Keras | `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` |
| Batch Normalization | Entender y aplicar `BatchNormalization` |
| Transfer Learning | Fine-tuning de modelo preentrenado (VGG16/ResNet) |

---

## 🔑 API Funcional de Keras (CRÍTICO para CSCA 5642)

```python
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

# Definir arquitectura con API Funcional
inputs = Input(shape=(784,))
x = Dense(256, activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(10, activation='softmax')(x)

# Crear modelo
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar con callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=callbacks)
```

---

## ⚡ Inicio Rápido

```bash
# Semana 16: Fundamentos
jupyter notebook Notebooks_Keras/01_perceptron_scratch.ipynb
jupyter notebook Notebooks_Keras/04_backprop_manual.ipynb

# Semana 17: Keras APIs (CRÍTICO)
jupyter notebook Notebooks_Keras/02_mlp_keras_sequential.ipynb
jupyter notebook Notebooks_Keras/03_mlp_keras_functional.ipynb  # PRIORITARIO

# Semana 18: CNNs
jupyter notebook Notebooks_Keras/05_cnn_keras.ipynb
streamlit run Laboratorios_Interactivos/cnn_filter_visualization_app.py

# Semana 19: RNNs
jupyter notebook Notebooks_Keras/06_rnn_lstm_keras.ipynb

# Semana 20: Regularización y Transfer Learning
jupyter notebook Notebooks_Keras/07_regularizacion_callbacks.ipynb
jupyter notebook Notebooks_Keras/08_transfer_learning_keras.ipynb

# OPCIONAL: Track PyTorch Avanzado
jupyter notebook Advanced_Track_PyTorch/01_tensors_autograd.ipynb
```

---

## ✅ Entregables del Módulo

- [ ] `neural_network.py` con backprop manual (from scratch)
- [ ] MLP en Keras usando API Funcional
- [ ] CNN para MNIST con >98% accuracy (Keras)
- [ ] LSTM para clasificación de texto (Keras)
- [ ] Modelo con EarlyStopping y ModelCheckpoint
- [ ] Experimento de Transfer Learning documentado

---

## 📚 Recursos

### Documentación Oficial
- **Keras Documentation**: https://keras.io/
- **TensorFlow Tutorials**: https://www.tensorflow.org/tutorials

### Lecturas Recomendadas
1. **Deep Learning with Python** (François Chollet) - Autor de Keras
2. **CS231n Stanford** - CNNs for Visual Recognition
3. **CS224n Stanford** - NLP with Deep Learning

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [M06 No Supervisado](../M06_Aprendizaje_No_Supervisado/) | [README](../README.md) | [M08 Proyecto Final →](../M08_Proyecto_Integrador/) |
