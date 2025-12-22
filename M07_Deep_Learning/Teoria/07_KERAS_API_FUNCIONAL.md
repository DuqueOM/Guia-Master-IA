# Módulo 7.2: API Funcional de Keras - Guía Completa

> **Semanas:** 17-20 | **Curso Alineado:** CSCA 5642 - Deep Learning
> **Framework:** TensorFlow/Keras (stack oficial del curso)

---

## 🎯 Objetivos de Aprendizaje

Al finalizar este módulo serás capaz de:

1. **Dominar** la API Funcional de Keras para arquitecturas complejas
2. **Construir** modelos con múltiples entradas/salidas
3. **Implementar** skip connections y arquitecturas residuales
4. **Aplicar** callbacks para entrenamiento profesional
5. **Realizar** transfer learning con modelos preentrenados

---

## 1. ¿Por Qué API Funcional?

### 1.1 Limitaciones de Sequential

```python
# Sequential: solo modelos lineales (capa tras capa)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
# Limitación: ¿Qué si necesitamos múltiples entradas? ¿Skip connections?
```

### 1.2 Capacidades de la API Funcional

```
┌─────────────────────────────────────────────────────────────────┐
│                 API FUNCIONAL: ARQUITECTURAS POSIBLES           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────┐     ┌─────┐    ┌─────┐                                 │
│  │Input│───▶│Layer│───▶│Layer│───▶ Output   (Sequential-like) │
│  └─────┘     └─────┘    └─────┘                                 │
│                                                                 │
│  ┌─────┐                                                        │
│  │Input│──┐                                                     │
│  └─────┘  │   ┌──────────┐                                      │
│           ├─▶│Concatenate│───▶ Output    (Multi-input)         │
│  ┌─────┐  │   └──────────┘                                      │
│  │Input│──┘                                                     │
│  └─────┘                                                        │
│                                                                 │
│  ┌─────┐     ┌─────┐    ┌─────┐                                 │
│  │Input│───▶│Layer│─┬─▶│Layer│───▶ Output1                    │
│  └─────┘     └─────┘ │  └─────┘       (Multi-output)            │
│                      │   ┌─────┐                                │
│                      └─▶│Layer│───▶ Output2                    │
│                          └─────┘                                │
│                                                                 │
│  ┌─────┐     ┌─────┐         ┌─────┐                            │
│  │Input│───▶│Layer│────┬───▶│ Add │───▶ Output                │
│  └─────┘     └─────┘    │    └─────┘       (Skip connection)    │
│                │        │       ▲                               │
│                └────────┴───────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Sintaxis Fundamental

### 2.1 Patrón Básico

```python
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

# 1. Definir entrada
inputs = Input(shape=(784,), name='input_layer')

# 2. Construir capas (notación funcional: layer()(tensor))
x = Dense(256, activation='relu', name='hidden_1')(inputs)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', name='hidden_2')(x)
x = Dropout(0.3)(x)

# 3. Definir salida
outputs = Dense(10, activation='softmax', name='output')(x)

# 4. Crear modelo
model = Model(inputs=inputs, outputs=outputs, name='mi_modelo')

# 5. Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Inspeccionar arquitectura
model.summary()
```

### 2.2 Anatomía de una Capa Funcional

```python
# Sintaxis: output_tensor = Layer(parámetros)(input_tensor)

# Desglose:
layer = Dense(64, activation='relu')  # Instanciar capa (no conectada)
output = layer(input_tensor)           # Conectar capa a tensor

# Forma compacta (más común):
output = Dense(64, activation='relu')(input_tensor)
```

---

## 3. Arquitecturas Multi-Input

### 3.1 Ejemplo: Modelo con Datos Tabulares + Texto

```python
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Flatten
from tensorflow.keras.models import Model

# Entrada 1: Datos numéricos (edad, salario, etc.)
numeric_input = Input(shape=(10,), name='numeric_features')
x1 = Dense(32, activation='relu')(numeric_input)

# Entrada 2: Texto (secuencia de palabras)
text_input = Input(shape=(100,), name='text_sequence')  # max_length=100
x2 = Embedding(input_dim=10000, output_dim=64)(text_input)
x2 = LSTM(32)(x2)

# Combinar ambas ramas
combined = Concatenate()([x1, x2])
x = Dense(64, activation='relu')(combined)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid', name='prediction')(x)

# Crear modelo
model = Model(
    inputs=[numeric_input, text_input],
    outputs=output,
    name='hybrid_model'
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento con múltiples inputs
model.fit(
    [X_numeric, X_text],  # Lista de arrays
    y,
    epochs=10,
    batch_size=32
)
```

### 3.2 Ejemplo: Modelo Siamés (Comparación de Similitud)

```python
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K

# Subred compartida (pesos compartidos)
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Dense(128, activation='relu')(input)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    return Model(input, x)

# Crear red base
base_network = create_base_network((784,))

# Dos entradas para comparar
input_a = Input(shape=(784,), name='input_a')
input_b = Input(shape=(784,), name='input_b')

# Procesar ambas con LA MISMA red (pesos compartidos)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Distancia euclidiana
distance = Lambda(
    lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True))
)([processed_a, processed_b])

# Modelo siamés
siamese_model = Model(inputs=[input_a, input_b], outputs=distance)
```

---

## 4. Arquitecturas Multi-Output

### 4.1 Ejemplo: Clasificación + Regresión Simultánea

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Input compartido
inputs = Input(shape=(100,), name='features')

# Backbone compartido
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)

# Output 1: Clasificación (categoría de producto)
category_output = Dense(10, activation='softmax', name='category')(x)

# Output 2: Regresión (precio)
price_output = Dense(1, activation='linear', name='price')(x)

# Modelo con múltiples salidas
model = Model(inputs=inputs, outputs=[category_output, price_output])

# Compilar con diferentes losses por output
model.compile(
    optimizer='adam',
    loss={
        'category': 'sparse_categorical_crossentropy',
        'price': 'mse'
    },
    loss_weights={
        'category': 1.0,
        'price': 0.5  # Menor peso a regresión
    },
    metrics={
        'category': 'accuracy',
        'price': 'mae'
    }
)

# Entrenar
model.fit(
    X_train,
    {'category': y_category, 'price': y_price},
    epochs=20
)
```

---

## 5. Skip Connections y Arquitecturas Residuales

### 5.1 Residual Block (ResNet-style)

```python
from tensorflow.keras.layers import Input, Dense, Add, BatchNormalization, Activation
from tensorflow.keras.models import Model

def residual_block(x, units):
    """
    Bloque residual: output = F(x) + x

    La conexión "skip" permite que el gradiente fluya directamente,
    facilitando el entrenamiento de redes muy profundas.
    """
    # Guardar input para skip connection
    shortcut = x

    # Rama principal
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(units)(x)
    x = BatchNormalization()(x)

    # Skip connection: sumar input original
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

# Construir red con bloques residuales
inputs = Input(shape=(256,))
x = Dense(256, activation='relu')(inputs)

# Apilar bloques residuales
x = residual_block(x, 256)
x = residual_block(x, 256)
x = residual_block(x, 256)

outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
```

### 5.2 Dense Connections (DenseNet-style)

```python
from tensorflow.keras.layers import Concatenate

def dense_block(x, units, num_layers=4):
    """
    Bloque denso: cada capa recibe features de TODAS las capas anteriores.
    """
    features = [x]

    for _ in range(num_layers):
        # Concatenar todas las features anteriores
        if len(features) > 1:
            x = Concatenate()(features)
        else:
            x = features[0]

        # Nueva capa
        new_features = Dense(units, activation='relu')(x)
        features.append(new_features)

    return Concatenate()(features)
```

---

## 6. Callbacks: Entrenamiento Profesional

### 6.1 Los Callbacks Esenciales

```python
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)
import datetime

# 1. Early Stopping: detener si no mejora
early_stop = EarlyStopping(
    monitor='val_loss',        # Métrica a monitorear
    patience=10,               # Épocas sin mejora antes de parar
    restore_best_weights=True, # Restaurar mejores pesos
    verbose=1
)

# 2. Model Checkpoint: guardar mejor modelo
checkpoint = ModelCheckpoint(
    filepath='models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,       # Solo guardar si mejora
    save_weights_only=False,   # Guardar modelo completo
    verbose=1
)

# 3. Reduce LR: reducir learning rate si plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                # Nuevo LR = LR * factor
    patience=5,                # Épocas antes de reducir
    min_lr=1e-7,              # LR mínimo
    verbose=1
)

# 4. TensorBoard: visualización
log_dir = f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,          # Histogramas de pesos
    write_graph=True
)

# 5. CSV Logger: guardar métricas en archivo
csv_logger = CSVLogger('training_log.csv', append=True)

# Usar todos los callbacks
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint, reduce_lr, tensorboard, csv_logger]
)
```

### 6.2 Custom Callback

```python
from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):
    """
    Callback personalizado para logging avanzado.
    """

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n--- Iniciando época {epoch + 1} ---")

    def on_epoch_end(self, epoch, logs=None):
        # logs contiene: loss, accuracy, val_loss, val_accuracy, etc.
        if logs.get('val_accuracy', 0) > 0.95:
            print(f"\n¡Accuracy > 95%! Considera detener.")

    def on_batch_end(self, batch, logs=None):
        # Se llama después de cada batch
        if batch % 100 == 0:
            print(f"  Batch {batch}: loss = {logs.get('loss', 'N/A'):.4f}")

    def on_train_end(self, logs=None):
        print("\n=== Entrenamiento finalizado ===")
```

---

## 7. Transfer Learning

### 7.1 Usando Modelos Preentrenados

```python
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Cargar modelo base (sin la cabeza de clasificación)
base_model = VGG16(
    weights='imagenet',        # Pesos preentrenados en ImageNet
    include_top=False,         # Excluir capas fully-connected
    input_shape=(224, 224, 3)
)

# Congelar pesos del modelo base
base_model.trainable = False

# Agregar nuestra cabeza de clasificación
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # training=False: modo inferencia
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)  # 10 clases nuevas

model = Model(inputs, outputs)

# Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Parámetros totales: {model.count_params():,}")
print(f"Parámetros entrenables: {sum(p.numpy().size for p in model.trainable_weights):,}")
```

### 7.2 Fine-Tuning

```python
# Después de entrenar la cabeza, descongelar algunas capas del base
base_model.trainable = True

# Congelar todas excepto las últimas N capas
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Re-compilar con learning rate muy bajo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR bajo
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continuar entrenamiento
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

### 7.3 Transfer Learning para Texto con BERT

```python
# Requiere: pip install transformers
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

# Cargar modelo y tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Inputs para BERT
input_ids = Input(shape=(128,), dtype='int32', name='input_ids')
attention_mask = Input(shape=(128,), dtype='int32', name='attention_mask')

# BERT output
bert_output = bert_model(input_ids, attention_mask=attention_mask)
pooled_output = bert_output.pooler_output  # [CLS] token

# Clasificador
x = Dense(256, activation='relu')(pooled_output)
x = Dropout(0.3)(x)
outputs = Dense(2, activation='softmax')(x)  # Clasificación binaria

# Modelo completo
model = Model(inputs=[input_ids, attention_mask], outputs=outputs)

# Congelar BERT inicialmente
bert_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 8. Regularización en Keras

### 8.1 Técnicas de Regularización

```python
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2

# 1. Dropout: desactivar neuronas aleatoriamente
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.5)(x)  # 50% de neuronas desactivadas

# 2. L1/L2 Regularization en pesos
x = Dense(
    128,
    activation='relu',
    kernel_regularizer=l2(0.01),      # L2 en pesos
    bias_regularizer=l1(0.01),        # L1 en biases
    activity_regularizer=l1_l2(l1=0.01, l2=0.01)  # En activaciones
)(inputs)

# 3. Batch Normalization: normalizar activaciones
x = Dense(128)(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# 4. Spatial Dropout (para CNNs)
from tensorflow.keras.layers import SpatialDropout2D
x = Conv2D(64, 3, activation='relu')(inputs)
x = SpatialDropout2D(0.3)(x)  # Dropea canales completos
```

### 8.2 Data Augmentation

```python
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential

# Capa de augmentation (solo durante entrenamiento)
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

# Integrar en modelo
inputs = Input(shape=(224, 224, 3))
x = data_augmentation(inputs)  # Solo se aplica en training=True
x = base_model(x)
# ... resto del modelo
```

---

## 9. Ejercicios Prácticos

### Ejercicio 1: Red Multi-Input para Predicción de Precios

```python
"""
Dataset: House Prices (Kaggle)
Tarea: Combinar features numéricas + descripción textual

1. Input 1: Features numéricas (area, habitaciones, etc.)
2. Input 2: Descripción de la propiedad (texto)
3. Fusionar y predecir precio

Arquitectura sugerida:
- Rama numérica: Dense layers
- Rama texto: Embedding + LSTM
- Concatenate + Dense + Output (regresión)
"""
```

### Ejercicio 2: Implementar ResNet-18 desde Cero

```python
"""
Implementar ResNet-18 usando API Funcional:
- 4 bloques residuales
- Skip connections con proyección cuando cambia dimensión
- BatchNorm + ReLU después de cada conv

Entrenar en CIFAR-10 y comparar con modelo sin skip connections.
"""
```

### Ejercicio 3: Transfer Learning para Clasificación de Plantas

```python
"""
Dataset: PlantVillage (Kaggle)
Tarea: Clasificar enfermedades de plantas

1. Usar MobileNetV2 como base (eficiente)
2. Fine-tuning progresivo:
   - Fase 1: Solo entrenar cabeza (10 épocas)
   - Fase 2: Descongelar últimos 20 layers (10 épocas)
   - Fase 3: Fine-tune completo con LR muy bajo (5 épocas)

Reportar accuracy por fase.
"""
```

---

## 10. Patrones Comunes y Best Practices

### 10.1 Checklist de Arquitectura

- [ ] ¿Input shape correcto?
- [ ] ¿BatchNormalization antes o después de activación? (controversial, probar ambos)
- [ ] ¿Dropout rate apropiado? (0.2-0.5 típico)
- [ ] ¿Regularización L2 en capas densas?
- [ ] ¿Activación final correcta? (softmax para multiclass, sigmoid para binary)

### 10.2 Checklist de Entrenamiento

- [ ] ¿EarlyStopping configurado?
- [ ] ¿ModelCheckpoint guardando mejor modelo?
- [ ] ¿ReduceLROnPlateau para fine-tuning automático de LR?
- [ ] ¿Validation split o validation_data?
- [ ] ¿Batch size apropiado para GPU memory?

### 10.3 Debugging Tips

```python
# 1. Verificar shapes
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# 2. Probar con batch pequeño
test_input = np.random.randn(2, *model.input_shape[1:])
test_output = model.predict(test_input)
print(f"Test output shape: {test_output.shape}")

# 3. Visualizar modelo
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

# 4. Gradient check (verificar que gradientes fluyen)
with tf.GradientTape() as tape:
    predictions = model(X_batch, training=True)
    loss = loss_fn(y_batch, predictions)
grads = tape.gradient(loss, model.trainable_variables)
for var, grad in zip(model.trainable_variables, grads):
    if grad is None:
        print(f"WARNING: No gradient for {var.name}")
```

---

## 11. Resumen

| Concepto | Sintaxis Clave |
|----------|----------------|
| **Input** | `inputs = Input(shape=(n,))` |
| **Layer funcional** | `x = Dense(64, activation='relu')(x)` |
| **Model** | `Model(inputs=inputs, outputs=outputs)` |
| **Multi-input** | `Model(inputs=[in1, in2], outputs=out)` |
| **Multi-output** | `Model(inputs=in, outputs=[out1, out2])` |
| **Skip connection** | `x = Add()([x, shortcut])` |
| **Transfer Learning** | `base_model.trainable = False` |

---

## 12. Recursos

- **Keras Documentation**: https://keras.io/guides/functional_api/
- **TensorFlow Tutorials**: https://www.tensorflow.org/tutorials
- **Deep Learning with Python (Chollet)**: Capítulos 7-8
- **Stanford CS231n**: CNN architectures

---

*Material desarrollado para el MS-AI Pathway - University of Colorado Boulder*
*Semanas 17-20 - CSCA 5642: Deep Learning*
