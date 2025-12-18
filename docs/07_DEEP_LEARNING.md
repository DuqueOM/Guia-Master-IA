# Módulo 07 - Deep Learning

> **🎯 Objetivo:** Implementar MLP con backprop + CNN forward (NumPy) + entrenamiento CNN con PyTorch
> **Fase:** 2 - Núcleo de ML | **Semanas 17-20**
> **Curso del Pathway:** Introduction to Deep Learning

---

<a id="m07-0"></a>

## 🧭 Cómo usar este módulo (modo 0→100)

**Propósito:** que puedas construir y depurar una red neuronal desde cero:

- forward pass
- backpropagation
- optimización (SGD/Momentum/Adam)
- sanity checks (overfit test)

### Objetivos de aprendizaje (medibles)

Al terminar este módulo podrás:

- **Implementar** un MLP que resuelva XOR.
- **Explicar** backprop como chain rule aplicada a un grafo computacional.
- **Depurar** entrenamiento con overfit test (si no memoriza, hay bug).
- **Implementar** forward pass de una CNN simple (convolución + pooling) en NumPy para dominar dimensiones.
- **Entrenar** una CNN equivalente usando PyTorch (`torch.nn`) sin implementar backward manual.

Enlaces rápidos:

- [03_CALCULO_MULTIVARIANTE.md](03_CALCULO_MULTIVARIANTE.md) (Chain Rule)
- [GLOSARIO.md](GLOSARIO.md)
- [RECURSOS.md](RECURSOS.md)
- [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
- [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)
- Evaluación (rúbrica): [study_tools/RUBRICA_v1.md](../study_tools/RUBRICA_v1.md) (scope `M07` en `rubrica.csv`; cierre Semana 20)

### Recursos (cuándo usarlos)

| Prioridad | Recurso | Cuándo usarlo en este módulo | Para qué |
|----------|---------|------------------------------|----------|
| **Obligatorio** | [03_CALCULO_MULTIVARIANTE.md](03_CALCULO_MULTIVARIANTE.md) | Antes de implementar `backward()` (Semana 18) | Asegurar Chain Rule y gradientes básicos |
| **Obligatorio** | `study_tools/DRYRUN_BACKPROPAGATION.md` | Justo antes de tu primera implementación completa de Backprop | Hacer “dry-run” y detectar errores de gradiente antes del código |
| **Obligatorio** | `study_tools/EXAMEN_ADMISION_SIMULADO.md` | Después de que tu MLP resuelva XOR y antes de cerrar el módulo | Validación tipo examen (sin IDE/internet) |
| **Complementario** | [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Semana 17–18, cuando necesites intuición de backprop | Visualizar forward/backward y por qué aprende |
| **Complementario** | [TensorFlow Playground](https://playground.tensorflow.org/) | Semana 17–18, cuando estudies por qué una capa lineal no resuelve XOR y cómo las activaciones cambian la geometría | Ver en tiempo real cómo la red “dobla” el espacio para separar clases |
| **Complementario** | [Deep Learning Book](https://www.deeplearningbook.org/) | Semana 19–20 (CNNs/entrenamiento), si quieres rigor | Referencia profunda (gratis) |
| **Opcional** | [RECURSOS.md](RECURSOS.md) | Al terminar el módulo (para profundizar en DL/CNNs) | Seleccionar refuerzos sin romper el plan |

---

## 🧠 ¿Por Qué Deep Learning?

```
DEEP LEARNING = Redes Neuronales Multicapa + Arquitecturas Especializadas

Ventajas sobre ML clásico:
├── Aprende features automáticamente (no feature engineering manual)
├── Puede modelar relaciones NO LINEALES complejas
├── Escala con más datos y más compute
└── Estado del arte en visión (CNNs), NLP (Transformers), etc.

Desventajas:
├── Requiere más datos
├── "Caja negra" - menos interpretable
└── Costoso computacionalmente
```

### Intuición geométrica: Deep Learning como “doblar el espacio” (origami)

Una capa lineal `z = Wx + b` solo puede **rotar, estirar o inclinar** el espacio: siempre produce una frontera de decisión lineal (un hiperplano). Por eso un modelo lineal no puede separar XOR.

La no linealidad (ReLU/sigmoid/tanh) es lo que permite “doblar” el espacio:

- después del primer doblez, puntos que antes estaban mezclados pueden quedar en regiones separables
- con varias capas, encadenas dobleces hasta que en la última capa los datos son separables con un hiperplano

Visualización sugerida:

- dibuja XOR en 2D
- intenta separarlo con una sola línea (imposible)
- luego imagina un doblez que junta los puntos de la misma clase

---

## 📚 Contenido del Módulo

| Semana | Tema | Entregable |
|--------|------|------------|
| 17 | Perceptrón y MLP | `activations.py` + forward pass |
| 18 | Backpropagation | `backward()` con Chain Rule |
| 19 | **CNNs: Teoría + Forward (NumPy)** | Convolución/pooling (forward) + quiz de dimensiones |
| 20 | **PyTorch para CNNs + Sequence Modeling (Light)** | `scripts/train_cnn_pytorch.py` + `scripts/simple_rnn_forward.py` |

---

## 🧵 Semana 20 (extra): Sequence Modeling (Light) — RNN forward pass

**Objetivo:** entender dimensiones en datos secuenciales sin entrenar.

- **Ejecutable:**
  - `python3 scripts/simple_rnn_forward.py`
- **Qué debes entender:**
  - `x.shape = (batch, time, features)`
  - `h.shape = (batch, time, hidden)`
  - `y.shape = (batch, time, out)`

---

## 💻 Parte 1: Perceptrón y Activaciones

### 1.1 La Neurona Artificial

```python
import numpy as np  # Importa NumPy para operaciones eficientes con arrays

"""
NEURONA ARTIFICIAL (Perceptrón)

Inspiración biológica:
- Recibe señales de entrada (dendrites)
- Procesa y decide si "dispara" (soma)
- Envía señal de salida (axon)

Modelo matemático:
    z = Σ wᵢxᵢ + b = w·x + b  (combinación lineal)
    a = σ(z)                    (activación)

Donde:
- x: vector de entradas
- w: vector de pesos (learnable)
- b: bias (learnable)
- σ: función de activación (introduce no-linealidad)
"""

def perceptron(x: np.ndarray, w: np.ndarray, b: float) -> float:  # Perceptrón: calcula z=w·x+b y aplica función escalón (clasificación lineal)
    """
    Un perceptrón simple.

    Args:
        x: entrada (n_features,)
        w: pesos (n_features,)
        b: bias

    Returns:
        salida activada
    """
    z = np.dot(w, x) + b  # Calcula combinación lineal de entradas y pesos
    return 1 if z > 0 else 0  # Función escalón: 1 si z>0, 0 si z≤0
```

### 1.2 Funciones de Activación

```python
import numpy as np  # Importa NumPy para operaciones matemáticas

class Activations:  # Agrupa activaciones típicas de redes neuronales y sus derivadas (API educativa/organizada)
    """Funciones de activación y sus derivadas."""  # Docstring de clase: documenta propósito; no cambia el cálculo en runtime

    @staticmethod  # Define método estático: no necesita `self`/estado; se usa como Activations.sigmoid(z)
    def sigmoid(z: np.ndarray) -> np.ndarray:  # Sigmoide: mapea logits reales a (0,1), típica en salida binaria
        """
        Sigmoid: σ(z) = 1 / (1 + e^(-z))

        Rango: (0, 1)
        Uso: Capa de salida para clasificación binaria
        Problema: Vanishing gradient para |z| grande
        """
        z = np.clip(z, -500, 500)  # Previene overflow en exp() con valores extremos
        return 1 / (1 + np.exp(-z))  # Fórmula matemática de la sigmoide

    @staticmethod  # Método estático: la derivada depende solo de la activación `a` ya calculada
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:  # Derivada de sigmoide: usada en backprop para propagar gradientes
        """σ'(z) = σ(z) · (1 - σ(z)) = a · (1 - a)"""  # Docstring: recuerda identidad; no afecta el valor devuelto
        return a * (1 - a)  # Derivada simplificada usando salida ya calculada

    @staticmethod  # Método estático: ReLU no requiere estado interno
    def relu(z: np.ndarray) -> np.ndarray:  # ReLU: activa solo valores positivos; es estándar en capas ocultas
        """
        ReLU: f(z) = max(0, z)

        Rango: [0, ∞)
        Uso: Capas ocultas (default moderno)
        Ventaja: No vanishing gradient para z > 0
        Problema: "Dying ReLU" si z < 0 siempre
        """
        return np.maximum(0, z)  # Implementación directa de ReLU

    @staticmethod  # Método estático: derivada depende de z (pre-activación) para crear la máscara
    def relu_derivative(z: np.ndarray) -> np.ndarray:  # Derivada de ReLU: 1 en z>0, 0 en z<=0 (define dónde fluye el gradiente)
        """ReLU'(z) = 1 si z > 0, 0 si z ≤ 0"""  # Docstring: especifica la regla de la derivada; ayuda a depuración
        return (z > 0).astype(float)  # Convierte booleano a float (1.0 o 0.0)

    @staticmethod  # Método estático: tanh tampoco requiere estado
    def tanh(z: np.ndarray) -> np.ndarray:  # Tanh: alternativa centrada en 0; puede usarse en capas ocultas
        """
        Tanh: f(z) = (e^z - e^(-z)) / (e^z + e^(-z))

        Rango: (-1, 1)
        Uso: Alternativa a sigmoid (centrado en 0)
        """
        return np.tanh(z)  # Usa implementación NumPy optimizada

    @staticmethod  # Método estático: derivada depende de la salida `a=tanh(z)` para evitar recomputar tanh
    def tanh_derivative(a: np.ndarray) -> np.ndarray:  # Derivada de tanh: usada en backprop; decrece cerca de saturación
        """tanh'(z) = 1 - tanh²(z) = 1 - a²"""  # Docstring: identidad matemática base para el cálculo
        return 1 - a ** 2  # Derivada usando identidad matemática

    @staticmethod  # Método estático: softmax opera por fila/eje y no requiere estado
    def softmax(z: np.ndarray) -> np.ndarray:  # Softmax: convierte logits en distribución de probabilidad multiclase (suma 1)
        """
        Softmax: softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)

        Rango: (0, 1), suma = 1
        Uso: Capa de salida para clasificación multiclase
        Output: probabilidades de cada clase
        """
        # Restar máximo para estabilidad numérica (previene overflow en exp)
        z_shifted = z - np.max(z, axis=-1, keepdims=True)  # Centra logits restando el máximo por fila: no cambia softmax y evita overflow
        exp_z = np.exp(z_shifted)  # Calcula exponenciales de valores estabilizados
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)  # Normaliza para que suma = 1


# Demo de funciones de activación
z = np.array([-2, -1, 0, 1, 2])  # Valores de prueba
act = Activations()  # Instancia clase de activaciones

print("z:", z)  # Muestra valores originales
print("sigmoid:", act.sigmoid(z))  # Muestra sigmoid aplicada
print("relu:", act.relu(z))  # Muestra ReLU aplicada
print("tanh:", act.tanh(z))  # Muestra tanh aplicada
print("softmax:", act.softmax(z))  # Muestra softmax aplicada
```

### 1.3 El Problema XOR

```python
"""
XOR: La limitación del Perceptrón Simple

XOR truth table:
    x1  x2  |  y
    0   0   |  0
    0   1   |  1
    1   0   |  1
    1   1   |  0

Un perceptrón simple NO puede resolver XOR porque:
- XOR no es linealmente separable
- No existe una línea que separe las clases

Solución: Red multicapa (MLP)
- Una capa oculta puede aprender features intermedias
- Combinación de features no lineales resuelve XOR
"""

# Datos XOR - problema clásico no linealmente separable
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entradas binarias
y_xor = np.array([0, 1, 1, 0])  # Salidas XOR

# Un perceptrón simple no puede aprender esto
# Necesitamos una red con al menos una capa oculta
```

---

## 💻 Parte 2: Forward Propagation

### 2.1 Arquitectura MLP

```python
"""
MLP - Multilayer Perceptron

Arquitectura típica:
    Input Layer → Hidden Layer(s) → Output Layer

Ejemplo para clasificación binaria:
    x (n_features) → h (n_hidden) → y (1)

Forward Pass:
    z₁ = W₁x + b₁        (capa 1: lineal)
    a₁ = σ(z₁)           (capa 1: activación)
    z₂ = W₂a₁ + b₂       (capa 2: lineal)
    a₂ = σ(z₂)           (capa 2: activación = output)

Dimensiones:
    x: (n_features,)
    W₁: (n_hidden, n_features)
    b₁: (n_hidden,)
    z₁, a₁: (n_hidden,)
    W₂: (n_output, n_hidden)
    b₂: (n_output,)
    z₂, a₂: (n_output,)
"""
```

### 2.2 Implementación Forward Pass

```python
import numpy as np  # Importa NumPy: operaciones vectorizadas y funciones matemáticas para el forward pass
from typing import List, Dict  # Importa tipos: documenta estructuras (lista de capas, cache) sin afectar runtime

class Layer:  # Capa densa: implementa forward (z=Wx+b) y aplica una activación
    """Una capa de la red neuronal."""  # Docstring de clase: documenta responsabilidad; se ejecuta como literal de string

    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):  # Inicializa parámetros y tipo de activación
        """
        Args:
            input_size: número de entradas
            output_size: número de neuronas
            activation: 'relu', 'sigmoid', 'tanh', 'softmax', 'linear'
        """
        self.input_size = input_size  # Guarda dimensión de entrada: útil para entender shapes y depurar
        self.output_size = output_size  # Guarda dimensión de salida: número de neuronas de la capa
        self.activation = activation  # Guarda activación: controla la no linealidad aplicada en forward

        # Inicialización Xavier/He
        if activation == 'relu':  # Selecciona He init para ReLU (varianza estable para activaciones)
            # He initialization para ReLU
            std = np.sqrt(2.0 / input_size)  # Std He: sqrt(2/fan_in)
        else:  # Para activaciones suaves (tanh/sigmoid) suele usarse Xavier para evitar saturación
            # Xavier initialization
            std = np.sqrt(1.0 / input_size)  # Std Xavier simplificado: sqrt(1/fan_in)

        self.W = np.random.randn(output_size, input_size) * std  # Pesos: shape (out,in) con escala de init
        self.b = np.zeros(output_size)  # Bias: vector (out,) inicializado a cero

        # Cache para backprop
        self.cache = {}  # Diccionario de cache: guarda x/z/a del forward para usar luego en backward

    def forward(self, x: np.ndarray) -> np.ndarray:  # Forward: computa activación de la capa para un input
        """
        Forward pass de una capa.

        z = Wx + b
        a = activation(z)
        """
        self.cache['x'] = x  # Guarda input: necesario para gradientes de W en backprop

        # Transformación lineal
        z = self.W @ x + self.b  # Pre-activación: (out,in)@(in,) + (out,) => (out,)
        self.cache['z'] = z  # Guarda z: útil para derivadas (ReLU) y depuración

        # Activación
        if self.activation == 'relu':  # ReLU: común en capas ocultas
            a = np.maximum(0, z)  # max(0,z) elemento a elemento
        elif self.activation == 'sigmoid':  # Sigmoid: útil como salida binaria
            a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid estable: clip evita overflow
        elif self.activation == 'tanh':  # tanh: activación centrada en 0
            a = np.tanh(z)  # Aplica tanh elemento a elemento
        elif self.activation == 'softmax':  # Softmax: salida multiclase
            z_shifted = z - np.max(z)  # Estabilización: resta máximo para prevenir overflow
            exp_z = np.exp(z_shifted)  # Exponencia logits estabilizados
            a = exp_z / np.sum(exp_z)  # Normaliza para obtener probabilidades que suman 1
        else:  # linear
            a = z  # Identidad: sin no linealidad

        self.cache['a'] = a  # Guarda activación: útil para derivadas (sigmoid/tanh) y capa siguiente
        return a  # Devuelve la salida de la capa


class NeuralNetwork:  # Red multicapa: compone varias Layer y realiza forward secuencial
    """Red Neuronal Multicapa."""  # Docstring de clase: describe el contenedor de capas; no afecta el resultado del forward

    def __init__(self, layer_sizes: List[int], activations: List[str]):  # Construye la red a partir de tamaños y activaciones
        """
        Args:
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            activations: ['relu', 'relu', ..., 'sigmoid'] para cada capa
        """
        assert len(activations) == len(layer_sizes) - 1  # Invariante: una activación por capa (excepto input)

        self.layers = []  # Lista de capas en orden: output de una alimenta la siguiente
        for i in range(len(layer_sizes) - 1):  # Itera pares consecutivos (in->out)
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])  # Crea capa i con su activación
            self.layers.append(layer)  # Agrega la capa a la red

    def forward(self, x: np.ndarray) -> np.ndarray:  # Forward de la red: propaga la entrada por todas las capas
        """Forward pass a través de todas las capas."""  # Docstring de método: describe la función; es una cadena literal en runtime
        a = x  # Activación inicial: la entrada del modelo
        for layer in self.layers:  # Recorre capas en orden forward
            a = layer.forward(a)  # Propaga activación a través de la capa
        return a  # Devuelve salida final

    def predict(self, X: np.ndarray) -> np.ndarray:  # Predicción batch: aplica forward y convierte a clases
        """Predicción para múltiples muestras."""  # Docstring de método: explica uso de predict en batch
        predictions = []  # Acumula predicciones por muestra
        for x in X:  # Itera muestras del batch
            output = self.forward(x)  # Forward por muestra
            if len(output) == 1:  # Caso binario: una sola salida
                predictions.append(1 if output[0] > 0.5 else 0)  # Umbral 0.5 para sigmoid
            else:  # Caso multiclase: vector de scores/probabilidades
                predictions.append(np.argmax(output))  # Selecciona índice del máximo
        return np.array(predictions)  # Devuelve ndarray para usar en métricas


# Demo
net = NeuralNetwork(  # Instancia red de demostración (sin entrenamiento) para probar el forward
    layer_sizes=[2, 4, 1],  # 2 inputs → 4 hidden → 1 output
    activations=['relu', 'sigmoid']  # ReLU en oculta, sigmoid en salida
)  # Cierra construcción de la red demo

# Forward pass
x = np.array([0.5, 0.3])  # Input de ejemplo: vector 2D
output = net.forward(x)  # Ejecuta forward: salida depende de pesos aleatorios
print(f"Input: {x}")  # Imprime input para referencia
print(f"Output: {output}")  # Imprime output de la red (sin entrenar)
```

---

## 💻 Parte 3: Backpropagation

### 3.0 Backpropagation — Nivel: intermedio/avanzado

**Propósito:** este bloque te lleva de “sé que backprop existe” a **poder derivarlo, implementarlo y depurarlo** bajo condiciones tipo examen.

#### Objetivos de aprendizaje (medibles)

Al terminar este bloque podrás:

- **Recordar** la notación estándar de una capa (`z = Wx + b`, `a = φ(z)`) y el rol de cada variable.
- **Explicar** por qué backprop es simplemente *regla de la cadena aplicada a un grafo computacional*.
- **Aplicar** backprop para calcular `∂L/∂W` y `∂L/∂b` en una red MLP de 2 capas.
- **Analizar** fallas típicas (signos, shapes, overflow) usando pruebas de sanidad.
- **Evaluar** si tu implementación es correcta con un *overfit test* y (cuando aplique) *gradient checking*.
- **Crear** una implementación mínima (NumPy) de forward + backward y entrenarla en un toy dataset.

#### Motivación / por qué importa

Backpropagation es el mecanismo que hace posible que redes con millones de parámetros se ajusten a datos. En práctica:

- **Visión (CV):** CNNs y modelos de clasificación/segmentación se entrenan con backprop.
- **NLP:** aunque los Transformers no se implementan aquí, el entrenamiento sigue siendo backprop sobre un grafo computacional.
- **Industria:** cuando un entrenamiento “no aprende”, casi siempre el diagnóstico comienza revisando gradientes, estabilidad numérica y shapes.

#### Prerrequisitos y nivel de entrada

- **Cálculo:** derivadas, derivadas parciales, regla de la cadena.
- **Álgebra lineal:** multiplicación matriz-vector, transpuesta.
- **Probabilidad / pérdidas:** cross-entropy como pérdida para clasificación.

Mini-recordatorio (enlaces directos):

- [GLOSARIO: Chain Rule](GLOSARIO.md#chain-rule)
- [GLOSARIO: Gradient](GLOSARIO.md#gradient)
- [GLOSARIO: Backpropagation](GLOSARIO.md#backpropagation)
- [GLOSARIO: Binary Cross-Entropy](GLOSARIO.md#binary-cross-entropy)

#### Resumen ejecutivo (big idea)

Backpropagation calcula gradientes **de manera eficiente** reutilizando resultados intermedios del forward pass. En vez de derivar a mano una expresión enorme, modelas el cálculo como un **grafo** de operaciones simples (sumas, productos, activaciones). Luego aplicas la regla de la cadena localmente y propagas “responsabilidad del error” desde la salida hasta los parámetros.

La idea operacional es:

- Haces un **forward pass** guardando `x`, `z`, `a` de cada capa.
- Calculas la pérdida `L`.
- Empiezas en la salida con un gradiente inicial y haces un **backward pass** capa por capa:
  - `δ = ∂L/∂z` (el “error” local)
  - `∂L/∂W = δ ⊗ x` y `∂L/∂b = δ`
  - propagas hacia atrás: `∂L/∂x = Wᵀ δ`

#### Visualización crítica: el grafo computacional de Backprop (hacer clic mental)

Para entender backprop, no mires fórmulas planas: mira el grafo.

Una neurona simple:

`L(a)  ←  a = σ(z)  ←  z = w·x + b`

El gradiente fluye río arriba (de derecha a izquierda):

1) **Llegada del error:** recibes `∂L/∂a`.
2) **Compuerta sigmoide:** multiplicas por la derivada local `σ'(z)`.
3) **Señal en z:**

`δ = ∂L/∂z = (∂L/∂a) · σ'(z)`

4) **Bifurcación lineal (`z = w·x + b`):**

- Hacia `w`: `∂L/∂w = δ · x`
- Hacia `b`: `∂L/∂b = δ`
- Hacia `x`: `∂L/∂x = δ · w`

Regla mnemotécnica:

- Gradiente del **peso** = **error local (`δ`) × entrada (`x`)**
- Gradiente hacia atrás = **error local (`δ`) × peso (`w`)**

#### Mapa del contenido y tiempo estimado

- **Intuición + vocabulario:** 20–35 min
- **Formalización (notación + shapes):** 30–45 min
- **Derivación guiada (2 capas):** 45–75 min
- **Worked example numérico (paso a paso):** 45–60 min
- **Implementación práctica (NumPy) + pruebas de sanidad:** 2–4 h

#### Núcleo: explicación progresiva por capas

##### a) Intuición / metáfora

Piensa en una red como una fábrica con varias estaciones. La salida está mal (pérdida alta) y quieres saber **cuánto contribuyó cada perilla** (peso) al error. Backprop es un procedimiento para *repartir la culpa* desde el error final hacia atrás, estación por estación.

##### b) Conceptos clave (glosario mínimo)

- **Forward pass:** computar `z` y `a` desde la entrada hasta la salida.
- **Loss `L`:** número que mide “qué tan mal” predice el modelo.
- **Gradiente:** vector de derivadas que indica cómo cambia `L` si mueves parámetros.
- **Delta `δ`:** gradiente local `∂L/∂z` en una capa (la señal que se propaga hacia atrás).

##### c) Formalización (fórmulas + shapes)

Para una capa totalmente conectada:

- `z = Wx + b`
- `a = φ(z)`

Shapes recomendados (para evitar errores silenciosos):

- `x`: `(n_in,)` o `(n_in, 1)`
- `W`: `(n_out, n_in)`
- `b`: `(n_out,)` o `(n_out, 1)`
- `z, a`: `(n_out,)` o `(n_out, 1)`

##### d) Demostración / derivación (idea central)

En cada capa usas regla de la cadena:

- `∂L/∂W = ∂L/∂z · ∂z/∂W`
- `∂L/∂b = ∂L/∂z · ∂z/∂b`
- `∂L/∂x = ∂L/∂z · ∂z/∂x`

Y como `z = Wx + b`:

- `∂z/∂W` depende de `x`
- `∂z/∂b = 1`
- `∂z/∂x = W`

Esto produce el patrón computacional:

```
dL_da  →  (multiplicar por φ'(z))  →  δ = dL_dz
                       │
                       ├── dL_dW = δ ⊗ x
                       ├── dL_db = δ
                       └── dL_dx = Wᵀ δ
```

##### e) Ejemplo resuelto (worked example) paso a paso

Objetivo del ejemplo: una red **2-2-1** (2 entradas, 2 ocultas, 1 salida) con sigmoid en salida para clasificación binaria. El entregable es poder escribir:

- forward: `z1, a1, z2, a2`
- backward: `δ2, dW2, db2, δ1, dW1, db1`

Guía de trabajo (sin números para que puedas rellenar tú):

1. **Forward**
   - `z1 = W1 x + b1`
   - `a1 = φ(z1)`
   - `z2 = W2 a1 + b2`
   - `a2 = σ(z2)`
2. **Loss**
   - `L = BCE(y, a2)`
3. **Backward**
   - Para sigmoid + BCE (caso típico): `δ2 = a2 - y`
   - `dW2 = δ2 ⊗ a1`
   - `db2 = δ2`
   - `δ1 = (W2ᵀ δ2) ⊙ φ'(z1)`
   - `dW1 = δ1 ⊗ x`
   - `db1 = δ1`

Ejemplo numérico completo (forward y backward, con números):

Definimos:

- Entrada: `x = [1.0, -2.0]`
- Etiqueta: `y = 1`
- Activación oculta: `φ = ReLU`
- Activación salida: `σ` (sigmoid)

Parámetros:

- `W1 = [[0.1, -0.2], [0.4, 0.3]]`, `b1 = [0.0, 0.1]`
- `W2 = [[-0.3, 0.2]]`, `b2 = [0.05]`

1) Forward

- `z1 = W1x + b1`
  - `z1_1 = 0.1·1 + (-0.2)·(-2) + 0.0 = 0.5`
  - `z1_2 = 0.4·1 + 0.3·(-2) + 0.1 = -0.1`
  - `z1 = [0.5, -0.1]`
- `a1 = ReLU(z1) = [0.5, 0.0]`

- `z2 = W2a1 + b2 = (-0.3)·0.5 + 0.2·0.0 + 0.05 = -0.10`
- `a2 = σ(z2) ≈ 0.4750`

2) Loss (Binary Cross-Entropy)

- `L = -log(a2) ≈ -log(0.4750) ≈ 0.744`

3) Backward

- Para sigmoid + BCE: `δ2 = a2 - y ≈ 0.4750 - 1 = -0.5250`

- Gradientes en salida:
  - `dW2 = δ2 ⊗ a1 = [-0.5250·0.5, -0.5250·0.0] ≈ [-0.2625, 0.0]`
  - `db2 = δ2 ≈ -0.5250`

- Propagación a la capa oculta:
  - `dL/da1 = W2ᵀ δ2 = [-0.3, 0.2]ᵀ · (-0.5250) ≈ [0.1575, -0.1050]`
  - `ReLU'(z1) = [1, 0]` (porque `z1_1>0` y `z1_2<0`)
  - `δ1 = dL/da1 ⊙ ReLU'(z1) ≈ [0.1575, 0.0]`

- Gradientes en primera capa:
  - `dW1 = δ1 ⊗ x`
    - para neurona 1: `[0.1575·1.0, 0.1575·(-2.0)] ≈ [0.1575, -0.3150]`
    - para neurona 2: `[0, 0]`
  - `db1 = δ1 ≈ [0.1575, 0.0]`

Chequeo mental:

- Los gradientes “se apagan” donde `ReLU'(z)=0`.
- `dW` siempre tiene la misma shape que `W`.

#### Protocolo (Semana 18): grafo computacional + shapes explícitos (antes de programar `backward()`)

Antes de escribir cualquier `backward()`, fija dos cosas:

- **Tu grafo computacional** (qué nodos existen y quién depende de quién).
- **Tus shapes** (para que cada gradiente tenga una shape única y verificable).

##### 1) Elige una convención y no la mezcles (recomendado: batch-first 2D)

- `X`: `(n, d_in)`
- `W`: `(d_in, d_out)`
- `b`: `(d_out,)` (se “broadcastea” a `(n, d_out)`)
- `Z = XW + b`: `(n, d_out)`
- Activaciones `A`: `(n, d_out)`

Evita mezclar `(d,)` y `(d,1)` a menos que decidas usar columna-vectores en TODO.

##### 2) Red de 2 capas: shapes del forward que debes poder escribir de memoria

Red (batch):

- `Z1 = XW1 + b1`, `A1 = relu(Z1)`
- `Z2 = A1W2 + b2`, `P = sigmoid(Z2)`

Tabla de shapes:

| Símbolo | Significado | Shape |
|---|---|---|
| `X` | batch de entrada | `(n, d_in)` |
| `W1` | pesos capa 1 | `(d_in, d_h)` |
| `b1` | bias capa 1 | `(d_h,)` |
| `Z1`, `A1` | pre/post activación | `(n, d_h)` |
| `W2` | pesos capa 2 | `(d_h, d_out)` |
| `b2` | bias capa 2 | `(d_out,)` |
| `Z2`, `P` | logits / probabilidades | `(n, d_out)` |
| `y` | targets | `(n, d_out)` |

##### 3) Invariantes de gradientes (no negociables)

Si `Z = XW + b` con las shapes batch-first:

- `dW` **debe** tener la misma shape que `W`.
- `db` **debe** tener la misma shape que `b`.
- `dX` **debe** tener la misma shape que `X`.

Para la red de 2 capas:

| Gradiente | Shape |
|---|---|
| `dZ2` | `(n, d_out)` |
| `dW2 = A1.T @ dZ2` | `(d_h, d_out)` |
| `db2 = sum(dZ2, axis=0)` | `(d_out,)` |
| `dA1 = dZ2 @ W2.T` | `(n, d_h)` |
| `dZ1 = dA1 * relu'(Z1)` | `(n, d_h)` |
| `dW1 = X.T @ dZ1` | `(d_in, d_h)` |
| `db1 = sum(dZ1, axis=0)` | `(d_h,)` |

##### 4) Protocolo de depuración (antes de “tocar hyperparams”)

- Agrega `assert` de shapes.
- Haz **gradient checking** en 1–3 coordenadas.
- Haz un **overfit test** en un dataset mini: si no memoriza, es bug.

##### 4.1 Cápsula: Shape checks (decorator + asserts)

Regla práctica: si una función consume tensores/arrays, valida **shapes** al inicio (y, si aplica, valida la salida). Esto reduce bugs silenciosos en `forward()`/`backward()`.

```python
import numpy as np  # Importa NumPy: provee np.asarray, generación de datos aleatorios y operaciones vectorizadas usadas en el ejemplo
from typing import Any, Callable, Dict, Optional, Sequence, Tuple  # Importa tipos: se usan para anotar shapes/firmas del decorator (solo documentación/IDE; no cambia el runtime)

def assert_shape(x: np.ndarray, shape: Sequence[Optional[int]], name: str = "x") -> np.ndarray:  # Valida que `x` tenga la dimensionalidad/shape esperada (con `None` como comodín)
    x = np.asarray(x)  # Fuerza conversión a ndarray: normaliza inputs (listas/tuplas) y garantiza que `ndim/shape` existan
    assert x.ndim == len(shape), f"{name}.ndim={x.ndim}, expected={len(shape)}"  # Verifica #dims: si falla, se detiene con AssertionError explicando el mismatch
    for i, (got, exp) in enumerate(zip(x.shape, shape)):  # Itera por dimensión i: compara la shape real vs la esperada dimensión-a-dimensión
        if exp is not None:  # `None` significa “no validar esta dimensión” (útil para batch variable)
            assert got == exp, f"{name}.shape[{i}]={got}, expected={exp}"  # Verifica dimensión i: si falla, se corta temprano evitando bugs silenciosos de broadcasting
    return x  # Devuelve el mismo array (ya normalizado): permite encadenar validación dentro de pipelines/funciones

def shape_check(  # Factory de decorator: construye un wrapper que valida shapes de kwargs (y opcionalmente la salida)
    spec: Dict[str, Sequence[Optional[int]]],  # Especificación: mapping nombre_argumento -> shape esperada (con None como comodín)
    out: Optional[Sequence[Optional[int]]] = None,  # Shape esperada de la salida (si se pasa): útil para validar invariantes post-forward
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:  # Devuelve un decorator que, al aplicarse, produce una función wrapped con asserts
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:  # Recibe la función objetivo y devuelve una versión instrumentada con checks
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # Wrapper: intercepta llamada para validar inputs/salida sin modificar la lógica interna
            for k, shp in spec.items():  # Recorre las claves especificadas en `spec` (solo valida las entradas que se esperan)
                if k in kwargs:  # Valida únicamente si el argumento fue pasado por keyword (este helper está diseñado para kwargs)
                    assert_shape(kwargs[k], shp, name=k)  # Aplica assert_shape al argumento: si no coincide, falla antes de ejecutar el cálculo
            y = fn(*args, **kwargs)  # Ejecuta la función original con los mismos args/kwargs: no altera el resultado, solo lo captura
            if out is not None:  # Si se definió un shape esperado de salida, habilita chequeo posterior
                assert_shape(y, out, name="out")  # Verifica shape del output: detecta errores de dimensiones inmediatamente tras el forward
            return y  # Devuelve el resultado original: el wrapper es transparente salvo por los asserts

        return wrapper  # Retorna la función decorada: es la que reemplazará a `fn` en tiempo de import/definición

    return deco  # Retorna el decorator configurado con `spec/out`: permite reutilizar la misma regla en múltiples funciones

@shape_check({"X": (None, 3), "W": (3, 4), "b": (4,)}, out=(None, 4))  # Valida shapes de inputs/outputs: detecta bugs de dimensiones antes de que propaguen
def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:  # Forward de capa densa: aplica transformación afín Z = XW + b (batch-first)
    return X @ W + b  # Multiplica (n,3)@(3,4)->(n,4) y suma bias (4,) por broadcasting: produce logits/activaciones pre-no-lineales

X = np.random.randn(5, 3)  # Crea un batch de 5 muestras con 3 features: ejemplo que cumple la spec (None,3)
W = np.random.randn(3, 4)  # Crea matriz de pesos (3->4): compatible con X para producto matricial
b = np.random.randn(4)  # Crea bias de salida (4,): se sumará a cada fila del batch vía broadcasting
Z = dense_forward(X=X, W=W, b=b)  # Ejecuta forward validado por decorator: asserts corren antes/después y luego se calcula Z
assert Z.shape == (5, 4)  # Sanity check final: confirma que la salida cumple la shape esperada (batch=5, d_out=4)
```

##### 4.2 Cápsula: Inicialización (Xavier vs He/Kaiming)

Regla práctica (MLP):

- Activaciones tipo `tanh/sigmoid` suelen ir mejor con **Xavier/Glorot**.
- Activaciones tipo `ReLU` suelen ir mejor con **He/Kaiming**.

```python
import numpy as np  # Importa NumPy: se usa para RNG, sqrt y generar matrices de pesos con distribución normal
from typing import Literal, Optional  # Importa tipos: restringe `mode` a valores válidos y hace `seed` opcional (anotaciones)

def init_linear(  # Inicializa pesos de una capa lineal controlando la varianza según la activación (Xavier vs He/Kaiming)
    fan_in: int,  # Número de unidades de entrada: determina la escala recomendada de inicialización para evitar exploding/vanishing
    fan_out: int,  # Número de unidades de salida: determina la shape final de W (fan_in, fan_out)
    mode: Literal["xavier", "kaiming"] = "xavier",  # Selecciona esquema: Xavier (tanh/sigmoid) o Kaiming (ReLU)
    seed: Optional[int] = None,  # Semilla opcional: si se pasa, la inicialización será reproducible
) -> np.ndarray:  # Devuelve matriz de pesos W con shape (fan_in, fan_out)
    rng = np.random.default_rng(seed)  # Crea generador RNG moderno: evita global state de np.random y permite reproducibilidad por seed

    if mode == "kaiming":  # He/Kaiming: recomendado para ReLU porque mantiene varianza al pasar por la no-linealidad rectificada
        std = np.sqrt(2.0 / fan_in)  # Desv. estándar: sqrt(2/fan_in) para compensar que ReLU “apaga” ~mitad de activaciones
    else:  # Xavier/Glorot: recomendado para tanh/sigmoid (más simétricas), busca preservar varianza entre capas
        std = np.sqrt(1.0 / fan_in)  # Desv. estándar: sqrt(1/fan_in) (forma simplificada) para mantener escala estable en forward/backward

    W = rng.standard_normal((fan_in, fan_out)) * std  # Muestra N(0,1) y escala por std: produce pesos con varianza controlada
    return W  # Devuelve pesos: se usarán en la capa lineal; una mala escala puede causar saturación o gradientes inestables

d_in, d_out = 784, 128  # Dimensiones ejemplo (MNIST->capa oculta): 784 entradas (28x28) y 128 unidades de salida
W_relu = init_linear(d_in, d_out, mode="kaiming", seed=0)  # Inicializa pesos para red con ReLU: usa He/Kaiming
W_tanh = init_linear(d_in, d_out, mode="xavier", seed=0)  # Inicializa pesos para red con tanh/sigmoid: usa Xavier/Glorot
assert W_relu.shape == (d_in, d_out)  # Sanity check: verifica que la shape de W coincide con (fan_in, fan_out)
assert W_tanh.shape == (d_in, d_out)  # Sanity check: confirma lo mismo para la inicialización Xavier
```

##### f) Implementación práctica (laboratorio)

Checklist mínimo de implementación (sin “magia”):

- una clase/capa que guarde `x`, `z`, `a` en cache
- un `backward()` que devuelva `dL_dx`, `dL_dW`, `dL_db`
- un training loop que muestre una curva de pérdida descendente

Protocolos de ejecución (integración v4/v5):

- **v4.0 (Semana 18):** antes de programar, completar `study_tools/DRYRUN_BACKPROPAGATION.md`.
- **v5.0 (validación):** si el entrenamiento no converge, hacer:
  - *Overfit on small batch* (este módulo ya lo incluye más abajo).
  - si el error persiste, revisar *gradient checking* (ver checklist general en `CHECKLIST.md`).

##### g) Variantes, limitaciones y casos frontera

- **Softmax + Cross-Entropy:** el gradiente de salida también se simplifica a `y_pred - y_true` (cuando `y_true` es one-hot).
- **Sigmoid en capas ocultas:** riesgo de *vanishing gradients* si `|z|` crece.
- **ReLU:** riesgo de *dying ReLU* (neurona que queda en 0 siempre).
- **Estabilidad numérica:** usar `clip`, restar `max(z)` en softmax, y `eps` en logs.

#### Visuales (para estudiar y recordar)

Grafo computacional mínimo (una capa):

```
x ──► (Wx + b) ──► z ──► φ(z) ──► a ──► L
         ▲                    ▲
         │                    │
         W,b                  φ'
```

#### Diagrama de flujo: forward (verde) / backward (rojo)

```
FORWARD (verde)
x → z1=W1x+b1 → a1=φ(z1) → z2=W2a1+b2 → a2 → L

BACKWARD (rojo)
L → dL/da2 → δ2=dL/dz2 → dW2,db2 → δ1 → dW1,db1
```

Regla práctica para implementarlo:

- **Forward:** guarda en cache `x, z, a` por capa.
- **Backward:** empieza por el último `δ` y propaga hacia atrás con `Wᵀ`.

#### Fallas típicas (con visual): Vanishing Gradient vs Dying ReLU

**1) Vanishing gradient (sigmoid/tanh en capas ocultas)**

Si `|z|` es grande, `σ(z)` se satura y `σ'(z) ≈ 0`.
En backprop, multiplicas muchas derivadas pequeñas:

```
δ1 = (W2ᵀ δ2) ⊙ φ'(z1)
δ0 = (W1ᵀ δ1) ⊙ φ'(z0)
...

si φ'(z) ≈ 0 en varias capas → δ se vuelve ~0
```

Síntomas:

- loss baja muy lento
- pesos de capas tempranas casi no cambian

Mitigación (en este nivel):

- usa ReLU en ocultas (o inicializaciones cuidadosas)
- normaliza features

**2) Dying ReLU**

ReLU: `φ(z)=max(0,z)` y `φ'(z)=0` si `z<0`.

Si una neurona queda siempre con `z<0`, su gradiente se vuelve 0 y “muere”:

```
z < 0  →  a = 0
φ'(z)=0  →  δ = δ_next ⊙ 0 = 0
```

Síntomas:

- muchas activaciones exactamente 0
- algunas neuronas nunca “reviven”

Mitigación (en este nivel):

- baja learning rate
- inicializa pesos con escalas razonables
- considera LeakyReLU (conceptual)

#### Actividades activas (aprendizaje activo)

- **Retrieval practice (5–10 min):** sin mirar notas, escribe las 6 ecuaciones: `δ2`, `dW2`, `db2`, `δ1`, `dW1`, `db1`.
- **Interleaving:** alterna ejercicios de backprop con ejercicios de shapes (recomendado: `study_tools/DRILL_DIMENSIONES_NUMPY.md`).
- **Generación:** crea tu propio mini-ejemplo con una red 3-3-1 y verifica a mano una iteración.

#### Evaluación (formativa y sumativa)

- **Quiz conceptual:**
  - ¿Qué representa `δ` y por qué es útil?
  - ¿Por qué `δ2 = a2 - y` en sigmoid+BCE?
- **Prueba práctica:** tu red debe:
  - resolver XOR, y
  - pasar el *overfit test* sobre un minibatch.

#### Cheat sheet (repaso rápido)

- `z = Wx + b`
- `a = φ(z)`
- `δ = ∂L/∂z = (∂L/∂a) ⊙ φ'(z)`
- `∂L/∂W = δ ⊗ x`
- `∂L/∂b = δ`
- `∂L/∂x = Wᵀ δ`

#### Errores comunes y FAQs

- **(Shapes)** confundir `(n,)` con `(n,1)` y obtener gradientes transpuestos.
- **(Signos)** usar `y - y_pred` en lugar de `y_pred - y` y “subir” la loss.
- **(Softmax)** implementar softmax sin restar el máximo → overflow.
- **(Debug)** si la red no puede memorizar 4 puntos de XOR, *no* es “falta de datos”; es un bug.

#### Recursos complementarios (orientados a práctica)

- [RECURSOS.md](RECURSOS.md)
- `study_tools/DRYRUN_BACKPROPAGATION.md`
- `study_tools/EXAMEN_ADMISION_SIMULADO.md`
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### Consolidación (Backpropagation)

#### Errores comunes

- **Shapes mal:** confundir `(n,)` con `(n,1)` y obtener transpuestas inesperadas.
- **Signo del gradiente:** si actualizas con `+ lr * grad`, subes la loss.
- **No cachear:** si no guardas `x, z, a`, terminas recomputando o usando valores incorrectos.
- **Explosión numérica:** logits grandes → `exp` overflow → `nan`.

#### Debugging / validación (v5)

- **Overfit on small batch:** si no puede memorizar 4 puntos (XOR), asume bug.
- Revisa `nan/inf`:
  - `np.exp` sin `clip`
  - `np.log` sin `eps`
- Registra hallazgos en `study_tools/DIARIO_ERRORES.md`.
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

#### Reto Feynman (tablero blanco)

Explica en 5 líneas o menos:

1) ¿Qué es `δ` y por qué es la señal que “viaja hacia atrás”?
2) ¿Por qué `dW = δ ⊗ x` tiene sentido dimensionalmente?
3) ¿Cómo distinguirías vanishing gradient vs dying ReLU en logs/activaciones?

---

## 💻 Parte 4: Optimizadores

### 4.1 SGD (Stochastic Gradient Descent)

```python
class SGD:  # Optimizador SGD básico: aplica descenso por gradiente con learning rate fijo (sin momentum ni adaptividad)
    """Vanilla Stochastic Gradient Descent."""  # Docstring: describe el algoritmo; es un literal de string y no cambia la actualización

    def __init__(self, learning_rate: float = 0.01):  # Constructor: almacena el learning rate que se usará en cada paso
        self.lr = learning_rate  # Guarda lr: escala el update; valores extremos causan divergencia o aprendizaje lento

    def update(self, layer, dW: np.ndarray, db: np.ndarray):  # Update in-place de parámetros del layer usando gradientes dW/db
        layer.W -= self.lr * dW  # Pesos: W <- W - lr*dW (descenso por gradiente)
        layer.b -= self.lr * db  # Bias: b <- b - lr*db
```

### 4.2 SGD con Momentum

```python
class SGDMomentum:  # Define SGD con momentum: mantiene una “velocidad” (EMA del gradiente) para suavizar y acelerar el descenso
    """
    SGD con Momentum.

    v_t = β·v_{t-1} + (1-β)·∇L
    θ = θ - lr·v_t

    Momentum ayuda a:
    - Acelerar convergencia
    - Escapar de mínimos locales
    - Reducir oscilaciones
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):  # Inicializa hiperparámetros y el estado (velocidades) por capa
        self.lr = learning_rate  # Guarda learning rate: escala el tamaño de paso al aplicar la velocidad a los parámetros
        self.momentum = momentum  # Guarda β (momentum): controla cuánto del “pasado” se conserva en la velocidad (suavizado)
        self.velocities = {}  # Diccionario layer_id -> {'W': vW, 'b': vb}: buffers persistentes para aplicar momentum por parámetro

    def update(self, layer, dW: np.ndarray, db: np.ndarray, layer_id: int):  # Aplica un paso de actualización con momentum a W/b del layer
        if layer_id not in self.velocities:  # Inicializa buffers si es la primera vez que se actualiza este layer (por id estable)
            self.velocities[layer_id] = {  # Crea estructura de velocidad: se mantiene entre iteraciones para acumular gradientes suavizados
                'W': np.zeros_like(dW),  # vW inicial en 0: mismo shape que dW para poder acumular EMA de gradientes de pesos
                'b': np.zeros_like(db)  # vb inicial en 0: mismo shape que db para acumular EMA de gradientes de bias
            }  # Fin de inicialización: si no se hace, el primer update no tendría historial y habría KeyError

        v = self.velocities[layer_id]  # Recupera referencia a los buffers del layer: se actualizarán in-place para persistir entre pasos

        # Actualizar velocidad
        v['W'] = self.momentum * v['W'] + (1 - self.momentum) * dW  # Actualiza velocidad W: EMA del gradiente; reduce oscilación en ravines
        v['b'] = self.momentum * v['b'] + (1 - self.momentum) * db  # Actualiza velocidad b: mismo principio para bias

        # Actualizar parámetros
        layer.W -= self.lr * v['W']  # Actualiza pesos usando velocidad: paso efectivo incorpora historial (momentum)
        layer.b -= self.lr * v['b']  # Actualiza bias: se mantiene consistente con update de W
```

### 4.3 Adam Optimizer

```python
class Adam:  # Define optimizador Adam: mantiene promedios móviles (1er y 2do momento) por parámetro para pasos adaptativos
    """
    Adam: Adaptive Moment Estimation.

    Combina:
    - Momentum (primer momento)
    - RMSprop (segundo momento)

    m_t = β₁·m_{t-1} + (1-β₁)·g_t       (momentum)
    v_t = β₂·v_{t-1} + (1-β₂)·g_t²      (velocidad adaptativa)
    m̂_t = m_t / (1 - β₁^t)              (corrección de bias)
    v̂_t = v_t / (1 - β₂^t)
    θ = θ - lr · m̂_t / (√v̂_t + ε)
    """

    def __init__(  # Inicializa hiperparámetros y estados internos del optimizador (m, v, y contador de paso t)
        self,  # Referencia a la instancia: permite guardar hiperparámetros y buffers entre actualizaciones
        learning_rate: float = 0.001,  # Paso base (lr): escala la magnitud del update; muy alto puede divergir, muy bajo aprende lento
        beta1: float = 0.9,  # Decaimiento del 1er momento (momentum): controla suavizado de gradientes en `m`
        beta2: float = 0.999,  # Decaimiento del 2do momento (RMS): controla suavizado de gradiente^2 en `v`
        epsilon: float = 1e-8  # Término numérico: evita división por cero cuando sqrt(v_hat) es muy pequeño
    ):  # Cierra firma: al instanciarse una vez, estos valores quedan fijos para todo el entrenamiento
        self.lr = learning_rate  # Guarda lr: se reutiliza en cada update para escalar el paso
        self.beta1 = beta1  # Guarda β1: controla cuánto “recuerda” el 1er momento el pasado
        self.beta2 = beta2  # Guarda β2: controla cuánto “recuerda” el 2do momento el pasado
        self.epsilon = epsilon  # Guarda ε: estabiliza la división en la regla de actualización
        self.m = {}  # Diccionario de 1er momento por layer_id: cada entrada guarda arrays para 'W' y 'b'
        self.v = {}  # Diccionario de 2do momento por layer_id: acumula promedio de gradiente al cuadrado
        self.t = 0  # Paso global: se usa para corrección de bias (β^t) en momentos iniciales

    def update(self, layer, dW: np.ndarray, db: np.ndarray, layer_id: int):  # Aplica un paso de Adam a los parámetros del `layer` usando gradientes dW/db
        if layer_id not in self.m:  # Inicializa estados si es la primera vez que se actualiza este layer_id
            self.m[layer_id] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}  # m=0: mismo shape que gradientes para acumular 1er momento
            self.v[layer_id] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}  # v=0: mismo shape que gradientes para acumular 2do momento

        self.t += 1  # Incrementa paso: importante para corrección de bias; si no se incrementa, m_hat/v_hat quedan mal escalados
        m, v = self.m[layer_id], self.v[layer_id]  # Recupera buffers del layer: referencias mutables para actualizar in-place

        # Actualizar momentos
        m['W'] = self.beta1 * m['W'] + (1 - self.beta1) * dW  # 1er momento (W): EMA del gradiente; suaviza ruido y acelera en direcciones consistentes
        m['b'] = self.beta1 * m['b'] + (1 - self.beta1) * db  # 1er momento (b): mismo cálculo para bias
        v['W'] = self.beta2 * v['W'] + (1 - self.beta2) * dW**2  # 2do momento (W): EMA de gradiente^2; aproxima varianza para normalizar paso
        v['b'] = self.beta2 * v['b'] + (1 - self.beta2) * db**2  # 2do momento (b): mismo cálculo para bias

        # Corrección de bias
        m_hat_W = m['W'] / (1 - self.beta1**self.t)  # Corrige bias en m_W: al inicio m está sesgado hacia 0 por inicialización en cero
        m_hat_b = m['b'] / (1 - self.beta1**self.t)  # Corrige bias en m_b: misma idea para bias
        v_hat_W = v['W'] / (1 - self.beta2**self.t)  # Corrige bias en v_W: evita subestimar magnitud al principio
        v_hat_b = v['b'] / (1 - self.beta2**self.t)  # Corrige bias en v_b: misma idea para bias

        # Actualizar parámetros
        layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)  # Update W: paso adaptativo por coordenada (divide por RMS) + ε para estabilidad
        layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)  # Update b: mismo update para bias; requiere que `np` esté en el namespace
```

---

## 🎯 Ejercicios por tema (progresivos) + Soluciones

Reglas:

- **Intenta primero** sin mirar la solución.
- **Timebox sugerido:** 30–75 min por ejercicio.
- **Éxito mínimo:** tu solución debe pasar los `assert`.

---

### Ejercicio 7.1: Activaciones y derivadas (chequeo numérico)

#### Enunciado

1) **Básico**

- Implementa `sigmoid(z)` y `relu(z)`.

2) **Intermedio**

- Implementa derivadas: `sigmoid'(z)` y `relu'(z)`.

3) **Avanzado**

- Verifica `sigmoid'(z)` con diferencias finitas centrales.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para asarray/clip/exp/maximum y para generar datos aleatorios del chequeo numérico

def sigmoid(z: np.ndarray) -> np.ndarray:  # Sigmoide: transforma logits reales en valores (0,1) de forma elemento-a-elemento
    z = np.asarray(z, dtype=float)  # Normaliza entrada a ndarray float: evita dtype entero y garantiza operaciones vectorizadas estables
    z = np.clip(z, -500, 500)  # Recorta extremos para evitar overflow/underflow en exp(-z) cuando |z| es grande
    return 1.0 / (1.0 + np.exp(-z))  # Calcula σ(z)=1/(1+e^{-z}); devuelve array con misma shape que z


def sigmoid_deriv(z: np.ndarray) -> np.ndarray:  # Derivada de sigmoide respecto a z: necesaria para backprop cuando activación es sigmoid
    a = sigmoid(z)  # Reutiliza la salida de sigmoid: permite usar identidad σ'(z)=σ(z)(1-σ(z)) sin recomputar exp manualmente
    return a * (1.0 - a)  # Calcula σ'(z): si esto está mal, el gradiente tendrá signo/magnitud erróneos y el entrenamiento fallará


def relu(z: np.ndarray) -> np.ndarray:  # ReLU: pone a 0 los valores negativos y deja pasar los positivos; estándar en capas ocultas
    return np.maximum(0.0, np.asarray(z, dtype=float))  # Convierte a float y aplica max(0,z) vectorizado; devuelve misma shape que z


def relu_deriv(z: np.ndarray) -> np.ndarray:  # Derivada de ReLU: máscara binaria (1 donde z>0, 0 donde z<=0)
    z = np.asarray(z, dtype=float)  # Normaliza z: asegura comparación numérica consistente y broadcasting esperado
    return (z > 0.0).astype(float)  # Convierte booleano a float: produce gradiente 1/0 para multiplicar en backprop


def num_derivative(f, z: np.ndarray, h: float = 1e-6) -> np.ndarray:  # Derivada numérica central: aproxima f'(z) usando diferencias finitas
    return (f(z + h) - f(z - h)) / (2.0 * h)  # Fórmula central: (f(z+h)-f(z-h))/(2h); más precisa que forward diff pero sensible a h


np.random.seed(0)  # Fija semilla global: hace reproducible el vector de prueba `z` y, por tanto, el resultado del test
z = np.random.randn(10)  # Genera 10 valores gaussianos: sirve como input genérico para comparar derivada numérica vs analítica
g_num = num_derivative(sigmoid, z)  # Calcula derivada numérica de sigmoid en z: referencia “aproximada” para validar implementación
g_ana = sigmoid_deriv(z)  # Calcula derivada analítica implementada: debe coincidir con la numérica dentro de tolerancias
assert np.allclose(g_num, g_ana, rtol=1e-5, atol=1e-6)  # Sanity check: si falla, hay bug en sigmoid_deriv o inestabilidad numérica
```

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 7.1: Activaciones y derivadas (chequeo numérico)</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M07-E07_1`
- **Duración estimada:** 20–45 min
- **Nivel:** Intermedio

#### 2) Objetivos
- Entender la diferencia entre **activación** `f(z)` y **derivada** `f'(z)`.
- Validar una derivada con **diferencias finitas centrales**.

#### 3) Errores comunes
- Usar diferencias hacia delante (más error) en lugar de centrales.
- Elegir `h` demasiado grande (sesgo) o demasiado pequeño (error numérico).
- No “clipear” `z` en sigmoid y obtener `inf/NaN`.

#### 4) Nota docente
- Pide que el alumno explique por qué el chequeo numérico es una prueba de sanidad (no una demostración formal).
</details>

---

### Ejercicio 7.2: Forward de una capa densa (batch) + shapes

#### Enunciado

1) **Básico**

- Implementa `dense_forward(X, W, b)` con `X:(n,d_in)`, `W:(d_in,d_out)`, `b:(d_out,)`.

2) **Intermedio**

- Verifica shapes de salida `Z:(n,d_out)`.

3) **Avanzado**

- Verifica que coincide con una implementación con loop (para un caso pequeño).

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para RNG, matrices, producto @, zeros_like y comparaciones numéricas (allclose)

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:  # Forward de capa densa: Z = XW + b (batch-first)
    return X @ W + b  # Multiplica (n,d_in)@(d_in,d_out)->(n,d_out) y suma bias (d_out,) por broadcasting en el eje batch


np.random.seed(1)  # Fija semilla global: hace reproducibles los datos/parametrización del ejemplo
n, d_in, d_out = 5, 3, 4  # Define shapes: batch=5, input_dim=3, output_dim=4 (contrato básico de capa densa)
X = np.random.randn(n, d_in)  # Genera batch de entradas: shape (n,d_in)
W = np.random.randn(d_in, d_out)  # Genera matriz de pesos: shape (d_in,d_out) compatible con X @ W
b = np.random.randn(d_out)  # Genera bias: shape (d_out,) se sumará a cada fila de Z vía broadcasting

Z = dense_forward(X, W, b)  # Calcula salida vectorizada: referencia “correcta” (sin loops explícitos)
assert Z.shape == (n, d_out)  # Invariante de shape: si falla, hay error en el contrato de dimensiones o en el broadcasting de b

Z_loop = np.zeros_like(Z)  # Inicializa buffer para versión con loop: mismo shape/dtype que Z para comparar resultados
for i in range(n):  # Recorre cada ejemplo del batch: implementa el mismo cálculo pero de forma escalar por fila
    Z_loop[i] = X[i] @ W + b  # Calcula Z para la fila i: (d_in,)@(d_in,d_out)->(d_out,) y suma bias

assert np.allclose(Z, Z_loop)  # Sanity check: versión vectorizada y versión con loop deben coincidir (tolerancias numéricas)
```

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 7.2: Forward denso (batch) y contratos de shape</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M07-E07_2`
- **Duración estimada:** 20–40 min
- **Nivel:** Intermedio

#### 2) Idea clave
- En convención batch-first, `X @ W + b` responde a:
  - `X:(n,d_in)`, `W:(d_in,d_out)`, `b:(d_out,)` → `Z:(n,d_out)`.

#### 3) Errores comunes
- Poner `W` como `(d_out,d_in)` y luego forzar traspuestas por “arreglo rápido”.
- Confundir `axis` al sumar bias (debe broadcast a la segunda dimensión).

#### 4) Nota docente
- Pide que el alumno escriba los shapes de memoria antes de correr el código.
</details>

---

### Ejercicio 7.3: Softmax estable + Cross-Entropy (multiclase)

#### Enunciado

1) **Básico**

- Implementa `logsumexp` y `softmax` estable.

2) **Intermedio**

- Implementa `categorical_cross_entropy` para `y_true` one-hot.

3) **Avanzado**

- Verifica:
  - `softmax(z)` suma 1.
  - CCE baja cuando aumenta la probabilidad de la clase correcta.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para operaciones vectorizadas (max/sum/exp/log), conversión a arrays y asserts numéricos

def logsumexp(z: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:  # Calcula log(sum(exp(z))) de forma estable (evita overflow) a lo largo de `axis`
    z = np.asarray(z, dtype=float)  # Convierte a ndarray float: normaliza el tipo y asegura que exp/log funcionen con precisión estable
    m = np.max(z, axis=axis, keepdims=True)  # Extrae el máximo por eje: se usa para “centrar” logits sin cambiar el resultado (invariante por suma)
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))  # Implementa identidad estable: logsumexp(z)=m+log(sum(exp(z-m)))
    return out if keepdims else np.squeeze(out, axis=axis)  # Mantiene o elimina dimensión reducida: `keepdims` controla broadcasting posterior


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:  # Calcula softmax estable: devuelve probabilidades que suman 1 a lo largo de `axis`
    z = np.asarray(z, dtype=float)  # Normaliza logits a ndarray float: evita sorpresas de dtype/broadcasting
    lse = logsumexp(z, axis=axis, keepdims=True)  # Calcula logsumexp estable con keepdims: permite restar con broadcasting (misma rank)
    return np.exp(z - lse)  # softmax(z)=exp(z-logsumexp(z)): estable y garantiza normalización (sum=1) salvo error numérico mínimo


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:  # CCE para one-hot: penaliza baja prob. asignada a la clase correcta
    y_true = np.asarray(y_true, dtype=float)  # Convierte labels one-hot a float: asegura multiplicación/log coherentes
    y_pred = np.asarray(y_pred, dtype=float)  # Convierte predicciones a float: deben ser probabilidades (o aproximación) por clase
    y_pred = np.clip(y_pred, eps, 1.0)  # Evita log(0): recorta p en [eps,1]; sin esto, la loss puede volverse inf/nan
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))  # CCE=-E[sum_k y_k log p_k]; con one-hot queda -E[log p_clase]


z = np.array([[10.0, 0.0, -10.0]])  # Logits de ejemplo (1x3): gran separación para probar estabilidad y ranking de probabilidades
p = softmax(z)  # Convierte logits a probabilidades: debe asignar casi toda la masa a la clase de logit máximo (10.0)
assert np.isclose(np.sum(p), 1.0)  # Invariante softmax: las probabilidades suman ~1 (tolerancia numérica)
assert np.argmax(p) == 0  # Invariante de ranking: la clase 0 tiene el mayor logit, por lo tanto debe tener la mayor probabilidad

y_true = np.array([[1.0, 0.0, 0.0]])  # Target one-hot: la clase correcta es la 0 (prob=1 en índice 0)
loss_good = categorical_cross_entropy(y_true, np.array([[0.9, 0.05, 0.05]]))  # Caso “bueno”: alta prob. en clase correcta -> loss baja
loss_bad = categorical_cross_entropy(y_true, np.array([[0.4, 0.3, 0.3]]))  # Caso “malo”: menos masa en clase correcta -> loss más alta
assert loss_good < loss_bad  # Sanity check: CCE debe penalizar más cuando baja la probabilidad de la clase correcta
```

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 7.3: Softmax estable + Cross-Entropy</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M07-E07_3`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio

#### 2) Idea clave
- Estabilidad: `softmax(z) = exp(z - logsumexp(z))` evita overflow.
- Para clasificación, lo importante es **comparar probabilidades** sin caer en `NaN`.

#### 3) Errores comunes
- Hacer `exp(z)` directamente con logits grandes.
- Olvidar `eps` al hacer `log(y_pred)`.
- Confundir CCE para `y_true` one-hot con BCE binaria.

#### 4) Nota docente
- Pide que el alumno explique por qué restar el máximo no cambia el resultado de softmax.
</details>

---

### Ejercicio 7.4: Backprop de 2 capas (gradiente) + gradient checking

#### Enunciado

Red (batch):

- `Z1 = XW1 + b1`, `A1 = relu(Z1)`
- `Z2 = A1W2 + b2`, `P = sigmoid(Z2)`
- Loss BCE: `L = -mean(y log(P) + (1-y) log(1-P))`

1) **Básico**

- Implementa forward + loss.

2) **Intermedio**

- Implementa backward: gradientes `dW1, db1, dW2, db2`.

3) **Avanzado**

- Verifica una coordenada de `dW2` con diferencias centrales.

#### Solución

```python
import numpy as np  # Importa NumPy: operaciones vectorizadas para forward/backward y generación de datos

def sigmoid(z: np.ndarray) -> np.ndarray:  # Sigmoid estable para salida binaria/probabilidades
    z = np.clip(z, -500, 500)  # Clipping: evita overflow en exp para |z| grande
    return 1.0 / (1.0 + np.exp(-z))  # σ(z)=1/(1+e^-z)


def relu(z: np.ndarray) -> np.ndarray:  # ReLU: no linealidad común en capas ocultas
    return np.maximum(0.0, z)  # Aplica max(0,z) elemento a elemento


def relu_deriv(z: np.ndarray) -> np.ndarray:  # Derivada de ReLU (subgradiente)
    return (z > 0.0).astype(float)  # 1 si z>0, 0 si z<=0


def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:  # Binary Cross-Entropy para targets {0,1}
    p = np.clip(p, eps, 1.0 - eps)  # Clipping: evita log(0) que produce inf/NaN
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))  # BCE media sobre el batch


def forward(X, W1, b1, W2, b2):  # Forward de red 2-capas: lineal+ReLU y lineal+sigmoid
    Z1 = X @ W1 + b1  # Pre-activación 1: (n,d_in)@(d_in,d_h)+(d_h,) -> (n,d_h)
    A1 = relu(Z1)  # Activación oculta: aplica ReLU
    Z2 = A1 @ W2 + b2  # Pre-activación 2: (n,d_h)@(d_h,1)+(1,) -> (n,1)
    P = sigmoid(Z2)  # Probabilidad predicha: sigmoid sobre logits
    cache = (X, Z1, A1, Z2, P)  # Cachea tensores para backward sin recomputar
    return P, cache  # Devuelve predicción y cache para backprop


def loss_fn(X, y, W1, b1, W2, b2):  # Función de pérdida: forward + BCE
    P, _ = forward(X, W1, b1, W2, b2)  # Calcula probabilidades con forward
    return bce(y, P)  # Evalúa BCE sobre el batch


def backward(y, cache, W2):  # Backward de la red: calcula gradientes de W1/b1/W2/b2
    X, Z1, A1, Z2, P = cache  # Desempaqueta cache: variables del forward
    n = X.shape[0]  # Tamaño de batch: normaliza gradientes (media)
    # BCE with sigmoid output: dZ2 = (P - y) / n
    dZ2 = (P - y) / n  # Para BCE+sigmoid: dZ2=(P-y)/n (batch mean)
    dW2 = A1.T @ dZ2  # Gradiente W2: (d_h,n)@(n,1) -> (d_h,1)
    db2 = np.sum(dZ2, axis=0)  # Gradiente b2: suma sobre batch -> (1,)
    dA1 = dZ2 @ W2.T  # Propaga a activación oculta: (n,1)@(1,d_h) -> (n,d_h)
    dZ1 = dA1 * relu_deriv(Z1)  # Aplica derivada ReLU: enmascara gradiente donde Z1<=0
    dW1 = X.T @ dZ1  # Gradiente W1: (d_in,n)@(n,d_h) -> (d_in,d_h)
    db1 = np.sum(dZ1, axis=0)  # Gradiente b1: suma sobre batch -> (d_h,)
    return dW1, db1, dW2, db2  # Devuelve gradientes para actualización/chequeo


np.random.seed(0)  # Semilla fija: reproducibilidad del grad-check
n, d_in, d_h = 8, 3, 5  # Dimensiones: batch=8, input=3, hidden=5
X = np.random.randn(n, d_in)  # Datos de entrada aleatorios: shape (n,d_in)
y = (np.random.rand(n, 1) < 0.5).astype(float)  # Labels binarios aleatorios: shape (n,1)
W1 = np.random.randn(d_in, d_h) * 0.1  # Pesos 1: init pequeño para estabilidad
b1 = np.zeros(d_h)  # Bias 1: vector (d_h,)
W2 = np.random.randn(d_h, 1) * 0.1  # Pesos 2: shape (d_h,1)
b2 = np.zeros(1)  # Bias 2: vector (1,)

P, cache = forward(X, W1, b1, W2, b2)  # Forward: obtiene probabilidades y cache
dW1, db1, dW2, db2 = backward(y, cache, W2)  # Backward: calcula gradientes analíticos

# Gradient check on one W2 coordinate
i, j = 2, 0  # Coordenada de W2 a chequear: índice (fila,col)
h = 1e-6  # Paso pequeño para diferencias finitas centrales
E = np.zeros_like(W2)  # Matriz base para perturbar una coordenada de W2
E[i, j] = 1.0  # Marca la coordenada (i,j) a perturbar
L_plus = loss_fn(X, y, W1, b1, W2 + h * E, b2)  # Loss con W2(i,j)+h
L_minus = loss_fn(X, y, W1, b1, W2 - h * E, b2)  # Loss con W2(i,j)-h
g_num = (L_plus - L_minus) / (2.0 * h)  # Gradiente numérico (diferencia central)
assert np.isclose(dW2[i, j], g_num, rtol=1e-4, atol=1e-6)  # Verifica gradiente analítico vs numérico
```

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 7.4: Backprop + gradient checking</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M07-E07_4`
- **Duración estimada:** 60–120 min
- **Nivel:** Avanzado

#### 2) Invariante principal
- Para `Z = XW + b` (batch-first):
  - `dW` tiene shape de `W`, `db` de `b`, `dX` de `X`.

#### 3) Gradient checking (mínimo viable)
- Chequea 1 coordenada (o pocas) de un gradiente grande (`dW2`) con diferencias centrales.
- Ajusta `h` y tolerancias si estás en float64 vs float32.

#### 4) Errores comunes
- Olvidar dividir por `n` en la loss (o en `dZ2`) y “mover” el bug de lugar.
- Mezclar `y` como `(n,)` con `P` como `(n,1)`.

#### 5) Nota docente
- Pide que el alumno explique por qué un único chequeo no garantiza que TODO el gradiente esté correcto.
</details>

---

### Ejercicio 7.5: Overfit test (sanity check obligatorio)

#### Enunciado

1) **Básico**

- Construye un dataset tiny (8–16 ejemplos) linealmente separable.

2) **Intermedio**

- Entrena Logistic Regression (GD) y verifica que la pérdida baja.

3) **Avanzado**

- Verifica que logra accuracy alta (por ejemplo, > 95%).

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para vectores/matrices, RNG, funciones exp/log y operaciones de álgebra lineal

def sigmoid(z: np.ndarray) -> np.ndarray:  # Define sigmoide: mapea logits reales a probabilidades en (0,1) elemento-a-elemento
    z = np.clip(z, -500, 500)  # Recorta logits para evitar overflow/underflow numérico en exp(-z) cuando |z| es grande
    return 1.0 / (1.0 + np.exp(-z))  # Calcula σ(z)=1/(1+e^{-z}); produce salida con misma shape que z


def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:  # Define Binary Cross-Entropy: mide discrepancia entre labels y probabilidades (más baja es mejor)
    p = np.clip(p, eps, 1.0 - eps)  # Estabiliza logs evitando log(0): fuerza p a (eps,1-eps) para evitar inf/nan
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))  # Promedia BCE por ejemplo y devuelve escalar Python (float)


np.random.seed(1)  # Semilla fija: reproducibilidad del overfit test (mismo dataset)
n = 16  # Tamaño del dataset tiny: 16 ejemplos (8 positivos, 8 negativos)
X_pos = np.random.randn(n // 2, 2) + np.array([2.0, 2.0])  # Clase positiva: gaussiana centrada en (2,2)
X_neg = np.random.randn(n // 2, 2) + np.array([-2.0, -2.0])  # Clase negativa: gaussiana centrada en (-2,-2)
X = np.vstack([X_pos, X_neg])  # Concatena ejemplos: shape (n,2)
y = np.vstack([np.ones((n // 2, 1)), np.zeros((n // 2, 1))])  # Labels: 1 para pos, 0 para neg (shape (n,1))

w = np.zeros((2, 1))  # Pesos de logistic regression: shape (2,1)
b = 0.0  # Bias escalar
lr = 0.2  # Learning rate: suficientemente alto para converger en pocas iteraciones

loss0 = None  # Guarda la loss inicial (t=0) para comparar progreso
for t in range(400):  # Loop de entrenamiento: gradient descent batch para logistic regression
    logits = X @ w + b  # Logits: (n,2)@(2,1)+(scalar) -> (n,1)
    p = sigmoid(logits)  # Probabilidades predichas: shape (n,1)
    loss = bce(y, p)  # Loss actual: BCE sobre dataset
    if loss0 is None:  # Captura loss inicial una sola vez
        loss0 = loss  # Guarda baseline para validar que al final disminuye
    # gradients
    dz = (p - y) / n  # Gradiente wrt logits: (p-y)/n para BCE+sigmoid
    dw = X.T @ dz  # Gradiente wrt w: (2,n)@(n,1) -> (2,1)
    db = float(np.sum(dz))  # Gradiente wrt b: suma sobre batch (escalar)
    w -= lr * dw  # Update w: descenso por gradiente
    b -= lr * db  # Update b: descenso por gradiente

loss_end = bce(y, sigmoid(X @ w + b))  # Loss final: debería ser <= loss0 si aprende
pred = (sigmoid(X @ w + b) >= 0.5).astype(int)  # Predicción binaria: umbral 0.5 sobre probabilidad
acc = float(np.mean(pred == y.astype(int)))  # Accuracy final: proporción de aciertos

assert loss_end <= loss0  # Invariante: la optimización debe reducir la pérdida
assert acc > 0.95  # Invariante: en dataset separable debería lograr alta accuracy
```

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 7.5: Overfit test (sanity check)</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M07-E07_5`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio

#### 2) Regla de oro
- Si tu modelo no puede **memorizar** un dataset tiny, asume bug (no “mala suerte”).

#### 3) Errores comunes
- Learning rate demasiado bajo (parece bug, pero sólo no se mueve).
- Dataset no separable o etiquetas con shape inconsistente.
- Error en el gradiente (signo, normalización por `n`, broadcasting de `b`).

#### 4) Nota docente
- Pide que el alumno haga el mismo test con 2–3 seeds y compare estabilidad.
</details>

---

### Ejercicio 7.6: Optimizadores en una función cuadrática (SGD vs Adam)

#### Enunciado

Minimiza `f(w) = (w - 3)^2`.

1) **Básico**

- Implementa SGD.

2) **Intermedio**

- Implementa Adam.

3) **Avanzado**

- Verifica que ambos se acercan a `w≈3` y que Adam no diverge.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para sqrt y para mantener consistencia numérica en Adam (np.sqrt)

def grad_f(w: float) -> float:  # Define el gradiente de f(w)=(w-3)^2: derivada analítica para usar en SGD/Adam
    return 2.0 * (w - 3.0)  # d/dw (w-3)^2 = 2(w-3): si esto estuviera mal, el optimizador convergería al punto equivocado


def sgd(w0: float, lr: float, steps: int) -> float:  # Implementa SGD 1D: aplica descenso por gradiente con paso constante
    w = float(w0)  # Convierte el inicial a float nativo: garantiza aritmética escalar y evita tipos raros (p.ej., np scalar)
    for _ in range(steps):  # Itera un número fijo de pasos: cada iteración aplica una actualización usando el gradiente actual
        w -= lr * grad_f(w)  # Update SGD: w <- w - lr * g(w); el signo es crítico (si fuera +, diverge)
    return w  # Devuelve el w final: aproximación al mínimo (idealmente cercano a 3)


def adam(w0: float, lr: float, steps: int, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> float:  # Implementa Adam 1D: momentos + normalización por RMS con corrección de bias
    w = float(w0)  # Estado del parámetro: se actualiza in-place en cada paso de optimización
    m = 0.0  # Primer momento (EMA del gradiente): actúa como momentum, suavizando ruido
    v = 0.0  # Segundo momento (EMA del gradiente^2): estima escala/varianza para ajustar el paso
    t = 0  # Contador de tiempo: necesario para corrección de bias (1 - beta^t) en los primeros pasos
    for _ in range(steps):  # Ejecuta N pasos: en problemas reales, esto sería por batch/iteración de entrenamiento
        t += 1  # Avanza el tiempo: si se omite, m_hat/v_hat quedan mal corregidos y el paso se sesga
        g = grad_f(w)  # Calcula gradiente actual en w: dirección local de máxima subida (queremos bajar)
        m = beta1 * m + (1 - beta1) * g  # Actualiza 1er momento: EMA del gradiente (promedio con decaimiento)
        v = beta2 * v + (1 - beta2) * (g ** 2)  # Actualiza 2do momento: EMA de g^2 (magnitud típica del gradiente)
        m_hat = m / (1 - beta1 ** t)  # Corrección de bias en m: compensa inicialización en cero, importante cuando t es pequeño
        v_hat = v / (1 - beta2 ** t)  # Corrección de bias en v: evita subestimar la escala del gradiente al inicio
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)  # Update Adam: paso adaptativo por RMS; eps evita división por cero/inestabilidad
    return w  # Devuelve el w final: debería acercarse al mínimo en w=3 si el update está bien implementado


w_sgd = sgd(w0=10.0, lr=0.1, steps=50)  # Ejecuta SGD desde w0=10: espera converger hacia 3 con lr moderado
w_adam = adam(w0=10.0, lr=0.2, steps=50)  # Ejecuta Adam desde w0=10: suele tolerar lr mayor por normalización adaptativa

assert abs(w_sgd - 3.0) < 1e-2  # Sanity check: SGD debe terminar suficientemente cerca del óptimo w=3
assert abs(w_adam - 3.0) < 1e-2  # Sanity check: Adam también debe converger; si falla, hay bug en momentos/corrección/update
```

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 7.6: SGD vs Adam (intuición)</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M07-E07_6`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio

#### 2) Idea clave
- SGD usa el gradiente “tal cual”.
- Adam introduce momentos (media y varianza) y suele ser más estable en problemas mal condicionados.

#### 3) Errores comunes
- Olvidar corrección de bias (`m_hat`, `v_hat`).
- Elegir `lr` de Adam igual que el de SGD sin validar.

#### 4) Nota docente
- Pide que el alumno grafique `w_t` para comparar trayectorias.
</details>

---

### Ejercicio 7.7: Gradient clipping (evitar exploding gradients)

#### Enunciado

1) **Básico**

- Implementa clipping por norma: si `||g|| > max_norm`, entonces `g <- g * (max_norm/||g||)`.

2) **Intermedio**

- Verifica que tras clipping la norma es `<= max_norm`.

3) **Avanzado**

- Verifica que si la norma ya es pequeña, el gradiente no cambia.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para asarray, norma L2 (linalg.norm), arrays de prueba y allclose

def clip_by_norm(g: np.ndarray, max_norm: float) -> np.ndarray:  # Clipping por norma: re-escala g para que ||g|| <= max_norm (si excede)
    g = np.asarray(g, dtype=float)  # Normaliza entrada a ndarray float: asegura que la norma y escalado sean numéricamente consistentes
    n = np.linalg.norm(g)  # Calcula norma L2: mide magnitud global del gradiente (no por componente)
    if n == 0.0:  # Caso borde: gradiente cero (no hay dirección de descenso);
        return g  # Retorna sin cambio: evita división por cero y preserva semántica (0 sigue siendo 0)
    if n <= max_norm:  # Si ya está bajo el umbral, no se debe tocar (evita introducir sesgo innecesario)
        return g  # Retorna el gradiente original: clipping sólo actúa cuando hay riesgo de pasos gigantes
    return g * (max_norm / n)  # Re-escala manteniendo dirección: multiplica por factor <1 para que la nueva norma sea exactamente max_norm


g_big = np.array([3.0, 4.0])  # norm=5
g_clip = clip_by_norm(g_big, max_norm=1.0)  # Aplica clipping: al ser ||g||=5>1, el resultado debe tener norma ~1
assert np.linalg.norm(g_clip) <= 1.0 + 1e-12  # Verifica invariante: tras clipping, la norma no debe exceder el umbral (con tolerancia)

g_small = np.array([0.3, 0.4])  # norm=0.5
g_keep = clip_by_norm(g_small, max_norm=1.0)  # Aplica clipping: como ||g||=0.5<=1, no debe modificar el gradiente
assert np.allclose(g_small, g_keep)  # Verifica que no hay cambio numérico: clipping no debe afectar gradientes ya pequeños
```

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 7.7: Gradient clipping</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M07-E07_7`
- **Duración estimada:** 20–45 min
- **Nivel:** Intermedio

#### 2) Idea clave
- Clipping por norma no “arregla” el gradiente: sólo evita pasos gigantes.

#### 3) Errores comunes
- Hacer clipping por componente (otra técnica) pensando que es lo mismo.
- No manejar el caso `||g||=0`.

#### 4) Nota docente
- Pide que el alumno explique por qué clipping puede estabilizar RNN/transformers (conceptual).
</details>

---

### Ejercicio 7.8: Convolución - cálculo de output shape (padding/stride)

#### Enunciado

1) **Básico**

- Implementa `conv2d_out(H, W, KH, KW, stride, padding)` para una conv sin dilatación.

2) **Intermedio**

- Verifica el caso MNIST: `28x28` con kernel `5x5`, `stride=1`, `padding=0` → `24x24`.

3) **Avanzado**

- Verifica un caso con padding: `28x28`, `5x5`, `stride=1`, `padding=2` → `28x28`.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para validación numérica en asserts y para mantener consistencia con el resto del módulo

def conv2d_out(H: int, W: int, KH: int, KW: int, stride: int = 1, padding: int = 0):  # Output shape (sin dilatación): fórmula estándar de conv para cada eje
    H_out = (H + 2 * padding - KH) // stride + 1  # Altura de salida: floor((H+2P-KH)/S)+1
    W_out = (W + 2 * padding - KW) // stride + 1  # Ancho de salida: floor((W+2P-KW)/S)+1
    return int(H_out), int(W_out)  # Devuelve (H_out,W_out): se usa para asserts y para dimensionar tensores

assert conv2d_out(28, 28, 5, 5, stride=1, padding=0) == (24, 24)  # Caso MNIST sin padding: 28-5+1=24
assert conv2d_out(28, 28, 5, 5, stride=1, padding=2) == (28, 28)  # Caso con padding=2 (aprox “same” para KH=5): mantiene 28
```

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 7.8: Output shape de conv (stride/padding)</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M07-E07_8`
- **Duración estimada:** 20–45 min
- **Nivel:** Intermedio

#### 2) Idea clave
- Fórmula sin dilatación: `H_out = (H + 2P - KH)//S + 1` (igual para `W_out`).
- Si no cuadra, normalmente el error está en `padding` o en entero vs float.

#### 3) Errores comunes
- Olvidar que `padding` aplica a ambos lados (por eso `2P`).
- Usar `/` en vez de `//` y obtener floats.

#### 4) Nota docente
- Pide que el alumno derive la fórmula a partir de “cuántas posiciones cabe el kernel”.
</details>

---

## 📦 Entregable del Módulo

### `neural_network.py`

```python
"""
Neural Network Module

Implementación desde cero de:
- MLP (Multilayer Perceptron)
- Backpropagation
- Optimizadores (SGD, Momentum, Adam)
- Funciones de activación

Autor: [Tu nombre]
Módulo: 06 - Deep Learning
"""

import numpy as np  # Importa NumPy: base de operaciones vectorizadas (matmul, exp, clip, etc.)
from typing import List, Tuple, Optional  # Importa tipos: documentación estática de firmas, no afecta runtime


# ============================================================
# ACTIVACIONES
# ============================================================

def sigmoid(z):  # Sigmoide: convierte logits a (0,1) aplicando una no linealidad suave (se usa típicamente en salida binaria)
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid estable: clip evita overflow en exp para |z| grande

def sigmoid_deriv(a):  # Derivada de sigmoide en función de la activación: útil en backprop para obtener da/dz sin recomputar exp
    return a * (1 - a)  # Derivada de sigmoid en función de la activación: σ'(z)=a(1-a)

def relu(z):  # ReLU: activa solo valores positivos (max(0,z)); estándar en capas ocultas por estabilidad de gradiente
    return np.maximum(0, z)  # ReLU: pasa valores positivos y anula negativos (no linealidad)

def relu_deriv(z):  # Derivada de ReLU: máscara 1/0 según z>0; controla por dónde fluye el gradiente en backprop
    return (z > 0).astype(float)  # Derivada de ReLU: 1 si z>0, 0 si z<=0 (subgradiente)

def tanh_deriv(a):  # Derivada de tanh en función de la activación: tanh'(z)=1-a^2, usada en backprop
    return 1 - a**2  # Derivada de tanh en función de la activación: tanh'(z)=1-a^2

def softmax(z):  # Softmax: normaliza logits a distribución (suma 1); se usa en salida multiclase
    exp_z = np.exp(z - np.max(z))  # Softmax (estabilizado restando max): reduce overflow en exp
    return exp_z / np.sum(exp_z)  # Normaliza para que la suma sea 1 (distribución de probabilidad)


# ============================================================
# CAPA
# ============================================================

class Layer:  # Define una capa densa simple: aplica W@x+b seguido de una activación y guarda cache para backward
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):  # Inicializa pesos/bias y configura activación
        self.activation = activation  # Guarda nombre de activación: define forward/backward de la capa
        scale = np.sqrt(2.0 / input_size) if activation == 'relu' else np.sqrt(1.0 / input_size)  # He para ReLU, Xavier simple para resto
        self.W = np.random.randn(output_size, input_size) * scale  # Pesos: (out,in) escalados para estabilidad inicial
        self.b = np.zeros(output_size)  # Bias: vector (out,) inicializado a cero
        self.cache = {}  # Cache: guarda x/z/a del forward para usar en backward sin recomputar

    def forward(self, x: np.ndarray) -> np.ndarray:  # Forward: computa z=W@x+b y aplica la activación
        self.cache['x'] = x  # Guarda input: se necesita para dW en backward (outer product con delta)
        z = self.W @ x + self.b  # Pre-activación: combinación lineal (out,in)@(in,) + (out,) -> (out,)
        self.cache['z'] = z  # Guarda z: derivada depende de z (ReLU) o se usa para depuración

        if self.activation == 'relu':  # Rama ReLU: típica en capas ocultas
            a = relu(z)  # Aplica ReLU elemento a elemento
        elif self.activation == 'sigmoid':  # Rama sigmoid: típica en salida binaria
            a = sigmoid(z)  # Convierte logits a probabilidad (0,1)
        elif self.activation == 'tanh':  # Rama tanh: no linealidad centrada en 0
            a = np.tanh(z)  # Aplica tanh elemento a elemento
        elif self.activation == 'softmax':  # Rama softmax: salida multiclase como distribución
            a = softmax(z)  # Normaliza logits a probabilidades
        else:  # Rama lineal/identidad: sin no linealidad
            a = z  # Identidad: útil en regresión o como logits antes de softmax externa

        self.cache['a'] = a  # Guarda activación: se usa en backward (sigmoid/tanh derivan de a)
        return a  # Devuelve salida de la capa para alimentar la siguiente

    def backward(self, dL_da: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Backward: propaga gradiente y produce dW/db
        z, x, a = self.cache['z'], self.cache['x'], self.cache['a']  # Recupera forward cacheado para calcular derivadas

        if self.activation == 'sigmoid':  # Derivada de sigmoid se expresa con la activación a
            da_dz = sigmoid_deriv(a)  # da/dz para sigmoid
        elif self.activation == 'relu':  # Derivada de ReLU depende de z (signo)
            da_dz = relu_deriv(z)  # da/dz para ReLU
        elif self.activation == 'tanh':  # Derivada de tanh se expresa con la activación a
            da_dz = tanh_deriv(a)  # da/dz para tanh
        else:  # Activación lineal: derivada 1
            da_dz = np.ones_like(z)  # da/dz=1 para identidad (misma shape que z)

        delta = dL_da * da_dz  # Regla de la cadena: dL/dz = dL/da * da/dz (elementwise)
        dL_dW = np.outer(delta, x)  # Gradiente de W: outer(delta(out,), x(in,)) -> (out,in)
        dL_db = delta  # Gradiente de b: dL/db = dL/dz (por neurona), sin suma porque es single-sample
        dL_dx = self.W.T @ delta  # Gradiente hacia atrás: (in,out)@(out,) -> (in,)

        return dL_dx, dL_dW, dL_db  # Devuelve gradientes: input, pesos, bias (para propagación y optimización)


# ============================================================
# OPTIMIZADORES
# ============================================================

class SGD:  # Optimizer SGD “vanilla”: actualiza parámetros restando lr * gradiente en cada paso
    def __init__(self, lr=0.01):  # Constructor SGD: fija la tasa de aprendizaje
        self.lr = lr  # Guarda learning rate: escala del update en cada step

    def step(self, layers, gradients):  # Aplica un paso de SGD a una lista de capas
        for layer, (dW, db) in zip(layers, gradients):  # Recorre capas y sus gradientes alineados
            layer.W -= self.lr * dW  # Update SGD: W <- W - lr * dW
            layer.b -= self.lr * db  # Update SGD: b <- b - lr * db


class Adam:  # Optimizer Adam: mantiene momentos (m,v) y aplica corrección de bias para updates adaptativos por parámetro
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):  # Inicializa hiperparámetros de Adam
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps  # Guarda lr y decays de momentos + epsilon numérico
        self.m, self.v, self.t = {}, {}, 0  # Estado por capa: momentos m/v y contador de pasos t

    def step(self, layers, gradients):  # Paso Adam: actualiza cada capa con momentos y bias correction
        self.t += 1  # Incrementa timestep: necesario para corrección de sesgo (bias correction)
        for i, (layer, (dW, db)) in enumerate(zip(layers, gradients)):  # Itera capas con índice para almacenar estado
            if i not in self.m:  # Inicializa estado si es la primera vez que se ve esta capa
                self.m[i] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}  # m: primer momento (media móvil del gradiente)
                self.v[i] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}  # v: segundo momento (media móvil del gradiente^2)

            self.m[i]['W'] = self.beta1 * self.m[i]['W'] + (1 - self.beta1) * dW  # Actualiza m(W): EMA del gradiente
            self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * db  # Actualiza m(b): EMA del gradiente
            self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1 - self.beta2) * dW**2  # Actualiza v(W): EMA del gradiente^2
            self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * db**2  # Actualiza v(b): EMA del gradiente^2

            m_hat_W = self.m[i]['W'] / (1 - self.beta1**self.t)  # Bias correction de m(W): corrige arranque en 0
            m_hat_b = self.m[i]['b'] / (1 - self.beta1**self.t)  # Bias correction de m(b)
            v_hat_W = self.v[i]['W'] / (1 - self.beta2**self.t)  # Bias correction de v(W)
            v_hat_b = self.v[i]['b'] / (1 - self.beta2**self.t)  # Bias correction de v(b)

            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)  # Update Adam W: step adaptativo por componente
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)  # Update Adam b: mismo update para bias


# ============================================================
# RED NEURONAL
# ============================================================

class NeuralNetwork:  # Red feedforward (MLP): compone capas, ejecuta forward/backward y entrena con SGD/Adam
    def __init__(self, layer_sizes: List[int], activations: List[str]):  # Construye una red feedforward a partir de tamaños
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activations[i])  # Crea Layer i con fan-in/out y activación
                       for i in range(len(layer_sizes)-1)]  # Itera pares consecutivos de tamaños para construir todas las capas
        self.loss_history = []  # Historial de pérdida por época: útil para depuración (convergencia)

    def forward(self, x: np.ndarray) -> np.ndarray:  # Forward de la red: aplica forward secuencial de cada capa
        for layer in self.layers:  # Recorre capas en orden: la salida de una es entrada de la siguiente
            x = layer.forward(x)  # Propaga activaciones: actualiza x con la salida de la capa
        return x  # Devuelve la salida final (probabilidad/logits según última activación)

    def backward(self, y_true: np.ndarray) -> List[Tuple]:  # Backprop: calcula gradientes de parámetros en todas las capas
        y_pred = self.layers[-1].cache['a']  # Usa activación del último forward: evita recalcular predicción
        dL_da = y_pred - y_true  # Gradiente inicial (MSE simplificada): cambia si cambias la función de pérdida

        gradients = []  # Lista de gradientes por capa (dW, db) en orden forward
        for layer in reversed(self.layers):  # Recorre capas de atrás hacia adelante (regla de la cadena)
            dL_da, dW, db = layer.backward(dL_da)  # Backward capa: devuelve gradiente para capa anterior y sus dW/db
            gradients.insert(0, (dW, db))  # Inserta al inicio para alinear con self.layers (misma indexación)
        return gradients  # Devuelve gradientes listos para el optimizador

    def fit(self, X, y, epochs=1000, lr=0.1, optimizer='sgd', verbose=True):  # Entrenamiento por SGD/Adam (muestra a muestra)
        opt = Adam(lr) if optimizer == 'adam' else SGD(lr)  # Selecciona optimizador según string: cambia dinámica de convergencia

        for epoch in range(epochs):  # Loop principal de entrenamiento: una iteración por época
            total_loss = 0  # Acumulador de pérdida total de la época (para promedio/monitoreo)
            for xi, yi in zip(X, y):  # Itera dataset ejemplo a ejemplo (SGD puro, no mini-batch)
                yi_arr = np.atleast_1d(yi)  # Asegura y como vector: evita errores si yi es escalar
                output = self.forward(xi)  # Forward: predicción actual con parámetros actuales

                # BCE loss
                output_clip = np.clip(output, 1e-15, 1-1e-15)  # Clipping: evita log(0) -> inf/NaN en BCE
                loss = -np.sum(yi_arr * np.log(output_clip) + (1-yi_arr) * np.log(1-output_clip))  # BCE binaria por muestra
                total_loss += loss  # Suma pérdidas: luego se promedia por número de muestras

                gradients = self.backward(yi_arr)  # Backprop: calcula gradientes de todas las capas
                opt.step(self.layers, gradients)  # Update: aplica optimizador a parámetros usando los gradientes

            self.loss_history.append(total_loss / len(X))  # Guarda pérdida media de la época para trazado/diagnóstico
            if verbose and epoch % (epochs//10) == 0:  # Loggea ~10 veces (ojo: epochs//10 debe ser >0)
                print(f"Epoch {epoch}: Loss = {self.loss_history[-1]:.4f}")  # Imprime pérdida: ayuda a detectar estancamiento

    def predict(self, X: np.ndarray) -> np.ndarray:  # Predicción binaria: umbraliza la salida de forward
        return np.array([1 if self.forward(x)[0] > 0.5 else 0 for x in X])  # 0.5 como umbral estándar para sigmoid

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # Accuracy: métrica simple para clasificación binaria
        return np.mean(self.predict(X) == y)  # Proporción de aciertos (promedio de booleanos)


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":  # Entry point: ejecuta un test rápido cuando se corre este archivo como script
    print("=== Test: XOR Problem ===")  # Banner: indica inicio del test de XOR
    X = np.array([[0,0], [0,1], [1,0], [1,1]])  # Dataset XOR (4 ejemplos): no linealmente separable
    y = np.array([0, 1, 1, 0])  # Etiquetas XOR: 1 si bits difieren, 0 si son iguales

    net = NeuralNetwork([2, 4, 1], ['tanh', 'sigmoid'])  # Red 2→4→1: suficiente para aprender XOR con no linealidad
    net.fit(X, y, epochs=5000, lr=0.5, verbose=True)  # Entrena muchas épocas: en dataset pequeño debe converger

    print("\nPredicciones:")  # Encabezado: muestra predicciones finales tras entrenamiento
    for xi, yi in zip(X, y):  # Itera ejemplos para inspección manual de outputs
        pred = net.forward(xi)[0]  # Forward sobre un ejemplo: toma componente 0 porque salida es (1,)
        print(f"{xi} -> {pred:.4f} (target: {yi})")  # Imprime predicción vs target para ver si memoriza XOR

    print(f"\nAccuracy: {net.score(X, y):.2%}")  # Accuracy final: debería acercarse a 100% si aprendió
    print("\n✓ Test XOR completado!")  # Mensaje final: indica que terminó el bloque de pruebas
```

---

## 💻 Parte 5: CNNs - Redes Convolucionales (Semana 19)

> ⚠️ **Nota:** En este módulo implementas **solo el forward pass** de una CNN simple en NumPy (para dominar dimensiones). El entrenamiento completo de una CNN se hace con **PyTorch** (sin implementar backward manual de CNN).

### Protocolo D (visualización generativa): convolución sobre una imagen real

Para que “convolución” no sea solo una fórmula, ejecuta el script:

- [`visualizations/viz_convolution.py`](../visualizations/viz_convolution.py)

Uso recomendado (con una imagen propia):

```bash
python3 visualizations/viz_convolution.py /ruta/a/tu_imagen.png
```

Qué debes observar:

- el **Sobel X** responde fuerte a bordes verticales
- la **magnitud** combina bordes en varias direcciones

Entregable sugerido: captura de *input vs feature map* + explicación en 5 líneas de qué patrón detecta el filtro.

### 5.1 ¿Por Qué CNNs para Imágenes?

```
PROBLEMA CON MLP PARA IMÁGENES:

Imagen MNIST: 28x28 = 784 píxeles
MLP fully connected a capa de 256 neuronas:
  → 784 × 256 = 200,704 parámetros (¡solo primera capa!)

Imagen HD: 1920x1080x3 = 6,220,800 píxeles
  → Imposible conectar todo con todo

SOLUCIÓN: CONVOLUCIÓN
- Procesar regiones locales (no toda la imagen)
- Compartir pesos (el mismo filtro en toda la imagen)
- Detectar patrones sin importar su posición
```

### 5.2 La Operación de Convolución

```python
import numpy as np  # Importa NumPy: se usa para arrays, zeros, sum y construir el ejemplo de imagen/kernel

def convolve2d_simple(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:  # Define convolución 2D “valid” (sin padding) para entender el mecanismo
    """
    Convolución 2D simplificada (para entender el concepto).

    La convolución desliza un kernel (filtro) sobre la imagen
    y calcula el producto punto en cada posición.

    Args:
        image: Imagen de entrada (H, W)
        kernel: Filtro (kH, kW), típicamente 3x3 o 5x5

    Returns:
        Feature map (H-kH+1, W-kW+1)
    """
    H, W = image.shape  # Extrae alto/ancho de la imagen: define el espacio sobre el que el kernel puede deslizarse
    kH, kW = kernel.shape  # Extrae alto/ancho del kernel: define el tamaño de la ventana local que se multiplica por la imagen

    # Tamaño del output (sin padding)
    out_H = H - kH + 1  # Alto del feature map (valid): cantidad de posiciones verticales posibles del kernel
    out_W = W - kW + 1  # Ancho del feature map (valid): cantidad de posiciones horizontales posibles del kernel

    output = np.zeros((out_H, out_W))  # Inicializa salida en 0: aquí se acumulará el producto punto región·kernel en cada posición

    for i in range(out_H):  # Recorre filas de la salida: i indica el desplazamiento vertical del kernel sobre la imagen
        for j in range(out_W):  # Recorre columnas de la salida: j indica el desplazamiento horizontal del kernel sobre la imagen
            # Extraer región de la imagen
            region = image[i:i+kH, j:j+kW]  # Toma ventana local (kH,kW): la porción de imagen bajo el kernel en esta posición
            # Producto punto con el kernel
            output[i, j] = np.sum(region * kernel)  # Multiplica elemento a elemento y suma: implementa correlación/convolución simplificada

    return output  # Devuelve el feature map: respuesta del filtro para cada posición (sin padding)


# Ejemplo: Detección de bordes verticales
image = np.array([  # Define una “imagen” toy: matriz 4x6 con un borde vertical (cambio de 0 a 1) en la mitad derecha
    [0, 0, 0, 1, 1, 1],  # Fila 0: patrón de borde vertical (izquierda oscura, derecha clara)
    [0, 0, 0, 1, 1, 1],  # Fila 1: repite patrón para que el filtro detecte borde consistente
    [0, 0, 0, 1, 1, 1],  # Fila 2: repite patrón
    [0, 0, 0, 1, 1, 1],  # Fila 3: repite patrón
])  # Cierra el array: dtype se infiere; aquí son enteros 0/1

# Kernel Sobel para bordes verticales
sobel_vertical = np.array([  # Define kernel Sobel vertical: responde fuerte donde hay cambios en la dirección x (vertical edges)
    [-1, 0, 1],  # Fila superior: diferencia izquierda-derecha (detecta gradiente horizontal)
    [-2, 0, 2],  # Fila central: mayor peso en el centro para robustez
    [-1, 0, 1]  # Fila inferior: completa el patrón simétrico del filtro
])  # Cierra el kernel: shape (3,3), se aplicará en cada región 3x3 de la imagen

edges = convolve2d_simple(image, sobel_vertical)  # Aplica convolución: produce mapa de activaciones donde el borde vertical es más intenso
print("Feature map (bordes verticales):")  # Imprime etiqueta: facilita interpretar la salida en consola
print(edges)  # Imprime el feature map numérico: valores altos/magnitudes indican detección del borde
```

### 5.3 Conceptos Clave de CNNs

```
┌─────────────────────────────────────────────────────────────────┐
│  VOCABULARIO CNN                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  KERNEL (FILTRO)                                                │
│  ├── Matriz pequeña (3x3, 5x5) que detecta patrones             │
│  ├── Los valores del kernel son APRENDIDOS (backprop)           │
│  └── Diferentes kernels detectan diferentes features            │
│                                                                 │
│  STRIDE                                                         │
│  ├── Cuántos píxeles se mueve el kernel en cada paso            │
│  ├── stride=1: mueve 1 píxel (output grande)                    │
│  └── stride=2: mueve 2 píxeles (output más pequeño)             │
│                                                                 │
│  PADDING                                                        │
│  ├── Añadir ceros alrededor de la imagen                        │
│  ├── 'valid': sin padding (output más pequeño)                  │
│  └── 'same': padding para mantener tamaño                       │
│                                                                 │
│  POOLING                                                        │
│  ├── Reduce dimensiones (downsampling)                          │
│  ├── Max Pooling: toma el máximo de cada región                 │
│  └── Average Pooling: toma el promedio                          │
│                                                                 │
│  FEATURE MAP                                                    │
│  └── Output de aplicar un filtro (lo que "ve" el filtro)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Cálculo de Dimensiones (Importante para Exámenes)

```python
def output_size(input_size: int, kernel_size: int,  # Tamaño por eje (conv/pooling): útil para H_out o W_out en problemas de examen
                stride: int = 1, padding: int = 0) -> int:  # Usa división piso (//) para obtener un entero válido
    """
    Fórmula para calcular tamaño del output de convolución.

    output_size = floor((input + 2*padding - kernel) / stride) + 1
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1  # Aplica floor((in+2P-K)/S)+1: si no cuadra, revisa padding/stride


# Ejemplos típicos de examen:
print("=== Ejercicios de dimensiones ===")  # Encabezado: imprime separador para ver los resultados de los casos en consola

# Ejemplo 1: MNIST sin padding
# Input: 28x28, Kernel: 5x5, Stride: 1, Padding: 0
out = output_size(28, 5, stride=1, padding=0)  # Esperado 24: 28-5+1
print(f"MNIST 28x28, kernel 5x5, stride 1: output = {out}x{out}")  # 24x24

# Ejemplo 2: Con padding 'same'
# Para mantener tamaño con kernel 3x3, necesitas padding=1
out = output_size(28, 3, stride=1, padding=1)  # Esperado 28: padding=1 compensa kernel 3x3 con stride 1
print(f"MNIST 28x28, kernel 3x3, padding 1: output = {out}x{out}")  # 28x28

# Ejemplo 3: Max Pooling 2x2 stride 2
out = output_size(24, 2, stride=2, padding=0)  # Esperado 12: pooling 2 con stride 2 reduce a la mitad
print(f"24x24, pooling 2x2 stride 2: output = {out}x{out}")  # 12x12

```


### 5.5 Arquitectura Típica de CNN

```
┌─────────────────────────────────────────────────────────────────┐
│  ARQUITECTURA LENET-5 (Clásica para MNIST)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: 28x28x1 (imagen grayscale)                              │
│         │                                                       │
│         ▼                                                       │
│  [CONV 5x5, 6 filtros] → 24x24x6                                │
│         │                                                       │
│         ▼                                                       │
│  [ReLU]                                                         │
│         │                                                       │
│         ▼                                                       │
│  [MaxPool 2x2] → 12x12x6                                        │
│         │                                                       │
│         ▼                                                       │
│  [CONV 5x5, 16 filtros] → 8x8x16                                │
│         │                                                       │
│         ▼                                                       │
│  [ReLU]                                                         │
│         │                                                       │
│         ▼                                                       │
│  [MaxPool 2x2] → 4x4x16 = 256 neuronas                          │
│         │                                                       │
│         ▼                                                       │
│  [Flatten] → 256                                                │
│         │                                                       │
│         ▼                                                       │
│  [FC 120] → 120                                                 │
│         │                                                       │
│         ▼                                                       │
│  [FC 84] → 84                                                   │
│         │                                                       │
│         ▼                                                       │
│  [FC 10 + Softmax] → 10 clases (dígitos 0-9)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 Max Pooling

```python
def max_pool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:  # Define max pooling 2D: reduce resolución tomando el máximo por ventana (downsampling)
    """
    Max Pooling 2D.

    Reduce dimensiones tomando el máximo de cada región.
    Hace la red más robusta a pequeñas traslaciones.

    Args:
        x: Feature map (H, W)
        pool_size: Tamaño de la ventana (típicamente 2)

    Returns:
        Pooled output (H//pool_size, W//pool_size)
    """
    H, W = x.shape  # Extrae dimensiones de entrada: altura/ancho del feature map (asume 2D)
    out_H, out_W = H // pool_size, W // pool_size  # Calcula dimensiones de salida (stride=pool_size): reduce por factor entero

    output = np.zeros((out_H, out_W))  # Inicializa salida: guardará el máximo de cada región (out_H,out_W)

    for i in range(out_H):  # Itera filas de salida: cada i corresponde a una ventana vertical de `pool_size` píxeles
        for j in range(out_W):  # Itera columnas de salida: cada j corresponde a una ventana horizontal de `pool_size` píxeles
            region = x[i*pool_size:(i+1)*pool_size,  # Extrae ventana en filas: desde i*pool_size hasta (i+1)*pool_size (no inclusivo)
                      j*pool_size:(j+1)*pool_size]  # Extrae ventana en columnas: define el bloque local cuyo máximo representará la región
            output[i, j] = np.max(region)  # Agrega máximo de la región: implementa invarianza parcial a traslaciones pequeñas

    return output  # Devuelve mapa pooled: reduce tamaño y conserva activaciones más salientes por región


# Ejemplo
feature_map = np.array([  # Define feature map toy 4x4: valores sencillos para verificar visualmente el max pooling
    [1, 3, 2, 4],  # Fila 0: contiene máximo 4 en la esquina derecha
    [5, 6, 1, 2],  # Fila 1: contiene máximo 6 que debería sobrevivir al pooling en la ventana superior izquierda
    [3, 2, 1, 0],  # Fila 2: valores decrecientes para probar pooling en la parte inferior izquierda
    [1, 2, 3, 4]  # Fila 3: contiene máximo 4 en la ventana inferior derecha
])  # Cierra el array: shape (4,4) compatible con pool_size=2

pooled = max_pool2d(feature_map, pool_size=2)  # Ejecuta max pooling 2x2: reduce 4x4 -> 2x2 tomando máximos por bloque
print("Original 4x4:")  # Imprime etiqueta del input: facilita lectura del ejemplo en consola
print(feature_map)  # Muestra matriz original: permite comparar contra salida pooled
print("\nMax Pooled 2x2:")  # Imprime etiqueta con salto de línea: separa visualmente input y output
print(pooled)  # [[6, 4], [3, 4]]
```

### 5.7 Por Qué Funcionan las CNNs

```
INTUICIÓN:

1. CAPAS INICIALES: Detectan features simples
   - Bordes horizontales, verticales, diagonales
   - Cambios de color, texturas

2. CAPAS MEDIAS: Combinan features simples
   - Esquinas, curvas, patrones

3. CAPAS PROFUNDAS: Features de alto nivel
   - Partes de objetos (ojos, ruedas, letras)

4. CAPAS FINALES: Objetos completos
   - "Esto es un 7", "Esto es un gato"

VENTAJAS CLAVE:
├── Parameter sharing: mismo filtro en toda la imagen
├── Sparse connectivity: cada output depende de región local
├── Translation invariance: detecta patrones sin importar posición
└── Hierarchical features: de simple a complejo
```

### 5.8 Recursos para Profundizar en CNNs

| Recurso | Descripción |
|---------|-------------|
| [3B1B - But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) | Intuición visual |
| [CS231n Stanford](http://cs231n.stanford.edu/) | Curso completo de CNNs |
| Deep Learning Book, Cap. 9 | Teoría formal |

---

## 📝 Derivación Analítica: Backpropagation a Mano (v3.2)

> 🎓 **Simulación de Examen:** *"Derive las ecuaciones de backpropagation para una red de 2 capas"*. Este es un clásico de exámenes de posgrado.

### Red de 2 Capas: Derivación Completa

**Arquitectura:**
- Input: $x$ (vector de features)
- Capa 1: $z_1 = W_1 x + b_1$, $a_1 = \sigma(z_1)$
- Capa 2: $z_2 = W_2 a_1 + b_2$, $\hat{y} = \sigma(z_2)$
- Loss: $L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$

#### Paso 1: Gradiente de la Capa de Salida

$$\frac{\partial L}{\partial z_2} = \hat{y} - y = \delta_2$$

(Resultado elegante gracias a la combinación sigmoid + cross-entropy)

$$\frac{\partial L}{\partial W_2} = \delta_2 \cdot a_1^T$$

$$\frac{\partial L}{\partial b_2} = \delta_2$$

#### Paso 2: Propagar el Error Hacia Atrás (Capa Oculta)

$$\frac{\partial L}{\partial a_1} = W_2^T \delta_2$$

$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \odot \sigma'(z_1) = W_2^T \delta_2 \odot a_1 \odot (1 - a_1) = \delta_1$$

$$\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^T$$

$$\frac{\partial L}{\partial b_1} = \delta_1$$

#### Resumen: Las 4 Ecuaciones de Backprop

```
┌─────────────────────────────────────────────────────────────┐
│ ECUACIONES DE BACKPROPAGATION                               │
│                                                             │
│ 1. δ_L = ∇_a L ⊙ σ'(z_L)     Error en capa final           │
│                                                             │
│ 2. δ_l = (W_{l+1}^T δ_{l+1}) ⊙ σ'(z_l)   Propagar atrás    │
│                                                             │
│ 3. ∂L/∂W_l = δ_l · a_{l-1}^T   Gradiente de pesos           │
│                                                             │
│ 4. ∂L/∂b_l = δ_l              Gradiente de bias             │
└─────────────────────────────────────────────────────────────┘
```

### Tu Entregable

Escribe en un documento (Markdown o LaTeX):
1. Derivación completa de backprop para red de 2 capas
2. Por qué $\delta_L = \hat{y} - y$ cuando usamos sigmoid + cross-entropy
3. Diagrama de grafo computacional mostrando el flujo de gradientes

---

## 🧪 Overfit on Small Batch: Debugging de Redes Neuronales (v3.3)

> ⚠️ **CRÍTICO:** Esta es la técnica #1 de debugging en Deep Learning. Si tu red no puede hacer overfitting en 10 ejemplos, tiene un bug.

### El Principio

```
REGLA DE ORO DEL DEBUGGING EN DL:

Una red neuronal DEBE poder memorizar un dataset pequeño.

Si entrenas con:
- 10 ejemplos
- Muchas épocas (1000+)
- Sin regularización

El loss DEBE llegar a ~0.00 (o muy cercano).

Si NO llega a 0 → TU IMPLEMENTACIÓN TIENE UN BUG
```

### Por Qué Funciona

```
┌─────────────────────────────────────────────────────────────┐
│ OVERFIT TEST                                                │
│                                                             │
│ Dataset pequeño (10 ejemplos):                              │
│ - Capacidad de la red >> complejidad del dataset            │
│ - La red puede "memorizar" cada ejemplo perfectamente       │
│ - Loss debe → 0 si backprop funciona                        │
│                                                             │
│ Si loss NO baja:                                            │
│ - Gradiente mal calculado                                   │
│ - Learning rate incorrecto                                  │
│ - Arquitectura rota (dimensiones)                           │
│ - Bug en forward o backward pass                            │
└─────────────────────────────────────────────────────────────┘
```

### Script: `overfit_test.py` (Entregable Obligatorio v3.3)

```python
"""
Overfit Test - Validación de Redes Neuronales
Si tu red no puede hacer overfit en 10 ejemplos, está rota.

Autor: [Tu nombre]
Módulo: 07 - Deep Learning
"""
import numpy as np  # Importa NumPy: se usa para arrays, MSE y operaciones numéricas del test
from typing import List, Tuple  # Importa tipos: documenta la firma (retorna passed e histórico de loss)


def overfit_test(  # Test diagnóstico: fuerza al modelo a memorizar un dataset mínimo para validar backprop
    model,  # Modelo a evaluar: debe implementar .forward(), .backward() y .update() según este runner
    X_small: np.ndarray,  # Features del dataset pequeño: típicamente (n_samples, n_features)
    y_small: np.ndarray,  # Labels del dataset pequeño: shape compatible con output del modelo
    epochs: int = 2000,  # Número de épocas: debe ser alto para dar margen a que la loss baje a target
    target_loss: float = 0.01,  # Umbral de aprobación: si la loss final es < target_loss, consideramos que memoriza
    verbose: bool = True  # Controla prints: útil para debugging sin afectar el cálculo
) -> Tuple[bool, List[float]]:  # Retorna (passed, loss_history) para automatizar validación y diagnóstico
    """
    Test de overfitting: la red debe memorizar un dataset pequeño.

    Args:
        model: Tu red neuronal (debe tener .fit() y .forward())
        X_small: Dataset pequeño (10-20 ejemplos)
        y_small: Labels del dataset
        epochs: Épocas de entrenamiento
        target_loss: Loss objetivo (default: 0.01)
        verbose: Mostrar progreso

    Returns:
        (passed, loss_history)
    """
    if verbose:  # Si verbose está activo, mostramos banner y parámetros del test para facilitar debugging
        print("=" * 60)  # Separador visual: mejora legibilidad del log
        print("OVERFIT TEST: ¿Puede tu red memorizar 10 ejemplos?")  # Mensaje: define el objetivo del test
        print("=" * 60)  # Cierra el banner superior
        print(f"Dataset size: {len(y_small)}")  # Reporta tamaño del dataset: ayuda a asegurar que es realmente “pequeño”
        print(f"Epochs: {epochs}")  # Reporta épocas: si es bajo, el test puede fallar por falta de entrenamiento
        print(f"Target loss: {target_loss}")  # Reporta umbral: criterio de éxito/fracaso
        print("-" * 60)  # Separador antes de comenzar loop de entrenamiento

    # Entrenar
    loss_history = []  # Guarda la loss media por época: permite ver si converge y detectar estancamientos
    for epoch in range(epochs):  # Loop de épocas: repetimos varias pasadas sobre el dataset pequeño
        # Forward pass para todos los ejemplos
        total_loss = 0.0  # Acumula pérdida de la época: se promediará al final
        for i in range(len(y_small)):  # Itera cada ejemplo del dataset: entrenamiento muestra a muestra
            output = model.forward(X_small[i])  # Forward: predicción actual para el ejemplo i
            loss = np.mean((output - y_small[i]) ** 2)  # MSE: loss simple para comprobar que el gradiente aprende
            total_loss += loss  # Suma loss por ejemplo: permite calcular la media por época

            # Backward y update (asumiendo que model tiene estos métodos)
            model.backward(y_small[i])  # Backward: calcula gradientes internos usando el target del ejemplo i
            model.update(learning_rate=0.1)  # Update: aplica un paso de optimización con LR fijo (ajustable)

        avg_loss = total_loss / len(y_small)  # Promedio de la época: métrica comparable entre épocas
        loss_history.append(avg_loss)  # Guarda histórico: útil para graficar y para criterio final

        if verbose and epoch % 500 == 0:  # Log cada 500 épocas: balance entre visibilidad y ruido
            print(f"Epoch {epoch:4d}: Loss = {avg_loss:.6f}")  # Reporta loss: ayuda a ver si desciende hacia target

    final_loss = loss_history[-1]  # Loss final: se usa como criterio de aprobación del test
    passed = final_loss < target_loss  # Condición de éxito: si puede memorizar, el gradiente y updates probablemente están bien

    if verbose:  # Imprime el diagnóstico final: ayuda a decidir si hay bug o si solo falta tuning
        print("-" * 60)  # Separador: delimita fin del entrenamiento
        print(f"Final Loss: {final_loss:.6f}")  # Reporta la loss final alcanzada por el modelo
        if passed:  # Rama éxito: la red pudo memorizar el dataset pequeño
            print("✓ PASSED: Tu red puede hacer overfitting")  # Indicador: criterio de overfit cumplido
            print("  → El forward y backward pass funcionan correctamente")  # Interpretación: gradiente/update parecen correctos
        else:  # Rama fallo: no memorizó, típicamente hay bug o hiperparámetros mal elegidos
            print("✗ FAILED: Tu red NO puede hacer overfitting")  # Indicador: criterio de overfit no cumplido
            print("  → Revisa tu implementación de backprop")  # Sugerencia principal: backprop suele ser el culpable
            print("  Posibles causas:")  # Lista de causas comunes para orientar el debugging
            print("  - Gradiente mal calculado")  # Error típico: derivadas incorrectas o signos invertidos
            print("  - Learning rate muy bajo")  # Si LR es demasiado bajo, la loss puede bajar muy lento
            print("  - Bug en forward pass")  # Si forward está mal, backward también será incorrecto
            print("  - Dimensiones incorrectas")  # Shapes incorrectas rompen el gradiente o el update

    return passed, loss_history  # Devuelve resultado + curva: permite asserts y análisis de convergencia


# ============================================================
# EJEMPLO: Test con XOR (debe pasar)
# ============================================================

def test_xor_overfit():  # Demo: prueba el overfit_test con el dataset XOR para validar el runner
    """Test: Una red pequeña debe resolver XOR perfectamente."""  # Docstring: criterio de éxito del test (memorizar XOR en toy dataset)
    print("\n" + "=" * 60)  # Banner: separa visualmente el test del resto del output
    print("TEST: Overfit on XOR Problem")  # Mensaje: indica que se está probando overfit en XOR
    print("=" * 60)  # Cierra banner

    # XOR dataset (4 ejemplos)
    X = np.array([  # Inputs XOR: todas las combinaciones posibles de 2 bits
        [0, 0],  # Caso 00
        [0, 1],  # Caso 01
        [1, 0],  # Caso 10
        [1, 1]  # Caso 11
    ], dtype=np.float64)  # Fuerza float64: estabilidad numérica y consistencia en operaciones

    y = np.array([  # Targets XOR: 1 si bits difieren, 0 si son iguales
        [0],  # XOR(0,0)=0
        [1],  # XOR(0,1)=1
        [1],  # XOR(1,0)=1
        [0]  # XOR(1,1)=0
    ], dtype=np.float64)  # Misma precisión que X: evita casts implícitos

    # Crear red simple (2 -> 8 -> 1)
    # NOTA: Reemplaza esto con tu clase NeuralNetwork
    class SimpleNet:  # Red mínima de 2 capas (2→8→1): suficiente capacidad para memorizar XOR
        def __init__(self):  # Inicializa parámetros y cache para backprop
            np.random.seed(42)  # Semilla fija: hace reproducible el resultado (misma inicialización)
            self.W1 = np.random.randn(8, 2) * 0.5  # Pesos capa 1: (hidden=8, in=2) escalados para evitar saturación
            self.b1 = np.zeros((8, 1))  # Bias capa 1: (8,1) para broadcasting con z1
            self.W2 = np.random.randn(1, 8) * 0.5  # Pesos capa 2: (out=1, hidden=8)
            self.b2 = np.zeros((1, 1))  # Bias salida: (1,1)

            # Cache para backprop
            self.cache = {}  # Guarda tensores intermedios del forward: necesarios para el backward

        def sigmoid(self, z):  # Activación sigmoid estable: evita overflow en exp
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # σ(z)=1/(1+e^-z) con clipping de z

        def forward(self, x):  # Forward: computa predicción pasando por 2 capas con sigmoid
            x = x.reshape(-1, 1)  # Asegura vector columna (2,1): requerido para shapes de matmul
            z1 = self.W1 @ x + self.b1  # Logits capa 1: (8,2)@(2,1)+(8,1) -> (8,1)
            a1 = self.sigmoid(z1)  # Activación oculta: introduce no linealidad
            z2 = self.W2 @ a1 + self.b2  # Logit salida: (1,8)@(8,1)+(1,1) -> (1,1)
            a2 = self.sigmoid(z2)  # Salida: probabilidad en (0,1)

            self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}  # Cachea intermedios para backprop
            return a2.flatten()  # Devuelve vector 1D: compatible con el cálculo de MSE del runner

        def backward(self, y_true):  # Backward: calcula gradientes dW/db para ambas capas
            y_true = np.array(y_true).reshape(-1, 1)  # Normaliza target a columna (1,1) para restas
            a2 = self.cache['a2']  # Recupera activación de salida
            a1 = self.cache['a1']  # Recupera activación oculta
            x = self.cache['x']  # Recupera input (columna)

            # Gradientes
            dz2 = a2 - y_true  # Error en salida (simplificado): sirve como delta para MSE/gradiente aproximado
            self.dW2 = dz2 @ a1.T  # Gradiente W2: (1,1)@(1,8) -> (1,8)
            self.db2 = dz2  # Gradiente b2: (1,1)

            da1 = self.W2.T @ dz2  # Propaga delta hacia capa oculta: (8,1)@(1,1)->(8,1)
            dz1 = da1 * a1 * (1 - a1)  # Delta oculta: multiplica por derivada de sigmoid σ'(z)=a(1-a)
            self.dW1 = dz1 @ x.T  # Gradiente W1: (8,1)@(1,2)->(8,2)
            self.db1 = dz1  # Gradiente b1: (8,1)

        def update(self, learning_rate):  # Update: aplica descenso por gradiente con el LR dado
            self.W1 -= learning_rate * self.dW1  # Actualiza W1
            self.b1 -= learning_rate * self.db1  # Actualiza b1
            self.W2 -= learning_rate * self.dW2  # Actualiza W2
            self.b2 -= learning_rate * self.db2  # Actualiza b2

    # Ejecutar test
    model = SimpleNet()  # Instancia la red simple: se usará para validar overfit en XOR
    passed, history = overfit_test(model, X, y, epochs=2000, target_loss=0.01)  # Ejecuta el runner: debería llegar a loss < 0.01

    # Verificar predicciones finales
    print("\nPredicciones finales:")  # Encabezado: muestra predicciones después del entrenamiento
    for i in range(len(X)):  # Itera cada ejemplo de XOR
        pred = model.forward(X[i])  # Predice usando el modelo entrenado
        print(f"  Input: {X[i]} → Pred: {pred[0]:.3f} (Target: {y[i][0]})")  # Compara predicción vs target

    return passed  # Devuelve si pasó: permite integrarlo en asserts/pytest o validación manual


if __name__ == "__main__":  # Entry point: permite ejecutar este archivo como script
    test_xor_overfit()  # Lanza el test XOR para verificar el overfit_test end-to-end
```

### Checklist de Debugging con Overfit Test

| Síntoma | Diagnóstico | Solución |
|---------|-------------|----------|
| Loss no baja | Gradiente = 0 o NaN | Verificar derivadas con grad_check |
| Loss baja muy lento | Learning rate muy bajo | Aumentar LR (probar 0.1, 0.5, 1.0) |
| Loss oscila mucho | Learning rate muy alto | Reducir LR |
| Loss sube | Signos invertidos en gradiente | Revisar forward/backward |
| Loss = NaN | Overflow en exp/softmax | Usar versiones numéricamente estables |

---

## 🎯 El Reto del Tablero Blanco (Metodología Feynman)

Explica en **máximo 5 líneas** sin jerga técnica:

1. **¿Qué es backpropagation?**
   > Pista: Piensa en "culpar" a cada peso por el error.

2. **¿Por qué ReLU es mejor que sigmoid en capas ocultas?**
   > Pista: Piensa en qué pasa con el gradiente de sigmoid cuando z es muy grande o muy pequeño.

3. **¿Qué hace una convolución en una imagen?**
   > Pista: Piensa en "deslizar una lupa" buscando un patrón específico.

4. **¿Por qué usamos pooling?**
   > Pista: Piensa en "resumir" una región y hacerla más pequeña.

---

## ✅ Checklist de Finalización (v3.3)

### Conocimiento
- [ ] Entiendo la analogía neurona biológica → neurona artificial
- [ ] Implementé sigmoid, ReLU, tanh, softmax y sus derivadas
- [ ] Entiendo por qué XOR no es linealmente separable
- [ ] Implementé forward pass para MLP
- [ ] Entiendo la Chain Rule aplicada a backpropagation
- [ ] Implementé backward pass calculando gradientes
- [ ] Implementé SGD, SGD+Momentum y Adam
- [ ] Mi red resuelve el problema XOR

### CNNs (Teoría)
- [ ] Entiendo qué es convolución, stride, padding y pooling
- [ ] Puedo calcular dimensiones de output de una CNN
- [ ] Conozco la arquitectura LeNet-5

### CNNs (Práctica)
- [ ] Implementé forward pass (NumPy) de convolución + pooling para una arquitectura tipo LeNet
- [ ] Entrené una CNN equivalente con PyTorch usando `scripts/train_cnn_pytorch.py`

### Sequence Modeling (Light)
- [ ] Ejecuté `scripts/simple_rnn_forward.py` y verifiqué shapes `(batch,time,features)`
- [ ] Puedo explicar qué cambia al variar `batch`, `time` y `hidden`

### Entregables de Código
- [ ] `neural_network.py` con tests pasando
- [ ] `mypy src/` pasa sin errores
- [ ] `pytest tests/` pasa sin errores

### Overfit Test (v3.3 - Obligatorio)
- [ ] **`overfit_test.py` implementado**
- [ ] **Mi red hace overfit en XOR (loss < 0.01)**
- [ ] Si el test falla, debuggeé con grad_check

### Derivación Analítica (Obligatorio)
- [ ] Derivé las ecuaciones de backprop a mano
- [ ] Documento con derivación completa (Markdown o LaTeX)
- [ ] Diagrama de grafo computacional

### Metodología Feynman
- [ ] Puedo explicar backpropagation en 5 líneas sin jerga
- [ ] Puedo explicar ReLU vs sigmoid en 5 líneas
- [ ] Puedo explicar convolución en 5 líneas
- [ ] Puedo explicar pooling en 5 líneas

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [06_UNSUPERVISED_LEARNING](06_UNSUPERVISED_LEARNING.md) | [00_INDICE](00_INDICE.md) | [08_PROYECTO_MNIST](08_PROYECTO_MNIST.md) |
