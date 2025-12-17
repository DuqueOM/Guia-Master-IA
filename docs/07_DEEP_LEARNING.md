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

def perceptron(x: np.ndarray, w: np.ndarray, b: float) -> float:
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

class Activations:
    """Funciones de activación y sus derivadas."""

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid: σ(z) = 1 / (1 + e^(-z))

        Rango: (0, 1)
        Uso: Capa de salida para clasificación binaria
        Problema: Vanishing gradient para |z| grande
        """
        z = np.clip(z, -500, 500)  # Previene overflow en exp() con valores extremos
        return 1 / (1 + np.exp(-z))  # Fórmula matemática de la sigmoide

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """σ'(z) = σ(z) · (1 - σ(z)) = a · (1 - a)"""
        return a * (1 - a)  # Derivada simplificada usando salida ya calculada

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """
        ReLU: f(z) = max(0, z)

        Rango: [0, ∞)
        Uso: Capas ocultas (default moderno)
        Ventaja: No vanishing gradient para z > 0
        Problema: "Dying ReLU" si z < 0 siempre
        """
        return np.maximum(0, z)  # Implementación directa de ReLU

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """ReLU'(z) = 1 si z > 0, 0 si z ≤ 0"""
        return (z > 0).astype(float)  # Convierte booleano a float (1.0 o 0.0)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """
        Tanh: f(z) = (e^z - e^(-z)) / (e^z + e^(-z))

        Rango: (-1, 1)
        Uso: Alternativa a sigmoid (centrado en 0)
        """
        return np.tanh(z)  # Usa implementación NumPy optimizada

    @staticmethod
    def tanh_derivative(a: np.ndarray) -> np.ndarray:
        """tanh'(z) = 1 - tanh²(z) = 1 - a²"""
        return 1 - a ** 2  # Derivada usando identidad matemática

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax: softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)

        Rango: (0, 1), suma = 1
        Uso: Capa de salida para clasificación multiclase
        Output: probabilidades de cada clase
        """
        # Restar máximo para estabilidad numérica (previene overflow en exp)
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
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
import numpy as np
from typing import List, Dict

class Layer:
    """Una capa de la red neuronal."""

    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        """
        Args:
            input_size: número de entradas
            output_size: número de neuronas
            activation: 'relu', 'sigmoid', 'tanh', 'softmax', 'linear'
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Inicialización Xavier/He
        if activation == 'relu':
            # He initialization para ReLU
            std = np.sqrt(2.0 / input_size)
        else:
            # Xavier initialization
            std = np.sqrt(1.0 / input_size)

        self.W = np.random.randn(output_size, input_size) * std
        self.b = np.zeros(output_size)

        # Cache para backprop
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass de una capa.

        z = Wx + b
        a = activation(z)
        """
        self.cache['x'] = x

        # Transformación lineal
        z = self.W @ x + self.b
        self.cache['z'] = z

        # Activación
        if self.activation == 'relu':
            a = np.maximum(0, z)
        elif self.activation == 'sigmoid':
            a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            a = np.tanh(z)
        elif self.activation == 'softmax':
            z_shifted = z - np.max(z)
            exp_z = np.exp(z_shifted)
            a = exp_z / np.sum(exp_z)
        else:  # linear
            a = z

        self.cache['a'] = a
        return a


class NeuralNetwork:
    """Red Neuronal Multicapa."""

    def __init__(self, layer_sizes: List[int], activations: List[str]):
        """
        Args:
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            activations: ['relu', 'relu', ..., 'sigmoid'] para cada capa
        """
        assert len(activations) == len(layer_sizes) - 1

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass a través de todas las capas."""
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicción para múltiples muestras."""
        predictions = []
        for x in X:
            output = self.forward(x)
            if len(output) == 1:
                predictions.append(1 if output[0] > 0.5 else 0)
            else:
                predictions.append(np.argmax(output))
        return np.array(predictions)


# Demo
net = NeuralNetwork(
    layer_sizes=[2, 4, 1],  # 2 inputs → 4 hidden → 1 output
    activations=['relu', 'sigmoid']
)

# Forward pass
x = np.array([0.5, 0.3])
output = net.forward(x)
print(f"Input: {x}")
print(f"Output: {output}")
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
import numpy as np
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

def assert_shape(x: np.ndarray, shape: Sequence[Optional[int]], name: str = "x") -> np.ndarray:
    x = np.asarray(x)
    assert x.ndim == len(shape), f"{name}.ndim={x.ndim}, expected={len(shape)}"
    for i, (got, exp) in enumerate(zip(x.shape, shape)):
        if exp is not None:
            assert got == exp, f"{name}.shape[{i}]={got}, expected={exp}"
    return x

def shape_check(
    spec: Dict[str, Sequence[Optional[int]]],
    out: Optional[Sequence[Optional[int]]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for k, shp in spec.items():
                if k in kwargs:
                    assert_shape(kwargs[k], shp, name=k)
            y = fn(*args, **kwargs)
            if out is not None:
                assert_shape(y, out, name="out")
            return y

        return wrapper

    return deco

@shape_check({"X": (None, 3), "W": (3, 4), "b": (4,)}, out=(None, 4))
def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b

X = np.random.randn(5, 3)
W = np.random.randn(3, 4)
b = np.random.randn(4)
Z = dense_forward(X=X, W=W, b=b)
assert Z.shape == (5, 4)
```

##### 4.2 Cápsula: Inicialización (Xavier vs He/Kaiming)

Regla práctica (MLP):

- Activaciones tipo `tanh/sigmoid` suelen ir mejor con **Xavier/Glorot**.
- Activaciones tipo `ReLU` suelen ir mejor con **He/Kaiming**.

```python
import numpy as np
from typing import Literal, Optional

def init_linear(
    fan_in: int,
    fan_out: int,
    mode: Literal["xavier", "kaiming"] = "xavier",
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if mode == "kaiming":
        std = np.sqrt(2.0 / fan_in)
    else:
        std = np.sqrt(1.0 / fan_in)

    W = rng.standard_normal((fan_in, fan_out)) * std
    return W

d_in, d_out = 784, 128
W_relu = init_linear(d_in, d_out, mode="kaiming", seed=0)
W_tanh = init_linear(d_in, d_out, mode="xavier", seed=0)
assert W_relu.shape == (d_in, d_out)
assert W_tanh.shape == (d_in, d_out)
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
class SGD:
    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def update(self, layer, dW: np.ndarray, db: np.ndarray):
        layer.W -= self.lr * dW
        layer.b -= self.lr * db
```

### 4.2 SGD con Momentum

```python
class SGDMomentum:
    """
    SGD con Momentum.

    v_t = β·v_{t-1} + (1-β)·∇L
    θ = θ - lr·v_t

    Momentum ayuda a:
    - Acelerar convergencia
    - Escapar de mínimos locales
    - Reducir oscilaciones
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update(self, layer, dW: np.ndarray, db: np.ndarray, layer_id: int):
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                'W': np.zeros_like(dW),
                'b': np.zeros_like(db)
            }

        v = self.velocities[layer_id]

        # Actualizar velocidad
        v['W'] = self.momentum * v['W'] + (1 - self.momentum) * dW
        v['b'] = self.momentum * v['b'] + (1 - self.momentum) * db

        # Actualizar parámetros
        layer.W -= self.lr * v['W']
        layer.b -= self.lr * v['b']
```

### 4.3 Adam Optimizer

```python
class Adam:
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

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer, dW: np.ndarray, db: np.ndarray, layer_id: int):
        if layer_id not in self.m:
            self.m[layer_id] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}
            self.v[layer_id] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}

        self.t += 1
        m, v = self.m[layer_id], self.v[layer_id]

        # Actualizar momentos
        m['W'] = self.beta1 * m['W'] + (1 - self.beta1) * dW
        m['b'] = self.beta1 * m['b'] + (1 - self.beta1) * db
        v['W'] = self.beta2 * v['W'] + (1 - self.beta2) * dW**2
        v['b'] = self.beta2 * v['b'] + (1 - self.beta2) * db**2

        # Corrección de bias
        m_hat_W = m['W'] / (1 - self.beta1**self.t)
        m_hat_b = m['b'] / (1 - self.beta1**self.t)
        v_hat_W = v['W'] / (1 - self.beta2**self.t)
        v_hat_b = v['b'] / (1 - self.beta2**self.t)

        # Actualizar parámetros
        layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
        layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
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
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    a = sigmoid(z)
    return a * (1.0 - a)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.asarray(z, dtype=float))


def relu_deriv(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return (z > 0.0).astype(float)


def num_derivative(f, z: np.ndarray, h: float = 1e-6) -> np.ndarray:
    return (f(z + h) - f(z - h)) / (2.0 * h)


np.random.seed(0)
z = np.random.randn(10)
g_num = num_derivative(sigmoid, z)
g_ana = sigmoid_deriv(z)
assert np.allclose(g_num, g_ana, rtol=1e-5, atol=1e-6)
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
import numpy as np

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b


np.random.seed(1)
n, d_in, d_out = 5, 3, 4
X = np.random.randn(n, d_in)
W = np.random.randn(d_in, d_out)
b = np.random.randn(d_out)

Z = dense_forward(X, W, b)
assert Z.shape == (n, d_out)

Z_loop = np.zeros_like(Z)
for i in range(n):
    Z_loop[i] = X[i] @ W + b

assert np.allclose(Z, Z_loop)
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
import numpy as np

def logsumexp(z: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    m = np.max(z, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    lse = logsumexp(z, axis=axis, keepdims=True)
    return np.exp(z - lse)


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))


z = np.array([[10.0, 0.0, -10.0]])
p = softmax(z)
assert np.isclose(np.sum(p), 1.0)
assert np.argmax(p) == 0

y_true = np.array([[1.0, 0.0, 0.0]])
loss_good = categorical_cross_entropy(y_true, np.array([[0.9, 0.05, 0.05]]))
loss_bad = categorical_cross_entropy(y_true, np.array([[0.4, 0.3, 0.3]]))
assert loss_good < loss_bad
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
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    P = sigmoid(Z2)
    cache = (X, Z1, A1, Z2, P)
    return P, cache


def loss_fn(X, y, W1, b1, W2, b2):
    P, _ = forward(X, W1, b1, W2, b2)
    return bce(y, P)


def backward(y, cache, W2):
    X, Z1, A1, Z2, P = cache
    n = X.shape[0]
    # BCE with sigmoid output: dZ2 = (P - y) / n
    dZ2 = (P - y) / n
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    return dW1, db1, dW2, db2


np.random.seed(0)
n, d_in, d_h = 8, 3, 5
X = np.random.randn(n, d_in)
y = (np.random.rand(n, 1) < 0.5).astype(float)
W1 = np.random.randn(d_in, d_h) * 0.1
b1 = np.zeros(d_h)
W2 = np.random.randn(d_h, 1) * 0.1
b2 = np.zeros(1)

P, cache = forward(X, W1, b1, W2, b2)
dW1, db1, dW2, db2 = backward(y, cache, W2)

# Gradient check on one W2 coordinate
i, j = 2, 0
h = 1e-6
E = np.zeros_like(W2)
E[i, j] = 1.0
L_plus = loss_fn(X, y, W1, b1, W2 + h * E, b2)
L_minus = loss_fn(X, y, W1, b1, W2 - h * E, b2)
g_num = (L_plus - L_minus) / (2.0 * h)
assert np.isclose(dW2[i, j], g_num, rtol=1e-4, atol=1e-6)
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
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


np.random.seed(1)
n = 16
X_pos = np.random.randn(n // 2, 2) + np.array([2.0, 2.0])
X_neg = np.random.randn(n // 2, 2) + np.array([-2.0, -2.0])
X = np.vstack([X_pos, X_neg])
y = np.vstack([np.ones((n // 2, 1)), np.zeros((n // 2, 1))])

w = np.zeros((2, 1))
b = 0.0
lr = 0.2

loss0 = None
for t in range(400):
    logits = X @ w + b
    p = sigmoid(logits)
    loss = bce(y, p)
    if loss0 is None:
        loss0 = loss
    # gradients
    dz = (p - y) / n
    dw = X.T @ dz
    db = float(np.sum(dz))
    w -= lr * dw
    b -= lr * db

loss_end = bce(y, sigmoid(X @ w + b))
pred = (sigmoid(X @ w + b) >= 0.5).astype(int)
acc = float(np.mean(pred == y.astype(int)))

assert loss_end <= loss0
assert acc > 0.95
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
import numpy as np

def grad_f(w: float) -> float:
    return 2.0 * (w - 3.0)


def sgd(w0: float, lr: float, steps: int) -> float:
    w = float(w0)
    for _ in range(steps):
        w -= lr * grad_f(w)
    return w


def adam(w0: float, lr: float, steps: int, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> float:
    w = float(w0)
    m = 0.0
    v = 0.0
    t = 0
    for _ in range(steps):
        t += 1
        g = grad_f(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return w


w_sgd = sgd(w0=10.0, lr=0.1, steps=50)
w_adam = adam(w0=10.0, lr=0.2, steps=50)

assert abs(w_sgd - 3.0) < 1e-2
assert abs(w_adam - 3.0) < 1e-2
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
import numpy as np

def clip_by_norm(g: np.ndarray, max_norm: float) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    n = np.linalg.norm(g)
    if n == 0.0:
        return g
    if n <= max_norm:
        return g
    return g * (max_norm / n)


g_big = np.array([3.0, 4.0])  # norm=5
g_clip = clip_by_norm(g_big, max_norm=1.0)
assert np.linalg.norm(g_clip) <= 1.0 + 1e-12

g_small = np.array([0.3, 0.4])  # norm=0.5
g_keep = clip_by_norm(g_small, max_norm=1.0)
assert np.allclose(g_small, g_keep)
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
import numpy as np

def conv2d_out(H: int, W: int, KH: int, KW: int, stride: int = 1, padding: int = 0):
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1
    return int(H_out), int(W_out)

assert conv2d_out(28, 28, 5, 5, stride=1, padding=0) == (24, 24)
assert conv2d_out(28, 28, 5, 5, stride=1, padding=2) == (28, 28)
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

import numpy as np
from typing import List, Tuple, Optional


# ============================================================
# ACTIVACIONES
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def tanh_deriv(a):
    return 1 - a**2

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


# ============================================================
# CAPA
# ============================================================

class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.activation = activation
        scale = np.sqrt(2.0 / input_size) if activation == 'relu' else np.sqrt(1.0 / input_size)
        self.W = np.random.randn(output_size, input_size) * scale
        self.b = np.zeros(output_size)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        z = self.W @ x + self.b
        self.cache['z'] = z

        if self.activation == 'relu':
            a = relu(z)
        elif self.activation == 'sigmoid':
            a = sigmoid(z)
        elif self.activation == 'tanh':
            a = np.tanh(z)
        elif self.activation == 'softmax':
            a = softmax(z)
        else:
            a = z

        self.cache['a'] = a
        return a

    def backward(self, dL_da: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z, x, a = self.cache['z'], self.cache['x'], self.cache['a']

        if self.activation == 'sigmoid':
            da_dz = sigmoid_deriv(a)
        elif self.activation == 'relu':
            da_dz = relu_deriv(z)
        elif self.activation == 'tanh':
            da_dz = tanh_deriv(a)
        else:
            da_dz = np.ones_like(z)

        delta = dL_da * da_dz
        dL_dW = np.outer(delta, x)
        dL_db = delta
        dL_dx = self.W.T @ delta

        return dL_dx, dL_dW, dL_db


# ============================================================
# OPTIMIZADORES
# ============================================================

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers, gradients):
        for layer, (dW, db) in zip(layers, gradients):
            layer.W -= self.lr * dW
            layer.b -= self.lr * db


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, layers, gradients):
        self.t += 1
        for i, (layer, (dW, db)) in enumerate(zip(layers, gradients)):
            if i not in self.m:
                self.m[i] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}
                self.v[i] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}

            self.m[i]['W'] = self.beta1 * self.m[i]['W'] + (1 - self.beta1) * dW
            self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * db
            self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1 - self.beta2) * dW**2
            self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * db**2

            m_hat_W = self.m[i]['W'] / (1 - self.beta1**self.t)
            m_hat_b = self.m[i]['b'] / (1 - self.beta1**self.t)
            v_hat_W = self.v[i]['W'] / (1 - self.beta2**self.t)
            v_hat_b = self.v[i]['b'] / (1 - self.beta2**self.t)

            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


# ============================================================
# RED NEURONAL
# ============================================================

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], activations: List[str]):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
                       for i in range(len(layer_sizes)-1)]
        self.loss_history = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true: np.ndarray) -> List[Tuple]:
        y_pred = self.layers[-1].cache['a']
        dL_da = y_pred - y_true

        gradients = []
        for layer in reversed(self.layers):
            dL_da, dW, db = layer.backward(dL_da)
            gradients.insert(0, (dW, db))
        return gradients

    def fit(self, X, y, epochs=1000, lr=0.1, optimizer='sgd', verbose=True):
        opt = Adam(lr) if optimizer == 'adam' else SGD(lr)

        for epoch in range(epochs):
            total_loss = 0
            for xi, yi in zip(X, y):
                yi_arr = np.atleast_1d(yi)
                output = self.forward(xi)

                # BCE loss
                output_clip = np.clip(output, 1e-15, 1-1e-15)
                loss = -np.sum(yi_arr * np.log(output_clip) + (1-yi_arr) * np.log(1-output_clip))
                total_loss += loss

                gradients = self.backward(yi_arr)
                opt.step(self.layers, gradients)

            self.loss_history.append(total_loss / len(X))
            if verbose and epoch % (epochs//10) == 0:
                print(f"Epoch {epoch}: Loss = {self.loss_history[-1]:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([1 if self.forward(x)[0] > 0.5 else 0 for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":
    print("=== Test: XOR Problem ===")
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])

    net = NeuralNetwork([2, 4, 1], ['tanh', 'sigmoid'])
    net.fit(X, y, epochs=5000, lr=0.5, verbose=True)

    print("\nPredicciones:")
    for xi, yi in zip(X, y):
        pred = net.forward(xi)[0]
        print(f"{xi} -> {pred:.4f} (target: {yi})")

    print(f"\nAccuracy: {net.score(X, y):.2%}")
    print("\n✓ Test XOR completado!")
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
import numpy as np

def convolve2d_simple(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
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
    H, W = image.shape
    kH, kW = kernel.shape

    # Tamaño del output (sin padding)
    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            # Extraer región de la imagen
            region = image[i:i+kH, j:j+kW]
            # Producto punto con el kernel
            output[i, j] = np.sum(region * kernel)

    return output


# Ejemplo: Detección de bordes verticales
image = np.array([
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
])

# Kernel Sobel para bordes verticales
sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

edges = convolve2d_simple(image, sobel_vertical)
print("Feature map (bordes verticales):")
print(edges)
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
def output_size(input_size: int, kernel_size: int,
                stride: int = 1, padding: int = 0) -> int:
    """
    Fórmula para calcular tamaño del output de convolución.

    output_size = floor((input + 2*padding - kernel) / stride) + 1
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


# Ejemplos típicos de examen:
print("=== Ejercicios de dimensiones ===")

# Ejemplo 1: MNIST sin padding
# Input: 28x28, Kernel: 5x5, Stride: 1, Padding: 0
out = output_size(28, 5, stride=1, padding=0)
print(f"MNIST 28x28, kernel 5x5, stride 1: output = {out}x{out}")  # 24x24

# Ejemplo 2: Con padding 'same'
# Para mantener tamaño con kernel 3x3, necesitas padding=1
out = output_size(28, 3, stride=1, padding=1)
print(f"MNIST 28x28, kernel 3x3, padding 1: output = {out}x{out}")  # 28x28

# Ejemplo 3: Max Pooling 2x2 stride 2
out = output_size(24, 2, stride=2, padding=0)
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
def max_pool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
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
    H, W = x.shape
    out_H, out_W = H // pool_size, W // pool_size

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = x[i*pool_size:(i+1)*pool_size,
                      j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)

    return output


# Ejemplo
feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [3, 2, 1, 0],
    [1, 2, 3, 4]
])

pooled = max_pool2d(feature_map, pool_size=2)
print("Original 4x4:")
print(feature_map)
print("\nMax Pooled 2x2:")
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
import numpy as np
from typing import List, Tuple


def overfit_test(
    model,
    X_small: np.ndarray,
    y_small: np.ndarray,
    epochs: int = 2000,
    target_loss: float = 0.01,
    verbose: bool = True
) -> Tuple[bool, List[float]]:
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
    if verbose:
        print("=" * 60)
        print("OVERFIT TEST: ¿Puede tu red memorizar 10 ejemplos?")
        print("=" * 60)
        print(f"Dataset size: {len(y_small)}")
        print(f"Epochs: {epochs}")
        print(f"Target loss: {target_loss}")
        print("-" * 60)

    # Entrenar
    loss_history = []
    for epoch in range(epochs):
        # Forward pass para todos los ejemplos
        total_loss = 0.0
        for i in range(len(y_small)):
            output = model.forward(X_small[i])
            loss = np.mean((output - y_small[i]) ** 2)  # MSE
            total_loss += loss

            # Backward y update (asumiendo que model tiene estos métodos)
            model.backward(y_small[i])
            model.update(learning_rate=0.1)

        avg_loss = total_loss / len(y_small)
        loss_history.append(avg_loss)

        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch:4d}: Loss = {avg_loss:.6f}")

    final_loss = loss_history[-1]
    passed = final_loss < target_loss

    if verbose:
        print("-" * 60)
        print(f"Final Loss: {final_loss:.6f}")
        if passed:
            print("✓ PASSED: Tu red puede hacer overfitting")
            print("  → El forward y backward pass funcionan correctamente")
        else:
            print("✗ FAILED: Tu red NO puede hacer overfitting")
            print("  → Revisa tu implementación de backprop")
            print("  Posibles causas:")
            print("  - Gradiente mal calculado")
            print("  - Learning rate muy bajo")
            print("  - Bug en forward pass")
            print("  - Dimensiones incorrectas")

    return passed, loss_history


# ============================================================
# EJEMPLO: Test con XOR (debe pasar)
# ============================================================

def test_xor_overfit():
    """Test: Una red pequeña debe resolver XOR perfectamente."""
    print("\n" + "=" * 60)
    print("TEST: Overfit on XOR Problem")
    print("=" * 60)

    # XOR dataset (4 ejemplos)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float64)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=np.float64)

    # Crear red simple (2 -> 8 -> 1)
    # NOTA: Reemplaza esto con tu clase NeuralNetwork
    class SimpleNet:
        def __init__(self):
            np.random.seed(42)
            self.W1 = np.random.randn(8, 2) * 0.5
            self.b1 = np.zeros((8, 1))
            self.W2 = np.random.randn(1, 8) * 0.5
            self.b2 = np.zeros((1, 1))

            # Cache para backprop
            self.cache = {}

        def sigmoid(self, z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        def forward(self, x):
            x = x.reshape(-1, 1)
            z1 = self.W1 @ x + self.b1
            a1 = self.sigmoid(z1)
            z2 = self.W2 @ a1 + self.b2
            a2 = self.sigmoid(z2)

            self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
            return a2.flatten()

        def backward(self, y_true):
            y_true = np.array(y_true).reshape(-1, 1)
            a2 = self.cache['a2']
            a1 = self.cache['a1']
            x = self.cache['x']

            # Gradientes
            dz2 = a2 - y_true
            self.dW2 = dz2 @ a1.T
            self.db2 = dz2

            da1 = self.W2.T @ dz2
            dz1 = da1 * a1 * (1 - a1)
            self.dW1 = dz1 @ x.T
            self.db1 = dz1

        def update(self, learning_rate):
            self.W1 -= learning_rate * self.dW1
            self.b1 -= learning_rate * self.db1
            self.W2 -= learning_rate * self.dW2
            self.b2 -= learning_rate * self.db2

    # Ejecutar test
    model = SimpleNet()
    passed, history = overfit_test(model, X, y, epochs=2000, target_loss=0.01)

    # Verificar predicciones finales
    print("\nPredicciones finales:")
    for i in range(len(X)):
        pred = model.forward(X[i])
        print(f"  Input: {X[i]} → Pred: {pred[0]:.3f} (Target: {y[i][0]})")

    return passed


if __name__ == "__main__":
    test_xor_overfit()
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
