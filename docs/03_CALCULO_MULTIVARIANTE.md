# Módulo 03 - Cálculo Multivariante para Deep Learning

> **🎯 Objetivo:** Dominar derivadas, gradientes y la Chain Rule para entender Backpropagation
> **Fase:** 1 - Fundamentos Matemáticos | **Semanas 6-8**
> **Prerrequisitos:** Módulo 02 (Álgebra Lineal para ML)

---

<a id="m03-0"></a>

## 🧭 Cómo usar este módulo (modo 0→100)

**Propósito:** que puedas hacer 3 cosas sin depender de “fe”:

- derivar gradientes de pérdidas comunes (MSE, BCE)
- implementar y depurar optimización (gradient descent)
- entender por qué backprop es chain rule aplicada a un grafo

### Objetivos de aprendizaje (medibles)

Al terminar este módulo podrás:

- **Calcular** derivadas y derivadas parciales (a mano y con verificación numérica).
- **Aplicar** gradiente y dirección de máximo descenso para optimizar funciones.
- **Implementar** gradient descent con criterios de convergencia razonables.
- **Explicar** la Chain Rule y usarla para derivar gradientes compuestos.
- **Validar** derivadas con gradient checking (error relativo pequeño).

### Prerrequisitos

- `Módulo 02` (producto matricial, normas, intuición geométrica).

Enlaces rápidos:

- [GLOSARIO: Derivative](GLOSARIO.md#derivative)
- [GLOSARIO: Gradient](GLOSARIO.md#gradient)
- [GLOSARIO: Gradient Descent](GLOSARIO.md#gradient-descent)
- [GLOSARIO: Chain Rule](GLOSARIO.md#chain-rule)
- [RECURSOS.md](RECURSOS.md)

### Integración con Plan v4/v5

- Visualización de optimización: `study_tools/VISUALIZACION_GRADIENT_DESCENT.md`
- Simulacros: `study_tools/SIMULACRO_EXAMEN_TEORICO.md`
- Evaluación (rúbrica): [study_tools/RUBRICA_v1.md](../study_tools/RUBRICA_v1.md) (scope `M03` en `rubrica.csv`)
- Protocolo completo:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Recursos (cuándo usarlos)

| Prioridad | Recurso | Cuándo usarlo en este módulo | Para qué |
|----------|---------|------------------------------|----------|
| **Obligatorio** | `study_tools/VISUALIZACION_GRADIENT_DESCENT.md` | Al implementar Gradient Descent (cuando ajustes `learning_rate` y criterios de parada) | Ver si “baja” o diverge y por qué |
| **Complementario** | [`visualizations/viz_gradient_3d.py`](../visualizations/viz_gradient_3d.py) | Semana 7, cuando ya entiendas `∇J` pero el `learning_rate` se sienta “mágico” | Generar un HTML interactivo con superficie 3D + trayectoria (convergencia/overshooting) |
| **Complementario** | [3Blue1Brown: Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Antes de Chain Rule (o si derivar se siente mecánico) | Intuición visual de derivadas y composición |
| **Complementario** | [Mathematics for ML: Multivariate Calculus](https://www.coursera.org/learn/multivariate-calculus-machine-learning) | Cuando pases de derivadas 1D a gradiente/derivadas parciales | Práctica estructurada con ejercicios |
| **Obligatorio** | `study_tools/SIMULACRO_EXAMEN_TEORICO.md` | Tras terminar Chain Rule (antes de saltar a M05/M07) | Verificar que puedes derivar sin mirar apuntes |
| **Opcional** | [RECURSOS.md](RECURSOS.md) | Al cerrar el módulo (para refuerzo) | Elegir material extra sin perder foco |

### Criterio de salida (cuándo puedes avanzar)

- Puedes derivar y verificar (numérico vs analítico) gradientes de MSE y BCE.
- Puedes explicar chain rule en 5 líneas y aplicarla a una composición.
- Puedes ejecutar gradient checking y entender qué significa el error relativo.

## 🧠 ¿Por Qué Cálculo para ML?

### ⚠️ CRÍTICO: Sin Chain Rule No Hay Deep Learning

```
El algoritmo de Backpropagation ES la Regla de la Cadena aplicada
a funciones compuestas de redes neuronales.

Si no entiendes:
  ∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w

NO entenderás por qué funciona una red neuronal y
probablemente REPROBARÁS el curso de Deep Learning.
```

### Conexión con el Pathway

| Concepto | Uso en ML | Curso del Pathway |
|----------|-----------|-------------------|
| **Derivada** | Tasa de cambio, pendiente | Todos |
| **Gradiente** | Dirección de máximo ascenso | Supervised Learning |
| **Gradient Descent** | Optimización de parámetros | Supervised + Deep Learning |
| **Chain Rule** | Backpropagation | Deep Learning |

---

## 🧭 Intuición geométrica (para que no sea mecánico)

### 1) El gradiente como brújula en una montaña

Piensa en la función de pérdida `J(θ)` como un terreno (montaña/valle) y tú como alguien parado en un punto.

- `J` te dice la altura.
- El **gradiente** `∇J` apunta hacia donde el terreno sube más rápido.
- Si quieres bajar (minimizar), te mueves en la dirección opuesta:

`θ_{t+1} = θ_t - α ∇J(θ_t)`

Visualización sugerida (hazlo en papel):

- curvas de nivel (contornos) alrededor de un valle
- un vector `∇J` perpendicular a las curvas de nivel

### 2) La regla de la cadena como engranajes (ratios de cambio)

Imagina tres engranajes conectados:

`x  →  g(x)  →  f(g(x))`

Si giras un poquito el primer engranaje (`x`), el último (`f`) gira según dos “ratios”:

- cuánto cambia `f` si cambia `g` (`df/dg`)
- cuánto cambia `g` si cambia `x` (`dg/dx`)

Y la regla es:

`df/dx = (df/dg) · (dg/dx)`

Backprop es esto mismo, pero aplicado a un grafo con muchas piezas: multiplicas ratios locales y propagas desde el final al inicio.

Diagrama sugerido (dibújalo): un grafo pequeño con nodos `z = Wx + b`, `a = φ(z)`, `L(a)` y flechas con gradientes “río arriba”.

## 📚 Contenido del Módulo

### Semana 6: Derivadas y Derivadas Parciales
### Semana 7: Gradiente y Gradient Descent
### Semana 8: Chain Rule y Preparación para Backprop

---

## 💻 Parte 1: Derivadas

### 1.1 Concepto de Derivada

```python
import numpy as np
import matplotlib.pyplot as plt

"""
DERIVADA: Tasa de cambio instantánea de una función.

Definición formal:
    f'(x) = lim[h→0] (f(x+h) - f(x)) / h

Interpretación geométrica: pendiente de la recta tangente.

Notaciones equivalentes:
    f'(x) = df/dx = d/dx f(x) = Df(x)
"""

def numerical_derivative(f, x: float, h: float = 1e-7) -> float:
    """
    Calcula la derivada numérica usando diferencias finitas.

    Método: diferencia central (más preciso)
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    """
    return (f(x + h) - f(x - h)) / (2 * h)


# Ejemplo: f(x) = x²
def f(x):
    return x ** 2

# Derivada analítica: f'(x) = 2x
def f_prime_analytical(x):
    return 2 * x

# Comparar
x = 3.0
numerical = numerical_derivative(f, x)
analytical = f_prime_analytical(x)

print(f"f(x) = x² en x={x}")
print(f"Derivada numérica:  {numerical:.6f}")
print(f"Derivada analítica: {analytical:.6f}")
print(f"Error: {abs(numerical - analytical):.2e}")
```

### 1.2 Derivadas Comunes en ML

```python
import numpy as np

"""
DERIVADAS QUE NECESITAS MEMORIZAR PARA ML:

1. Constante:     d/dx(c) = 0
2. Lineal:        d/dx(x) = 1
3. Potencia:      d/dx(xⁿ) = n·x^(n-1)
4. Exponencial:   d/dx(eˣ) = eˣ
5. Logaritmo:     d/dx(ln x) = 1/x
6. Suma:          d/dx(f+g) = f' + g'
7. Producto:      d/dx(f·g) = f'g + fg'
8. Cociente:      d/dx(f/g) = (f'g - fg')/g²
9. Cadena:        d/dx(f(g(x))) = f'(g(x))·g'(x)
"""

# Funciones de activación y sus derivadas

def sigmoid(x: np.ndarray) -> np.ndarray:
    """σ(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    d/dx σ(x) = σ(x) · (1 - σ(x))

    Derivación:
    σ(x) = (1 + e^(-x))^(-1)
    σ'(x) = -1·(1 + e^(-x))^(-2) · (-e^(-x))
          = e^(-x) / (1 + e^(-x))²
          = σ(x) · (1 - σ(x))
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU(x) = max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    d/dx ReLU(x) = { 1 si x > 0
                  { 0 si x < 0
                  { indefinido si x = 0 (usamos 0)
    """
    return (x > 0).astype(float)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    d/dx tanh(x) = 1 - tanh²(x)
    """
    return 1 - np.tanh(x) ** 2


# Verificar con derivada numérica
def verify_derivative(f, f_prime, x, name):
    numerical = (f(x + 1e-7) - f(x - 1e-7)) / (2e-7)
    analytical = f_prime(x)
    error = np.abs(numerical - analytical).max()
    print(f"{name}: error máximo = {error:.2e}")

x = np.array([-2, -1, 0.5, 1, 2])
verify_derivative(sigmoid, sigmoid_derivative, x, "Sigmoid")
verify_derivative(np.tanh, tanh_derivative, x, "Tanh")
```

### 1.3 Derivadas Parciales

```python
import numpy as np

"""
DERIVADA PARCIAL: Derivada respecto a UNA variable,
manteniendo las otras constantes.

Para f(x, y):
    ∂f/∂x = derivada respecto a x, tratando y como constante
    ∂f/∂y = derivada respecto a y, tratando x como constante

Notación: ∂ (partial) en lugar de d
"""

def f(x: float, y: float) -> float:
    """f(x, y) = x² + 3xy + y²"""
    return x**2 + 3*x*y + y**2

# Derivadas parciales analíticas:
# ∂f/∂x = 2x + 3y
# ∂f/∂y = 3x + 2y

def df_dx(x: float, y: float) -> float:
    """∂f/∂x = 2x + 3y"""
    return 2*x + 3*y

def df_dy(x: float, y: float) -> float:
    """∂f/∂y = 3x + 2y"""
    return 3*x + 2*y


# Derivada parcial numérica
def partial_derivative(f, var_idx: int, point: list, h: float = 1e-7) -> float:
    """
    Calcula ∂f/∂xᵢ en un punto dado.

    Args:
        f: función
        var_idx: índice de la variable (0 para x, 1 para y, etc.)
        point: punto donde evaluar [x, y, ...]
        h: paso pequeño
    """
    point_plus = point.copy()
    point_minus = point.copy()
    point_plus[var_idx] += h
    point_minus[var_idx] -= h
    return (f(*point_plus) - f(*point_minus)) / (2 * h)


# Verificar
point = [2.0, 3.0]
print(f"Punto: x={point[0]}, y={point[1]}")
print(f"f(x,y) = {f(*point)}")
print(f"\n∂f/∂x:")
print(f"  Analítica: {df_dx(*point)}")
print(f"  Numérica:  {partial_derivative(f, 0, point):.6f}")
print(f"\n∂f/∂y:")
print(f"  Analítica: {df_dy(*point)}")
print(f"  Numérica:  {partial_derivative(f, 1, point):.6f}")
```

---

## 💻 Parte 2: Gradiente

### 2.1 Definición del Gradiente

```python
import numpy as np

"""
GRADIENTE: Vector de todas las derivadas parciales.

Para f: Rⁿ → R (función de n variables que retorna un escalar):

∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

Propiedades importantes:
1. El gradiente apunta en la dirección de MÁXIMO ASCENSO
2. La magnitud indica qué tan rápido aumenta f en esa dirección
3. -∇f apunta en la dirección de MÁXIMO DESCENSO (usado en optimización)
"""

def compute_gradient(f, point: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """
    Calcula el gradiente de f en un punto usando diferencias finitas.

    Args:
        f: función f(x) donde x es un array
        point: punto donde calcular el gradiente
        h: paso para diferencias finitas

    Returns:
        gradiente como array
    """
    n = len(point)
    gradient = np.zeros(n)

    for i in range(n):
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[i] += h
        point_minus[i] -= h
        gradient[i] = (f(point_plus) - f(point_minus)) / (2 * h)

    return gradient


# Ejemplo: f(x, y) = x² + y²
def paraboloid(p: np.ndarray) -> float:
    """Paraboloide: f(x,y) = x² + y²"""
    return p[0]**2 + p[1]**2

# Gradiente analítico: ∇f = [2x, 2y]
def paraboloid_gradient_analytical(p: np.ndarray) -> np.ndarray:
    return np.array([2*p[0], 2*p[1]])


# Verificar
point = np.array([3.0, 4.0])
grad_numerical = compute_gradient(paraboloid, point)
grad_analytical = paraboloid_gradient_analytical(point)

print(f"Punto: {point}")
print(f"f(punto) = {paraboloid(point)}")
print(f"Gradiente numérico:  {grad_numerical}")
print(f"Gradiente analítico: {grad_analytical}")
```

### 2.2 Visualización del Gradiente

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_gradient():
    """Visualiza el gradiente como campo vectorial."""

    # Crear grid
    x = np.linspace(-3, 3, 15)
    y = np.linspace(-3, 3, 15)
    X, Y = np.meshgrid(x, y)

    # Función: f(x,y) = x² + y²
    Z = X**2 + Y**2

    # Gradiente: ∇f = [2x, 2y]
    U = 2 * X  # ∂f/∂x
    V = 2 * Y  # ∂f/∂y

    # Normalizar para visualización
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 0.1)
    V_norm = V / (magnitude + 0.1)

    plt.figure(figsize=(10, 8))

    # Contornos de nivel
    plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
    plt.colorbar(label='f(x,y) = x² + y²')

    # Flechas del gradiente
    plt.quiver(X, Y, U_norm, V_norm, magnitude, cmap='Reds', alpha=0.8)

    # Punto mínimo
    plt.plot(0, 0, 'g*', markersize=15, label='Mínimo global')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradiente de f(x,y) = x² + y²\nLas flechas apuntan hacia ARRIBA (máximo ascenso)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()

# visualize_gradient()  # Descomentar para ejecutar


---

### Intuición: Gradient Descent como “bajar una montaña en la niebla”

Imagina que estás en una montaña con niebla: no ves el valle (mínimo), pero puedes **sentir la pendiente local**.

- **El gradiente** `∇f(x)` apunta hacia el “subir más rápido”.
- Para bajar, te mueves en la dirección opuesta: `-∇f(x)`.
- El `learning_rate (α)` es el tamaño del paso: demasiado grande → te pasas/oscillas; demasiado pequeño → avanzas lento.

Checklist de diagnóstico rápido:

- **Si diverge:** `α` es demasiado grande o tu gradiente está mal.
- **Si converge muy lento:** `α` demasiado pequeño.
- **Si el loss baja y luego sube:** posible oscilación (reduce `α`).
- **Si no baja nunca:** gradiente incorrecto (haz gradient checking).

## 💻 Parte 3: Gradient Descent

### 3.1 Algoritmo Básico

#### Código generador de intuición (Protocolo D): superficie 3D + slider de `learning_rate`

Ejecuta el script (genera un HTML interactivo):

- [`visualizations/viz_gradient_3d.py`](../visualizations/viz_gradient_3d.py)

Ejemplos:

```bash
python3 visualizations/viz_gradient_3d.py --lr 0.01 --steps 30 --html-out artifacts/gd_lr0_01.html
python3 visualizations/viz_gradient_3d.py --lr 1.0 --steps 30 --html-out artifacts/gd_lr1_0.html
```

Checklist de uso:

- cambia `lr` a valores pequeños (ej. `0.01`) y observa convergencia suave
- sube `lr` (ej. `0.5` o `1.0`) y observa oscilación/divergencia

Objetivo: que puedas explicar la frase:

> “El learning rate no es un número mágico: controla cuánto avanzas en la dirección del gradiente, y si te pasas, rebotas.”

"""
GRADIENT DESCENT: Algoritmo de optimización iterativo.

Idea: Para minimizar f(x), moverse en dirección opuesta al gradiente.

Algoritmo:
    1. Inicializar x₀
    2. Repetir hasta convergencia:
       x_{t+1} = x_t - α · ∇f(x_t)

Donde α (alpha) es el "learning rate" (tasa de aprendizaje).
"""

def gradient_descent(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Gradient Descent para minimizar f.

    Args:
        f: función a minimizar
        grad_f: gradiente de f
        x0: punto inicial
        learning_rate: tasa de aprendizaje (α)
        max_iterations: máximo de iteraciones
        tolerance: criterio de parada (norma del gradiente)

    Returns:
        x_final: solución encontrada
        history_x: trayectoria de x
        history_f: valores de f en cada paso
    """
    x = x0.copy()
    history_x = [x.copy()]
    history_f = [f(x)]

    for i in range(max_iterations):
        # Calcular gradiente
        grad = grad_f(x)

        # Verificar convergencia
        if np.linalg.norm(grad) < tolerance:
            print(f"Convergió en iteración {i}")
            break

        # Actualizar x
        x = x - learning_rate * grad

        # Guardar historia
        history_x.append(x.copy())
        history_f.append(f(x))

    return x, history_x, history_f


# Ejemplo: Minimizar f(x,y) = x² + y²
def f(p: np.ndarray) -> float:
    return p[0]**2 + p[1]**2

def grad_f(p: np.ndarray) -> np.ndarray:
    return np.array([2*p[0], 2*p[1]])

# Ejecutar
x0 = np.array([4.0, 3.0])
x_final, history_x, history_f = gradient_descent(f, grad_f, x0, learning_rate=0.1)

print(f"\nPunto inicial: {x0}")
print(f"Mínimo encontrado: {x_final}")
print(f"f(mínimo) = {f(x_final):.6f}")
print(f"Iteraciones: {len(history_f)}")


### 3.2 Efecto del Learning Rate

"""
El learning rate (α) controla la velocidad de convergencia.

- α muy pequeño: Convergencia lenta
- α óptimo: Convergencia rápida y estable
- α muy grande: Oscilaciones, puede diverger
"""

import numpy as np
import matplotlib.pyplot as plt

def compare_learning_rates():
    """Compara diferentes learning rates."""

    def f(p):
        return p[0]**2 + p[1]**2

    def grad_f(p):
        return np.array([2*p[0], 2*p[1]])

    x0 = np.array([4.0, 3.0])

    learning_rates = [0.01, 0.1, 0.5, 0.9]

    plt.figure(figsize=(12, 4))

    for i, lr in enumerate(learning_rates):
        x_final, history_x, history_f = gradient_descent(
            f, grad_f, x0, learning_rate=lr, max_iterations=50
        )

        plt.subplot(1, 4, i+1)
        plt.plot(history_f, 'b-o', markersize=3)
        plt.xlabel('Iteración')
        plt.ylabel('f(x)')
        plt.title(f'α = {lr}')
        plt.yscale('log')
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle('Efecto del Learning Rate en Gradient Descent', y=1.02)
    plt.show()

    """
    Observaciones:
    - α muy pequeño (0.01): Convergencia muy lenta
    - α óptimo (0.1-0.5): Convergencia rápida y estable
    - α muy grande (0.9): Oscilaciones, puede diverger
    - α > 1: Generalmente diverge para este problema
    """

# compare_learning_rates()  # Descomentar para ejecutar


### 3.3 Funciones de Pérdida en ML

"""
FUNCIONES DE PÉRDIDA COMUNES Y SUS GRADIENTES

En ML, minimizamos una "función de pérdida" (loss function)
que mide qué tan mal están nuestras predicciones.
"""

# 1. MSE (Mean Squared Error) - Regresión
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Gradiente de MSE respecto a y_pred."""
    n = len(y_true)
    return 2 * (y_pred - y_true) / n


# 2. Binary Cross-Entropy - Clasificación binaria
def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Binary Cross-Entropy."""
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Evitar log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Gradiente de BCE respecto a y_pred."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)


# Demo
np.random.seed(42)
y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.2, 0.8, 0.9])

print("MSE Loss:", mse_loss(y_true, y_pred))
print("BCE Loss:", binary_cross_entropy(y_true, y_pred))


---

## 💻 Parte 4: Regla de la Cadena (Chain Rule)

### 4.0 Visualización: Grafo computacional (computational graph)

En Deep Learning, casi todo es una composición de funciones. El truco mental es pensar en un **grafo**:

```
x ──► z = w·x + b ──► a = σ(z) ──► L(a, y)

(forward)  verde: x→z→a→L
(backward) rojo:  dL/da → da/dz → dz/dw, dz/db
```

Regla de oro (chain rule):

```
dL/dw = dL/da · da/dz · dz/dw
dL/db = dL/da · da/dz · dz/db
```

### 4.0.1 Derivación paso a paso: `f(x) = x²`

Si `f(x) = x²`, entonces:

```
f'(x) = lim_{h→0} [(x+h)² - x²] / h
      = lim_{h→0} [x² + 2xh + h² - x²] / h
      = lim_{h→0} [2xh + h²] / h
      = lim_{h→0} [2x + h]
      = 2x
```

### 4.0.2 Derivación paso a paso: sigmoide `σ(z)`

Definición:

```
σ(z) = 1 / (1 + e^{-z})
```

Resultado clave:

```
σ'(z) = σ(z)(1 - σ(z))
```

Consejo práctico: cuando ya tienes `a = σ(z)`, usa `a(1-a)` para derivar, en vez de re-calcular `exp`.

### 4.1 Chain Rule en 1D

!!! note "REGLA DE LA CADENA (Chain Rule)"
    Si `y = f(g(x))`, entonces:

    `dy/dx = df/dg · dg/dx`

    O en notación de composición:

    `(f ∘ g)'(x) = f'(g(x)) · g'(x)`

    Esto es **fundamental** para Backpropagation.

```text
Ejemplo: y = (x² + 1)³

Sea g(x) = x² + 1  y  f(u) = u³
Entonces y = f(g(x))

dy/dx = f'(g(x)) · g'(x)
      = 3(x² + 1)² · 2x
      = 6x(x² + 1)²
```

```python
def g(x):
    return x**2 + 1


def f(u):
    return u**3


def y(x):
    return f(g(x))


def dy_dx_analytical(x):
    """Derivada usando chain rule."""
    return 6 * x * (x**2 + 1)**2


def dy_dx_numerical(x, h=1e-7):
    """Derivada numérica."""
    return (y(x + h) - y(x - h)) / (2 * h)


# Verificar
x = 2.0
print(f"y({x}) = {y(x)}")
print(f"dy/dx analítica:  {dy_dx_analytical(x)}")
print(f"dy/dx numérica:   {dy_dx_numerical(x):.6f}")
```


### 4.2 Chain Rule para Funciones Compuestas (Backprop Preview)

!!! note "CHAIN RULE PARA REDES NEURONALES"
    Una capa de red neuronal:

    `z = Wx + b` (transformación lineal)

    `a = σ(z)` (activación)

    Si `L` es la pérdida, necesitamos:

    `∂L/∂W`, `∂L/∂b` (para actualizar los pesos)

    Usando Chain Rule:

    `∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W`

    `∂L/∂b = ∂L/∂a · ∂a/∂z · ∂z/∂b`

```python
def simple_forward_backward():
    """
    Ejemplo simplificado de forward y backward pass.

    Red: x → [z = wx + b] → [a = sigmoid(z)] → [L = (a - y)²]
    """
    # Datos
    x = 2.0          # Input
    y_true = 1.0     # Target

    # Parámetros
    w = 0.5
    b = 0.1

    # ========== FORWARD PASS ==========
    z = w * x + b                    # z = wx + b
    a = 1 / (1 + np.exp(-z))         # a = sigmoid(z)
    L = (a - y_true) ** 2            # L = MSE

    print("=== FORWARD PASS ===")
    print(f"z = w*x + b = {w}*{x} + {b} = {z}")
    print(f"a = sigmoid(z) = {a:.4f}")
    print(f"L = (a - y)² = ({a:.4f} - {y_true})² = {L:.4f}")

    # ========== BACKWARD PASS (Chain Rule) ==========
    # Objetivo: calcular ∂L/∂w y ∂L/∂b

    # Paso 1: ∂L/∂a
    dL_da = 2 * (a - y_true)

    # Paso 2: ∂a/∂z = sigmoid'(z) = a(1-a)
    da_dz = a * (1 - a)

    # Paso 3: ∂z/∂w = x,  ∂z/∂b = 1
    dz_dw = x
    dz_db = 1

    # Aplicar Chain Rule
    dL_dz = dL_da * da_dz           # ∂L/∂z = ∂L/∂a · ∂a/∂z
    dL_dw = dL_dz * dz_dw           # ∂L/∂w = ∂L/∂z · ∂z/∂w
    dL_db = dL_dz * dz_db           # ∂L/∂b = ∂L/∂z · ∂z/∂b

    print("\n=== BACKWARD PASS (Chain Rule) ===")
    print(f"∂L/∂a = 2(a - y) = {dL_da:.4f}")
    print(f"∂a/∂z = a(1-a) = {da_dz:.4f}")
    print(f"∂z/∂w = x = {dz_dw}")
    print(f"∂z/∂b = 1")
    print(f"\n∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w = {dL_dw:.4f}")
    print(f"∂L/∂b = ∂L/∂a · ∂a/∂z · ∂z/∂b = {dL_db:.4f}")

    # ========== VERIFICACIÓN NUMÉRICA ==========
    h = 1e-7

    # ∂L/∂w numérica
    z_plus = (w + h) * x + b
    a_plus = 1 / (1 + np.exp(-z_plus))
    L_plus = (a_plus - y_true) ** 2

    z_minus = (w - h) * x + b
    a_minus = 1 / (1 + np.exp(-z_minus))
    L_minus = (a_minus - y_true) ** 2

    dL_dw_numerical = (L_plus - L_minus) / (2 * h)

    print(f"\n=== VERIFICACIÓN ===")
    print(f"∂L/∂w analítica: {dL_dw:.6f}")
    print(f"∂L/∂w numérica:  {dL_dw_numerical:.6f}")
    print(f"Error: {abs(dL_dw - dL_dw_numerical):.2e}")

    return dL_dw, dL_db

simple_forward_backward()

```


### 4.3 Backpropagation en una Red de 2 Capas

!!! note "RED NEURONAL DE 2 CAPAS"
    Arquitectura:

    `x (input) → z₁ = W₁x + b₁ → a₁ = sigmoid(z₁) → z₂ = W₂a₁ + b₂ → a₂ = sigmoid(z₂) → L = MSE(a₂, y)`

    Backpropagation usa Chain Rule repetidamente:

    `∂L/∂W₂ = ∂L/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂W₂`

    `∂L/∂W₁ = ∂L/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁`

```python
class SimpleNeuralNet:
    """Red neuronal de 2 capas para demostrar backprop."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Inicializar pesos (Xavier initialization)
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(output_size)

        # Cache para backprop
        self.cache = {}

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass guardando valores intermedios."""
        # Capa 1
        z1 = self.W1 @ x + self.b1
        a1 = self.sigmoid(z1)

        # Capa 2
        z2 = self.W2 @ a1 + self.b2
        a2 = self.sigmoid(z2)

        # Guardar para backprop
        self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

        return a2

    def backward(self, y_true: np.ndarray) -> dict:
        """
        Backward pass usando Chain Rule.

        Returns:
            Gradientes de todos los parámetros
        """
        x = self.cache['x']
        a1 = self.cache['a1']
        a2 = self.cache['a2']

        # ∂L/∂a₂ (MSE)
        dL_da2 = 2 * (a2 - y_true)

        # ∂a₂/∂z₂
        da2_dz2 = self.sigmoid_derivative(a2)

        # ∂L/∂z₂ = ∂L/∂a₂ · ∂a₂/∂z₂
        dL_dz2 = dL_da2 * da2_dz2

        # Gradientes de capa 2
        # ∂z₂/∂W₂ = a₁, ∂z₂/∂b₂ = 1
        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2

        # Propagar hacia atrás a capa 1
        # ∂z₂/∂a₁ = W₂
        dL_da1 = self.W2.T @ dL_dz2

        # ∂a₁/∂z₁
        da1_dz1 = self.sigmoid_derivative(a1)

        # ∂L/∂z₁
        dL_dz1 = dL_da1 * da1_dz1

        # Gradientes de capa 1
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1

        return {
            'dW1': dL_dW1, 'db1': dL_db1,
            'dW2': dL_dW2, 'db2': dL_db2
        }

    def update(self, gradients: dict, learning_rate: float):
        """Actualiza parámetros usando gradient descent."""
        self.W1 -= learning_rate * gradients['dW1']
        self.b1 -= learning_rate * gradients['db1']
        self.W2 -= learning_rate * gradients['dW2']
        self.b2 -= learning_rate * gradients['db2']


# Demo: XOR problem
def demo_xor():
    """Entrena la red para resolver XOR."""
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # 2x4
    y = np.array([[0], [1], [1], [0]]).T              # 1x4

    # Crear red
    net = SimpleNeuralNet(input_size=2, hidden_size=4, output_size=1)

    # Entrenar
    losses = []
    for epoch in range(10000):
        total_loss = 0
        for i in range(4):
            # Forward
            output = net.forward(X[:, i])
            loss = (output - y[:, i]) ** 2
            total_loss += loss[0]

            # Backward
            gradients = net.backward(y[:, i])

            # Update
            net.update(gradients, learning_rate=0.5)

        losses.append(total_loss / 4)

        if epoch % 2000 == 0:
            print(f"Epoch {epoch}: Loss = {losses[-1]:.4f}")

    # Test
    print("\n=== Resultados XOR ===")
    for i in range(4):
        pred = net.forward(X[:, i])
        print(f"Input: {X[:, i]} → Pred: {pred[0]:.3f} (Target: {y[0, i]})")

demo_xor()

```


---

## 🎯 Ejercicios por tema (progresivos) + Soluciones

Reglas:

- **Intenta primero** sin mirar la solución.
- **Timebox sugerido:** 15–30 min por ejercicio.
- **Éxito mínimo:** tu solución debe pasar los `assert`.

---

### Ejercicio 3.1: Derivada numérica (diferencias finitas) vs derivada analítica

#### Enunciado

1) **Básico**

- Implementa la derivada numérica central: `f'(x) ≈ (f(x+h)-f(x-h))/(2h)`.

2) **Intermedio**

- Para `f(x) = x^3 + 2x`, implementa `f'(x)` analítica y compara en varios puntos.

3) **Avanzado**

- Prueba `h=1e-2, 1e-4, 1e-6` y verifica que el error no crece de forma absurda.

#### Solución

```python
import numpy as np

def num_derivative_central(f, x: float, h: float = 1e-6) -> float:
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def f(x: float) -> float:
    return x**3 + 2.0 * x


def f_prime(x: float) -> float:
    return 3.0 * x**2 + 2.0


xs = [-2.0, -0.5, 0.0, 1.0, 3.0]
for x in xs:
    approx = num_derivative_central(f, x, h=1e-6)
    exact = f_prime(x)
    assert np.isclose(approx, exact, rtol=1e-6, atol=1e-6)


x0 = 1.234
errs = []
for h in [1e-2, 1e-4, 1e-6]:
    approx = num_derivative_central(f, x0, h=h)
    errs.append(abs(approx - f_prime(x0)))
assert errs[1] <= errs[0] + 1e-6
```

---

### Ejercicio 3.2: Derivadas parciales y gradiente (2D)

#### Enunciado

Sea `f(x, y) = x^2 y + sin(y)`.

1) **Básico**

- Deriva analíticamente `∂f/∂x` y `∂f/∂y`.

2) **Intermedio**

- Implementa el gradiente `∇f(x,y)` y evalúalo en un punto.

3) **Avanzado**

- Verifica con gradiente numérico (diferencias centrales) que tu gradiente analítico es correcto.

#### Solución

```python
import numpy as np

def f_xy(x: float, y: float) -> float:
    return x**2 * y + np.sin(y)


def grad_f_xy(x: float, y: float) -> np.ndarray:
    dfdx = 2.0 * x * y
    dfdy = x**2 + np.cos(y)
    return np.array([dfdx, dfdy], dtype=float)


def num_grad_2d(f, x: float, y: float, h: float = 1e-6) -> np.ndarray:
    dfdx = (f(x + h, y) - f(x - h, y)) / (2.0 * h)
    dfdy = (f(x, y + h) - f(x, y - h)) / (2.0 * h)
    return np.array([dfdx, dfdy], dtype=float)


x0, y0 = 1.2, -0.7
g_anal = grad_f_xy(x0, y0)
g_num = num_grad_2d(f_xy, x0, y0)
assert np.allclose(g_anal, g_num, rtol=1e-5, atol=1e-6)
```

---

### Ejercicio 3.3: Derivada direccional (intuición: el gradiente manda)

#### Enunciado

1) **Básico**

- Para `f(x,y)=x^2 y + sin(y)`, calcula `∇f(x0,y0)`.

2) **Intermedio**

- Dado un vector dirección unitario `u`, calcula la derivada direccional `D_u f = ∇f · u`.

3) **Avanzado**

- Verifica numéricamente `D_u f` con diferencias finitas sobre `p(t)=p0 + t u`.

#### Solución

```python
import numpy as np

def f_xy(x: float, y: float) -> float:
    return x**2 * y + np.sin(y)


def grad_f_xy(x: float, y: float) -> np.ndarray:
    return np.array([2.0 * x * y, x**2 + np.cos(y)], dtype=float)


x0, y0 = 0.5, 1.0
g = grad_f_xy(x0, y0)

u = np.array([3.0, 4.0], dtype=float)
u = u / np.linalg.norm(u)

dir_anal = float(np.dot(g, u))

h = 1e-6
f_plus = f_xy(x0 + h * u[0], y0 + h * u[1])
f_minus = f_xy(x0 - h * u[0], y0 - h * u[1])
dir_num = float((f_plus - f_minus) / (2.0 * h))

assert np.isclose(dir_anal, dir_num, rtol=1e-5, atol=1e-6)
```

---

### Ejercicio 3.4: Jacobiano (función vectorial)

#### Enunciado

Sea `g(x1,x2) = [x1^2 + x2, sin(x1 x2)]`.

1) **Básico**

- Escribe el Jacobiano `J` (matriz 2x2) a mano.

2) **Intermedio**

- Implementa `J_analytical(x)`.

3) **Avanzado**

- Verifica con Jacobiano numérico (diferencias centrales) que `J` coincide.

#### Solución

```python
import numpy as np

def g(x: np.ndarray) -> np.ndarray:
    x1, x2 = float(x[0]), float(x[1])
    return np.array([x1**2 + x2, np.sin(x1 * x2)], dtype=float)


def J_analytical(x: np.ndarray) -> np.ndarray:
    x1, x2 = float(x[0]), float(x[1])
    dg1_dx1 = 2.0 * x1
    dg1_dx2 = 1.0
    dg2_dx1 = np.cos(x1 * x2) * x2
    dg2_dx2 = np.cos(x1 * x2) * x1
    return np.array([[dg1_dx1, dg1_dx2], [dg2_dx1, dg2_dx2]], dtype=float)


def J_numeric(g, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    x = x.astype(float)
    m = g(x).shape[0]
    n = x.shape[0]
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        J[:, j] = (g(x + h * e) - g(x - h * e)) / (2.0 * h)
    return J


x0 = np.array([0.7, -1.1])
Ja = J_analytical(x0)
Jn = J_numeric(g, x0)
assert np.allclose(Ja, Jn, rtol=1e-5, atol=1e-6)
```

---

### Ejercicio 3.5: Hessiano (curvatura local) + convexidad

#### Enunciado

Sea `f(x1,x2) = x1^2 + 2 x2^2`.

1) **Básico**

- Calcula el Hessiano `H`.

2) **Intermedio**

- Verifica que `H` es simétrico.

3) **Avanzado**

- Verifica que `H` es definido positivo (eigenvalores > 0).

#### Solución

```python
import numpy as np

H = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=float)
assert np.allclose(H, H.T)

eigvals = np.linalg.eigvals(H)
assert np.all(eigvals > 0)
```

---

### Ejercicio 3.6: Gradient Descent 1D (convergencia)

#### Enunciado

Minimiza `f(x) = (x - 3)^2` con Gradient Descent.

1) **Básico**

- Implementa la regla de actualización: `x <- x - α f'(x)`.

2) **Intermedio**

- Registra `x_t` y `f(x_t)`.

3) **Avanzado**

- Usa un criterio de parada por `|grad| < tol`.

#### Solución

```python
import numpy as np

def f(x: float) -> float:
    return (x - 3.0) ** 2


def grad_f(x: float) -> float:
    return 2.0 * (x - 3.0)


x = 10.0
alpha = 0.1
history = []
for _ in range(200):
    g = grad_f(x)
    history.append((x, f(x)))
    if abs(g) < 1e-8:
        break
    x = x - alpha * g

assert abs(x - 3.0) < 1e-4
assert history[-1][1] <= history[0][1]
```

---

### Ejercicio 3.7: Efecto del learning rate (estabilidad)

#### Enunciado

Minimiza `f(x)=x^2` con Gradient Descent desde `x0=1`.

1) **Básico**

- Deriva la actualización: `x_{t+1} = (1 - 2α) x_t`.

2) **Intermedio**

- Prueba con `α=0.25` y verifica que `|x_t|` decrece.

3) **Avanzado**

- Prueba con `α=1.1` y verifica divergencia (`|x_t|` crece).

#### Solución

```python
import numpy as np

def run_gd_x2(alpha: float, steps: int = 10) -> np.ndarray:
    x = 1.0
    xs = [x]
    for _ in range(steps):
        grad = 2.0 * x
        x = x - alpha * grad
        xs.append(x)
    return np.array(xs)


xs_good = run_gd_x2(alpha=0.25, steps=10)
assert abs(xs_good[-1]) < abs(xs_good[0])

xs_bad = run_gd_x2(alpha=1.1, steps=10)
assert abs(xs_bad[-1]) > abs(xs_bad[0])
```

---

### Ejercicio 3.8: Gradient checking (vector) + error relativo

#### Enunciado

1) **Básico**

- Implementa gradiente numérico (diferencias centrales) para `f(w)`.

2) **Intermedio**

- Usa `f(w)=∑ w_i^3` cuyo gradiente analítico es `3 w_i^2`.

3) **Avanzado**

- Calcula error relativo `||g_num - g_anal|| / (||g_num|| + ||g_anal|| + eps)`.

#### Solución

```python
import numpy as np

def f(w: np.ndarray) -> float:
    return float(np.sum(w ** 3))


def grad_analytical(w: np.ndarray) -> np.ndarray:
    return 3.0 * (w ** 2)


def grad_numeric(f, w: np.ndarray, h: float = 1e-6) -> np.ndarray:
    w = w.astype(float)
    g = np.zeros_like(w)
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = 1.0
        g[i] = (f(w + h * e) - f(w - h * e)) / (2.0 * h)
    return g


np.random.seed(0)
w = np.random.randn(5)
g_a = grad_analytical(w)
g_n = grad_numeric(f, w)

eps = 1e-12
rel_err = np.linalg.norm(g_n - g_a) / (np.linalg.norm(g_n) + np.linalg.norm(g_a) + eps)
assert rel_err < 1e-7
```

---

### Ejercicio 3.9: Chain Rule (neurona + MSE) + verificación numérica

#### Enunciado

Una neurona:

- `z = w·x + b`
- `ŷ = σ(z)`
- `L = (ŷ - y)^2`

1) **Básico**

- Deriva `dL/dz` usando chain rule.

2) **Intermedio**

- Deriva `dL/dw` y `dL/db`.

3) **Avanzado**

- Verifica tus gradientes con diferencias centrales.

#### Solución

```python
import numpy as np

def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def loss_mse(y_hat: float, y: float) -> float:
    return float((y_hat - y) ** 2)


def forward(w: np.ndarray, b: float, x: np.ndarray, y: float) -> float:
    z = float(np.dot(w, x) + b)
    y_hat = sigmoid(z)
    return loss_mse(y_hat, y)


def grads_analytical(w: np.ndarray, b: float, x: np.ndarray, y: float):
    z = float(np.dot(w, x) + b)
    y_hat = sigmoid(z)

    dL_dyhat = 2.0 * (y_hat - y)
    dyhat_dz = y_hat * (1.0 - y_hat)
    dL_dz = dL_dyhat * dyhat_dz

    dL_dw = dL_dz * x
    dL_db = dL_dz
    return dL_dw.astype(float), float(dL_db)


def grads_numeric(w: np.ndarray, b: float, x: np.ndarray, y: float, h: float = 1e-6):
    gw = np.zeros_like(w, dtype=float)
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = 1.0
        gw[i] = (forward(w + h * e, b, x, y) - forward(w - h * e, b, x, y)) / (2.0 * h)

    gb = (forward(w, b + h, x, y) - forward(w, b - h, x, y)) / (2.0 * h)
    return gw, float(gb)


np.random.seed(1)
w = np.random.randn(3)
b = 0.1
x = np.random.randn(3)
y = 1.0

gw_a, gb_a = grads_analytical(w, b, x, y)
gw_n, gb_n = grads_numeric(w, b, x, y)

assert np.allclose(gw_a, gw_n, rtol=1e-5, atol=1e-6)
assert np.isclose(gb_a, gb_n, rtol=1e-5, atol=1e-6)
```

---

## Entregable del Módulo

### Script: `gradient_descent_demo.py`

```python
"""
Gradient Descent Demo - Visualización de Optimización

Este script implementa Gradient Descent desde cero y visualiza
la trayectoria de optimización en diferentes funciones.

Autor: [Tu nombre]
Módulo: 03 - Cálculo Multivariante
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-8
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Implementación de Gradient Descent.

    Args:
        f: función objetivo
        grad_f: gradiente de f
        x0: punto inicial
        learning_rate: α
        max_iterations: máximo de iteraciones
        tolerance: criterio de convergencia

    Returns:
        x_final, history_x, history_f
    """
    x = x0.copy().astype(float)
    history_x = [x.copy()]
    history_f = [f(x)]

    for i in range(max_iterations):
        grad = grad_f(x)

        if np.linalg.norm(grad) < tolerance:
            break

        x = x - learning_rate * grad
        history_x.append(x.copy())
        history_f.append(f(x))

    return x, history_x, history_f


def visualize_optimization(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    learning_rate: float,
    title: str,
    xlim: Tuple[float, float] = (-5, 5),
    ylim: Tuple[float, float] = (-5, 5)
):
    """Visualiza la trayectoria de optimización."""

    x_final, history_x, history_f = gradient_descent(
        f, grad_f, x0, learning_rate, max_iterations=50
    )

    # Crear grid para contornos
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Contornos y trayectoria
    ax1 = axes[0]
    contour = ax1.contour(X, Y, Z, levels=30, cmap='viridis')
    ax1.clabel(contour, inline=True, fontsize=8)

    # Trayectoria
    history_x = np.array(history_x)
    ax1.plot(history_x[:, 0], history_x[:, 1], 'r.-', markersize=8, linewidth=1.5)
    ax1.plot(history_x[0, 0], history_x[0, 1], 'go', markersize=12, label='Inicio')
    ax1.plot(history_x[-1, 0], history_x[-1, 1], 'r*', markersize=15, label='Final')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'{title}\nα = {learning_rate}')
    ax1.legend()
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # Plot 2: Convergencia
    ax2 = axes[1]
    ax2.semilogy(history_f, 'b-o', markersize=4)
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('f(x) (escala log)')
    ax2.set_title('Convergencia')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'gd_{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()

    print(f"\n{title}")
    print(f"  Punto inicial: {x0}")
    print(f"  Mínimo encontrado: {x_final}")
    print(f"  f(mínimo): {f(x_final):.6f}")
    print(f"  Iteraciones: {len(history_f)}")


def main():
    """Ejecutar demos."""

    # === Función 1: Paraboloide ===
    def paraboloid(p):
        return p[0]**2 + p[1]**2

    def grad_paraboloid(p):
        return np.array([2*p[0], 2*p[1]])

    visualize_optimization(
        paraboloid, grad_paraboloid,
        x0=np.array([4.0, 3.0]),
        learning_rate=0.1,
        title="Paraboloide f(x,y) = x² + y²"
    )

    # === Función 2: Rosenbrock (más difícil) ===
    def rosenbrock(p):
        return (1 - p[0])**2 + 100*(p[1] - p[0]**2)**2

    def grad_rosenbrock(p):
        dx = -2*(1 - p[0]) - 400*p[0]*(p[1] - p[0]**2)
        dy = 200*(p[1] - p[0]**2)
        return np.array([dx, dy])

    visualize_optimization(
        rosenbrock, grad_rosenbrock,
        x0=np.array([-1.0, 1.0]),
        learning_rate=0.001,
        title="Rosenbrock f(x,y) = (1-x)² + 100(y-x²)²",
        xlim=(-2, 2),
        ylim=(-1, 3)
    )

    # === Función 3: Cuadrática elíptica ===
    def elliptic(p):
        return p[0]**2 + 10*p[1]**2

    def grad_elliptic(p):
        return np.array([2*p[0], 20*p[1]])

    visualize_optimization(
        elliptic, grad_elliptic,
        x0=np.array([4.0, 2.0]),
        learning_rate=0.05,
        title="Elíptica f(x,y) = x² + 10y²"
    )


if __name__ == "__main__":
    main()

```


---
## Entregable Obligatorio v3.3

### Script: `grad_check.py`

```python
"""
Gradient Checking - Validación de Derivadas
Técnica estándar de CS231n Stanford para debugging de backprop.

Autor: [Tu nombre]
Módulo: 03 - Cálculo Multivariante
"""
import numpy as np
from typing import Callable, Dict, Tuple


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Calcula el gradiente numérico usando diferencias centrales.

    Args:
        f: Función escalar f(x) -> float
        x: Punto donde calcular el gradiente
        epsilon: Tamaño del paso (default: 1e-5)

    Returns:
        Gradiente numérico aproximado
    """
    grad = np.zeros_like(x)

    # Iterar sobre cada dimensión
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        # f(x + epsilon)
        x[idx] = old_value + epsilon
        fx_plus = f(x)

        # f(x - epsilon)
        x[idx] = old_value - epsilon
        fx_minus = f(x)

        # Diferencias centrales: (f(x+ε) - f(x-ε)) / 2ε
        grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)

        # Restaurar valor original
        x[idx] = old_value
        it.iternext()

    return grad


def gradient_check(
    analytic_grad: np.ndarray,
    numerical_grad: np.ndarray,
    threshold: float = 1e-7
) -> Tuple[bool, float]:
    """
    Compara gradiente analítico vs numérico.

    Args:
        analytic_grad: Gradiente calculado con backprop
        numerical_grad: Gradiente calculado numéricamente
        threshold: Umbral de error aceptable

    Returns:
        (passed, relative_error)
    """
    # Error relativo: ||a - n|| / (||a|| + ||n||)
    diff = np.linalg.norm(analytic_grad - numerical_grad)
    norm_sum = np.linalg.norm(analytic_grad) + np.linalg.norm(numerical_grad)

    if norm_sum == 0:
        relative_error = 0.0
    else:
        relative_error = diff / norm_sum

    passed = relative_error < threshold
    return passed, relative_error


# === EJEMPLO: Validar gradiente de MSE Loss ===

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_pred - y_true) ** 2))

def mse_gradient_analytic(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Gradiente analítico de MSE respecto a y_pred."""
    n = len(y_true)
    return 2 * (y_pred - y_true) / n


def test_mse_gradient():
    """Test: Validar gradiente de MSE."""
    print("=" * 60)
    print("GRADIENT CHECK: MSE Loss")
    print("=" * 60)

    np.random.seed(42)
    y_pred = np.random.randn(10)
    y_true = np.random.randn(10)

    # Gradiente analítico
    grad_analytic = mse_gradient_analytic(y_pred, y_true)

    # Gradiente numérico
    def loss_fn(pred):
        return mse_loss(pred, y_true)

    grad_numerical = numerical_gradient(loss_fn, y_pred.copy())

    # Comparar
    passed, error = gradient_check(grad_analytic, grad_numerical)

    print(f"Gradiente Analítico: {grad_analytic[:3]}...")
    print(f"Gradiente Numérico:  {grad_numerical[:3]}...")
    print(f"Error Relativo: {error:.2e}")
    print(f"Resultado: {'✓ PASSED' if passed else '✗ FAILED'}")

    return passed


# === EJEMPLO: Validar gradiente de Sigmoid ===

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative_analytic(z: np.ndarray) -> np.ndarray:
    """Derivada analítica: σ'(z) = σ(z)(1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)


def test_sigmoid_gradient():
    """Test: Validar derivada de sigmoid."""
    print("\n" + "=" * 60)
    print("GRADIENT CHECK: Sigmoid Derivative")
    print("=" * 60)

    np.random.seed(42)
    z = np.random.randn(5)

    # Derivada analítica
    grad_analytic = sigmoid_derivative_analytic(z)

    # Derivada numérica (para cada elemento)
    def sigmoid_element(z_arr):
        return float(np.sum(sigmoid(z_arr)))  # Suma para tener escalar

    grad_numerical = numerical_gradient(sigmoid_element, z.copy())

    # Comparar
    passed, error = gradient_check(grad_analytic, grad_numerical)

    print(f"Derivada Analítica: {grad_analytic}")
    print(f"Derivada Numérica:  {grad_numerical}")
    print(f"Error Relativo: {error:.2e}")
    print(f"Resultado: {'✓ PASSED' if passed else '✗ FAILED'}")

    return passed


# === EJEMPLO: Validar gradiente de una capa lineal ===

def test_linear_layer_gradient():
    """Test: Validar gradiente de capa lineal y = Wx + b."""
    print("\n" + "=" * 60)
    print("GRADIENT CHECK: Linear Layer (y = Wx + b)")
    print("=" * 60)

    np.random.seed(42)

    # Dimensiones
    n_in, n_out = 4, 3

    # Parámetros
    W = np.random.randn(n_out, n_in)
    b = np.random.randn(n_out)
    x = np.random.randn(n_in)
    y_true = np.random.randn(n_out)

    # Forward + Loss
    def forward_and_loss(W_flat):
        W_reshaped = W_flat.reshape(n_out, n_in)
        y_pred = W_reshaped @ x + b
        return mse_loss(y_pred, y_true)

    # Gradiente analítico de W
    y_pred = W @ x + b
    dL_dy = 2 * (y_pred - y_true) / n_out  # Gradiente de MSE
    dL_dW_analytic = np.outer(dL_dy, x)    # ∂L/∂W = ∂L/∂y · x^T

    # Gradiente numérico de W
    dL_dW_numerical = numerical_gradient(forward_and_loss, W.flatten().copy())
    dL_dW_numerical = dL_dW_numerical.reshape(n_out, n_in)

    # Comparar
    passed, error = gradient_check(
        dL_dW_analytic.flatten(),
        dL_dW_numerical.flatten()
    )

    print(f"Error Relativo: {error:.2e}")
    print(f"Resultado: {'✓ PASSED' if passed else '✗ FAILED'}")

    return passed


def main():
    """Ejecutar todos los gradient checks."""
    print("\n" + "=" * 60)
    print("       GRADIENT CHECKING SUITE")
    print("       Validación Matemática v3.3")
    print("=" * 60)

    results = []
    results.append(("MSE Loss", test_mse_gradient()))
    results.append(("Sigmoid", test_sigmoid_gradient()))
    results.append(("Linear Layer", test_linear_layer_gradient()))

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("-" * 60)
    if all_passed:
        print("✓ TODOS LOS GRADIENT CHECKS PASARON")
        print("  Tu implementación de derivadas es correcta.")
    else:
        print("✗ ALGUNOS GRADIENT CHECKS FALLARON")
        print("  Revisa tu implementación de backprop.")

    return all_passed


if __name__ == "__main__":
    main()

```
---
## 🧩 Consolidación (errores comunes + debugging v5 + reto Feynman)

### Errores comunes

- **Confundir derivada local con “dirección global”:** el gradiente solo te da información local.
- **`learning_rate` demasiado grande:** puede oscilar o divergir aunque el gradiente sea correcto.
- **Estabilidad numérica:** `exp(z)` puede overflow; usa `np.clip` cuando aplique.
- **Gradient checking mal aplicado:** `ε` demasiado pequeño puede amplificar ruido numérico.

### Debugging / validación (v5)

- Si tu entrenamiento es inestable o no baja el loss, valida derivadas con `grad_check.py`.
- Registra hallazgos en `study_tools/DIARIO_ERRORES.md`.
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Reto Feynman (tablero blanco)

Explica en 5 líneas o menos:

1) ¿Qué significa “seguir `-∇f`” y por qué eso baja la función?
2) Dibuja el grafo `x→z→a→L` y explica por qué multiplicas derivadas.
3) ¿Por qué gradient checking detecta bugs de backprop?

---

## ✅ Checklist de Finalización (v3.3)

### Conocimiento
- [ ] Puedo calcular derivadas de funciones comunes (polinomios, exp, log)
- [ ] Entiendo derivadas parciales y puedo calcularlas
- [ ] Puedo calcular el gradiente de una función multivariable
- [ ] Implementé Gradient Descent desde cero
- [ ] Entiendo el efecto del learning rate
- [ ] Puedo aplicar la Chain Rule a funciones compuestas
- [ ] Entiendo cómo la Chain Rule se aplica en Backpropagation
- [ ] Puedo derivar ∂L/∂w para una neurona simple

### Entregables v3.3
- [ ] `gradient_descent_demo.py` funcional
- [ ] **`grad_check.py` implementado y todos los tests pasan**
- [ ] Validé mis derivadas de sigmoid, MSE y capa lineal

### Metodología Feynman
- [ ] Puedo explicar Chain Rule en 5 líneas sin jerga
- [ ] Puedo explicar por qué gradient checking funciona

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [02_ALGEBRA_LINEAL_ML](02_ALGEBRA_LINEAL_ML.md) | [00_INDICE](00_INDICE.md) | [04_PROBABILIDAD_ML](04_PROBABILIDAD_ML.md) |
