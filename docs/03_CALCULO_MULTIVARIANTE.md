# Módulo 03 - Cálculo Multivariante para Deep Learning

> **🎯 Objetivo:** Dominar derivadas, gradientes y la Chain Rule para entender Backpropagation  
> **Fase:** 1 - Fundamentos Matemáticos | **Semanas 6-8**  
> **Prerrequisitos:** Módulo 02 (Álgebra Lineal para ML)

---

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
```

---

## 💻 Parte 3: Gradient Descent

### 3.1 Algoritmo Básico

```python
import numpy as np
from typing import Callable, List, Tuple

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
```

### 3.2 Efecto del Learning Rate

```python
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
```

### 3.3 Funciones de Pérdida en ML

```python
import numpy as np

"""
FUNCIONES DE PÉRDIDA COMUNES Y SUS GRADIENTES

En ML, minimizamos una "función de pérdida" (loss function)
que mide qué tan mal están nuestras predicciones.
"""

# 1. MSE (Mean Squared Error) - Regresión
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE = (1/n) Σ (y_true - y_pred)²
    """
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    ∂MSE/∂y_pred = (2/n) Σ (y_pred - y_true)
                 = (2/n) (y_pred - y_true)
    """
    n = len(y_true)
    return (2 / n) * (y_pred - y_true)


# 2. Binary Cross-Entropy - Clasificación binaria
def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    BCE = -(1/n) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Evitar log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    ∂BCE/∂y_pred = (1/n) · (y_pred - y_true) / (y_pred · (1 - y_pred))
    
    Simplificación cuando y_pred = σ(z):
    ∂BCE/∂z = (1/n) · (y_pred - y_true)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)


# Demo
np.random.seed(42)
y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.2, 0.8, 0.9])

print("MSE Loss:", mse_loss(y_true, y_pred))
print("BCE Loss:", binary_cross_entropy(y_true, y_pred))
```

---

## 💻 Parte 4: Regla de la Cadena (Chain Rule)

### 4.1 Chain Rule en 1D

```python
import numpy as np

"""
REGLA DE LA CADENA (Chain Rule)

Si y = f(g(x)), entonces:
    dy/dx = df/dg · dg/dx
    
O en notación de composición:
    (f ∘ g)'(x) = f'(g(x)) · g'(x)

Esto es FUNDAMENTAL para Backpropagation.
"""

# Ejemplo: y = (x² + 1)³
# 
# Sea g(x) = x² + 1  y  f(u) = u³
# Entonces y = f(g(x))
#
# dy/dx = f'(g(x)) · g'(x)
#       = 3(x² + 1)² · 2x
#       = 6x(x² + 1)²

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

```python
import numpy as np

"""
CHAIN RULE PARA REDES NEURONALES

Una capa de red neuronal:
    z = Wx + b       (transformación lineal)
    a = σ(z)         (activación)
    
Si L es la pérdida, necesitamos:
    ∂L/∂W, ∂L/∂b     (para actualizar los pesos)

Usando Chain Rule:
    ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
    ∂L/∂b = ∂L/∂a · ∂a/∂z · ∂z/∂b
"""

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

```python
import numpy as np

"""
RED NEURONAL DE 2 CAPAS

Arquitectura:
    x (input) 
    → z₁ = W₁x + b₁ 
    → a₁ = sigmoid(z₁) 
    → z₂ = W₂a₁ + b₂ 
    → a₂ = sigmoid(z₂) 
    → L = MSE(a₂, y)

Backpropagation usa Chain Rule repetidamente:
    ∂L/∂W₂ = ∂L/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂W₂
    ∂L/∂W₁ = ∂L/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
"""

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

## 📦 Entregable del Módulo

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

## 🔬 Gradient Checking: Validación Matemática (v3.3)

> ⚠️ **CRÍTICO:** El mayor riesgo en ML es implementar backpropagation incorrectamente. El código puede correr, el loss puede bajar, pero el gradiente estar mal. **Esta técnica es estándar en CS231n de Stanford.**

### Concepto: Derivada Numérica vs Analítica

```
GRADIENT CHECKING

Tu gradiente analítico (backprop):
    ∂L/∂w = [valor calculado con Chain Rule]

Gradiente numérico (aproximación):
    ∂L/∂w ≈ [L(w + ε) - L(w - ε)] / (2ε)

Si |analítico - numérico| > 10⁻⁷ → TU IMPLEMENTACIÓN TIENE UN BUG
```

### Script: `grad_check.py` (Entregable Obligatorio v3.3)

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


# ============================================================
# EJEMPLO: Validar gradiente de MSE Loss
# ============================================================

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


# ============================================================
# EJEMPLO: Validar gradiente de Sigmoid
# ============================================================

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


# ============================================================
# EJEMPLO: Validar gradiente de una capa lineal
# ============================================================

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

### Cómo Usar Gradient Checking

```python
# En tu código de backprop:

# 1. Calcula el gradiente analítico (tu implementación)
grad_analytic = my_backward_pass(...)

# 2. Calcula el gradiente numérico
def loss_wrapper(params):
    return forward_pass(params, ...)

grad_numerical = numerical_gradient(loss_wrapper, params)

# 3. Compara
passed, error = gradient_check(grad_analytic, grad_numerical)
if not passed:
    raise ValueError(f"Gradient check failed! Error: {error:.2e}")
```

### Reglas del Gradient Checking

| Error Relativo | Interpretación |
|----------------|----------------|
| < 10⁻⁷ | ✓ Excelente - tu gradiente es correcto |
| 10⁻⁷ a 10⁻⁵ | ⚠️ Sospechoso - revisa tu código |
| > 10⁻⁵ | ✗ Bug - tu backprop está mal |

> ⚠️ **Nota:** Desactiva gradient checking durante el entrenamiento real (es lento). Solo úsalo para validar tu implementación.

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
