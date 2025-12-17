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

### Ritmo semanal recomendado (aplicado a Semanas 6–8)

- **Lunes y Martes (Concepto):** prioriza Chain Rule. Si solo dominas 1 cosa de cálculo para DL, es esta.
- **Miércoles y Jueves (Implementación):** implementa y valida: gradientes analíticos + diferencias finitas.
- **Viernes (Romper cosas):** fuerza fallos típicos y explícales con teoría:
  - sube `learning_rate` hasta divergir y describe la señal en `history_f`
  - prueba entradas grandes en sigmoide (`z` muy positivo/negativo) y explica saturación (gradiente ~ 0)
  - cambia `epsilon` en diferencias finitas y observa cuándo se rompe (ruido numérico)

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 1.1: Concepto de Derivada</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Derivada como pendiente local + verificación numérica
- **ID (opcional):** `M03-T01_1`
- **Duración estimada:** 60–90 min
- **Nivel:** Fundamentos
- **Dependencias:** Álgebra básica, intuición de función

#### 2) Objetivo(s) de aprendizaje (medibles)
- Calcular una derivada analítica simple (ej. `x²`) y **validarla** con diferencias finitas.
- Explicar qué representa `h` y cómo afecta el error numérico.

#### 3) Relevancia y contexto
- En ML, el gradiente es “la derivada” que guía la optimización; si no controlas el concepto, backprop se vuelve magia.

#### 4) Mapa conceptual / conceptos clave
- derivada = tasa de cambio local
- recta tangente
- diferencias finitas (central)

#### 5) Definiciones y fórmulas esenciales
- `f'(x) = lim[h→0] (f(x+h) - f(x-h)) / (2h)` (central).

#### 6) Explicación didáctica (2 niveles)
- **Intuición:** “qué tan inclinada está la curva en ese punto”.
- **Operativa:** compara derivada analítica vs numérica y mira el error.

#### 7) Ejemplo modelado
- `f(x)=x²` → `f'(x)=2x`; valida en `x=3`.

#### 8) Práctica guiada
- Cambia `x` (ej. `-2`, `0.5`, `10`) y observa el error.

#### 9) Práctica independiente / transferencia
- Repite con `f(x)=x³` y `f(x)=sin(x)` (deriva y verifica).

#### 10) Evaluación
- ¿Por qué la diferencia central suele ser más precisa que la forward difference?

#### 11) Errores comunes
- Elegir `h` demasiado pequeño (ruido numérico) o demasiado grande (sesgo).

#### 12) Retención
- (día 2) escribe el esquema “analítica vs numérica → error” y explica qué valida.

#### 13) Diferenciación
- Avanzado: prueba funciones con cambios bruscos (ej. `abs`) y discute no-diferenciabilidad.

#### 14) Recursos
- GLOSARIO: Derivative.

#### 15) Nota docente
- Exigir siempre “derivada + validación numérica” al introducir un nuevo gradiente.
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 1.2: Derivadas Comunes en ML</strong></summary>

#### 1) Metadatos
- **Título:** “derivadas que debes memorizar” + verificación automática
- **ID (opcional):** `M03-T01_2`
- **Duración estimada:** 60–120 min
- **Nivel:** Fundamentos
- **Dependencias:** 1.1

#### 2) Objetivos
- Memorizar y aplicar derivadas de `exp`, `log`, potencias y activaciones (sigmoid/tanh/ReLU).
- Implementar un verificador numérico y usar el error para detectar bugs.

#### 3) Relevancia
- Estas derivadas aparecen en backprop: activación + loss + capa lineal.

#### 4) Conceptos clave
- `σ'(x) = σ(x)(1-σ(x))`
- `tanh'(x) = 1-tanh(x)^2`
- ReLU derivada por tramos

#### 5) Fórmulas esenciales
- Regla de la cadena (adelanto): derivadas se multiplican en composiciones.

#### 6) Explicación didáctica
- **Patrón ML:** implementa `f`, implementa `f'`, valida con diferencias finitas.

#### 7) Ejemplo modelado
- Verificación de `sigmoid` y `tanh` con error máximo.

#### 8) Práctica guiada
- Añade `relu` al verificador y discute el punto `x=0`.

#### 9) Práctica independiente
- Implementa `softplus(x)=log(1+exp(x))` y su derivada; verifica numéricamente.

#### 10) Evaluación
- ¿Por qué en `sigmoid_derivative` conviene reutilizar `σ(x)` en lugar de re-computar `exp`?

#### 11) Errores comunes
- Overflow en `exp` para `x` grande (necesidad de estabilidad numérica).

#### 12) Retención
- (día 7) recita 5 derivadas clave sin mirar (potencia, exp, log, sigmoid, tanh).

#### 13) Diferenciación
- Avanzado: explica por qué ReLU “funciona” pese a no ser derivable en 0.

#### 14) Recursos
- GLOSARIO: Gradient, Chain Rule.

#### 15) Nota docente
- Requerir “tabla personal” de derivadas + mini test de verificación.
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 1.3: Derivadas Parciales</strong></summary>

#### 1) Metadatos
- **Título:** Parciales como “congelar variables” + check numérico
- **ID (opcional):** `M03-T01_3`
- **Duración estimada:** 60–120 min
- **Nivel:** Fundamentos
- **Dependencias:** 1.1

#### 2) Objetivos
- Calcular `∂f/∂x` y `∂f/∂y` y verificarlas numéricamente en un punto.
- Interpretar “mantener constante” y su conexión con gradiente.

#### 3) Relevancia
- Backprop calcula parciales “locales” en cada nodo del grafo.

#### 4) Conceptos clave
- parcial vs total
- punto de evaluación
- diferencias finitas por coordenada

#### 5) Fórmulas
- `∂f/∂x ≈ (f(x+h,y)-f(x-h,y)) / (2h)`.

#### 6) Explicación didáctica
- Cada parcial es “cómo cambia la salida si muevo solo una coordenada”.

#### 7) Ejemplo modelado
- `f(x,y)=x²+3xy+y²` con parciales analíticas y check.

#### 8) Práctica guiada
- Cambia el punto (ej. `[0,0]`, `[1,-2]`) y compara parciales.

#### 9) Práctica independiente
- Define `g(x,y)=sin(xy)+x` y deriva parciales; valida.

#### 10) Evaluación
- ¿Por qué el gradiente junta todas las parciales en un vector?

#### 11) Errores comunes
- Confundir `df/dx` (1D) con `∂f/∂x` (multivariable).

#### 12) Retención
- (día 2) explica la idea de “congelar variables” con un ejemplo propio.

#### 13) Diferenciación
- Avanzado: relacionar parciales con derivada direccional (preview del gradiente).

#### 14) Recursos
- GLOSARIO: Gradient.

#### 15) Nota docente
- Repetir: “primero analítica, luego numérica, luego interpretación”.
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 2.1: Definición del Gradiente</strong></summary>

#### 1) Metadatos
- **Título:** Gradiente como vector de derivadas parciales + verificación numérica
- **ID (opcional):** `M03-T02_1`
- **Duración estimada:** 60–120 min
- **Nivel:** Fundamentos
- **Dependencias:** 1.3 (parciales)

#### 2) Objetivos
- Calcular un gradiente analítico simple (paraboloide) y **validarlo** con diferencias finitas.
- Interpretar `∇f` como dirección de máximo ascenso y `-∇f` como dirección de descenso.

#### 3) Relevancia
- El gradiente es la señal que guía el entrenamiento en ML; si el gradiente está mal, el modelo no aprende.

#### 4) Conceptos clave
- `∇f` (vector)
- norma del gradiente
- diferencia central por coordenada

#### 5) Fórmulas esenciales
- `∇f = [∂f/∂x₁, …, ∂f/∂xₙ]`.

#### 6) Explicación didáctica
- **Mentalidad de debugging:** primero deriva, luego valida numéricamente, luego interpreta.

#### 7) Ejemplo modelado
- `f(x,y)=x²+y²` → `∇f=[2x,2y]`.

#### 8) Práctica guiada
- Cambia el punto (ej. `[1,1]`, `[-3,0]`) y compara gradiente analítico vs numérico.

#### 9) Práctica independiente
- Define `f(x,y)=x²+10y²` y deriva `∇f`; valida numéricamente.

#### 10) Evaluación
- ¿Por qué `∇f` es perpendicular a las curvas de nivel?

#### 11) Errores comunes
- Confundir gradiente (vector) con “derivada” (escalar).

#### 12) Retención
- (día 2) explica en 2 frases qué te dice la dirección de `-∇f`.

#### 13) Diferenciación
- Avanzado: conecta `||∇f||` con “qué tan empinada” es la superficie.

#### 14) Recursos
- GLOSARIO: Gradient.

#### 15) Nota docente
- Exigir `allclose`/comparación numérica para gradientes nuevos (hábito tipo “grad-check mini”).
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 2.2: Visualización del Gradiente</strong></summary>

#### 1) Metadatos
- **Título:** Campo vectorial y contornos: ver `∇f` en acción
- **ID (opcional):** `M03-T02_2`
- **Duración estimada:** 45–90 min
- **Nivel:** Fundamentos
- **Dependencias:** 2.1

#### 2) Objetivos
- Interpretar un campo vectorial del gradiente y relacionarlo con contornos de nivel.
- Explicar por qué las flechas apuntan hacia máximo ascenso.

#### 3) Relevancia
- Evita que Gradient Descent se convierta en “receta”: aquí ves el porqué geométrico.

#### 4) Conceptos clave
- curvas de nivel
- dirección perpendicular
- normalización para visualización

#### 5) Fórmulas esenciales
- Para `f(x,y)=x²+y²`: `∇f=[2x,2y]`.

#### 6) Explicación didáctica
- Contornos = “misma altura”; gradiente apunta al cambio más rápido → cruza contornos en ángulo recto.

#### 7) Ejemplo modelado
- Flechas alrededor del origen apuntan hacia afuera (sube); para bajar, irías hacia adentro.

#### 8) Práctica guiada
- Cambia `Z` a `X**2 + 10*Y**2` y observa cómo cambia el campo.

#### 9) Práctica independiente
- Prueba una función con “valle” (tipo Rosenbrock) y discute por qué el gradiente puede zigzaguear.

#### 10) Evaluación
- ¿Por qué normalizar `U,V` ayuda a visualizar pero no cambia la dirección?

#### 11) Errores comunes
- Interpretar el tamaño de flecha sin considerar la normalización.

#### 12) Retención
- (día 7) dibuja a mano contornos y gradiente para una función simple.

#### 13) Diferenciación
- Avanzado: conecta con Hessiano (curvatura) (preview de ejercicios).

#### 14) Recursos
- `visualizations/viz_gradient_3d.py` (para trayectoria + superficie).

#### 15) Nota docente
- Pedir una explicación oral: “por qué el gradiente es perpendicular a contornos”.
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 3.1: Algoritmo Básico (Gradient Descent)</strong></summary>

#### 1) Metadatos
- **Título:** Gradient Descent como iteración `x ← x - α∇f(x)`
- **ID (opcional):** `M03-T03_1`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** 2.1 (gradiente), 2.2 (intuición geométrica)

#### 2) Objetivos
- Implementar GD 2D y **explicar** el rol de `α` y el criterio `||∇f|| < tol`.
- Diagnosticar convergencia/overshooting a partir del historial de `f`.

#### 3) Relevancia
- Es el núcleo de entrenamiento en ML (con variantes: SGD, Adam).

#### 4) Conceptos clave
- `learning_rate` (α)
- criterio de parada
- trayectoria (historia)

#### 5) Fórmulas
- `x_{t+1} = x_t - α ∇f(x_t)`.

#### 6) Didáctica
- Siempre guarda `history_x` y `history_f` para “ver” si aprende.

#### 7) Ejemplo modelado
- `f(x,y)=x²+y²` converge al origen.

#### 8) Práctica guiada
- Cambia `α` y observa número de iteraciones.

#### 9) Transferencia
- Usa el mismo patrón con una función elíptica (mal condicionada).

#### 10) Evaluación
- ¿Por qué `-∇f` baja localmente la función?

#### 11) Errores comunes
- `α` grande → diverge; `α` pequeño → lento.

#### 12) Retención
- (día 2) escribe el update rule y nombra cada término.

#### 13) Diferenciación
- Avanzado: diferencia entre stopping por `||∇f||` vs cambio en `f`.

#### 14) Recursos
- `visualizations/viz_gradient_3d.py`.

#### 15) Nota docente
- Pedir “reporte de diagnóstico”: converge/divege y por qué.
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 3.2: Efecto del Learning Rate</strong></summary>

#### 1) Metadatos
- **Título:** Estabilidad: cómo `α` controla convergencia vs oscilación
- **ID (opcional):** `M03-T03_2`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 3.1

#### 2) Objetivos
- Comparar curvas de `f(x)` para distintos `α` y **clasificar** el comportamiento.
- Identificar señales de inestabilidad (oscilación, diverge).

#### 3) Relevancia
- El ajuste de LR es una de las causas #1 de entrenamiento inestable.

#### 4) Conceptos clave
- escala log en loss
- overshooting
- sensibilidad a condiciones

#### 5) Fórmulas
- GD con `α` fijo: estabilidad depende de curvatura (idea cualitativa).

#### 6) Didáctica
- “Mira la curva”: suave → ok, serrucho → alto, explode → demasiado alto.

#### 7) Ejemplo modelado
- Comparación de `α ∈ {0.01,0.1,0.5,0.9}`.

#### 8) Práctica guiada
- Añade un `α=1.1` y observa.

#### 9) Transferencia
- Relaciona con entrenamiento de NN (LR schedules / Adam) (preview).

#### 10) Evaluación
- ¿Por qué usar escala log ayuda a comparar convergencia?

#### 11) Errores comunes
- Concluir “no aprende” cuando solo falta bajar `α`.

#### 12) Retención
- (día 7) escribe 3 síntomas y la acción correctiva.

#### 13) Diferenciación
- Avanzado: conecta `α` con “curvatura” (Hessiano) de forma conceptual.

#### 14) Recursos
- `study_tools/VISUALIZACION_GRADIENT_DESCENT.md`.

#### 15) Nota docente
- Exigir evidencia: plot + explicación del caso.
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 3.3: Funciones de Pérdida en ML</strong></summary>

#### 1) Metadatos
- **Título:** Loss + gradiente: contrato mínimo para entrenar
- **ID (opcional):** `M03-T03_3`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 1.2 (derivadas), 3.1

#### 2) Objetivos
- Implementar MSE y BCE y **derivar/validar** su gradiente respecto a `y_pred`.
- Explicar por qué se usa `clip` en BCE (estabilidad numérica).

#### 3) Relevancia
- Sin `loss` y su gradiente correcto, no hay entrenamiento fiable.

#### 4) Conceptos clave
- MSE (regresión)
- BCE (clasificación)
- estabilidad: `log(0)`

#### 5) Fórmulas esenciales
- `MSE = mean((y-ŷ)^2)`; `∂MSE/∂ŷ = 2(ŷ-y)/n`.

#### 6) Didáctica
- Separar: (1) definición de loss (2) gradiente (3) sanity-check numérico.

#### 7) Ejemplo modelado
- Dataset mini con `y_true` y `y_pred` y prints de losses.

#### 8) Práctica guiada
- Haz gradient checking de `mse_gradient` con diferencias finitas.

#### 9) Práctica independiente
- Conecta con `∂L/∂z` en una neurona (preview de Chain Rule).

#### 10) Evaluación
- ¿Qué problema evita `eps`/`clip` en BCE?

#### 11) Errores comunes
- confundir gradiente respecto a `ŷ` vs respecto a parámetros.

#### 12) Retención
- (día 2) escribe MSE y su gradiente sin mirar.

#### 13) Diferenciación
- Avanzado: discusión conceptual de saturación en sigmoid + BCE.

#### 14) Recursos
- CS231n: loss functions + numerical gradient check.

#### 15) Nota docente
- Hacer que el alumno identifique “dónde entra el `clip`” y por qué.
</details>

---
## 💻 Parte 4: Regla de la Cadena (Chain Rule)

### 4.0.0 Introducción

La Regla de la Cadena (Chain Rule) es un concepto fundamental en el cálculo que nos permite encontrar la derivada de una función compuesta. En el contexto del aprendizaje automático, esta regla es crucial para el entrenamiento de modelos, ya que nos permite calcular la derivada de la función de pérdida con respecto a los parámetros del modelo.

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.0: Visualización: Grafo computacional (computational graph)</strong></summary>

#### 1) Metadatos
- **Título:** Grafo computacional: pensar en “nodos” y “rutas” de derivación
- **ID (opcional):** `M03-T04_0`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 1.2 (derivadas), 3.3 (loss), 2.1 (gradiente)

#### 2) Objetivos
- Explicar con tus palabras la diferencia entre **forward** y **backward** en el grafo.
- Usar la regla de la cadena para obtener `dL/dw` y `dL/db` como producto de factores locales.

#### 3) Relevancia
- Este patrón mental es el corazón de backpropagation: derivadas locales + composición.

#### 4) Mapa conceptual mínimo
- **Composición:** `x → u → y`.
- **Backward:** se propagan derivadas desde `L` hacia los parámetros.

#### 5) Definiciones esenciales
- **Grafo computacional:** diagrama dirigido donde cada nodo es una operación/función.
- **Derivada local:** derivada de una operación respecto a su entrada inmediata.

#### 6) Explicación didáctica
- Regla práctica: para derivar respecto a una variable, multiplica las derivadas locales a lo largo del camino desde `L` hasta esa variable.

#### 7) Ejemplo modelado (micro)
- Si `z = w·x + b`, entonces:
  - `dz/dw = x`
  - `dz/db = 1`
  - y `dL/dw = dL/da · da/dz · x`

#### 8) Práctica guiada
- A partir del mismo grafo, deriva `dL/dx` y explica el significado (sensibilidad de la pérdida a la entrada).

#### 9) Práctica independiente
- Dibuja un grafo para `L = ( (w1·x + b1)² ) + (w2·x)` y deriva `dL/dw1`, `dL/db1`, `dL/dw2`.

#### 10) Autoevaluación
- ¿Qué factor te faltaría si olvidas el nodo `a = σ(z)`?

#### 11) Errores comunes
- Omitir un nodo intermedio (un factor) en el producto.
- Confundir `dL/dw` con `dw/dL` (dirección).

#### 12) Retención
- (día 2) Reproduce de memoria el grafo y escribe las fórmulas de `dL/dw` y `dL/db`.

#### 13) Diferenciación
- Avanzado: generaliza el patrón a `z = Wx + b` (vectores/matrices) y discute formas/dimensiones.

#### 14) Recursos
- Sección “computational graphs” de cursos intro de DL (p.ej., CS231n).

#### 15) Nota docente
- Pide una “narración” del backward: `L → a → z → (w,b)` y justificación de cada derivada local.
</details>

### 4.0.1 Derivación paso a paso: `f(x) = x²`

Si `f(x) = x²`, entonces:

```
f'(x) = lim_{h→0} [(x+h)² - x²] / h
      = lim_{h→0} [x² + 2xh + h² - x²] / h
      = lim_{h→0} [2xh + h²] / h
      = lim_{h→0} [2x + h]
      = 2x
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.0.1: Derivación paso a paso: f(x) = x²</strong></summary>

#### 1) Metadatos
- **Título:** Derivación por definición: intuición del límite
- **ID (opcional):** `M03-T04_0_1`
- **Duración estimada:** 30–60 min
- **Nivel:** Básico–Intermedio
- **Dependencias:** 1.2 (derivadas)

#### 2) Objetivos
- Reproducir el cálculo de `f'(x)` desde la definición de derivada.
- Explicar qué significa “tomar el límite” en términos de aproximación.

#### 3) Relevancia
- Esta derivación es un “patrón base” que luego se reutiliza en chain rule y gradientes.

#### 4) Mapa conceptual
- **Definición:** derivada = límite de cociente incremental.
- **Álgebra:** expandir, simplificar, cancelar, aplicar límite.

#### 5) Definiciones esenciales
- `f'(x) = lim_{h→0} (f(x+h)-f(x))/h`.

#### 6) Explicación didáctica
- La cancelación del término `x²` es la pista de que el cociente incremental “aísla” la variación.

#### 7) Ejemplo modelado
- Validación rápida: si `x=3`, entonces `f'(3)=6`.

#### 8) Práctica guiada
- Repite el proceso para `f(x)=x³` y compara el resultado con la regla conocida.

#### 9) Práctica independiente
- Deriva `f(x)=(x+1)²` por definición y simplifica.

#### 10) Autoevaluación
- ¿En qué paso aparece el requisito de `h→0` y por qué no puedes sustituir `h=0` antes?

#### 11) Errores comunes
- Sustituir `h=0` demasiado pronto (división por cero).
- Errores al expandir `(x+h)²`.

#### 12) Retención
- (día 2) escribe de memoria la expansión de `(x+h)²` y el resultado `2x`.

#### 13) Diferenciación
- Avanzado: conecta el resultado `2x` con la pendiente de la parábola en el plano.

#### 14) Recursos
- Sección de derivada por definición en cualquier texto de Cálculo I.

#### 15) Nota docente
- Pedir al alumno que explique cada cancelación (qué término desaparece y por qué).
</details>

### 4.0.2 Derivación paso a paso: sigmoide `σ(z)`

Definición:

```
σ(z) = 1 / (1 + e^{-z})
```

Resultado clave:

```
σ'(z) = σ(z)(1 - σ(z))
```

Derivación (paso a paso, conectada a código):

 1) Reescribe la sigmoide como potencia:

 ```
 σ(z) = (1 + e^{-z})^{-1}
 ```

 2) Deriva usando Chain Rule (derivada de `u^{-1}` y de `e^{-z}`):

 ```
 σ'(z) = - (1 + e^{-z})^{-2} · d/dz(1 + e^{-z})
       = - (1 + e^{-z})^{-2} · (-e^{-z})
       = e^{-z} / (1 + e^{-z})^2
 ```

 3) Demuestra que es equivalente a `σ(z)(1-σ(z))`:

 ```
 1 - σ(z) = 1 - 1/(1+e^{-z})
          = (1+e^{-z}-1)/(1+e^{-z})
          = e^{-z}/(1+e^{-z})

 σ(z)(1-σ(z)) = [1/(1+e^{-z})] · [e^{-z}/(1+e^{-z})]
              = e^{-z}/(1+e^{-z})^2
              = σ'(z)
 ```

 Conexión directa con `grad_check.py`:

 - En el script, esto aparece como:
   - `s = sigmoid(z)`
   - `return s * (1 - s)`
 - La razón práctica: si ya calculaste `s` en el forward, el backward usa `s(1-s)` y evita recalcular `exp`.

 Consejo práctico: cuando ya tienes `a = σ(z)`, usa `a(1-a)` para derivar, en vez de re-calcular `exp`.

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.0.2: Derivación paso a paso: sigmoide σ(z)</strong></summary>

#### 1) Metadatos
- **Título:** Derivada de la sigmoide: forma útil para backprop
- **ID (opcional):** `M03-T04_0_2`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio
- **Dependencias:** 4.0 (grafo), 1.2 (derivadas)

#### 2) Objetivos
- Justificar (al menos a nivel algebraico) por qué `σ'(z)=σ(z)(1-σ(z))`.
- Explicar por qué esta forma es computacionalmente conveniente.

#### 3) Relevancia
- La identidad `a(1-a)` aparece constantemente en redes con activación sigmoide.

#### 4) Mapa conceptual
- **Función:** `σ(z) = 1/(1+e^{-z})`
- **Derivada:** reescritura algebraica para expresar todo en función de `σ(z)`.

#### 5) Definiciones esenciales
- `σ(z)` y su derivada cerrada.

#### 6) Explicación didáctica
- Si ya computaste `a=σ(z)` en el forward, en el backward no recalculas exponenciales: usas `a(1-a)`.

#### 7) Ejemplo modelado
- Si `a=0.8`, entonces `σ'(z)=0.8·0.2=0.16`.

#### 8) Práctica guiada
- Calcula `σ(z)` y `σ'(z)` para `z ∈ {-2,0,2}` y compara magnitudes.

#### 9) Práctica independiente
- Explica con una frase por qué la sigmoide “satura” (derivada pequeña) en valores grandes de |z|.

#### 10) Autoevaluación
- ¿Qué ocurre con `σ'(z)` cuando `a≈0` o `a≈1`?

#### 11) Errores comunes
- Olvidar la regla de la cadena al derivar `e^{-z}`.
- Confundir `σ'(z)` con `1-σ(z)`.

#### 12) Retención
- (día 2) escribe de memoria: `σ'(z)=σ(z)(1-σ(z))`.

#### 13) Diferenciación
- Avanzado: conectar saturación con vanishing gradients en redes profundas.

#### 14) Recursos
- Notas de activaciones y derivadas (sigmoid/tanh/ReLU).

#### 15) Nota docente
- Pedir que el alumno derive la identidad y luego explique su utilidad computacional.
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.1: Chain Rule en 1D</strong></summary>

#### 1) Metadatos
- **Título:** Regla de la cadena en 1D: composición y “multiplicar derivadas locales”
- **ID (opcional):** `M03-T04_1`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 4.0 (grafo), 1.2 (derivadas)

#### 2) Objetivos
- Identificar `f` y `g` en una composición `y = f(g(x))`.
- Calcular `dy/dx` aplicando `dy/dx = f'(g(x))·g'(x)`.
- Verificar resultados comparando derivada analítica vs numérica.

#### 3) Relevancia
- Es el patrón exacto que se repite en backprop: derivadas locales encadenadas.

#### 4) Mapa conceptual
- **Composición:** `x → g(x) → f(g(x))`.
- **Derivación:** “derivar afuera” evaluado “adentro” y multiplicar por la derivada de adentro.

#### 5) Definiciones esenciales
- Si `y = f(u)` y `u = g(x)`, entonces `dy/dx = (dy/du)(du/dx)`.

#### 6) Explicación didáctica
- Técnica: escribe primero el camino `x → u → y`, y luego escribe derivadas a lo largo del camino.

#### 7) Ejemplo modelado
- `y=(x²+1)³`: identifica `u=x²+1`, `f(u)=u³`, luego `dy/dx=3u²·2x`.

#### 8) Práctica guiada
- Calcula `dy/dx` para `y = sin(x²)` y valida con diferencia finita.

#### 9) Práctica independiente
- Resuelve `y = exp( (3x-2)⁴ )` paso a paso, nombrando variables intermedias.

#### 10) Autoevaluación
- ¿Por qué aparece la evaluación `f'(g(x))` y no solo `f'(x)`?

#### 11) Errores comunes
- Olvidar el factor `g'(x)`.
- Derivar el “afuera” pero no evaluar en el “adentro”.

#### 12) Retención
- (día 2) escribe el patrón `f(g(x))' = f'(g(x))·g'(x)` y crea 2 ejemplos propios.

#### 13) Diferenciación
- Avanzado: usa notación de diferenciales `dy = f'(u)du`, `du=g'(x)dx`.

#### 14) Recursos
- Secciones de “funciones compuestas” en Cálculo I y notas de chain rule.

#### 15) Nota docente
- Pedir que el alumno “etiquete” cada subfunción con un nombre intermedio (u, v, …) antes de derivar.
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.2: Chain Rule para Funciones Compuestas (Backprop Preview)</strong></summary>

#### 1) Metadatos
- **Título:** Backprop como chain rule repetido (con verificación numérica)
- **ID (opcional):** `M03-T04_2`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** 4.0 (grafo), 4.1 (chain rule), 3.3 (loss), 4.0.2 (sigmoide)

#### 2) Objetivos
- Calcular `∂L/∂w` y `∂L/∂b` en una neurona: `z=wx+b`, `a=σ(z)`, `L=(a-y)²`.
- Entender el “pipeline” de derivadas locales: `∂L/∂a`, `∂a/∂z`, `∂z/∂w`, `∂z/∂b`.
- Validar la derivación con diferencias finitas (sanity check).

#### 3) Relevancia
- Backprop no es “magia”: es chain rule aplicado de forma sistemática.

#### 4) Mapa conceptual mínimo
- **Forward:** `x → z → a → L`.
- **Backward:** `dL/da → da/dz → dz/dw, dz/db`.

#### 5) Definiciones esenciales
- **Gradiente:** vector de derivadas parciales respecto a parámetros.
- **Gradient checking:** comparar gradiente analítico vs numérico.

#### 6) Explicación didáctica
- Regla práctica: en backward, cada paso “empuja” la derivada un nodo hacia atrás multiplicando por la derivada local.

#### 7) Ejemplo modelado
- Con `dz/dw = x` y `dz/db = 1`, se obtiene `∂L/∂w = ∂L/∂z · x` y `∂L/∂b = ∂L/∂z`.

#### 8) Práctica guiada
- Cambia la pérdida a `L = -[ y log(a) + (1-y)log(1-a) ]` (BCE) y escribe el nuevo `∂L/∂a`.

#### 9) Práctica independiente
- Implementa una función `grad_check` genérica que compare gradientes para distintos `h` y reporte error relativo.

#### 10) Autoevaluación
- ¿Por qué `h` no puede ser demasiado grande ni demasiado pequeño en diferencias finitas?

#### 11) Errores comunes
- Olvidar `σ'(z)=a(1-a)` y recalcular exponenciales innecesariamente.
- Implementar mal el gradiente numérico (forward vs central differences).

#### 12) Retención
- (día 2) escribe el pipeline: `dL/da`, `da/dz`, `dz/dw`, `dz/db` y cómo se combinan.

#### 13) Diferenciación
- Avanzado: extender de escalar a vector: `z = w·x + b`, `∂z/∂w = x` (vector).

#### 14) Recursos
- Sección “gradient checking” en cursos de DL (p.ej., CS231n).

#### 15) Nota docente
- Exigir evidencia: gradiente analítico + numérico + tolerancias (`rtol`, `atol`) y explicación del resultado.
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.3: Backpropagation en una Red de 2 Capas</strong></summary>

#### 1) Metadatos
- **Título:** Backprop en 2 capas: cache, gradientes y actualización
- **ID (opcional):** `M03-T04_3`
- **Duración estimada:** 120–180 min
- **Nivel:** Intermedio–Avanzado
- **Dependencias:** 4.2 (preview), 4.0 (grafo), 3.1–3.2 (GD y LR)

#### 2) Objetivos
- Entender por qué el **cache** (guardar `x, z1, a1, z2, a2`) es necesario para backprop.
- Derivar/interpretar `dW2, db2, dW1, db1` y sus dimensiones.
- Conectar el cálculo de gradientes con el update de Gradient Descent.

#### 3) Relevancia
- Este patrón (forward → cache → backward → update) es la base de cualquier entrenamiento de NN.

#### 4) Mapa conceptual mínimo
- **Forward:** `x → (z1,a1) → (z2,a2) → L`
- **Backward:** `dL/da2 → dL/dz2 → (dW2,db2) → dL/da1 → dL/dz1 → (dW1,db1)`

#### 5) Definiciones esenciales
- **Backpropagation:** aplicación sistemática de chain rule para obtener derivadas respecto a parámetros.
- **Outer product:** usado para formar `dW = δ ⊗ activación`.

#### 6) Explicación didáctica
- Regla práctica de formas:
  - Si `z = W a + b`, entonces `∂L/∂W = δ ⊗ a` y `∂L/∂b = δ`.
- Si te equivocas en shapes, casi siempre te falta un transpose.

#### 7) Ejemplo modelado
- El demo XOR muestra un loop completo: forward, loss, backward, update, y reporte periódico.

#### 8) Práctica guiada
- Imprime shapes (`W1.shape`, `dW1.shape`, etc.) y verifica coherencia en cada paso.

#### 9) Práctica independiente
- Cambia `hidden_size` y observa impacto en convergencia.
- Añade `learning_rate` más pequeño y compara estabilidad.

#### 10) Autoevaluación
- ¿Por qué `dW2 = outer(dL_dz2, a1)` y no `outer(a1, dL_dz2)`?

#### 11) Errores comunes
- Olvidar que `sigmoid_derivative` usa `a` (activación) y no `z`.
- Confundir el vector columna/fila y generar `dW` transpuesto.
- LR demasiado alto: diverge o se “queda oscilando”.

#### 12) Retención
- (día 2) Escribe el pipeline de 2 capas: `δ2 → dW2 → δ1 → dW1`.

#### 13) Diferenciación
- Avanzado: reemplaza MSE por BCE + sigmoid y discute estabilidad.

#### 14) Recursos
- Capítulos intro de backprop (computational graphs) y notas de “matrix calculus”.

#### 15) Nota docente
- Pedir evidencia de comprensión: diagrama + shapes + explicación de `outer`.
</details>

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


# Aproximamos derivadas numéricamente usando *diferencias centrales*.
# Intuición: medir la pendiente alrededor de x de forma simétrica (x+h y x-h)
# cancela términos de error de primer orden y suele ser más preciso que la
# diferencia hacia adelante.

def num_derivative_central(f, x: float, h: float = 1e-6) -> float:
    # f: función escalar f(x).
    # x: punto donde evaluamos la derivada.
    # h: tamaño de paso. Hay tradeoff:
    # - h grande => error de truncamiento (aproximación) domina
    # - h muy pequeño => cancelación numérica (floating point) domina
    # Devolvemos float para facilitar asserts y logs.
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def f(x: float) -> float:
    # Función de prueba (suave y derivable).
    return x**3 + 2.0 * x


def f_prime(x: float) -> float:
    # Derivada analítica:
    # d/dx (x^3) = 3x^2
    # d/dx (2x)  = 2
    return 3.0 * x**2 + 2.0


# Probamos varios puntos para evitar que pase “por casualidad” en un solo x.
xs = [-2.0, -0.5, 0.0, 1.0, 3.0]
for x in xs:
    # Aproximación numérica.
    approx = num_derivative_central(f, x, h=1e-6)
    # Valor exacto (analítico).
    exact = f_prime(x)
    # np.isclose compara igualdad aproximada con tolerancias:
    # - rtol: tolerancia relativa (escala con el tamaño)
    # - atol: tolerancia absoluta (útil cerca de 0)
    assert np.isclose(approx, exact, rtol=1e-6, atol=1e-6)


# Estudiamos cómo cambia el error con distintos h.
# Nota: no imponemos monotonía estricta porque h extremadamente pequeño puede
# empeorar por precisión de máquina.
x0 = 1.234
errs = []
for h in [1e-2, 1e-4, 1e-6]:
    # Misma x0, distinto paso.
    approx = num_derivative_central(f, x0, h=h)
    # Error absoluto vs derivada analítica.
    errs.append(abs(approx - f_prime(x0)))

# Sanidad mínima: al refinar de 1e-2 a 1e-4, no debería empeorar.
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
    # Función escalar de 2 variables:
    # f(x, y) = x^2 * y + sin(y)
    return x**2 * y + np.sin(y)


def grad_f_xy(x: float, y: float) -> np.ndarray:
    # Gradiente analítico (derivadas parciales):
    # ∂f/∂x = 2xy
    # ∂f/∂y = x^2 + cos(y)
    dfdx = 2.0 * x * y
    dfdy = x**2 + np.cos(y)
    # Empaquetamos como vector [df/dx, df/dy].
    return np.array([dfdx, dfdy], dtype=float)


def num_grad_2d(f, x: float, y: float, h: float = 1e-6) -> np.ndarray:
    # Gradiente numérico con diferencias centrales.
    # Para cada variable, perturbamos solo esa coordenada.
    dfdx = (f(x + h, y) - f(x - h, y)) / (2.0 * h)
    dfdy = (f(x, y + h) - f(x, y - h)) / (2.0 * h)
    # Vector gradiente.
    return np.array([dfdx, dfdy], dtype=float)


# Punto de evaluación (no trivial para evitar simetrías).
x0, y0 = 1.2, -0.7

# Gradiente analítico.
g_anal = grad_f_xy(x0, y0)

# Gradiente numérico (check independiente).
g_num = num_grad_2d(f_xy, x0, y0)

# Deben coincidir si las derivadas están bien.
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
    # Misma función del ejercicio anterior.
    return x**2 * y + np.sin(y)


def grad_f_xy(x: float, y: float) -> np.ndarray:
    # ∇f(x,y) = [∂f/∂x, ∂f/∂y]
    return np.array([2.0 * x * y, x**2 + np.cos(y)], dtype=float)


# Punto base p0 = (x0, y0).
x0, y0 = 0.5, 1.0

# Gradiente en p0.
g = grad_f_xy(x0, y0)

# Vector dirección (aún no unitario).
u = np.array([3.0, 4.0], dtype=float)

# La derivada direccional se define sobre u unitario: ||u|| = 1.
u = u / np.linalg.norm(u)

# Derivada direccional analítica: D_u f = ∇f · u.
dir_anal = float(np.dot(g, u))

# Verificación numérica: avanzamos/retrocedemos h sobre la recta p(t)=p0 + t u.
h = 1e-6
f_plus = f_xy(x0 + h * u[0], y0 + h * u[1])
f_minus = f_xy(x0 - h * u[0], y0 - h * u[1])

# Diferencia central en la dirección u.
dir_num = float((f_plus - f_minus) / (2.0 * h))

# Comparación con tolerancia.
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
    # Función vectorial g: R^2 -> R^2.
    # Convertimos a float para evitar dtypes raros (int) y asegurar operaciones reales.
    x1, x2 = float(x[0]), float(x[1])
    # Definimos:
    # g1 = x1^2 + x2
    # g2 = sin(x1 * x2)
    return np.array([x1**2 + x2, np.sin(x1 * x2)], dtype=float)


def J_analytical(x: np.ndarray) -> np.ndarray:
    # Jacobiano J: matriz de derivadas parciales.
    # J[i, j] = ∂g_i / ∂x_j
    # Aquí hay 2 salidas y 2 entradas => J es 2x2.
    x1, x2 = float(x[0]), float(x[1])

    # g1 = x1^2 + x2
    # ∂g1/∂x1 = 2x1
    # ∂g1/∂x2 = 1
    dg1_dx1 = 2.0 * x1
    dg1_dx2 = 1.0

    # g2 = sin(x1*x2)
    # Regla de la cadena:
    # ∂g2/∂x1 = cos(x1*x2) * x2
    # ∂g2/∂x2 = cos(x1*x2) * x1
    dg2_dx1 = np.cos(x1 * x2) * x2
    dg2_dx2 = np.cos(x1 * x2) * x1

    # Empaquetamos en una matriz 2x2.
    return np.array([[dg1_dx1, dg1_dx2], [dg2_dx1, dg2_dx2]], dtype=float)


def J_numeric(g, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    # Jacobiano numérico con diferencias centrales.
    # Para cada coordenada j, perturbamos x por ±h e_j y obtenemos la columna J[:, j].
    x = x.astype(float)
    # m: dimensión de salida, n: dimensión de entrada.
    m = g(x).shape[0]
    n = x.shape[0]
    # Inicializamos J.
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        # Vector base e_j.
        e = np.zeros(n)
        e[j] = 1.0
        # Diferencia central para todas las salidas a la vez.
        J[:, j] = (g(x + h * e) - g(x - h * e)) / (2.0 * h)
    return J


# Punto de prueba.
x0 = np.array([0.7, -1.1])

# Comparamos Jacobiano analítico vs numérico.
Ja = J_analytical(x0)
Jn = J_numeric(g, x0)

# Si la derivación está correcta, deben ser casi iguales.
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

# Para f(x1,x2)=x1^2 + 2x2^2:
# - ∂²f/∂x1² = 2
# - ∂²f/∂x2² = 4
# - derivadas cruzadas = 0
H = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=float)

# El Hessiano de una función escalar dos-veces derivable debe ser simétrico.
assert np.allclose(H, H.T)

# Hessiano definido positivo => función estrictamente convexa.
# En particular, un criterio suficiente aquí es: eigenvalores > 0.
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
    # Función convexa con mínimo global en x=3.
    return (x - 3.0) ** 2


def grad_f(x: float) -> float:
    # Derivada: d/dx (x-3)^2 = 2(x-3)
    return 2.0 * (x - 3.0)


# Inicialización.
x = 10.0

# Learning rate (tamaño de paso).
alpha = 0.1

# Historial de iteraciones para inspección y asserts.
history = []
for _ in range(200):
    # Gradiente en el punto actual.
    g = grad_f(x)
    # Guardamos (x, f(x)) antes de actualizar.
    history.append((x, f(x)))
    # Criterio de parada: gradiente cerca de 0 => cerca del mínimo.
    if abs(g) < 1e-8:
        break
    # Actualización de Gradient Descent.
    x = x - alpha * g

# Debe converger cerca de 3.
assert abs(x - 3.0) < 1e-4

# La pérdida final no debería ser mayor que la inicial.
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
    # Minimizamos f(x)=x^2 con GD. Su gradiente es 2x.
    x = 1.0
    # Guardamos la trayectoria.
    xs = [x]
    for _ in range(steps):
        # Gradiente en el punto actual.
        grad = 2.0 * x
        # Paso de GD.
        x = x - alpha * grad
        # Guardamos el nuevo x.
        xs.append(x)
    # Convertimos a np.array para análisis.
    return np.array(xs)


# Con alpha=0.25, el factor (1-2α)=0.5 => converge.
xs_good = run_gd_x2(alpha=0.25, steps=10)

# La magnitud debe decrecer.
assert abs(xs_good[-1]) < abs(xs_good[0])


# Con alpha=1.1, |1-2α| = |1-2.2| = 1.2 > 1 => diverge.
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
    # Función escalar sobre un vector: f(w) = sum_i w_i^3.
    # Convertimos a float para devolver un escalar Python.
    return float(np.sum(w ** 3))


def grad_analytical(w: np.ndarray) -> np.ndarray:
    # Gradiente analítico: ∂/∂w_i (w_i^3) = 3 w_i^2.
    return 3.0 * (w ** 2)


def grad_numeric(f, w: np.ndarray, h: float = 1e-6) -> np.ndarray:
    # Gradiente numérico con diferencias centrales.
    # Para cada coordenada i, perturbamos w por ±h e_i.
    w = w.astype(float)
    # Vector de gradientes numéricos.
    g = np.zeros_like(w)
    for i in range(w.size):
        # Vector base e_i.
        e = np.zeros_like(w)
        e[i] = 1.0
        # Diferencia central: ∂f/∂w_i ≈ (f(w+h e_i) - f(w-h e_i)) / (2h)
        g[i] = (f(w + h * e) - f(w - h * e)) / (2.0 * h)
    return g


# Semilla para reproducibilidad.
np.random.seed(0)

# Vector de prueba.
w = np.random.randn(5)

# Gradientes analítico y numérico.
g_a = grad_analytical(w)
g_n = grad_numeric(f, w)

# Error relativo: más robusto que el error absoluto porque normaliza escalas.
eps = 1e-12
rel_err = np.linalg.norm(g_n - g_a) / (np.linalg.norm(g_n) + np.linalg.norm(g_a) + eps)

# Si falla, normalmente indica error en derivada o un h inapropiado.
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
    # Sigmoide σ(z) = 1 / (1 + exp(-z)).
    # Convertimos a float para devolver escalar.
    return float(1.0 / (1.0 + np.exp(-z)))


def loss_mse(y_hat: float, y: float) -> float:
    # Pérdida MSE para un solo ejemplo: (ŷ - y)^2
    return float((y_hat - y) ** 2)


def forward(w: np.ndarray, b: float, x: np.ndarray, y: float) -> float:
    # Forward de una neurona:
    # z = w·x + b
    # ŷ = σ(z)
    # L = (ŷ - y)^2
    z = float(np.dot(w, x) + b)
    y_hat = sigmoid(z)
    return loss_mse(y_hat, y)


def grads_analytical(w: np.ndarray, b: float, x: np.ndarray, y: float):
    # Gradientes analíticos vía Chain Rule.
    z = float(np.dot(w, x) + b)
    y_hat = sigmoid(z)

    # dL/dŷ cuando L=(ŷ-y)^2.
    dL_dyhat = 2.0 * (y_hat - y)
    # dŷ/dz para sigmoide: σ'(z)=σ(z)(1-σ(z)).
    dyhat_dz = y_hat * (1.0 - y_hat)
    # Chain rule: dL/dz = dL/dŷ * dŷ/dz.
    dL_dz = dL_dyhat * dyhat_dz

    # z = w·x + b => dz/dw = x y dz/db = 1.
    # Entonces:
    # dL/dw = dL/dz * x
    # dL/db = dL/dz
    dL_dw = dL_dz * x
    dL_db = dL_dz
    return dL_dw.astype(float), float(dL_db)


def grads_numeric(w: np.ndarray, b: float, x: np.ndarray, y: float, h: float = 1e-6):
    # Gradientes numéricos por diferencias centrales.
    gw = np.zeros_like(w, dtype=float)
    for i in range(w.size):
        # Vector base e_i.
        e = np.zeros_like(w)
        e[i] = 1.0
        # ∂L/∂w_i ≈ (L(w+h e_i) - L(w-h e_i)) / (2h)
        gw[i] = (forward(w + h * e, b, x, y) - forward(w - h * e, b, x, y)) / (2.0 * h)

    # ∂L/∂b ≈ (L(b+h) - L(b-h)) / (2h)
    gb = (forward(w, b + h, x, y) - forward(w, b - h, x, y)) / (2.0 * h)
    return gw, float(gb)


# Reproducibilidad.
np.random.seed(1)

# Parámetros y entrada de ejemplo.
w = np.random.randn(3)
b = 0.1
x = np.random.randn(3)

# Etiqueta objetivo.
y = 1.0

# Comparamos gradientes.
gw_a, gb_a = grads_analytical(w, b, x, y)
gw_n, gb_n = grads_numeric(w, b, x, y)

# Si la derivación por chain rule está bien, deben coincidir.
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
