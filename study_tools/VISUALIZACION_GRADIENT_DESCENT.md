# 📊 Guía de Visualización 3D para Gradient Descent

> La intuición visual del "valle" y la "pelota rodando" es vital para entender optimizadores.
> Semanas 6-7: Usa GeoGebra o matplotlib 3D para visualizar.

---

## 🎯 Objetivo

Desarrollar intuición geométrica sobre:
- Cómo el gradiente indica la dirección de máximo ascenso
- Por qué el learning rate afecta la convergencia
- Qué son mínimos locales, globales y puntos silla
- Cómo funcionan diferentes optimizadores

---

## 🛠️ Herramientas Recomendadas

### 1. GeoGebra 3D (Online, Gratis)
- URL: https://www.geogebra.org/3d
- Ideal para: Exploración rápida e interactiva

### 2. Matplotlib 3D (Python)
- Ideal para: Visualizaciones programáticas y animaciones

### 3. Desmos (2D)
- URL: https://www.desmos.com/calculator
- Ideal para: Funciones 2D y curvas de nivel

---

## 📐 Ejercicio 1: Superficie Convexa Simple

### Código Python

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def visualizar_superficie_convexa():
    """Visualiza f(x,y) = x² + y² - un bowl perfecto."""

    # Crear grid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Función objetivo: bowl convexo
    Z = X**2 + Y**2

    # Crear figura
    fig = plt.figure(figsize=(14, 5))

    # Vista 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Superficie Convexa: f(x,y) = x² + y²')

    # Curvas de nivel (contour)
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap=cm.viridis)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Curvas de Nivel (Vista desde arriba)')
    ax2.set_aspect('equal')
    ax2.plot(0, 0, 'r*', markersize=15, label='Mínimo global')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('superficie_convexa.png', dpi=150)
    plt.show()

visualizar_superficie_convexa()
```

### Preguntas para Reflexionar
1. ¿Dónde está el mínimo? ¿Por qué es único?
2. ¿Hacia dónde apunta el gradiente en cualquier punto?
3. ¿Por qué las curvas de nivel son círculos perfectos?

---

## 📐 Ejercicio 2: Animación de Gradient Descent

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def gradient_descent_animation():
    """Anima el proceso de Gradient Descent."""

    # Función y gradiente
    def f(x, y):
        return x**2 + y**2

    def grad_f(x, y):
        return np.array([2*x, 2*y])

    # Parámetros
    learning_rate = 0.1
    n_steps = 30

    # Punto inicial
    path = [(2.5, 2.5)]

    # Ejecutar GD
    x, y = path[0]
    for _ in range(n_steps):
        grad = grad_f(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        path.append((x, y))

    path = np.array(path)

    # Crear grid para contour
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    # Animación
    fig, ax = plt.subplots(figsize=(8, 8))

    def init():
        ax.clear()
        ax.contour(X, Y, Z, levels=20, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Gradient Descent: f(x,y) = x² + y²')
        ax.set_aspect('equal')
        return []

    def animate(i):
        ax.clear()
        ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)

        # Dibujar camino hasta el paso actual
        ax.plot(path[:i+1, 0], path[:i+1, 1], 'ro-', markersize=4)
        ax.plot(path[i, 0], path[i, 1], 'r*', markersize=15)

        # Dibujar gradiente actual (invertido, apunta hacia descenso)
        if i < len(path) - 1:
            grad = grad_f(path[i, 0], path[i, 1])
            ax.arrow(path[i, 0], path[i, 1],
                    -0.3*grad[0], -0.3*grad[1],
                    head_width=0.1, head_length=0.05, fc='blue', ec='blue')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Gradient Descent - Paso {i}, f = {f(path[i,0], path[i,1]):.4f}')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')

        return []

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(path), interval=200, blit=True)

    # Para Jupyter: HTML(anim.to_jshtml())
    # Para script:
    anim.save('gradient_descent.gif', writer='pillow', fps=5)
    plt.show()

    return path

path = gradient_descent_animation()
print(f"Punto final: ({path[-1,0]:.6f}, {path[-1,1]:.6f})")
```

---

## 📐 Ejercicio 3: Efecto del Learning Rate

```python
import numpy as np
import matplotlib.pyplot as plt

def comparar_learning_rates():
    """Compara diferentes learning rates."""

    def f(x, y):
        return x**2 + y**2

    def grad_f(x, y):
        return np.array([2*x, 2*y])

    learning_rates = [0.01, 0.1, 0.5, 0.9, 1.1]
    colors = ['blue', 'green', 'orange', 'red', 'purple']

    fig, axes = plt.subplots(1, len(learning_rates), figsize=(20, 4))

    # Grid para contour
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    for ax, lr, color in zip(axes, learning_rates, colors):
        # Ejecutar GD
        path = [(2.5, 2.5)]
        x, y = path[0]

        for _ in range(30):
            grad = grad_f(x, y)
            x = x - lr * grad[0]
            y = y - lr * grad[1]

            # Limitar para evitar explosión
            x = np.clip(x, -10, 10)
            y = np.clip(y, -10, 10)
            path.append((x, y))

        path = np.array(path)

        # Dibujar
        ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
        ax.plot(path[:, 0], path[:, 1], f'{color[0]}o-', markersize=3)
        ax.plot(path[0, 0], path[0, 1], 'g*', markersize=15, label='Inicio')
        ax.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='Final')

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f'LR = {lr}\nFinal: ({path[-1,0]:.2f}, {path[-1,1]:.2f})')
        ax.legend(fontsize=8)

    plt.suptitle('Efecto del Learning Rate en Gradient Descent', fontsize=14)
    plt.tight_layout()
    plt.savefig('learning_rates.png', dpi=150)
    plt.show()

comparar_learning_rates()
```

### Observaciones Clave
| Learning Rate | Comportamiento |
|---------------|----------------|
| 0.01 | Muy lento, pero converge seguro |
| 0.1 | Buen balance velocidad/estabilidad |
| 0.5 | Rápido pero empieza a oscilar |
| 0.9 | Oscilaciones fuertes |
| 1.1 | DIVERGE - la "pelota salta fuera del bowl" |

---

## 📐 Ejercicio 4: Superficie No Convexa (Mínimos Locales)

```python
def visualizar_superficie_no_convexa():
    """Visualiza una función con múltiples mínimos."""

    # Función con múltiples mínimos
    def f(x, y):
        return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(14, 5))

    # Vista 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Superficie No Convexa\n(Múltiples mínimos locales)')

    # Curvas de nivel
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap=cm.coolwarm)
    plt.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Curvas de Nivel')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('superficie_no_convexa.png', dpi=150)
    plt.show()

visualizar_superficie_no_convexa()
```

### Pregunta Clave
¿Por qué el punto inicial importa en superficies no convexas?

---

## 📐 Ejercicio 5: Punto Silla (Saddle Point)

```python
def visualizar_punto_silla():
    """Visualiza f(x,y) = x² - y² (saddle point en origen)."""

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2

    fig = plt.figure(figsize=(14, 5))

    # Vista 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap=cm.RdYlBu, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Punto Silla: f(x,y) = x² - y²\n(Mínimo en x, máximo en y)')

    # Curvas de nivel
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap=cm.RdYlBu)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(0, 0, 'ko', markersize=10, label='Punto silla')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Curvas de Nivel\n(Hipérbolas)')
    ax2.set_aspect('equal')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('punto_silla.png', dpi=150)
    plt.show()

visualizar_punto_silla()
```

### Concepto Clave
En un punto silla:
- El gradiente es cero (parece un punto crítico)
- Pero NO es mínimo ni máximo
- La Hessiana tiene eigenvalores de diferente signo

---

## 📐 Ejercicio 6: Comparación de Optimizadores

```python
def comparar_optimizadores():
    """Compara SGD, Momentum y Adam en una superficie elongada."""

    # Función elongada (difícil para SGD vanilla)
    def f(x, y):
        return 10*x**2 + y**2

    def grad_f(x, y):
        return np.array([20*x, 2*y])

    # Implementaciones
    def sgd_step(x, y, lr):
        grad = grad_f(x, y)
        return x - lr * grad[0], y - lr * grad[1]

    def momentum_step(x, y, vx, vy, lr, beta=0.9):
        grad = grad_f(x, y)
        vx = beta * vx + (1 - beta) * grad[0]
        vy = beta * vy + (1 - beta) * grad[1]
        return x - lr * vx, y - lr * vy, vx, vy

    # Ejecutar optimizadores
    paths = {}

    # SGD
    x, y = 2.0, 2.0
    path = [(x, y)]
    for _ in range(50):
        x, y = sgd_step(x, y, 0.05)
        path.append((x, y))
    paths['SGD'] = np.array(path)

    # Momentum
    x, y = 2.0, 2.0
    vx, vy = 0, 0
    path = [(x, y)]
    for _ in range(50):
        x, y, vx, vy = momentum_step(x, y, vx, vy, 0.05)
        path.append((x, y))
    paths['Momentum'] = np.array(path)

    # Visualizar
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)

    for name, path in paths.items():
        ax.plot(path[:, 0], path[:, 1], 'o-', label=name, markersize=3)

    ax.plot(2, 2, 'g*', markersize=20, label='Inicio')
    ax.plot(0, 0, 'r*', markersize=20, label='Mínimo')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Comparación de Optimizadores\nf(x,y) = 10x² + y² (superficie elongada)')
    ax.legend()
    ax.set_aspect('equal')
    plt.savefig('comparacion_optimizadores.png', dpi=150)
    plt.show()

comparar_optimizadores()
```

---

## 📐 Ejercicio en GeoGebra

### Instrucciones para GeoGebra 3D

1. Abre https://www.geogebra.org/3d
2. En la barra de entrada, escribe:
   ```
   f(x,y) = x^2 + y^2
   ```
3. Rota la vista 3D con el mouse
4. Prueba otras funciones:
   - `f(x,y) = x^2 - y^2` (punto silla)
   - `f(x,y) = sin(x)*cos(y)` (múltiples mínimos)
   - `f(x,y) = 10*x^2 + y^2` (elipse elongada)

---

## 📊 Checklist de Comprensión

- [ ] Puedo dibujar mentalmente una superficie convexa vs no convexa
- [ ] Entiendo por qué el gradiente apunta "cuesta arriba"
- [ ] Sé qué pasa cuando el learning rate es muy grande/pequeño
- [ ] Puedo identificar un punto silla en una gráfica
- [ ] Entiendo por qué Momentum ayuda en superficies elongadas
- [ ] Puedo explicar la analogía de "la pelota rodando" a un principiante

---

## 🎯 Tarea Final

Ejecuta una visualización ya provista y registra observaciones:

- Ejecuta `visualizations/viz_gradient_3d.py` con `lr` pequeño y grande (convergencia vs divergencia).
- Cambia `steps` y describe cómo cambia la trayectoria.
- Entrega una captura/export del HTML y una explicación en 5 líneas.
