# 📐 Drill de Dimensiones NumPy

> La habilidad #1 para no atascarse en Deep Learning: predecir mentalmente el `.shape`

## 🎯 Objetivo
Dedicar 1 hora extra en Semanas 1-2 a predecir mentalmente el `.shape` de operaciones NumPy.

---

## 📋 Metodología del Drill

### Paso 1: Lee la operación
### Paso 2: Predice el `.shape` resultante EN PAPEL
### Paso 3: Verifica en Python
### Paso 4: Si fallaste, registra en el Diario de Errores

---

## 🔥 Ejercicios de Calentamiento (Nivel 1)

Predice el shape ANTES de ejecutar:

```python
import numpy as np

# Ejercicio 1
a = np.array([1, 2, 3])
# Predice: a.shape = ?

# Ejercicio 2
b = np.array([[1, 2, 3]])
# Predice: b.shape = ?

# Ejercicio 3
c = np.array([[1], [2], [3]])
# Predice: c.shape = ?

# Ejercicio 4
d = np.zeros((3, 4))
# Predice: d.shape = ?

# Ejercicio 5
e = np.ones((2, 3, 4))
# Predice: e.shape = ?
```

<details>
<summary>✅ Respuestas Nivel 1</summary>

```python
a.shape = (3,)       # Vector 1D
b.shape = (1, 3)     # Matriz fila
c.shape = (3, 1)     # Matriz columna
d.shape = (3, 4)     # Matriz 3x4
e.shape = (2, 3, 4)  # Tensor 3D
```
</details>

---

## 🔥 Operaciones Básicas (Nivel 2)

```python
A = np.random.randn(3, 4)  # shape: (3, 4)
B = np.random.randn(4, 5)  # shape: (4, 5)
C = np.random.randn(3, 4)  # shape: (3, 4)
v = np.random.randn(4)     # shape: (4,)
w = np.random.randn(3, 1)  # shape: (3, 1)

# Ejercicio 6: Producto matricial
resultado = A @ B
# Predice: resultado.shape = ?

# Ejercicio 7: Multiplicación elemento a elemento
resultado = A * C
# Predice: resultado.shape = ?

# Ejercicio 8: Producto matriz-vector
resultado = A @ v
# Predice: resultado.shape = ?

# Ejercicio 9: Broadcasting
resultado = A + w
# Predice: resultado.shape = ?

# Ejercicio 10: Transpuesta
resultado = A.T
# Predice: resultado.shape = ?
```

<details>
<summary>✅ Respuestas Nivel 2</summary>

```python
# 6: (3, 4) @ (4, 5) = (3, 5)
# 7: (3, 4) * (3, 4) = (3, 4)
# 8: (3, 4) @ (4,) = (3,)  # ¡OJO! Vector 1D, no (3, 1)
# 9: (3, 4) + (3, 1) = (3, 4)  # Broadcasting expande w
# 10: (3, 4).T = (4, 3)
```
</details>

---

## 🔥 Broadcasting Avanzado (Nivel 3)

```python
A = np.random.randn(3, 4)      # (3, 4)
B = np.random.randn(4)         # (4,)
C = np.random.randn(3, 1)      # (3, 1)
D = np.random.randn(1, 4)      # (1, 4)
E = np.random.randn(2, 3, 4)   # (2, 3, 4)

# Ejercicio 11
resultado = A + B
# Predice: resultado.shape = ?

# Ejercicio 12
resultado = C + D
# Predice: resultado.shape = ?

# Ejercicio 13
resultado = A * C
# Predice: resultado.shape = ?

# Ejercicio 14
resultado = E + A
# Predice: resultado.shape = ?

# Ejercicio 15
resultado = E * B
# Predice: resultado.shape = ?
```

<details>
<summary>✅ Respuestas Nivel 3</summary>

```python
# 11: (3, 4) + (4,) = (3, 4)
#     (4,) se broadcastea a (1, 4) → (3, 4)

# 12: (3, 1) + (1, 4) = (3, 4)
#     Broadcasting en ambas dimensiones

# 13: (3, 4) * (3, 1) = (3, 4)
#     (3, 1) se expande a (3, 4)

# 14: (2, 3, 4) + (3, 4) = (2, 3, 4)
#     (3, 4) se broadcastea a (1, 3, 4) → (2, 3, 4)

# 15: (2, 3, 4) * (4,) = (2, 3, 4)
#     (4,) → (1, 1, 4) → (2, 3, 4)
```
</details>

---

## 🔥 Operaciones de Reducción (Nivel 4)

```python
A = np.random.randn(3, 4, 5)  # (3, 4, 5)

# Ejercicio 16: Suma total
resultado = np.sum(A)
# Predice: resultado.shape = ?

# Ejercicio 17: Suma por eje
resultado = np.sum(A, axis=0)
# Predice: resultado.shape = ?

# Ejercicio 18: Suma por eje
resultado = np.sum(A, axis=1)
# Predice: resultado.shape = ?

# Ejercicio 19: Suma manteniendo dimensiones
resultado = np.sum(A, axis=1, keepdims=True)
# Predice: resultado.shape = ?

# Ejercicio 20: Suma por múltiples ejes
resultado = np.sum(A, axis=(0, 2))
# Predice: resultado.shape = ?
```

<details>
<summary>✅ Respuestas Nivel 4</summary>

```python
# 16: ()  # Escalar, shape vacío
# 17: (4, 5)  # Colapsa eje 0
# 18: (3, 5)  # Colapsa eje 1
# 19: (3, 1, 5)  # Mantiene dimensión como 1
# 20: (4,)  # Colapsa ejes 0 y 2
```
</details>

---

## 🔥 Reshape y Manipulación (Nivel 5)

```python
A = np.arange(24)  # (24,)

# Ejercicio 21
resultado = A.reshape(4, 6)
# Predice: resultado.shape = ?

# Ejercicio 22
resultado = A.reshape(2, 3, 4)
# Predice: resultado.shape = ?

# Ejercicio 23
resultado = A.reshape(-1, 6)
# Predice: resultado.shape = ?

# Ejercicio 24
B = np.random.randn(3, 4)
resultado = B.flatten()
# Predice: resultado.shape = ?

# Ejercicio 25
resultado = B.reshape(-1)
# Predice: resultado.shape = ?
```

<details>
<summary>✅ Respuestas Nivel 5</summary>

```python
# 21: (4, 6)
# 22: (2, 3, 4)
# 23: (4, 6)  # -1 calcula automáticamente: 24/6 = 4
# 24: (12,)  # 3*4 = 12
# 25: (12,)  # Equivalente a flatten()
```
</details>

---

## 🔥 Operaciones de ML (Nivel 6 - Crítico)

```python
# Escenario: Red Neuronal
X = np.random.randn(100, 784)    # 100 imágenes, 784 píxeles
W1 = np.random.randn(784, 128)   # Pesos capa 1
b1 = np.random.randn(128)        # Bias capa 1
W2 = np.random.randn(128, 10)    # Pesos capa 2
b2 = np.random.randn(10)         # Bias capa 2

# Ejercicio 26: Forward pass capa 1
Z1 = X @ W1 + b1
# Predice: Z1.shape = ?

# Ejercicio 27: Forward pass capa 2
A1 = np.maximum(0, Z1)  # ReLU
Z2 = A1 @ W2 + b2
# Predice: Z2.shape = ?

# Ejercicio 28: Softmax output
exp_scores = np.exp(Z2)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
# Predice: probs.shape = ?

# Ejercicio 29: Gradiente respecto a W2
# Si dZ2.shape = (100, 10), ¿cuál es dW2.shape?
dZ2 = np.random.randn(100, 10)
dW2 = A1.T @ dZ2
# Predice: dW2.shape = ?

# Ejercicio 30: Gradiente respecto a W1
dA1 = dZ2 @ W2.T
# Predice: dA1.shape = ?
```

<details>
<summary>✅ Respuestas Nivel 6</summary>

```python
# 26: (100, 784) @ (784, 128) + (128,) = (100, 128)
#     Broadcasting: (128,) → (1, 128) → (100, 128)

# 27: (100, 128) @ (128, 10) + (10,) = (100, 10)

# 28: (100, 10)
#     exp_scores: (100, 10)
#     sum con keepdims: (100, 1)
#     división con broadcasting: (100, 10)

# 29: (128, 100) @ (100, 10) = (128, 10)
#     ¡CORRECTO! dW2 tiene el mismo shape que W2

# 30: (100, 10) @ (10, 128) = (100, 128)
#     ¡CORRECTO! dA1 tiene el mismo shape que A1
```
</details>

---

## 📊 Tracking de Progreso

| Fecha | Nivel Completado | Errores | Tiempo |
|-------|------------------|---------|--------|
| | | | |
| | | | |
| | | | |

---

## 🧠 Reglas de Oro para Recordar

1. **Producto matricial `@`**: (m, n) @ (n, p) = (m, p)
2. **Broadcasting**: Se alinean dimensiones desde la derecha
3. **Reducción sin keepdims**: Elimina el eje
4. **Reducción con keepdims**: Mantiene eje como 1
5. **reshape(-1)**: Calcula automáticamente esa dimensión
6. **Vector 1D (n,)**: Se comporta diferente a (n, 1) o (1, n)

---

## 🎯 Ejercicio Diario (5 minutos)

Cada día de estudio, antes de codificar:
1. Abre Python
2. Crea 3 arrays aleatorios
3. Predice 5 operaciones en papel
4. Verifica en código
5. Registra errores en el Diario
