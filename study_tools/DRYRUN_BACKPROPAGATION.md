# ✏️ Dry Run en Papel: Backpropagation

> **Semana 18 - Obligatorio**: Hacer un paso de backprop completo a mano ANTES de codificar.
> El código saldrá bien a la primera si la matemática en papel está clara.

---

## 🎯 Objetivo

Antes de escribir una sola línea de código de backpropagation:
1. Dibujar el grafo computacional
2. Calcular el forward pass con números simples
3. Calcular el backward pass a mano
4. Verificar que las dimensiones son correctas

---

## 📐 Ejercicio 1: Red Neuronal Mínima (2 neuronas)

### Arquitectura
```
Input → [Neurona 1] → [Neurona 2] → Output → Loss
  x         z₁→a₁         z₂→ŷ        L
```

### Parámetros
- **Input**: x = 2
- **Pesos**: w₁ = 0.5, w₂ = 0.3
- **Bias**: b₁ = 0.1, b₂ = 0.2
- **Activación**: ReLU
- **Target**: y = 1
- **Loss**: MSE = (ŷ - y)²

---

### Paso 1: Dibujar Grafo Computacional

```
    x=2
     │
     ▼
    [×]◄── w₁=0.5
     │
     ▼
    [+]◄── b₁=0.1
     │
     ▼
   z₁=1.1
     │
     ▼
  [ReLU]
     │
     ▼
   a₁=1.1
     │
     ▼
    [×]◄── w₂=0.3
     │
     ▼
    [+]◄── b₂=0.2
     │
     ▼
   z₂=0.53
     │
     ▼
   ŷ=0.53
     │
     ▼
    [-]◄── y=1
     │
     ▼
    [²]
     │
     ▼
  L=0.2209
```

---

### Paso 2: Forward Pass (Calcular cada nodo)

| Paso | Operación | Cálculo | Resultado |
|------|-----------|---------|-----------|
| 1 | z₁ = x·w₁ + b₁ | 2·0.5 + 0.1 | **z₁ = 1.1** |
| 2 | a₁ = ReLU(z₁) | max(0, 1.1) | **a₁ = 1.1** |
| 3 | z₂ = a₁·w₂ + b₂ | 1.1·0.3 + 0.2 | **z₂ = 0.53** |
| 4 | ŷ = z₂ | (sin activación final) | **ŷ = 0.53** |
| 5 | L = (ŷ - y)² | (0.53 - 1)² | **L = 0.2209** |

---

### Paso 3: Backward Pass (Regla de la Cadena)

**Objetivo**: Calcular ∂L/∂w₁, ∂L/∂w₂, ∂L/∂b₁, ∂L/∂b₂

#### Derivadas de cada operación:

| Operación | Derivada local |
|-----------|----------------|
| L = (ŷ - y)² | ∂L/∂ŷ = 2(ŷ - y) |
| ŷ = z₂ | ∂ŷ/∂z₂ = 1 |
| z₂ = a₁·w₂ + b₂ | ∂z₂/∂w₂ = a₁, ∂z₂/∂b₂ = 1, ∂z₂/∂a₁ = w₂ |
| a₁ = ReLU(z₁) | ∂a₁/∂z₁ = 1 si z₁ > 0, sino 0 |
| z₁ = x·w₁ + b₁ | ∂z₁/∂w₁ = x, ∂z₁/∂b₁ = 1 |

#### Cálculo paso a paso:

```
PASO ATRÁS 1: ∂L/∂ŷ
─────────────────────
∂L/∂ŷ = 2(ŷ - y) = 2(0.53 - 1) = 2(-0.47) = -0.94

PASO ATRÁS 2: ∂L/∂z₂
─────────────────────
∂L/∂z₂ = ∂L/∂ŷ · ∂ŷ/∂z₂ = -0.94 · 1 = -0.94

PASO ATRÁS 3: ∂L/∂w₂ y ∂L/∂b₂
─────────────────────────────
∂L/∂w₂ = ∂L/∂z₂ · ∂z₂/∂w₂ = -0.94 · a₁ = -0.94 · 1.1 = -1.034
∂L/∂b₂ = ∂L/∂z₂ · ∂z₂/∂b₂ = -0.94 · 1 = -0.94

PASO ATRÁS 4: ∂L/∂a₁
────────────────────
∂L/∂a₁ = ∂L/∂z₂ · ∂z₂/∂a₁ = -0.94 · w₂ = -0.94 · 0.3 = -0.282

PASO ATRÁS 5: ∂L/∂z₁
────────────────────
∂L/∂z₁ = ∂L/∂a₁ · ∂a₁/∂z₁
Como z₁ = 1.1 > 0, ∂a₁/∂z₁ = 1
∂L/∂z₁ = -0.282 · 1 = -0.282

PASO ATRÁS 6: ∂L/∂w₁ y ∂L/∂b₁
─────────────────────────────
∂L/∂w₁ = ∂L/∂z₁ · ∂z₁/∂w₁ = -0.282 · x = -0.282 · 2 = -0.564
∂L/∂b₁ = ∂L/∂z₁ · ∂z₁/∂b₁ = -0.282 · 1 = -0.282
```

---

### Paso 4: Resumen de Gradientes

| Parámetro | Gradiente | Interpretación |
|-----------|-----------|----------------|
| ∂L/∂w₂ | -1.034 | Aumentar w₂ reduce el loss |
| ∂L/∂b₂ | -0.94 | Aumentar b₂ reduce el loss |
| ∂L/∂w₁ | -0.564 | Aumentar w₁ reduce el loss |
| ∂L/∂b₁ | -0.282 | Aumentar b₁ reduce el loss |

---

### Paso 5: Actualización de Pesos (SGD, lr=0.1)

```
w₂_new = w₂ - lr · ∂L/∂w₂ = 0.3 - 0.1·(-1.034) = 0.3 + 0.1034 = 0.4034
b₂_new = b₂ - lr · ∂L/∂b₂ = 0.2 - 0.1·(-0.94) = 0.2 + 0.094 = 0.294
w₁_new = w₁ - lr · ∂L/∂w₁ = 0.5 - 0.1·(-0.564) = 0.5 + 0.0564 = 0.5564
b₁_new = b₁ - lr · ∂L/∂b₁ = 0.1 - 0.1·(-0.282) = 0.1 + 0.0282 = 0.1282
```

---

## 📐 Ejercicio 2: Red con 2 Inputs (Verificar Dimensiones)

### Arquitectura
```
x = [x₁, x₂] → Capa 1 (2→2) → ReLU → Capa 2 (2→1) → ŷ → Loss
```

### Dimensiones

| Variable | Shape | Notas |
|----------|-------|-------|
| X | (1, 2) | 1 ejemplo, 2 features |
| W₁ | (2, 2) | 2 inputs → 2 neuronas |
| b₁ | (1, 2) | Un bias por neurona |
| Z₁ = XW₁ + b₁ | (1, 2) | (1,2)@(2,2) + (1,2) |
| A₁ = ReLU(Z₁) | (1, 2) | Misma shape que Z₁ |
| W₂ | (2, 1) | 2 inputs → 1 output |
| b₂ | (1, 1) | Un bias |
| Z₂ = A₁W₂ + b₂ | (1, 1) | (1,2)@(2,1) + (1,1) |
| ŷ | (1, 1) | Un escalar |
| y | (1, 1) | Target |
| L | () | Escalar (sin shape) |

### Dimensiones de Gradientes (CRÍTICO)

| Gradiente | Shape | Fórmula de Verificación |
|-----------|-------|-------------------------|
| ∂L/∂ŷ | (1, 1) | Misma shape que ŷ |
| ∂L/∂Z₂ | (1, 1) | Misma shape que Z₂ |
| ∂L/∂W₂ | (2, 1) | **Misma shape que W₂** |
| ∂L/∂b₂ | (1, 1) | Misma shape que b₂ |
| ∂L/∂A₁ | (1, 2) | Misma shape que A₁ |
| ∂L/∂Z₁ | (1, 2) | Misma shape que Z₁ |
| ∂L/∂W₁ | (2, 2) | **Misma shape que W₁** |
| ∂L/∂b₁ | (1, 2) | Misma shape que b₁ |

### Regla de Oro
> **El gradiente de un parámetro SIEMPRE tiene la misma shape que el parámetro.**

---

## 📐 Plantilla en Blanco para Tus Ejercicios

### Datos del Problema
```
Input: x = ___
Pesos capa 1: W₁ = ___
Bias capa 1: b₁ = ___
Activación: ___
Pesos capa 2: W₂ = ___
Bias capa 2: b₂ = ___
Target: y = ___
Loss function: ___
```

### Forward Pass
| Paso | Operación | Cálculo | Resultado |
|------|-----------|---------|-----------|
| 1 | z₁ = | | |
| 2 | a₁ = | | |
| 3 | z₂ = | | |
| 4 | ŷ = | | |
| 5 | L = | | |

### Backward Pass
| Paso | Gradiente | Cálculo | Resultado |
|------|-----------|---------|-----------|
| 1 | ∂L/∂ŷ = | | |
| 2 | ∂L/∂z₂ = | | |
| 3 | ∂L/∂w₂ = | | |
| 4 | ∂L/∂b₂ = | | |
| 5 | ∂L/∂a₁ = | | |
| 6 | ∂L/∂z₁ = | | |
| 7 | ∂L/∂w₁ = | | |
| 8 | ∂L/∂b₁ = | | |

### Verificación de Dimensiones
| Parámetro | Shape | ∂L/∂param Shape | ✓/✗ |
|-----------|-------|-----------------|-----|
| W₁ | | | |
| b₁ | | | |
| W₂ | | | |
| b₂ | | | |

---

## 🧪 Código de Verificación

Después de hacer el dry run en papel, verifica con este código:

```python
import numpy as np

def verificar_backprop_manual():
    """Verifica los cálculos manuales del Ejercicio 1."""
    
    # Datos
    x = 2.0
    w1, b1 = 0.5, 0.1
    w2, b2 = 0.3, 0.2
    y = 1.0
    
    # Forward
    z1 = x * w1 + b1
    a1 = max(0, z1)  # ReLU
    z2 = a1 * w2 + b2
    y_hat = z2
    L = (y_hat - y) ** 2
    
    print("=== FORWARD PASS ===")
    print(f"z1 = {z1:.4f}")
    print(f"a1 = {a1:.4f}")
    print(f"z2 = {z2:.4f}")
    print(f"ŷ = {y_hat:.4f}")
    print(f"L = {L:.4f}")
    
    # Backward
    dL_dy_hat = 2 * (y_hat - y)
    dL_dz2 = dL_dy_hat * 1
    dL_dw2 = dL_dz2 * a1
    dL_db2 = dL_dz2 * 1
    dL_da1 = dL_dz2 * w2
    dL_dz1 = dL_da1 * (1 if z1 > 0 else 0)  # ReLU derivative
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1 * 1
    
    print("\n=== BACKWARD PASS ===")
    print(f"∂L/∂ŷ = {dL_dy_hat:.4f}")
    print(f"∂L/∂z₂ = {dL_dz2:.4f}")
    print(f"∂L/∂w₂ = {dL_dw2:.4f}")
    print(f"∂L/∂b₂ = {dL_db2:.4f}")
    print(f"∂L/∂a₁ = {dL_da1:.4f}")
    print(f"∂L/∂z₁ = {dL_dz1:.4f}")
    print(f"∂L/∂w₁ = {dL_dw1:.4f}")
    print(f"∂L/∂b₁ = {dL_db1:.4f}")
    
    # Verificación con numerical gradient
    eps = 1e-5
    
    def compute_loss(x, w1, b1, w2, b2, y):
        z1 = x * w1 + b1
        a1 = max(0, z1)
        z2 = a1 * w2 + b2
        return (z2 - y) ** 2
    
    numerical_dw1 = (compute_loss(x, w1+eps, b1, w2, b2, y) - 
                    compute_loss(x, w1-eps, b1, w2, b2, y)) / (2*eps)
    
    print(f"\n=== VERIFICACIÓN NUMÉRICA ===")
    print(f"∂L/∂w₁ analítico: {dL_dw1:.6f}")
    print(f"∂L/∂w₁ numérico:  {numerical_dw1:.6f}")
    print(f"Diferencia: {abs(dL_dw1 - numerical_dw1):.2e}")

verificar_backprop_manual()
```

---

## ✅ Checklist Antes de Codificar

- [ ] Dibujé el grafo computacional completo
- [ ] Calculé forward pass con números de ejemplo
- [ ] Calculé backward pass paso a paso
- [ ] Verifiqué que cada gradiente tiene la shape correcta
- [ ] Los gradientes de parámetros tienen la misma shape que los parámetros
- [ ] Verifiqué con numerical gradients (eps = 1e-5)

---

## 🚫 Errores Comunes a Evitar

1. **Olvidar la derivada de ReLU**: Es 0 cuando z ≤ 0
2. **Confundir shapes en matmul**: (m,n) @ (n,p) = (m,p)
3. **No transponer correctamente**: ∂L/∂W = X.T @ ∂L/∂Z
4. **Sumar gradientes en batch**: dL/db = sum(dL/dz, axis=0)
5. **Olvidar el 2 en MSE**: ∂(ŷ-y)²/∂ŷ = **2**(ŷ-y)
