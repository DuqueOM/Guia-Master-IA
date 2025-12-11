# 📓 Diario de Errores - Master en IA

> "El experto en algo fue una vez un principiante que cometió todos los errores posibles."

## Instrucciones
Registra CADA error matemático o de código que cometas. Antes del examen, revisa este diario.
Este es tu recurso más valioso para no repetir errores.

---

## Plantilla de Registro

```markdown
### [FECHA] - [TEMA]

**🔴 Error cometido:**
[Descripción breve del error]

**💡 Causa raíz:**
[Por qué ocurrió - confusión conceptual, typo, etc.]

**✅ Solución:**
[Cómo lo arreglaste]

**🎯 Lección aprendida:**
[Qué hacer diferente la próxima vez]

**🏷️ Categoría:** [NumPy | Álgebra Lineal | Cálculo | Probabilidad | ML | DL]
```

---

## 📊 Registro de Errores

### Categorías Comunes de Errores

#### 🔢 NumPy y Operaciones con Arrays
| Fecha | Error | Frecuencia |
|-------|-------|------------|
| | | |

#### 📐 Álgebra Lineal
| Fecha | Error | Frecuencia |
|-------|-------|------------|
| | | |

#### 📈 Cálculo y Gradientes
| Fecha | Error | Frecuencia |
|-------|-------|------------|
| | | |

#### 🎲 Probabilidad y Estadística
| Fecha | Error | Frecuencia |
|-------|-------|------------|
| | | |

#### 🤖 Machine Learning
| Fecha | Error | Frecuencia |
|-------|-------|------------|
| | | |

#### 🧠 Deep Learning
| Fecha | Error | Frecuencia |
|-------|-------|------------|
| | | |

---

## 🚨 Errores Frecuentes (Top 10)

> Actualiza esta lista semanalmente con tus errores más repetidos

1. **[Pendiente]**
2. **[Pendiente]**
3. **[Pendiente]**
4. **[Pendiente]**
5. **[Pendiente]**
6. **[Pendiente]**
7. **[Pendiente]**
8. **[Pendiente]**
9. **[Pendiente]**
10. **[Pendiente]**

---

## 📝 Registro Detallado

<!-- Copia la plantilla y añade tus errores aquí -->

### [EJEMPLO] 2024-XX-XX - NumPy Broadcasting

**🔴 Error cometido:**
Confundí `*` (multiplicación elemento a elemento) con `@` (multiplicación de matrices)

```python
# Lo que escribí (INCORRECTO):
resultado = A * B  # Broadcasting elemento a elemento

# Lo que quería (CORRECTO):
resultado = A @ B  # Producto matricial
```

**💡 Causa raíz:**
En matemáticas usamos el mismo símbolo para ambas operaciones. NumPy las diferencia.

**✅ Solución:**
- `*` → Hadamard product (elemento a elemento)
- `@` → Producto matricial (matrix multiplication)
- `np.dot()` → También producto matricial pero menos claro

**🎯 Lección aprendida:**
Siempre verificar con `.shape` antes y después de operaciones matriciales.
Usar `@` explícitamente para productos matriciales.

**🏷️ Categoría:** NumPy

---

### [EJEMPLO] 2024-XX-XX - Dimensiones en Gradientes

**🔴 Error cometido:**
El gradiente de la pérdida tenía dimensiones invertidas, causando error silencioso en backprop.

```python
# INCORRECTO: (n_features, n_samples)
dW = X @ dL.T

# CORRECTO: (n_features, n_samples) @ (n_samples, 1) = (n_features, 1)
dW = X.T @ dL
```

**💡 Causa raíz:**
No verifiqué las dimensiones esperadas antes de codificar.

**✅ Solución:**
Escribir las dimensiones esperadas en comentarios ANTES de codificar:
```python
# X: (n_samples, n_features)
# dL: (n_samples, 1)
# dW debe ser: (n_features, 1)
dW = X.T @ dL  # (n_features, n_samples) @ (n_samples, 1) ✓
```

**🎯 Lección aprendida:**
SIEMPRE documentar shapes esperados en comentarios. Usar asserts:
```python
assert dW.shape == (n_features, 1), f"Expected {(n_features, 1)}, got {dW.shape}"
```

**🏷️ Categoría:** Deep Learning

---

## 📅 Resumen Semanal

| Semana | Total Errores | Categoría Más Problemática | Acción Correctiva |
|--------|---------------|---------------------------|-------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |

---

## 🎓 Notas Pre-Examen

> Revisa esta sección 24 horas antes de cualquier examen

### Errores que DEBO evitar:
1.
2.
3.

### Verificaciones que DEBO hacer:
1.
2.
3.
