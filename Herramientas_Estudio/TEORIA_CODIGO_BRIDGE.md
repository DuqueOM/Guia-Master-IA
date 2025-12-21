# 🔁 Teoría ↔ Código Bridge (semanal) – v5.1

> Objetivo: entrenar el criterio clave del Pathway: traducir matemática a implementación.

---

## ⏱️ Frecuencia

- 1 vez por semana (ideal: viernes o sábado antes del cierre).
- 20–30 min.

---

## ✅ Plantilla (copia/pega)

```text
SEMANA: __
TEMA: __

1) Teoría (en 1 línea):
__

2) Fórmula/expresión:
__

3) Traducción a NumPy (en 1-3 líneas):
__

4) Shapes esperados:
- X: __
- y: __
- salida: __

5) Test rápido / sanity check:
__

6) Error típico + señal de alerta:
__
```

---

## Ejemplos sugeridos (elige 1 por semana)

- Matriz de covarianza `Σ = (1/(n-1)) XᵀX` → implementación con `X_centered.T @ X_centered`.
- Gradiente MSE → `X.T @ (y_pred - y)`.
- Log-sum-exp → estabilidad numérica de softmax.
- Output shape en CNN (stride/padding) → cálculo de dimensiones.
