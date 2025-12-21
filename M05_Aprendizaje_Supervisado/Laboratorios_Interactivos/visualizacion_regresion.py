import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.title("Laboratorio Interactivo: Intuición de Regresión Lineal")
st.write("Ajusta la pendiente (m) y la intersección (b) para minimizar el error.")

# Generar datos sintéticos
rng = np.random.default_rng(42)
X = 2 * rng.random((100, 1))
y = 4 + 3 * X + rng.normal(size=(100, 1))

# Controles interactivos
m = st.slider("Pendiente (m)", 0.0, 10.0, 1.0)
b = st.slider("Intersección (b)", 0.0, 10.0, 0.0)

# Predicción del usuario
y_pred = m * X + b

# Cálculo del Error (MSE)
mse = float(np.mean((y - y_pred) ** 2))

# Visualización
fig, ax = plt.subplots()
ax.scatter(X, y, color="blue", alpha=0.5, label="Datos Reales")
ax.plot(X, y_pred, color="red", linewidth=2, label=f"Tu Modelo (MSE: {mse:.2f})")
ax.set_xlabel("Variable X")
ax.set_ylabel("Objetivo y")
ax.legend()

st.pyplot(fig)
st.success(f"Error Cuadrático Medio actual: {mse:.4f}")
