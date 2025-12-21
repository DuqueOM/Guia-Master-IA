import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def make_data(n: int = 80, noise: float = 0.25, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n,))
    y = np.sin(2 * np.pi * X) + rng.normal(0.0, noise, size=(n,))
    return X, y


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


st.set_page_config(page_title="Bias-Variance (Overfitting)", layout="centered")

st.title("Laboratorio Interactivo: Bias–Variance / Overfitting")
st.write(
    "Aumenta el grado del polinomio y observa cómo baja el error de entrenamiento, pero puede subir el de validación."
)

degree = st.slider("Grado del polinomio", min_value=1, max_value=20, value=3, step=1)
noise = st.slider("Ruido", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
seed = st.number_input("Semilla", value=42, step=1)

X, y = make_data(n=80, noise=float(noise), seed=int(seed))

# Split fijo reproducible
idx = np.arange(len(X))
rng = np.random.default_rng(int(seed))
rng.shuffle(idx)
train_size = int(0.8 * len(X))
train_idx = idx[:train_size]
val_idx = idx[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# Curva + errores por grado
degrees = np.arange(1, 21)
train_errs = []
val_errs = []

for d in degrees:
    coeffs = np.polyfit(X_train, y_train, deg=int(d))
    p = np.poly1d(coeffs)
    train_errs.append(mse(y_train, p(X_train)))
    val_errs.append(mse(y_val, p(X_val)))

# Modelo seleccionado
coeffs = np.polyfit(X_train, y_train, deg=int(degree))
p = np.poly1d(coeffs)

x_grid = np.linspace(-1.0, 1.0, 400)
y_grid = p(x_grid)

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.scatter(X_train, y_train, s=18, alpha=0.8, label="Train")
ax1.scatter(X_val, y_val, s=18, alpha=0.8, label="Val")
ax1.plot(x_grid, y_grid, color="red", linewidth=2, label=f"Polinomio grado {degree}")
ax1.set_title("Ajuste vs datos")
ax1.legend()
ax1.grid(True, alpha=0.2)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(degrees, train_errs, marker="o", label="MSE train")
ax2.plot(degrees, val_errs, marker="o", label="MSE val")
ax2.axvline(int(degree), color="black", linestyle="--", alpha=0.6)
ax2.set_title("Curva Bias–Variance (train vs val)")
ax2.set_xlabel("Grado")
ax2.set_ylabel("MSE")
ax2.legend()
ax2.grid(True, alpha=0.2)

st.pyplot(fig1)
st.pyplot(fig2)

st.success(
    f"MSE train (grado {degree}): {train_errs[int(degree)-1]:.4f} | MSE val: {val_errs[int(degree)-1]:.4f}"
)
