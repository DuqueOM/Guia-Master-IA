import numpy as np
import plotly.graph_objects as go
import streamlit as st


def rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def make_ellipsoid(n: int = 600, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    scales = np.array([3.0, 1.5, 0.5])
    X = Z * scales
    # Rotación fija para que el eje principal no esté alineado con x
    R0 = rotation_z(np.deg2rad(35.0)) @ rotation_y(np.deg2rad(20.0))
    return X @ R0.T


st.set_page_config(page_title="PCA manual (rotación)", layout="wide")

st.title("Laboratorio Interactivo: Intuición de PCA (rotación 3D → proyección 2D)")
st.write(
    "Rota el sistema de coordenadas e intenta alinear el eje x' con la dirección de máxima varianza. Luego compara con PCA real (SVD)."
)

seed = st.number_input("Semilla", value=42, step=1)
angle_z = st.slider("Rotación alrededor de Z (grados)", 0.0, 180.0, 0.0, 1.0)
angle_y = st.slider("Rotación alrededor de Y (grados)", 0.0, 180.0, 0.0, 1.0)

X = make_ellipsoid(seed=int(seed))
R = rotation_z(np.deg2rad(float(angle_z))) @ rotation_y(np.deg2rad(float(angle_y)))
Xr = X @ R.T

# Varianza por eje tras rotación manual
variances = np.var(Xr, axis=0, ddof=1)

# PCA real (SVD) como referencia
Xc = X - X.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
explained = (S**2) / (len(X) - 1)
explained_ratio = explained / explained.sum()

col_left, col_right = st.columns(2)

with col_left:
    fig3d = go.Figure(
        data=[
            go.Scatter3d(
                x=Xr[:, 0],
                y=Xr[:, 1],
                z=Xr[:, 2],
                mode="markers",
                marker={"size": 2, "opacity": 0.6},
            )
        ]
    )
    fig3d.update_layout(
        height=520,
        scene={"xaxis_title": "x'", "yaxis_title": "y'", "zaxis_title": "z'"},
        title="Nube 3D (rotada manualmente)",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    st.plotly_chart(fig3d, use_container_width=True)

with col_right:
    fig2d = go.Figure(
        data=[
            go.Scatter(
                x=Xr[:, 0],
                y=Xr[:, 1],
                mode="markers",
                marker={"size": 4, "opacity": 0.55},
            )
        ]
    )
    fig2d.update_layout(
        height=520,
        xaxis_title="x'",
        yaxis_title="y'",
        title="Proyección 2D (x', y')",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    st.plotly_chart(fig2d, use_container_width=True)

st.subheader("Señales")

st.write("Varianza por eje después de tu rotación (objetivo: var(x') grande):")
st.code(str(variances))

st.write("PCA real (SVD) — explained variance ratio (PC1, PC2, PC3):")
st.code(str(explained_ratio))

st.info(
    "Ejecuta: streamlit run interactive_labs/m06_unsupervised/pca_rotation_plotly_app.py"
)
