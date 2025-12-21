import importlib
import sys
from pathlib import Path

import numpy as np
import streamlit as st


def _load_plot_linear_transformation():
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return importlib.import_module(
        "visualizations.viz_transformations"
    ).plot_linear_transformation


plot_linear_transformation = _load_plot_linear_transformation()

st.set_page_config(page_title="Transformación lineal (2x2)", layout="centered")

st.title("Laboratorio Interactivo: Transformación Lineal (2x2)")
st.write(
    "Define una matriz 2×2 y observa cómo deforma el espacio. Activa eigenvectors para ver las direcciones que la transformación respeta."
)

col1, col2 = st.columns(2)

with col1:
    a11 = st.number_input("a11", value=1.0, step=0.1)
    a21 = st.number_input("a21", value=0.0, step=0.1)

with col2:
    a12 = st.number_input("a12", value=1.0, step=0.1)
    a22 = st.number_input("a22", value=1.0, step=0.1)

A = np.array([[a11, a12], [a21, a22]], dtype=float)

show_eigen = st.checkbox("Mostrar eigenvectors (si son reales)", value=True)
lim = st.slider("Escala visual", min_value=2.0, max_value=10.0, value=5.0, step=1.0)

fig, _ax = plot_linear_transformation(A, lim=float(lim), show_eigen=show_eigen)
st.pyplot(fig)

vals, vecs = np.linalg.eig(A)
vals = np.real_if_close(vals)
vecs = np.real_if_close(vecs)

st.subheader("Diagnóstico")
st.write("Matriz A:")
st.code(str(A))

st.write("Eigenvalues (λ):")
st.code(str(vals))

if not np.iscomplexobj(vecs):
    st.write("Eigenvectors (columnas de V):")
    st.code(str(vecs))
else:
    st.warning(
        "Los eigenvectors son complejos para esta matriz (no hay eigenvectors reales en R²)."
    )

st.info(
    "Ejecuta: streamlit run interactive_labs/m02_linear_algebra/transformacion_lineal_app.py"
)
