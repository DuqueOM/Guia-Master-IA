# Laboratorios Interactivos (Interactive Labs)

Este repositorio incluye **laboratorios ejecutables** para convertir el material en un laboratorio práctico:

- Apps con **Streamlit** (sliders + feedback visual en tiempo real)
- Animaciones con **Manim Community Edition** (videos/gifs)
- Visualizaciones 3D con **Plotly** (rotación/zoom dentro de notebooks o apps)

## Estructura

- `interactive_labs/m02_linear_algebra/`
- `interactive_labs/m05_supervised/`
- `interactive_labs/m06_unsupervised/`
- `interactive_labs/m07_deep_learning/`

## Instalación (recomendado)

Dependencias base (guía + notebooks + PDF):

```bash
pip install -r requirements.txt
```

Dependencias para laboratorios interactivos:

```bash
pip install -r requirements-visual.txt
```

Notas:

- `manim` puede requerir dependencias del sistema (por ejemplo `ffmpeg`). Si falla, instala esas dependencias y reintenta.
- Para labs de PyTorch: instala el extra `pytorch` (ver `pyproject.toml`).

## Ejecución rápida

Streamlit:

```bash
streamlit run interactive_labs/m05_supervised/visualizacion_regresion.py
```

Manim:

```bash
manim -pqh interactive_labs/m02_linear_algebra/animacion_matriz.py AnimacionMatriz
```
