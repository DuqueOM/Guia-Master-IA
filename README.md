# 🎓 Guía Master IA - Ecosistema Educativo Híbrido

> **Preparación de 6 meses para el MS in AI de CU Boulder**
> Metodología: **Teoría (.md) → Práctica (.ipynb/.py) → Visualización (Streamlit/Manim)**

---

## 🎯 Objetivo

Dominio absoluto de los **3 cursos clave del MS-AI Pathway**:

| Track | Curso | Módulo |
|-------|-------|--------|
| **ML Line 1** | Supervised Learning | M05 |
| **ML Line 1** | Unsupervised Algorithms | M06 |
| **ML Line 1** | Deep Learning | M07 |

---

## 🗺️ Estructura del Ecosistema

```
Guia-Master-IA/
├── README.md                          # Este archivo
├── plan_de_estudio_6_meses.md         # Cronograma semana a semana
├── requirements.txt                   # Dependencias
│
├── M01_Fundamentos_Python/            # Semanas 1-2
│   ├── Teoria/                        # Contenido teórico profundo
│   ├── Notebooks/                     # Ejercicios prácticos
│   └── Laboratorios_Interactivos/     # Apps Streamlit/Manim
│
├── M02_Algebra_Lineal/                # Semanas 3-5
│   ├── Teoria/
│   ├── Notebooks/
│   └── Laboratorios_Interactivos/
│
├── M03_Calculo_Optimizacion/          # Semanas 6-7
│   ├── Teoria/
│   ├── Notebooks/
│   └── Laboratorios_Interactivos/
│
├── M04_Probabilidad_Estadistica/      # Semana 8
│   ├── Teoria/
│   ├── Notebooks/
│   └── Laboratorios_Interactivos/
│
├── M05_Aprendizaje_Supervisado/       # Semanas 9-12 ⭐ PATHWAY
│   ├── Teoria/
│   ├── Notebooks/
│   └── Laboratorios_Interactivos/
│
├── M06_Aprendizaje_No_Supervisado/    # Semanas 13-16 ⭐ PATHWAY
│   ├── Teoria/
│   ├── Notebooks/
│   └── Laboratorios_Interactivos/
│
├── M07_Deep_Learning/                 # Semanas 17-20 ⭐ PATHWAY
│   ├── Teoria/
│   ├── Notebooks/
│   └── Laboratorios_Interactivos/
│
├── M08_Proyecto_Integrador/           # Semanas 21-24
│   ├── Teoria/
│   ├── Notebooks/
│   ├── Laboratorios_Interactivos/
│   └── Proyecto_Final/
│
├── Recursos_Adicionales/
│   ├── Glosarios/
│   ├── Planes_Estrategicos/
│   └── Cheat_Sheets/
│
├── Herramientas_Estudio/              # Metacognición y evaluación
│   ├── DIARIO_ERRORES.md
│   ├── SIMULACRO_EXAMEN_TEORICO.md
│   └── ...
│
└── Huerfanos/                         # Archivos pendientes de clasificar
```

---

## 🔄 Metodología de Aprendizaje

### El Ciclo de 3 Fases

```
┌─────────────────────────────────────────────────────────────────┐
│  FASE 1: TEORÍA (.md)                                           │
│  • Lee el contenido en Teoria/                                  │
│  • Estudia definiciones, fórmulas y analogías                   │
│  • Dibuja conceptos en papel (método Feynman)                   │
├─────────────────────────────────────────────────────────────────┤
│  FASE 2: PRÁCTICA (.ipynb / .py)                                │
│  • Ejecuta notebooks en Notebooks/                              │
│  • Implementa algoritmos desde cero                             │
│  • Valida con asserts y tests                                   │
├─────────────────────────────────────────────────────────────────┤
│  FASE 3: VISUALIZACIÓN (Streamlit/Manim)                        │
│  • Ejecuta apps en Laboratorios_Interactivos/                   │
│  • Manipula parámetros en tiempo real                           │
│  • Conecta intuición visual con matemáticas                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Inicio Rápido

### 1. Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/Guia-Master-IA.git
cd Guia-Master-IA

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar un laboratorio interactivo

```bash
# Ejemplo: Visualización de transformaciones lineales
streamlit run M02_Algebra_Lineal/Laboratorios_Interactivos/transformacion_lineal_app.py
```

### 3. Seguir el plan de estudio

Ver [plan_de_estudio_6_meses.md](plan_de_estudio_6_meses.md) para el cronograma detallado.

---

## 📊 Progreso por Módulo

| Módulo | Semanas | Estado | Descripción |
|--------|---------|--------|-------------|
| M01 | 1-2 | 📚 | Python Científico (NumPy, Pandas) |
| M02 | 3-5 | 📚 | Álgebra Lineal (SVD, Eigenvalues) |
| M03 | 6-7 | 📚 | Cálculo y Optimización (Gradientes) |
| M04 | 8 | 📚 | Probabilidad (Bayes, MLE) |
| M05 | 9-12 | ⭐ | Supervised Learning (Pathway) |
| M06 | 13-16 | ⭐ | Unsupervised Learning (Pathway) |
| M07 | 17-20 | ⭐ | Deep Learning (Pathway) |
| M08 | 21-24 | 🎯 | Proyecto Integrador MNIST |

---

## 🛠️ Tecnologías

- **Python 3.10+**
- **NumPy / Pandas** - Computación científica
- **Matplotlib / Plotly** - Visualización
- **Streamlit** - Apps interactivas
- **Manim** - Animaciones matemáticas
- **PyTorch** - Deep Learning (Semana 24)

---

## 📖 Recursos Adicionales

- [Glosario Matemático](Recursos_Adicionales/Glosarios/GLOSARIO.md)
- [Planes Estratégicos](Recursos_Adicionales/Planes_Estrategicos/)
- [Herramientas de Estudio](Herramientas_Estudio/README.md)

---

## 🎯 Perfil de Salida

Al completar este programa podrás:

1. ✅ Implementar algoritmos de ML desde cero (sin sklearn)
2. ✅ Leer y entender papers de ML/DL
3. ✅ Aprobar los 3 cursos del MS-AI Pathway
4. ✅ Construir un portafolio profesional

---

*Desarrollado como preparación para el MS in AI de la University of Colorado Boulder*
