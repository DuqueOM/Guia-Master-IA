# 📋 Plan de Acción Perfeccionado v5.0 – Validación y Certificación

> Este plan NO cambia el contenido académico de la guía.
> Añade una capa de **validación externa**, **rigor en datos** y **simulacro de examen de admisión** sobre las mismas 24 semanas.

---

## 🎯 Objetivo de v5.0

- Que **tú** sepas que dominas el contenido (v3.x + v4.0).
- Que un **tercero** (mentor/IA/entrevistador) pueda confirmar tu nivel.
- Que tu ejecución esté alineada con el **formato de examen** de la maestría.

v5.0 introduce 5 protocolos sobre la guía principal:

1. **Protocolo 1 – Data Rigor (Dirty Data Check)**
2. **Protocolo 2 – Validación Externa (Desafío del Tablero Blanco)**
3. **Protocolo 3 – Examen de Admisión Simulado**
4. **Protocolo D – Visualización Generativa (Intuición Geométrica por Código)**
5. **Protocolo E – Rescate Cognitivo y Ejecución (Metacognición + Puentes + Simulacros PB + Badges)**

---

## 📦 Relación con otros documentos

- Contenido base de 24 semanas: [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md)
- Estrategia de estudio diario y PyTorch: [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
- Simulacros teóricos: `study_tools/SIMULACRO_EXAMEN_TEORICO.md`
- **Nuevas herramientas v5.0:**
  - `study_tools/DIRTY_DATA_CHECK.md`
  - `study_tools/DESAFIO_TABLERO_BLANCO.md`
  - `study_tools/EXAMEN_ADMISION_SIMULADO.md`
  - `study_tools/CIERRE_SEMANAL.md`
  - `study_tools/DIARIO_METACOGNITIVO.md`
  - `study_tools/TEORIA_CODIGO_BRIDGE.md`
  - `study_tools/BADGES_CHECKPOINTS.md`
  - `study_tools/SIMULACRO_PERFORMANCE_BASED.md`

---

## 1️⃣ Protocolo 1 – Data Rigor (Dirty Data Check)

> *"El código es inútil si el dato es basura."*

### 1.1. Módulo 01 – Python/Pandas (Semanas 1–2)

**Cambio en el entregable de Módulo 01:**

Además de cargar un CSV y convertirlo a NumPy, el entregable incluye ahora un **Dirty Data Check**:

- Identificar y documentar **al menos 5 problemas reales** en el dataset:
  - Valores nulos / NaN
  - Outliers obvios
  - Tipos incorrectos (strings donde deberían ser números)
  - Codificaciones extrañas ("?", "N/A", "-999" como missing)
  - Duplicados
- Para cada problema:
  - Describir la **estrategia de limpieza elegida** (drop, imputación, corrección, etc.).
  - Justificar la decisión (impacto en el modelo, tamaño de muestra, etc.).

📄 Usa la plantilla:

- `study_tools/DIRTY_DATA_CHECK.md` (sección **Caso 1: Módulo 01 – CSV Inicial**).

### 1.2. Módulo 05 – Supervised Learning (Semanas 9–12)

En el primer proyecto de **Regresión Logística** (Módulo 05):

- Usar un **dataset real** con:
  - Variables categóricas (requieren One-Hot Encoding).
  - Variables numéricas que necesitan escalado (MinMax / StandardScaler manual).
- Implementar un **pipeline de preprocesamiento** claro **antes** del modelo:
  - Limpieza básica (missing, outliers).
  - Codificación de categóricas (one-hot).
  - Escalado de numéricas.
  - División train/test.
- Documentar este flujo en `DIRTY_DATA_CHECK.md` (Caso 2: Módulo 05 – Dataset Supervisado).

🔗 Referencia cruzada en la guía:
- Ver sección **Supervised Learning (Módulo 05)** en [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md).

---

## 2️⃣ Protocolo 2 – Validación Externa (Desafío del Tablero Blanco)

> *"Si no lo puedes explicar en 5 minutos, no lo entiendes de verdad."*
> – Método Feynman aplicado a ML

La guía v3.2 ya menciona el "Reto del Tablero Blanco". v5.0 lo **formaliza**:

### 2.1. Frecuencia (Fase 1 y 2)

- **4 sesiones obligatorias**:
  - Semana 4
  - Semana 8
  - Semana 12
  - Semana 16

### 2.2. Dinámica de cada sesión

1. Elegir **un concepto central** de las semanas previas, por ejemplo:
   - Regla de la Cadena
   - Gradient Descent
   - K-Means
   - PCA
   - Regresión Logística
   - Backpropagation
2. Preparar una **explicación de 5 minutos** como si hablaras con un colega.
3. Grabar un **video corto** (pantalla + voz, cámara opcional) explicando el concepto en:
   - Un tablero blanco físico,
   - Una tablet, o
   - Una pizarra digital sencilla.
4. Pedir **feedback externo**:
   - Mentor, colega, comunidad online, o
   - Una IA avanzada (pedir evaluación en claridad, precisión y rigor).

📄 Usa la plantilla:

- `study_tools/DESAFIO_TABLERO_BLANCO.md` para:
  - Registrar tema, fecha, links al video.
  - Autoevaluación + feedback recibido.

### 2.3. Criterio de dominio (Feynman)

> Si en 5 minutos **no puedes** explicar el concepto:
> - sin leer,
> - sin abusar de jerga,
> - y sin cometer errores conceptuales,
>
> entonces **no lo has dominado** todavía.

En ese caso:
- Volver al módulo correspondiente en [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md).
- Repetir ejercicios clave.
- Reintentar el desafío en 1 semana.

---

## 3️⃣ Protocolo 3 – Examen de Admisión Simulado (Semanas 22 y 23)

> *"Entrena como si mañana fuera el examen real."*

Este protocolo convierte las semanas 22 y 23 en un **campo de entrenamiento de examen**.

### 3.1. Formato del Examen Simulado

- **Duración:** 2 horas continuas.
- **Condiciones:**
  - Sin internet.
  - Sin IDE.
  - Solo papel, lápiz y calculadora básica.
- **Contenido:**
  - **40% Código (en pseudocódigo / pasos):**
    - PCA paso a paso,
    - o Backpropagation en una red simple,
    - o K-Means completo.
  - **60% Teórico:**
    - Derivación de una función de pérdida (e.g., Cross-Entropy).
    - Explicación gráfica de Bias–Variance.
    - Preguntas conceptuales de ML/DL.

📄 Detalle y plantilla:

- `study_tools/EXAMEN_ADMISION_SIMULADO.md`
  Incluye estructura sugerida, rúbrica de calificación y hojas para registrar resultados.

### 3.2. Calendario

- **Semana 22 – Simulacro 1 (diagnóstico):**
  - Objetivo: detectar debilidades antes de la última semana.
- **Semana 23 – Simulacro 2 (final):**
  - Objetivo: confirmar que ya estás en nivel de admisión.

### 3.3. Métrica de "Listo para Admisión"

- La nota del simulacro final (Semana 23) es tu **Puntaje de Admisión Simulado**.
- Criterio:
  - **≥ 80%:** nivel adecuado para presentarte con confianza.
  - **< 80%:** recomendar extender 2–4 semanas más, reforzar teoría y repetir simulacro.

---

## 4️⃣ Hoja de Ruta Integrada v5.0

| Fase | Tarea Principal (Contenido) | Mejora Estratégica (Ejecución v5.0) |
|------|-----------------------------|--------------------------------------|
| Semanas 1–8 | Fundamentos Matemáticos (Álgebra, Cálculo, Probabilidad). | **Protocolo 1 – Data Rigor:** Dirty Data Check en Módulo 01 (CSV real). |
| Semanas 9–20 | Core ML (Supervisado, No Supervisado, Deep Learning). | **Protocolo 1 + 2:** Dirty Data Check en proyecto supervisado + 4 desafíos de Tablero Blanco. |
| Semanas 21–24 | Proyecto Integrador (MNIST Analyst). | **Protocolo 3:** 2 simulacros de examen (teoría + pseudocódigo) en Semanas 22 y 23. |

La **transición a PyTorch** sigue estando detallada en [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md) y en `study_tools/PUENTE_NUMPY_PYTORCH.md`.

---

## 5️⃣ Cómo usar v4.0 y v5.0 juntos

- **v4.0** responde: *"¿Cómo estudio cada día para no abandonar?"*
  - Protocolo "sandwich".
  - Diario de errores.
  - Dry run de backprop.
  - Puente final a PyTorch.

- **v5.0** responde: *"¿Cómo demuestro (a otros y a un examen) que sí estoy listo?"*
  - Dirty Data Check con datasets reales.
  - Validación externa con explicaciones orales.
  - Simulacro de examen de admisión.

Usa ambos planes como **capas** sobre [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md):

1. Primero aseguras **ejecución diaria y proyectos** (v4.0).
2. Luego aseguras **validación externa y simulacro de examen** (v5.0).

---

## 6️⃣ Checklist Rápido v5.0

- [ ] Módulo 01: `DIRTY_DATA_CHECK.md` completado para un CSV real.
- [ ] Módulo 05: Dirty Data Check aplicado a dataset supervisado con categóricas + escalado.
- [ ] 4 videos del **Desafío del Tablero Blanco** (Semanas 4, 8, 12, 16) grabados y evaluados.
- [ ] Simulacro 1 (Semana 22) completado y analizado.
- [ ] Simulacro 2 (Semana 23) ≥ 80%.
- [ ] Diario de Errores actualizado con errores conceptuales detectados en simulacros y desafíos.

## 7️⃣ Plan de Acción Definitivo v5.0 – Protocolo A/B/C

> Esta sección resume cómo combinar los módulos y herramientas existentes en **tres protocolos operativos** durante las 24 semanas.

### Protocolo A – Verificación Triple del Código (Precisión Matemática)

| Módulo/Semana | Acción de Validación | Objetivo de Seguridad |
|---------------|----------------------|------------------------|
| **Módulo 03 – Cálculo** | **Gradient Checking Obligatorio**: antes de confiar en cualquier `backward()` manual, ejecutar un chequeo de gradiente por diferencias finitas (método de diferencia central) sobre tus derivadas clave. | Detectar errores sutiles de signo o de álgebra en la implementación de tus gradientes. |
| **Módulo 05 – Supervised Learning** | **Modo Sombra (Shadow Mode)**: los viernes, después de entrenar tu propia Regresión Lineal/Logística, entrenar el mismo modelo con `sklearn` y comparar accuracy y coeficientes (W, b). | Validar que tu implementación desde cero coincide con la librería estándar de la industria; si hay discrepancias grandes, el bug es tuyo. |
| **Módulo 07 – Deep Learning** | **Test de Overfitting**: entrenar tu red MLP manual para que memorice un minibatch pequeño (5–10 ejemplos) hasta pérdida casi cero. | Comprobar que la lógica de forward/backward es correcta; si no puede memorizar el minibatch, hay un fallo estructural en backprop. |

- Referencias:
  - Gradient checking: sección **"Gradient Checking: Validación Matemática (v3.3)"** en `03_CALCULO_MULTIVARIANTE.md`.
  - Shadow Mode: sección **"Shadow Mode: Validación con sklearn (v3.3)"** en `05_SUPERVISED_LEARNING.md`.
  - Overfitting: implementación y entrenamiento de MLP en `07_DEEP_LEARNING.md`.

### Protocolo B – Rigor Académico y Validación Externa (Examen)

| Frecuencia | Acción | Objetivo |
|-----------|--------|----------|
| **Diario (15 minutos)** | **Diario de Errores Críticos**: registrar a mano los 5 errores conceptuales más grandes del día (confusiones de definiciones, notación, dimensiones, interpretación de resultados, etc.). | Evitar repetir los mismos errores y construir un "resumen de examen" personalizado. |
| **Mensual (Sábado)** | **Desafío del Tablero Blanco (Feynman Challenge)**: explicar en 5–7 minutos, sin leer notas, un concepto complejo (PCA, K-Means, Gradient Descent, Regla de la Cadena, Backpropagation, etc.) y registrar feedback externo. | Probar dominio conceptual en lenguaje sencillo, similar a una entrevista técnica u oral de examen. |
| **Semanas 22–23** | **Simulacro de Examen de Admisión**: 2 horas, sin IDE ni internet, 40% pseudocódigo y dimensiones, 60% teoría y derivaciones. | Simular el Pathway real y obtener una métrica objetiva de "listo / no listo". |

- Herramientas asociadas:
  - Diario: `study_tools/DIARIO_ERRORES.md`.
  - Tablero Blanco: `study_tools/DESAFIO_TABLERO_BLANCO.md` (y sección 2 de este documento).
  - Examen: `study_tools/EXAMEN_ADMISION_SIMULADO.md` (y sección 3 de este documento).

### Protocolo C – Puente a la Realidad (Transición a Industria)

1. **Traducción del Core ML a PyTorch (Módulo 08)**
   - Después de completar el proyecto **MNIST Analyst** con tu propio código NumPy, reescribir tu red neuronal más avanzada usando PyTorch (`torch.nn.Linear`, `optim`, etc.).
   - Apoyarte en `study_tools/PUENTE_NUMPY_PYTORCH.md` y en las secciones de Deep Learning para mapear capas y operaciones.
   - Objetivo: ver cómo docenas de líneas de inicialización y backprop se condensan en unas pocas capas de alto nivel.

2. **Análisis de Errores con Datos Reales (Módulo 08)**
   - En el informe final del proyecto MNIST incluir una sección titulada **"Las 5 imágenes peor clasificadas"**.
   - Para cada imagen:
     - Mostrar la imagen y la predicción errónea del modelo.
     - Describir brevemente el tipo de error.
     - Argumentar si el fallo se debe principalmente a **Bias** (modelo demasiado simple) o **Variance** (modelo demasiado complejo / datos ruidosos o insuficientes), conectando con el análisis de Bias–Variance del proyecto.

Con estos tres protocolos, v5.0 pasa de ser solo una capa extra de herramientas a un **sistema completo de verificación, validación externa y transición a herramientas de producción** aplicado sobre las mismas 24 semanas del plan base.

---

### Protocolo E – Rescate Cognitivo y Ejecución (Metacognición + Puentes + Simulacros PB + Badges)

Objetivo: reducir **fatiga cognitiva**, aumentar **retención** y mantener **motivación visible** durante 24 semanas sin bajar el rigor.

Regla práctica:

- esto no es contenido extra: es **consolidación** y **transferencia** (teoría → código) que evita deserción.

Componentes:

1) **Cierre Cognitivo Semanal (Sábado, 1 hora)**
   - Herramienta: `study_tools/CIERRE_SEMANAL.md`
   - Incluye mapa mental, Feynman, 3 errores comunes, checklist y mini autoevaluación.

2) **Metacognición diaria (5 min)**
   - Herramienta: `study_tools/DIARIO_METACOGNITIVO.md`
   - Preguntas guía: qué entendí, qué no, patrón de error, acción correctiva.

3) **Puente Teoría ↔ Código (semanal, 20–30 min)**
   - Herramienta: `study_tools/TEORIA_CODIGO_BRIDGE.md`
   - Ejemplos: covarianza → NumPy, gradiente MSE → implementación, log-sum-exp → estabilidad.

4) **Badges y mini-victorias (por módulo)**
   - Herramienta: `study_tools/BADGES_CHECKPOINTS.md`
   - Criterio: cada badge requiere evidencia verificable (script/notebook/test/explicación breve).

5) **Simulacros performance-based (Semanas 8, 16, 23)**
   - Herramienta: `study_tools/SIMULACRO_PERFORMANCE_BASED.md`
   - Formato: 50% teoría + 50% pseudocódigo/código.


### Protocolo D – Visualización Generativa (Intuición Geométrica por Código)

Objetivo: construir intuición geométrica **mediante visualizaciones reproducibles** (sin depender de imágenes estáticas). El estudiante aprende a “ver”:

- matrices como deformación del espacio
- gradiente descent como trayectoria en un valle
- convolución como detector local de patrones

Regla práctica:

- no es “decoración”: cada visualización debe responder una pregunta conceptual

#### Semana 3 (Álgebra) – Transformaciones lineales y eigenvectors

- **Tarea:** ejecutar y modificar [`visualizations/viz_transformations.py`](../visualizations/viz_transformations.py)
- **Entregable:**
  - un plot que muestre una rejilla antes/después de `A`
  - y una breve nota: “¿cuál es el eigenvector (si existe) y qué le pasa?”

#### Semana 7 (Cálculo) – Descenso de gradiente 3D interactivo

- **Tarea:** ejecutar [`visualizations/viz_gradient_3d.py`](../visualizations/viz_gradient_3d.py) (Plotly, export a HTML)
- **Entregable:**
  - captura o export de la trayectoria con `lr` pequeño (convergencia)
  - captura o export con `lr` grande (divergencia)
  - explicación en 5 líneas del fenómeno

#### Semana 19 (CNNs) – Convolución y feature maps (Sobel)

- **Tarea:** ejecutar [`visualizations/viz_convolution.py`](../visualizations/viz_convolution.py)
- **Entregable:**
  - cargar una imagen propia (foto) y aplicar Sobel
  - mostrar input vs feature map
  - explicar qué patrón detecta el filtro

Herramientas requeridas:

- `matplotlib`
- `plotly`
- `ipywidgets`
