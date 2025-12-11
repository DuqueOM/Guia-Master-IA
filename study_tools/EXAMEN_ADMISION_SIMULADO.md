# 🎓 Examen de Admisión Simulado – v5.0

> *"Practica el examen antes de que el examen te practique a ti."*

Este documento define la estructura y rúbrica del **Examen de Admisión Simulado** que debes realizar en las **Semanas 22 y 23**.

---

## 🕒 Formato General

- **Duración total:** 2 horas.
- **Condiciones:**
  - Sin internet.
  - Sin IDE / Jupyter.
  - Solo papel, lápiz y calculadora básica.
- **Estructura recomendada:**
  - Parte A – Código en Pseudocódigo (40 pts).
  - Parte B – Teoría y Derivaciones (60 pts).

> Consejo: imprime este documento y úsalo como portada de tu examen.

---

## 📘 Parte A – Código en Pseudocódigo (40 pts)

Elige **UNO** de los siguientes problemas (o combina elementos de varios):

### Opción 1 – PCA (20–40 pts)

1. Escribe en pseudocódigo el algoritmo de PCA usando SVD:
   - Cálculo de la media.
   - Centrado de datos.
   - Cálculo de la matriz de covarianza o uso directo de SVD.
   - Selección de componentes principales.
   - Proyección de datos a espacio reducido.
2. Añade comentarios que expliquen **qué hace cada paso** y **por qué**.

### Opción 2 – K-Means (20–40 pts)

1. Escribe el pseudocódigo completo del algoritmo de K-Means:
   - Inicialización (idealmente K-Means++).
   - Asignación de puntos a centroides.
   - Re-cálculo de centroides.
   - Criterio de parada.
2. Explica cómo cambiaría si usas otra métrica de distancia.

### Opción 3 – Backpropagation (20–40 pts)

1. Considera una red MLP simple (ejemplo: 2–3–1).
2. Escribe el pseudocódigo del **forward pass** y del **backward pass** para MSE o cross-entropy.
3. Especifica dónde ocurren las multiplicaciones de matrices y las derivadas de activación.

> **Puntuación:**
> - 30–40 pts: algoritmo completo, ordenado y explicado.
> - 20–29 pts: idea correcta con huecos menores.
> - < 20 pts: omisiones importantes o pasos incorrectos.

---

## 📗 Parte B – Teoría y Derivaciones (60 pts)

### Sección 1 – Derivación de Función de Pérdida (30 pts)

Ejemplo recomendado: **Cross-Entropy para regresión logística binaria**.

1. Escribe la función de pérdida:
   \( L(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})] \)
   con \( \hat{y}^{(i)} = \sigma(\theta^T x^{(i)}) \).
2. Deriva \( \frac{\partial L}{\partial \theta} \) paso a paso usando la Regla de la Cadena.
3. Simplifica la expresión final y explica su interpretación.

> **Puntuación (30 pts):**
> - 25–30: derivación correcta, pasos claros y bien justificados.
> - 18–24: idea central correcta, algunos saltos o pequeños errores.
> - < 18: errores de concepto o pasos clave ausentes.

### Sección 2 – Bias–Variance y Generalización (30 pts)

1. Define **bias** y **variance** en el contexto de ML.
2. Dibuja (en papel) una gráfica conceptual de un modelo **subajustado**, **bien ajustado** y **sobreajustado**.
3. Explica cómo cambiarían las curvas de entrenamiento/validación si:
   - Aumentas la complejidad del modelo.
   - Aumentas el tamaño del dataset.
   - Aumentas la regularización.
4. Da **2 ejemplos concretos** (e.g., regresión lineal simple vs MLP grande en MNIST).

> **Puntuación (30 pts):**
> - 25–30: explicaciones precisas, gráficas coherentes, ejemplos sólidos.
> - 18–24: comprensión aceptable pero con lagunas.
> - < 18: confusión en el concepto o uso incorrecto de términos.

---

## 📊 Hoja de Calificación

### Resumen de Puntuación

| Parte | Máx. | Obtenido |
|-------|------|----------|
| A – Pseudocódigo | 40 |    |
| B1 – Derivación de pérdida | 30 |    |
| B2 – Bias–Variance | 30 |    |
| **Total** | **100** |    |

**Fecha del simulacro:**
**Semana:** [22 o 23]

---

## ✅ Criterio de Aprobación

- **Objetivo mínimo:** **80/100** en el simulacro de la **Semana 23**.
- Si obtienes **< 80**:
  - Identifica secciones débiles (¿Parte A, B1 o B2?).
  - Revisa los módulos correspondientes en `docs/`.
  - Repite el simulacro una semana después, si es posible.

---

## 🧠 Reflexión Post-Examen

Después de cada simulacro, responde:

1. ¿Qué parte se sintió **más difícil**? ¿Por qué?
2. ¿En qué momento te quedaste sin tiempo?
3. ¿Alguna derivación que creías dominar resultó difícil en papel?
4. ¿Qué cambiarás en tu estudio de la semana siguiente?

---

## 📅 Registro de Simulacros

| # | Semana | Fecha | Puntuación | Parte más débil | Acción correctiva |
|---|--------|-------|-----------|------------------|-------------------|
| 1 | 22 | | /100 | | |
| 2 | 23 | | /100 | | |
