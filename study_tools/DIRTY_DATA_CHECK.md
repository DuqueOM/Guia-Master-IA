# 🧹 Dirty Data Check – v5.0

> *"El modelo es tan bueno como el dato que le das."*

Este documento te guía para documentar rigurosamente los problemas de tus datasets y las decisiones de limpieza que tomas.

Usa una copia de esta plantilla para cada dataset importante (Módulo 01, Módulo 05, proyecto MNIST si lo deseas).

---

## 📌 Resumen del Dataset

- **Nombre del dataset:**
- **Fuente (URL / archivo local):**
- **Tamaño (filas, columnas):**
- **Objetivo del modelo (regresión / clasificación / otro):**

---

## 🔍 Perfilado Inicial

Completa esta sección con Pandas (Módulo 01) antes de empezar a limpiar.

- `df.info()` – tipos de datos, nulos.
- `df.describe()` – estadísticas básicas.
- Conteo de valores únicos por columna clave.

**Notas rápidas:**
- Columnas con muchos nulos:
- Columnas con valores raros (e.g., "?", "N/A", "-999"):
- Sospechas de outliers:

---

## 🧯 Problemas Detectados y Decisiones

### Estructura para cada problema

```markdown
### Problema #N – [Título breve]

**Columna(s) afectada(s):**
**Tipo de problema:** [nulos | outliers | tipo incorrecto | codificación | duplicados | otro]

**Evidencia:**
- [Ejemplo de salida de Pandas que muestra el problema]

**Opciones consideradas:**
- [Opción A] (pros / contras)
- [Opción B] (pros / contras)

**Decisión final:**
- [Qué hiciste y por qué]

**Impacto esperado en el modelo:**
- [Cómo crees que afecta a bias/variance, estabilidad, etc.]
```

---

## 🧪 Caso 1 – Módulo 01 (CSV Inicial)

> **Objetivo:** Mostrar que puedes hacer un análisis serio de calidad de datos con Pandas.

- Dataset utilizado (nombre / descripción):
- Mínimo **5 problemas documentados** usando la estructura anterior.

Checklist:
- [ ] Identifiqué y documenté ≥ 5 problemas.
- [ ] Justifiqué cada decisión de limpieza.
- [ ] Puedo explicar a otra persona por qué estas decisiones son razonables.

---

## 🧪 Caso 2 – Módulo 05 (Dataset Supervisado Real)

> **Objetivo:** Practicar un pipeline de preprocesamiento realista antes de Regresión Logística.

Requisitos del dataset:
- Al menos **1–2 columnas categóricas** → requiere **One-Hot Encoding**.
- Al menos **2–3 columnas numéricas** → requiere **escalado** (MinMax / StandardScaler manual).

Elementos obligatorios:
- [ ] Limpieza de nulos y valores raros.
- [ ] Diseño de features (crear, combinar o transformar columnas si es útil).
- [ ] One-Hot Encoding implementado a mano (sin sklearn).
- [ ] Escalado manual de features numéricos.
- [ ] División train/test definida después del preprocesamiento.

Documenta **al menos 5 decisiones clave** usando la estructura de problemas anterior.

---

## 📊 Resumen de Decisiones Clave

| # | Columna / Problema | Decisión | Justificación corta |
|---|--------------------|----------|---------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

---

## 🧠 Reflexión

- ¿Qué aprendiste sobre la **realidad de los datos** respecto a los ejemplos sintéticos?
- ¿Qué habrías hecho distinto si tuvieras más tiempo o recursos?
- ¿Cómo impacta la calidad del dato en la interpretación de tus resultados de ML?
