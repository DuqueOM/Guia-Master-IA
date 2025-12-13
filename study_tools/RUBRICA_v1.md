# 📏 Rúbrica de Evaluación (v1.0)

> Objetivo: evaluar entregables y simulacros con criterios consistentes, para calibrar “qué tan listo estoy” y detectar brechas temprano.

---

## 👥 Roles de evaluación

- **Autoevaluador (estudiante):** scoring semanal rápido + scoring completo en checkpoints.
- **Revisor IA/pareja:** revisión de código + scoring preliminar (ver `prompts/AI_CODE_REVIEWER.md`).
- **Mentor externo (si disponible):** validación en checkpoints grandes (ideal: PB-16 y PB-23).

---

## 🧮 Estructura general (100 puntos)

Puntaje base: **95 pts** + **5 pts bonus** (opcional) = **100**.

- **A. Dominio Teórico** — 25 pts
- **B. Implementación & Calidad de Código** — 25 pts
- **C. Evaluación en Simulacros (PB)** — 20 pts
- **D. Proyecto & Documentación Científica** — 15 pts
- **E. Prácticas Metacognitivas y Proceso** — 10 pts
- **Bonus (badges/challenges)** — 5 pts

---

## 📌 Niveles por criterio

Cada criterio se evalúa con 4 niveles. Regla de conversión sugerida:

- **Exceeds**: 100% del subpeso
- **Meets**: 75% del subpeso
- **Approaching**: 50% del subpeso
- **Not met**: 0–25% del subpeso

**Evidencia requerida:** archivos, tests (`pytest`), checks de calidad (`pre-commit`), notebooks, docs, y entregables de proceso (p.ej. `DIRTY_DATA_CHECK`).

---

## 🚫 Condiciones duras (no negociables)

- **PB-23 ≥ 80/100**: requisito para marcar estado **“Listo para admisión”**.
- **Entregables de código**: tests unitarios pasan + `pre-commit` pasa.
- **Dirty Data Check**: obligatorio en **Módulo 01** (Caso 1) y **Módulo 05** (Caso 2).

---

## 🗓️ Cuándo aplicar la rúbrica (cronograma)

- **Semana 0 (preparación):** crea y calibra rúbrica con 1 entregable pequeño.
- **Semanas 1–8:** scoring rápido semanal (en el cierre) + scoring de **PB-8**.
- **Semanas 9–20:** scoring completo al cierre de módulos (**Semanas 12, 16, 20**) + scoring de **PB-16**.
- **Semanas 21–24:** scoring del proyecto MNIST + scoring de PB-23 (examen simulado).

---

## 🧾 Plantilla de scoring rápido (cierre semanal)

Usa esto durante `study_tools/CIERRE_SEMANAL.md`.

```text
SEMANA: __
MÓDULO: __

A (Teoría) __/25
B (Código) __/25
C (Simulacros) __/20
D (Proyecto) __/15
E (Proceso) __/10
BONUS __/5

TOTAL: __/100
ESTADO: [Aún no listo | En progreso | Listo]  (si PB-23 >=80)
TOP 3 BRECHAS: 1) __ 2) __ 3) __
```

---

## 🧩 Ejemplo granular — Módulo 05 (Supervised Learning)

Peso sugerido dentro de una evaluación de módulo: **12 pts** (repartidos en A/B/C/E).

### A1. Derivación matemática (MSE / logística) — 4 pts

- **Exceeds (4):** deriva MSE y cross-entropy paso a paso, explica supuestos, responde preguntas de seguimiento.
- **Meets (3):** derivación correcta con 1 error menor de notación.
- **Approaching (2):** conceptos entendidos, faltan pasos o error no crítico.
- **Not met (0–1):** no puede derivar o hay errores conceptuales.

### B1. Implementación NumPy sin sklearn (logistic_regression.py) — 4 pts

- **Exceeds (4):** type hints, tests (edge cases), vectorizado, `pre-commit` ok.
- **Meets (3):** implementación correcta con faltantes menores.
- **Approaching (2):** funciona en casos simples pero falla en corner cases / shapes.
- **Not met (0–1):** no funciona o usa sklearn.

### C1. Validación / Metrics / CV — 2 pts

- **Exceeds (2):** K-fold CV + learning curves + interpretación de bias-variance.
- **Meets (1.5):** CV correcto + documentación breve.
- **Approaching (1):** solo train/test.
- **Not met (0):** sin evaluación.

### E1. Dirty Data Check aplicado — 2 pts

- Evidencia: `study_tools/DIRTY_DATA_CHECK.md` (Caso 2).

---

## 🧠 Ejemplo granular — Proyecto Final “MNIST Analyst”

Peso sugerido del proyecto: **25 pts** (agregando D/B/C).

### D1. Pipeline end-to-end reproducible — 8 pts

- **Exceeds (8):** pipeline reproducible + scripts + README en inglés + tests.
- **Meets (6):** pipeline y notebooks legibles + README mínimo.
- **Approaching (4):** pipeline incompleto o pasos manuales.
- **Not met (0–2):** no reproducible.

### B2. Calidad del código — 6 pts

- Evidencia: type hints, docstrings, `mypy`, tests, `pre-commit`.

### C2. Resultados / Métricas — 6 pts

- **Exceeds (6):** MLP > 92% y Logistic > 87% + análisis.
- **Meets (5):** MLP ≥ 90%, Logistic ≥ 85%.
- **Approaching (3):** MLP 85–90%, Logistic 80–85%.
- **Not met (0):** no alcanzadas.

### E2. Informe MODEL_COMPARISON.md — 5 pts

- **Exceeds (5):** análisis profundo + error analysis + gráficos + conclusiones.
- **Meets (4):** comparativa correcta y conclusiones.
- **Approaching (2):** superficial.
- **Not met (0):** faltante.

---

## 🧷 Regla de admisión (PB-23)

- Si **PB-23 < 80**, el estado es **“Aún no listo”** aunque el total global sea alto.
- Evidencia: scoring registrado (p.ej. en un reporte generado desde `rubrica.csv`).
