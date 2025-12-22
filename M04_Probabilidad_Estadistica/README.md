# Módulo 04: Probabilidad y Estadística para Machine Learning

> **Semana:** 8 | **Fase:** Fundamentos Matemáticos
> **Curso Alineado:** Preparación para CSCA 5622, 5632, 5642
> **Carga Estimada:** 10-12 horas

---

## 🎯 Objetivos de Aprendizaje

Al completar este módulo, el estudiante será capaz de:

1. **Fundamentos Probabilísticos**
   - Aplicar regla de Bayes en problemas de clasificación
   - Distinguir entre probabilidad frecuentista y bayesiana
   - Calcular probabilidades condicionales e independencia

2. **Estimación Estadística**
   - Derivar estimadores MLE (Maximum Likelihood Estimation)
   - Derivar estimadores MAP (Maximum A Posteriori)
   - Comparar sesgo, varianza y MSE de estimadores

3. **Cadenas de Markov**
   - Construir matrices de transición
   - Calcular distribuciones estacionarias
   - Analizar propiedades de ergodicidad y mixing time

4. **Métodos de Monte Carlo**
   - Implementar Monte Carlo simple para integración
   - Implementar Metropolis-Hastings y Gibbs Sampling
   - Diagnosticar convergencia (R-hat, ESS, trace plots)

---

## 📅 Syllabus Detallado (Semana 8)

### Día 1-2: Fundamentos de Probabilidad y Bayes

| Tema | Contenido | Ejercicio |
|------|-----------|-----------|
| Axiomas de probabilidad | Espacios muestrales, eventos, sigma-álgebras | Ejercicio 1.1 |
| Probabilidad condicional | P(A\|B), independencia, Bayes | Ejercicio 1.2 |
| Regla de Bayes | Prior, likelihood, posterior, evidencia | Ejercicio 1.3 |
| Variables aleatorias | Discretas vs continuas, PMF/PDF, CDF | Ejercicio 1.4 |

**Lectura obligatoria:** Murphy Cap. 2.1-2.4, Bishop Cap. 1.2

### Día 3-4: Distribuciones y Estimación (MLE/MAP)

| Tema | Contenido | Ejercicio |
|------|-----------|-----------|
| Distribuciones comunes | Bernoulli, Binomial, Gaussiana, Poisson | Ejercicio 2.1 |
| Esperanza y varianza | E[X], Var(X), propiedades | Ejercicio 2.2 |
| **MLE** | Derivación log-likelihood, ejemplos analíticos | **Lab 1** |
| **MAP** | Priors conjugados, regularización bayesiana | **Lab 1** |
| Sesgo-Varianza | Trade-off, MSE = Bias² + Variance | Ejercicio 2.3 |

**Lectura obligatoria:** Murphy Cap. 3.1-3.5, Bishop Cap. 2.1-2.3

### Día 5: Cadenas de Markov (Discrete-Time)

| Tema | Contenido | Ejercicio |
|------|-----------|-----------|
| Definición DTMC | Estados, transiciones, matriz P | Ejercicio 3.1 |
| Propiedades | Irreducibilidad, aperiodicidad, recurrencia | Ejercicio 3.2 |
| Distribución estacionaria | π = πP, existencia y unicidad | **Lab 3** |
| Teorema ergódico | Convergencia, mixing time | **Lab 3** |
| Aplicaciones ML | PageRank, HMM preview | Ejercicio 3.3 |

**Lectura obligatoria:** Levin & Peres Cap. 1-2, Murphy Cap. 17.2

### Día 6-7: Monte Carlo y MCMC

| Tema | Contenido | Ejercicio |
|------|-----------|-----------|
| Monte Carlo simple | Integración, estimación de π | Ejercicio 4.1 |
| Importance Sampling | Reducción de varianza | Ejercicio 4.2 |
| **Metropolis-Hastings** | Algoritmo, acceptance ratio, proposal | **Lab 2** |
| **Gibbs Sampling** | Caso especial, conditional sampling | **Lab 2** |
| Diagnósticos | Burn-in, thinning, R-hat, ESS, trace plots | **Lab 2** |

**Lectura obligatoria:** Murphy Cap. 24.1-24.3, Bishop Cap. 11.2-11.3

---

## 🧪 Laboratorios Obligatorios

### Lab 1: MLE/MAP y Estimadores (`Notebooks/Lab1_MLE_MAP.py`)

**Objetivos:**
- Derivar MLE para Bernoulli, Gaussiana, Poisson
- Implementar MLE numéricamente con scipy.optimize
- Comparar MLE vs MAP con diferentes priors
- Visualizar efecto del tamaño de muestra

**Entregables:**
- [ ] Derivación analítica de MLE para Gaussiana (μ, σ²)
- [ ] Implementación de MLE numérico
- [ ] Comparación MLE vs MAP con prior Beta-Binomial
- [ ] Gráfico: sesgo vs varianza vs n

### Lab 2: Monte Carlo y MCMC (`Notebooks/Lab2_MonteCarlo_MCMC.py`)

**Objetivos:**
- Estimar π usando Monte Carlo
- Implementar Metropolis-Hastings desde cero
- Implementar Gibbs Sampling para Gaussiana bivariada
- Diagnosticar convergencia con trace plots y R-hat

**Entregables:**
- [ ] Estimación de π con intervalos de confianza
- [ ] Muestreo de distribución objetivo con M-H
- [ ] Gibbs Sampler para mixture de Gaussianas
- [ ] Análisis de convergencia (burn-in, ESS)

### Lab 3: Cadenas de Markov (`Notebooks/Lab3_MarkovChains.py`)

**Objetivos:**
- Construir matriz de transición desde datos
- Calcular distribución estacionaria analítica y numéricamente
- Simular random walks y verificar convergencia
- Estimar mixing time empíricamente

**Entregables:**
- [ ] Matriz de transición para problema de clima
- [ ] Cálculo de eigenvector para π
- [ ] Simulación de 10,000 pasos y histograma
- [ ] Comparación: teórico vs empírico

---

## 📊 Datasets Recomendados

| Dataset | Uso | Fuente |
|---------|-----|--------|
| Iris | Estimación de parámetros Gaussianos | sklearn.datasets |
| Weather transitions | Cadenas de Markov | Sintético |
| Beta-Binomial | MLE vs MAP | Sintético |
| 2D Gaussian Mixture | MCMC sampling | Sintético |

---

## ✅ Checklist de Autoevaluación

### Teoría (antes de avanzar a M05)

- [ ] Puedo derivar la regla de Bayes y explicar cada término
- [ ] Puedo derivar MLE para distribución Gaussiana
- [ ] Entiendo la diferencia entre MLE y MAP
- [ ] Puedo construir una matriz de transición de Markov
- [ ] Entiendo qué significa "estacionaria" y "ergódica"
- [ ] Puedo explicar por qué funciona Metropolis-Hastings

### Práctica (Labs completados)

- [ ] Lab 1: MLE/MAP implementado y validado
- [ ] Lab 2: MCMC funcionando con diagnósticos
- [ ] Lab 3: Cadena de Markov simulada correctamente

### Ejercicios Tipo Examen

- [ ] Ejercicio 1.1-1.4 completados
- [ ] Ejercicio 2.1-2.3 completados
- [ ] Ejercicio 3.1-3.3 completados
- [ ] Ejercicio 4.1-4.2 completados

---

## 📁 Estructura del Módulo

```
M04_Probabilidad_Estadistica/
├── README.md                          # Este archivo (syllabus)
├── Teoria/
│   ├── 04_PROBABILIDAD_ML.md          # Fundamentos teóricos
│   └── markov_montecarlo.md           # DTMC + MCMC detallado
├── Notebooks/
│   ├── 01_distribuciones_mle.py       # Introducción
│   ├── Lab1_MLE_MAP.py                # Lab obligatorio 1
│   ├── Lab2_MonteCarlo_MCMC.py        # Lab obligatorio 2
│   └── Lab3_MarkovChains.py           # Lab obligatorio 3
├── Laboratorios_Interactivos/
│   └── gmm_3_gaussians_contours.py    # Visualización GMM
├── tests/
│   └── test_m04_labs.py               # Tests automáticos
└── assets/
```

---

## 📚 Referencias Obligatorias

| Recurso | Capítulos | Prioridad |
|---------|-----------|-----------|
| **Murphy - ML: A Probabilistic Perspective** | Cap. 2, 3, 17, 24 | ⭐⭐⭐ |
| **Bishop - Pattern Recognition and ML** | Cap. 1.2, 2, 11 | ⭐⭐⭐ |
| **Levin & Peres - Markov Chains and Mixing Times** | Cap. 1-4 | ⭐⭐ |
| **Goodfellow - Deep Learning** | Cap. 3 (Probability) | ⭐⭐ |

Ver [REFERENCES.md](../REFERENCES.md) para lista completa con enlaces.

---

## 🔗 Conexiones con Otros Módulos

| Módulo | Concepto de M04 | Aplicación |
|--------|-----------------|------------|
| **M05** | MLE/MAP | Regresión logística, regularización |
| **M06** | EM Algorithm | GMM, clustering probabilístico |
| **M06** | Markov Chains | Sistemas de recomendación |
| **M07** | Bayes | Dropout como aproximación bayesiana |
| **M07** | Monte Carlo | Dropout, variational inference |
| **M08** | Probabilidad | Naive Bayes, language models |

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [← M03: Cálculo](../M03_Calculo_Optimizacion/) | [README Principal](../README.md) | [M05: Supervisado →](../M05_Aprendizaje_Supervisado/) |
