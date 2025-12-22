# RUBRIC.md — Rúbrica de Evaluación Cuantitativa

> **Objetivo**: Criterios claros y medibles para evaluar el progreso del estudiante en el MS in AI pathway (CU Boulder/Coursera).

---

## 📊 Escala de Calificación

| Nota | Rango | Descripción |
|------|-------|-------------|
| **A** | 90-100% | Dominio excepcional, listo para investigación/industria |
| **B** | 80-89% | Competencia sólida, cumple requisitos del pathway |
| **C** | 70-79% | Comprensión básica, necesita refuerzo |
| **D** | 60-69% | Insuficiente, requiere repetir módulo |
| **F** | <60% | No demuestra competencia mínima |

---

## 🎯 Métricas por Módulo

### M01: Fundamentos de Python para ML

| Criterio | Peso | A (90%+) | B (80%+) | C (70%+) |
|----------|------|----------|----------|----------|
| **Sintaxis y estructuras** | 20% | Código idiomático, PEP8, type hints | Código funcional, estilo consistente | Código funciona pero estilo pobre |
| **NumPy vectorizado** | 30% | Sin loops explícitos, broadcasting correcto | Usa vectorización, algunos loops | Mezcla loops y vectorización |
| **Pandas fluido** | 25% | Operaciones encadenadas, sin SettingWithCopy | Manipulación correcta, warnings menores | Funcional pero ineficiente |
| **Visualización** | 15% | Gráficos publicables, etiquetas completas | Gráficos claros, algún detalle falta | Gráficos básicos legibles |
| **Testing** | 10% | pytest con >80% coverage, parametrizado | Tests unitarios básicos | Algunos tests manuales |

**Proyecto integrador M01**: Script ETL que procesa dataset real (>10k filas) en <5 segundos.

---

### M02: Álgebra Lineal

| Criterio | Peso | A (90%+) | B (80%+) | C (70%+) |
|----------|------|----------|----------|----------|
| **Operaciones matriciales** | 25% | Implementa desde cero + usa NumPy | Usa NumPy correctamente | Confunde dimensiones ocasionalmente |
| **Descomposiciones** | 25% | SVD, eigen, Cholesky aplicados a ML | Calcula descomposiciones, interpreta | Calcula pero no interpreta |
| **Espacios vectoriales** | 20% | Demuestra rango, null space, proyecciones | Conceptos claros, aplicación parcial | Definiciones correctas |
| **Aplicaciones ML** | 20% | PCA desde cero, regularización L2 derivada | Usa PCA, entiende regularización | Aplica sin entender derivación |
| **Eficiencia numérica** | 10% | Evita inversas explícitas, usa solve() | Código correcto, no óptimo | Funciona pero lento |

**Proyecto integrador M02**: Implementar PCA desde cero y comparar con sklearn (error < 1e-10).

---

### M03: Cálculo y Optimización

| Criterio | Peso | A (90%+) | B (80%+) | C (70%+) |
|----------|------|----------|----------|----------|
| **Gradientes analíticos** | 25% | Deriva funciones complejas, verifica numéricamente | Gradientes correctos para funciones estándar | Errores ocasionales en cadena |
| **Gradient Descent** | 30% | Implementa SGD, Adam, momentum desde cero | GD básico converge, tuning manual | GD funciona con hiperparámetros dados |
| **Backpropagation** | 25% | Implementa backprop para MLP arbitrario | Backprop para red de 2 capas | Entiende concepto, no implementa |
| **Diagnóstico** | 20% | Learning curves, gradient checking, early stopping | Monitorea loss, detecta problemas | Ejecuta sin diagnóstico |

**Proyecto integrador M03**: Red neuronal de 3 capas entrenada con backprop manual, accuracy >85% en MNIST subset.

---

### M04: Probabilidad y Estadística

| Criterio | Peso | A (90%+) | B (80%+) | C (70%+) |
|----------|------|----------|----------|----------|
| **Distribuciones** | 20% | Deriva MLE/MAP, entiende conjugados | Aplica MLE, interpreta parámetros | Usa distribuciones correctamente |
| **Cadenas de Markov** | 25% | Prueba convergencia, calcula mixing time | Simula cadenas, encuentra estacionaria | Entiende transiciones |
| **MCMC** | 30% | Metropolis-Hastings + Gibbs desde cero, R-hat < 1.1 | Implementa M-H, diagnostica convergencia | Usa MCMC de librería |
| **Inferencia Bayesiana** | 25% | Posterior analítico + aproximación MCMC | Calcula posteriors conjugados | Entiende Bayes, no calcula |

**Proyecto integrador M04**: Modelo jerárquico bayesiano con MCMC, ESS > 1000, R-hat < 1.05.

---

### M05: Aprendizaje Supervisado (CSCA 5622)

| Criterio | Peso | A (90%+) | B (80%+) | C (70%+) |
|----------|------|----------|----------|----------|
| **Regresión** | 20% | Ridge/Lasso desde cero, cross-validation | Usa sklearn, interpreta coeficientes | Aplica regresión lineal |
| **Clasificación** | 25% | SVM dual, kernels, logística multinomial | Logística + SVM con tuning | Clasifica con defaults |
| **Árboles/Ensembles** | 25% | Implementa RF/XGBoost, feature importance | Usa ensembles, tuning básico | Random Forest out-of-box |
| **Evaluación** | 20% | ROC-AUC, calibración, fairness metrics | Precision/Recall, F1, matriz confusión | Accuracy solamente |
| **Pipeline completo** | 10% | sklearn Pipeline reproducible, MLflow | Pipeline funcional | Scripts separados |

**Proyecto integrador M05**: Competencia Kaggle con F1 > 0.85 en clasificación multiclase.

---

### M06: Aprendizaje No Supervisado (CSCA 5632)

| Criterio | Peso | A (90%+) | B (80%+) | C (70%+) |
|----------|------|----------|----------|----------|
| **Clustering** | 30% | K-means++, DBSCAN, jerárquico + métricas internas | K-means con elbow/silhouette | Aplica clustering |
| **Reducción dimensionalidad** | 30% | PCA, t-SNE, UMAP con interpretación | Usa técnicas, visualiza | PCA básico |
| **Detección anomalías** | 20% | Isolation Forest, LOF, autoencoders | Un método con threshold tuning | Detecta outliers simples |
| **Modelos generativos** | 20% | GMM, VAE básico | GMM con BIC selection | Entiende mezclas |

**Proyecto integrador M06**: Sistema de detección de anomalías con precision@k > 0.7.

---

### M07: Deep Learning (CSCA 5642)

| Criterio | Peso | A (90%+) | B (80%+) | C (70%+) |
|----------|------|----------|----------|----------|
| **Fundamentos** | 20% | Backprop manual, inicialización Xavier/He | Entiende arquitecturas, usa frameworks | Entrena redes con tutoriales |
| **CNNs** | 25% | Diseña arquitectura, transfer learning fine-tune | Usa ResNet/VGG preentrenados | CNN básica funciona |
| **RNNs/Transformers** | 25% | Atención desde cero, fine-tune BERT | LSTM para secuencias, usa HuggingFace | RNN simple |
| **Regularización** | 15% | Dropout, batch norm, data augmentation | Aplica técnicas estándar | Overfitting no controlado |
| **MLOps básico** | 15% | Checkpoints, TensorBoard, reproducibilidad | Guarda modelos, logging básico | Entrenamiento ad-hoc |

**Proyecto integrador M07**: Modelo con >92% accuracy en CIFAR-10 o fine-tuned transformer para NLP.

---

### M08: Proyecto Integrador Final

| Criterio | Peso | A (90%+) | B (80%+) | C (70%+) |
|----------|------|----------|----------|----------|
| **Definición problema** | 15% | Problema novel, métricas justificadas | Problema claro, métricas estándar | Problema definido |
| **EDA y preprocesamiento** | 15% | Análisis exhaustivo, pipeline robusto | EDA completo, limpieza adecuada | Exploración básica |
| **Modelado** | 25% | Múltiples modelos, ablation study | Baseline + modelo avanzado | Un modelo funcional |
| **Evaluación rigurosa** | 20% | Test set separado, intervalos confianza | Validación cruzada correcta | Train/test split |
| **Documentación** | 15% | README, docstrings, notebook narrativo | Código comentado, README básico | Código sin documentar |
| **Presentación** | 10% | Demo interactiva, slides profesionales | Presentación clara | Explicación verbal |

---

## 📋 Checklist de Autoevaluación

### Antes de entregar cualquier módulo:

- [ ] Código pasa `ruff check` sin errores
- [ ] Código pasa `mypy --strict` (o con config del proyecto)
- [ ] Tests pasan con `pytest -v`
- [ ] Notebooks ejecutan de principio a fin sin errores
- [ ] README actualizado con instrucciones de ejecución
- [ ] Gráficos tienen títulos, ejes etiquetados, leyendas

### Para obtener B o superior:

- [ ] Type hints en todas las funciones públicas
- [ ] Docstrings con parámetros y retornos documentados
- [ ] Al menos 3 tests por función principal
- [ ] Análisis de resultados con interpretación
- [ ] Comparación con baseline o método alternativo

### Para obtener A:

- [ ] Implementación desde cero de al menos un algoritmo clave
- [ ] Análisis de complejidad temporal/espacial
- [ ] Experimentos de ablación o sensibilidad
- [ ] Código optimizado (profiling si aplica)
- [ ] Contribución original o extensión del material

---

## 🏆 Ejemplos de Trabajo Nivel "B"

### M04 Lab 2 (MCMC) — Ejemplo B:

```python
def metropolis_hastings(log_target, proposal_std, n_samples, x_init, burn_in):
    """
    Implementación básica de Metropolis-Hastings.

    - Proposal Gaussiano simétrico ✓
    - Burn-in implementado ✓
    - Retorna samples y acceptance rate ✓
    - Falta: diagnósticos avanzados (ESS, trace plots automáticos)
    """
    samples = np.zeros(n_samples + burn_in)
    samples[0] = x_init
    accepted = 0

    for i in range(1, n_samples + burn_in):
        proposal = samples[i-1] + np.random.normal(0, proposal_std)
        log_alpha = log_target(proposal) - log_target(samples[i-1])

        if np.log(np.random.random()) < log_alpha:
            samples[i] = proposal
            accepted += 1
        else:
            samples[i] = samples[i-1]

    return samples[burn_in:], accepted / (n_samples + burn_in)
```

**Por qué es B y no A**:
- ✓ Implementación correcta desde cero
- ✓ Burn-in y acceptance rate
- ✗ No incluye ESS automático
- ✗ No incluye Gelman-Rubin para múltiples cadenas
- ✗ No tiene adaptive proposal tuning

---

## 📈 Tracking de Progreso

Usa el siguiente formato en tu `PROGRESS.md` personal:

```markdown
## Semana X — Módulo Y

### Completado
- [x] Lab 1: MLE (2.5h)
- [x] Lab 2: MCMC (3h)
- [ ] Lab 3: Markov Chains (pendiente)

### Autoevaluación
- Distribuciones: B+ (entiendo conjugados, falta derivar posteriors complejos)
- MCMC: B (M-H funciona, R-hat implementado, ESS marginal)
- Markov: C+ (simulo cadenas, no domino mixing time)

### Plan de mejora
1. Revisar Murphy Cap. 24 para posteriors
2. Practicar más ejercicios de ESS
3. Estudiar Levin-Peres para mixing time
```

---

## 🎓 Correspondencia con Coursera

| Módulo Guía | Curso Coursera | Nota mínima requerida |
|-------------|----------------|----------------------|
| M04 | APPA 5002 (Markov & Monte Carlo) | B |
| M05 | CSCA 5622 (Supervised Learning) | B |
| M06 | CSCA 5632 (Unsupervised Learning) | B |
| M07 | CSCA 5642 (Deep Learning) | B |

**Nota**: Para mantener el pathway, se requiere B (80%) en cada curso. Esta rúbrica está calibrada para que un "B" aquí corresponda a un B en Coursera.

---

## 📝 Feedback y Mejora Continua

Después de completar cada módulo, responde en `M08_Proyecto_Integrador/FEEDBACK.md`:

1. ¿Qué concepto fue más difícil? ¿Por qué?
2. ¿Qué recurso te ayudó más? (libro, video, código)
3. ¿Te sientes preparado para el examen de Coursera? (1-5)
4. Sugerencias para mejorar el material

**Meta**: >80% de estudiantes responden "4" o "5" en preparación para examen.
