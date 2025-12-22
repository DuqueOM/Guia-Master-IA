# Módulo 5.5: Ética en IA e Interpretabilidad (XAI)

> **Semana:** 11 | **Curso Alineado:** CSCA 5622 - Supervised Learning
> **Prerequisitos:** Regresión Logística, Árboles de Decisión, Random Forest

---

## 1. ¿Por Qué Importa la Ética en ML?

### 1.1 La Falacia de la Objetividad Algorítmica

**Mito común:** "Los algoritmos son objetivos porque son matemáticos."

**Realidad:** Los algoritmos de ML aprenden de datos históricos que reflejan decisiones humanas pasadas, incluyendo sus sesgos. Un modelo entrenado con datos sesgados **amplificará** esos sesgos.

### 1.2 Casos Reales de Sesgo en ML

| Caso | Año | Problema | Impacto |
|------|-----|----------|---------|
| **Amazon Recruiting** | 2018 | Modelo de CV penalizaba "women's" | Discriminación de género |
| **COMPAS** | 2016 | Predicción de reincidencia | Falsos positivos 2x para afroamericanos |
| **Google Photos** | 2015 | Clasificación de imágenes | Etiquetó personas negras como "gorilas" |
| **Apple Card** | 2019 | Límites de crédito | Límites 10-20x menores para mujeres |

---

## 2. Sesgo Algorítmico: Taxonomía y Fuentes

### 2.1 Tipos de Sesgo en el Pipeline de ML

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **Historical Bias** | Datos reflejan inequidades pasadas | Menos mujeres en liderazgo historicamente |
| **Representation Bias** | Subgrupos sub-representados | Dataset de caras con 80% blancos |
| **Measurement Bias** | Proxy variables | Código postal como proxy de raza |
| **Aggregation Bias** | Un modelo para grupos heterogéneos | Diagnóstico médico igual para todas las edades |
| **Evaluation Bias** | Métricas no representativas | Evaluar solo en grupo mayoritario |

### 2.2 El Pipeline de Sesgo

```
Mundo Real → Recolección → Preprocesamiento → Entrenamiento → Evaluación → Deployment
    ↓            ↓              ↓                 ↓              ↓            ↓
 Historical   Sampling      Measurement      Aggregation     Evaluation   Feedback
   Bias        Bias           Bias             Bias           Bias         Loop
```

---

## 3. El Caso COMPAS: Un Estudio de Caso

### 3.1 Contexto

**COMPAS** (Correctional Offender Management Profiling for Alternative Sanctions) es un algoritmo propietario usado en el sistema judicial de EE.UU. para predecir la probabilidad de reincidencia criminal.

### 3.2 El Análisis de ProPublica (2016)

ProPublica analizó 7,000 casos en el Condado de Broward, Florida:

```python
# Resultados del análisis (simplificado)
resultados = {
    'afroamericanos': {
        'falsos_positivos': 44.9,  # Predichos como alto riesgo, no reincidieron
        'falsos_negativos': 28.0   # Predichos como bajo riesgo, sí reincidieron
    },
    'blancos': {
        'falsos_positivos': 23.5,
        'falsos_negativos': 47.7
    }
}
# El algoritmo era 2x más probable de etiquetar incorrectamente a afroamericanos como alto riesgo
```

### 3.3 La Paradoja de Fairness

**Descubrimiento crítico:** Es matemáticamente imposible satisfacer todas las definiciones de fairness simultáneamente cuando las tasas base difieren entre grupos.

| Métrica de Fairness | Definición | COMPAS |
|---------------------|------------|--------|
| **Calibration** | P(Y=1 dado score=s) igual entre grupos | ✅ Cumplía |
| **Equal FPR** | Tasa de falsos positivos igual | ❌ No cumplía |
| **Equal FNR** | Tasa de falsos negativos igual | ❌ No cumplía |

### 3.4 Lección Fundamental

> "Fairness is not a purely technical problem. It requires understanding the social context
> and making explicit value judgments about which trade-offs are acceptable."
> — Arvind Narayanan, Princeton

---

## 4. Interpretabilidad vs Explicabilidad

### 4.1 Definiciones

| Concepto | Definición | Audiencia |
|----------|------------|-----------|
| **Interpretabilidad** | El modelo es inherentemente comprensible | Data Scientists |
| **Explicabilidad** | Podemos generar explicaciones post-hoc | Usuarios finales, reguladores |
| **Transparencia** | El proceso completo es auditable | Auditores, sociedad |

### 4.2 El Trade-off Interpretabilidad vs Precisión

```
Alta Interpretabilidad                        Alta Precisión
        │                                           │
        ▼                                           ▼
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│  Linear   │  │  Decision │  │  Random   │  │   Deep    │
│Regression │→ │   Tree    │→ │  Forest   │→ │  Learning │
└───────────┘  └───────────┘  └───────────┘  └───────────┘
   "Glass         "Glass         "Gray           "Black
    Box"           Box"           Box"            Box"
```

### 4.3 ¿Cuándo es Crítica la Interpretabilidad?

| Dominio | Requisito | Razón |
|---------|-----------|-------|
| **Medicina** | Alto | Decisiones de vida/muerte requieren justificación |
| **Finanzas** | Alto | Regulaciones (GDPR Art. 22, ECOA) |
| **Justicia Criminal** | Muy Alto | Libertad individual en juego |
| **Recomendaciones** | Bajo | Impacto limitado de errores individuales |
| **Publicidad** | Bajo | Optimización comercial |

---

## 5. SHAP: SHapley Additive exPlanations

### 5.1 Fundamento Teórico: Valores de Shapley

Los valores de Shapley provienen de **teoría de juegos cooperativos** (Lloyd Shapley, Premio Nobel 2012).

**Analogía del Restaurante:**
Imagina que 3 amigos (A, B, C) abren un restaurante juntos y generan $120,000 de ganancia. ¿Cómo dividir las ganancias de forma "justa"?

```
Coaliciones posibles y sus ganancias:
- {} → $0
- {A} → $50,000  (A solo)
- {B} → $40,000  (B solo)
- {C} → $30,000  (C solo)
- {A,B} → $90,000
- {A,C} → $70,000
- {B,C} → $60,000
- {A,B,C} → $120,000

Valor de Shapley para A:
= Promedio de contribución marginal de A en todas las permutaciones
```

### 5.2 SHAP en Machine Learning

**Idea:** Cada feature es un "jugador" y la predicción es la "ganancia". SHAP calcula la contribución de cada feature a la predicción.

```python
# Ejemplo conceptual
prediccion = modelo.predict(x)  # Ej: $350,000 (precio de casa)
base_value = promedio_predicciones  # Ej: $280,000

# SHAP descompone la diferencia:
# prediccion - base_value = suma(shap_values)
# $350,000 - $280,000 = $70,000

shap_values = {
    'sqft': +$45,000,      # Tamaño aumenta precio
    'location': +$35,000,  # Buena ubicación
    'age': -$10,000,       # Casa vieja reduce precio
    # Suma: $70,000 ✓
}
```

### 5.3 Propiedades Matemáticas de SHAP

SHAP es el **único** método que satisface estas propiedades:

| Propiedad | Significado |
|-----------|-------------|
| **Local Accuracy** | Los SHAP values suman exactamente la predicción |
| **Missingness** | Features ausentes tienen SHAP = 0 |
| **Consistency** | Si una feature contribuye más, su SHAP aumenta |

### 5.4 Implementación Práctica

```python
import shap
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Cargar datos
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar modelo
model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Calcular SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizaciones
shap.summary_plot(shap_values, X_test)  # Importancia global
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])  # Explicación local
shap.dependence_plot("LSTAT", shap_values, X_test)  # Dependencia de una feature
```

### 5.5 Tipos de Gráficos SHAP

| Gráfico | Uso | Interpretación |
|---------|-----|----------------|
| **Summary Plot** | Importancia global | Features ordenadas por impacto promedio |
| **Force Plot** | Explicación individual | Cómo cada feature empuja la predicción |
| **Dependence Plot** | Relación feature-target | Efecto no lineal de una feature |
| **Waterfall Plot** | Descomposición secuencial | Paso a paso de base a predicción |

---

## 6. LIME: Local Interpretable Model-agnostic Explanations

### 6.1 Intuición

**Idea central:** Aunque el modelo global sea complejo (caja negra), localmente puede aproximarse con un modelo simple (interpretable).

```
                    Modelo Global Complejo
                           │
                           │  Zona local
                           ▼  alrededor de x
    ┌──────────────────────●──────────────────────┐
    │                   ╱    ╲                     │
    │                 ╱        ╲                   │
    │    Frontera   ╱   Aprox.   ╲                │
    │    compleja  │    lineal    │               │
    │               ╲            ╱                 │
    │                 ╲        ╱                   │
    │                   ╲    ╱                     │
    └─────────────────────────────────────────────┘

    LIME: "En la vecindad de x, el modelo se comporta
           aproximadamente como una regresión lineal"
```

### 6.2 Algoritmo LIME

```
1. Seleccionar instancia x a explicar
2. Generar N perturbaciones de x (vecinos artificiales)
3. Obtener predicciones del modelo original para cada perturbación
4. Pesar perturbaciones por proximidad a x (kernel)
5. Entrenar modelo interpretable (ej: regresión lineal) en datos pesados
6. Los coeficientes del modelo simple son la "explicación"
```

### 6.3 Implementación

```python
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Cargar datos y entrenar modelo
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Crear explicador LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    X,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode='classification'
)

# Explicar una instancia
idx = 42
exp = explainer.explain_instance(
    X[idx],
    model.predict_proba,
    num_features=4
)

# Visualizar
exp.show_in_notebook()
print(exp.as_list())
# Output: [('petal width (cm) > 1.75', 0.42),
#          ('petal length (cm) > 4.95', 0.31), ...]
```

### 6.4 LIME para Texto e Imágenes

```python
# LIME para texto
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['negative', 'positive'])
exp = explainer.explain_instance(
    "This movie was absolutely terrible and boring",
    classifier.predict_proba,
    num_features=6
)
# Resalta palabras que contribuyen a la predicción

# LIME para imágenes
from lime import lime_image
explainer = lime_image.LimeImageExplainer()
exp = explainer.explain_instance(
    image,
    model.predict,
    top_labels=3,
    num_samples=1000
)
# Resalta regiones de la imagen importantes para la predicción
```

### 6.5 SHAP vs LIME

| Aspecto | SHAP | LIME |
|---------|------|------|
| **Base teórica** | Teoría de juegos (axiomática) | Aproximación local ad-hoc |
| **Consistencia** | Garantizada matemáticamente | No garantizada |
| **Velocidad** | Más lento (exacto) | Más rápido (aproximado) |
| **Explicaciones** | Globales + locales | Solo locales |
| **Uso recomendado** | Análisis detallado, auditoría | Explicaciones rápidas, prototipado |

---

## 7. Métricas de Fairness

### 7.1 Definiciones Formales

Sea:
- `Y` = outcome real (0/1)
- `Ŷ` = predicción del modelo (0/1)
- `A` = atributo protegido (ej: raza, género)

### 7.2 Principales Métricas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **Demographic Parity** | P(Ŷ=1\|A=0) = P(Ŷ=1\|A=1) | Igual tasa de predicciones positivas |
| **Equalized Odds** | P(Ŷ=1\|Y=y,A=0) = P(Ŷ=1\|Y=y,A=1) ∀y | Igual TPR y FPR entre grupos |
| **Equal Opportunity** | P(Ŷ=1\|Y=1,A=0) = P(Ŷ=1\|Y=1,A=1) | Igual TPR (beneficio igualitario) |
| **Calibration** | P(Y=1\|Ŷ=s,A=0) = P(Y=1\|Ŷ=s,A=1) | Scores significan lo mismo para todos |

### 7.3 Implementación con Fairlearn

```python
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Calcular métricas por grupo
metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test  # Atributo protegido
)

print(metric_frame.by_group)
print(f"Demographic Parity Difference: {demographic_parity_difference(y_test, y_pred, sensitive_features=A_test)}")
```

### 7.4 Teorema de Imposibilidad de Fairness

**Teorema (Chouldechova, 2017; Kleinberg et al., 2016):**

> Cuando las tasas base difieren entre grupos (P(Y=1|A=0) ≠ P(Y=1|A=1)),
> es imposible satisfacer simultáneamente:
> 1. Calibration
> 2. Equal FPR
> 3. Equal FNR

**Implicación práctica:** Debemos elegir qué tipo de fairness priorizar según el contexto.

---

## 8. Ejercicios Prácticos

### Ejercicio 1: Análisis SHAP de Modelo de Crédito

```python
"""
Dataset: German Credit (UCI)
Tarea: Entrenar modelo de aprobación de crédito y analizar con SHAP

1. Cargar el dataset German Credit
2. Entrenar un Random Forest
3. Calcular SHAP values
4. Identificar las 5 features más importantes
5. Analizar si alguna feature es proxy de atributos protegidos
"""

# Tu código aquí
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
# ... continuar implementación
```

### Ejercicio 2: Auditoría de Fairness

```python
"""
Dataset: Adult Income (UCI)
Tarea: Evaluar fairness de un modelo de predicción de ingresos

1. Entrenar modelo para predecir income > 50K
2. Calcular métricas de fairness por género y raza
3. Identificar disparidades
4. Proponer estrategias de mitigación
"""

# Tu código aquí
```

### Ejercicio 3: LIME para Explicar Predicciones

```python
"""
Tarea: Usar LIME para explicar predicciones incorrectas

1. Entrenar modelo en dataset de tu elección
2. Identificar 5 predicciones incorrectas
3. Generar explicaciones LIME para cada una
4. Analizar: ¿Las explicaciones revelan problemas del modelo?
"""

# Tu código aquí
```

---

## 9. Checklist de Ética para ML

### 9.1 Fase de Diseño

- [ ] ¿El problema requiere realmente ML?
- [ ] ¿Quiénes son los stakeholders afectados?
- [ ] ¿Existen grupos vulnerables que podrían ser perjudicados?
- [ ] ¿Qué definición de fairness es apropiada para este contexto?

### 9.2 Fase de Datos

- [ ] ¿Los datos son representativos de la población objetivo?
- [ ] ¿Existen proxy variables de atributos protegidos?
- [ ] ¿Se ha documentado la procedencia de los datos?
- [ ] ¿Se han identificado posibles sesgos históricos?

### 9.3 Fase de Modelado

- [ ] ¿Se han evaluado múltiples métricas de fairness?
- [ ] ¿El modelo es suficientemente interpretable para el dominio?
- [ ] ¿Se pueden generar explicaciones para decisiones individuales?
- [ ] ¿Se ha realizado análisis de subgrupos?

### 9.4 Fase de Deployment

- [ ] ¿Existe un proceso de apelación para decisiones automatizadas?
- [ ] ¿Se monitoreará el modelo por drift de fairness?
- [ ] ¿Los usuarios entienden las limitaciones del modelo?
- [ ] ¿Hay un plan para auditorías periódicas?

---

## 10. Lecturas Obligatorias

1. **"Machine Bias" (ProPublica, 2016)** - https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing

2. **"A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)** - Paper original de SHAP

3. **"Why Should I Trust You?" (Ribeiro et al., 2016)** - Paper original de LIME

4. **"Fairness and Machine Learning" (Barocas, Hardt, Narayanan)** - Libro completo gratuito online

5. **"The Ethical Algorithm" (Kearns & Roth, 2019)** - Accesible para no-técnicos

---

## 11. Resumen

| Concepto | Punto Clave |
|----------|-------------|
| **Sesgo algorítmico** | Los modelos amplifican sesgos de los datos |
| **SHAP** | Explicaciones con garantías teóricas (Shapley) |
| **LIME** | Aproximación local interpretable |
| **Fairness** | Múltiples definiciones, imposible satisfacer todas |
| **Práctica** | Checklist + monitoreo continuo |

> **Reflexión final:** La ética en ML no es un problema técnico que se "resuelve" una vez.
> Es un proceso continuo de reflexión, evaluación y mejora que requiere
> la colaboración de técnicos, expertos de dominio y comunidades afectadas.

---

*Material desarrollado para el MS-AI Pathway - University of Colorado Boulder*
*Semana 11 - CSCA 5622: Supervised Learning*
