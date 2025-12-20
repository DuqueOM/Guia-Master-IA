# Módulo 04: Probabilidad Esencial para Machine Learning

> **Semana 8 | Prerequisito para entender Loss Functions y GMM**
> **Filosofía: Solo la probabilidad que necesitas para la Línea 1**

---

## 🎯 Objetivo del Módulo

Dominar los **conceptos mínimos de probabilidad** necesarios para:

1. Entender **Logistic Regression** como modelo probabilístico
2. Comprender **Cross-Entropy Loss** y por qué funciona
3. Prepararte para **Gaussian Mixture Models (GMM)** en Unsupervised
4. Entender **Softmax** como distribución de probabilidad

> ⚠️ **Nota:** Este NO es el curso completo de Probabilidad (Línea 2). Es solo lo esencial para ML.

---

<a id="m04-0"></a>

## 🧭 Cómo usar este módulo (modo 0→100)

**Propósito:** conectar probabilidad con lo que realmente usarás en el Pathway:

- pérdidas (cross-entropy) como *negative log-likelihood*
- clasificación probabilística (logistic/softmax)
- gaussianas como base de modelos generativos (GMM)
- estabilidad numérica (evitar `NaN`)

### Objetivos de aprendizaje (medibles)

Al terminar el módulo podrás:

- **Explicar** `P(A|B)` y el teorema de Bayes con un ejemplo de clasificación.
- **Aplicar** el punto de vista de MLE: “elegir parámetros que hacen los datos más probables”.
- **Derivar** por qué minimizar cross-entropy equivale a maximizar log-likelihood (binaria y multiclase).
- **Implementar** softmax y log-softmax de forma numéricamente estable (log-sum-exp).
- **Diagnosticar** fallos típicos: `log(0)`, overflow/underflow, probabilidades que no suman 1.

### Prerrequisitos

- De `Módulo 01`: NumPy (vectorización, `axis`, broadcasting).
- De `Módulo 03`: Chain Rule y gradiente (para entender el salto a `Módulo 05/07`).

Enlaces rápidos:

- [RECURSOS.md](RECURSOS.md)
- [GLOSARIO: Binary Cross-Entropy](GLOSARIO.md#binary-cross-entropy)
- [GLOSARIO: Softmax](GLOSARIO.md#softmax)
- [GLOSARIO: Chain Rule](GLOSARIO.md#chain-rule)

### Integración con Plan v4/v5

- [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
- [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)
- Registro de errores: `study_tools/DIARIO_ERRORES.md`
- Evaluación (rúbrica): [study_tools/RUBRICA_v1.md](../study_tools/RUBRICA_v1.md) (scope `M04` en `rubrica.csv`; incluye PB-8)

### Recursos (cuándo usarlos)

| Prioridad | Recurso | Cuándo usarlo en este módulo | Para qué |
|----------|---------|------------------------------|----------|
| **Obligatorio** | `study_tools/DIARIO_ERRORES.md` | Cada vez que aparezca `NaN`, `inf`, `log(0)` u overflow/underflow | Registrar el caso y crear un “fix” reproducible |
| **Obligatorio** | [StatQuest - Maximum Likelihood](https://www.youtube.com/watch?v=XepXtl9YKwc) | Antes (o durante) la sección de MLE y cross-entropy | Alinear intuición de “maximizar verosimilitud” |
| **Complementario** | [3Blue1Brown - Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) | Cuando Bayes se sienta “fórmula sin sentido” (día 3-4) | Visualizar prior/likelihood/posterior |
| **Complementario** | [Mathematics for ML (book)](https://mml-book.github.io/) | Al implementar Gaussiana multivariada y covarianza | Refuerzo de notación y derivaciones |
| **Opcional** | [RECURSOS.md](RECURSOS.md) | Al terminar el módulo (para planificar Línea 2 o profundizar) | Elegir rutas de estudio sin romper el foco de Línea 1 |

### Mapa conceptual (qué conecta con qué)

- **MLE → Cross-Entropy:** sustenta Logistic Regression (Módulo 05) y BCE/CCE en Deep Learning (Módulo 07).
- **Gaussiana multivariada:** es el “átomo” de GMM (Módulo 06).
- **Softmax + Log-Sum-Exp:** evita inestabilidad numérica en clasificación multiclase (Módulo 05/07).

### Ritmo semanal recomendado (Semana 8, sin extender)

- **Lunes y Martes (Concepto):** Bayes + MLE como idea central (qué maximizas y respecto a qué variable).
- **Miércoles y Jueves (Implementación):** implementa versiones estables (log-sum-exp, `clip/eps`) y valida con ejemplos pequeños.
- **Viernes (Romper cosas):** provoca `log(0)`, overflow/underflow y documenta el fix (esto se repite en M05/M07).

### Ajuste crítico de profundidad (Semana 8): MLE como “puente obligatorio” a Cross-Entropy

Este módulo es corto por diseño, pero MLE NO es opcional si quieres entender por qué usamos cross-entropy.

- Objetivo mínimo: poder explicar en 5–10 líneas por qué
  - **maximizar likelihood**
  - equivale a **minimizar negative log-likelihood (NLL)**
  - y por qué eso se ve como **cross-entropy** en clasificación.

Prompt sugerido (si usas IA):

- "Explícame Maximum Likelihood Estimation (MLE) y muéstrame cómo la cross-entropy es el negative log-likelihood para una Bernoulli (BCE) y para una distribución categórica (softmax). No te saltes pasos."

---

## 📚 Contenido

### Día 1-2: Fundamentos de Probabilidad

#### 1.1 Probabilidad Básica

```text
P(A) = casos favorables / casos totales

Propiedades:
- 0 ≤ P(A) ≤ 1
- P(Ω) = 1 (espacio muestral)
- P(∅) = 0 (evento imposible)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 1.1: Probabilidad Básica</strong></summary>

#### 1) Metadatos
- **Título:** Probabilidad como “regla de conteo” + axiomas mínimos
- **ID (opcional):** `M04-T01_1`
- **Duración estimada:** 45–90 min
- **Nivel:** Básico
- **Dependencias:** M01 (manejo básico de notación y números)

#### 2) Objetivos
- Calcular `P(A)` en ejemplos discretos simples y verificar que `0 ≤ P(A) ≤ 1`.
- Explicar qué son `Ω`, `∅` y por qué `P(Ω)=1`.

#### 3) Relevancia
- En ML casi todo termina siendo “probabilidad” o “log-probabilidad” (pérdidas como NLL).

#### 4) Mapa conceptual mínimo
- **Espacio muestral (`Ω`)** → posibles resultados.
- **Evento (`A`)** ⊆ `Ω` → subconjunto de resultados.
- **Probabilidad** → número en [0,1] que cuantifica qué tan “frecuente” es el evento.

#### 5) Definiciones esenciales
- `Ω`: conjunto de resultados posibles.
- `A`: evento.
- `P(A)`: probabilidad del evento.

#### 6) Explicación didáctica
- Regla de sanidad: si te da `P(A)>1` o negativa, tu modelado está mal.

#### 7) Ejemplo modelado
- Dado un dado justo: `P(A=“sale par”) = 3/6 = 0.5`.

#### 8) Práctica guiada
- Escribe 3 eventos distintos en un dado (por ejemplo `{1}`, `{1,2,3}`, `{2,4,6}`) y calcula `P`.

#### 9) Práctica independiente
- Baraja estándar: calcula `P(A=“carta roja”)` y `P(B=“corazón”)`.

#### 10) Autoevaluación
- ¿Por qué `P(∅)=0` es consistente con la idea de “casos favorables/casos totales”?

#### 11) Errores comunes
- Confundir “probabilidad” con “conteo” sin normalizar por el total.
- Olvidar definir el espacio muestral antes de calcular probabilidades.

#### 12) Retención
- (día 2) define `Ω`, `A` y escribe las 3 propiedades básicas (rango, `P(Ω)`, `P(∅)`).

#### 13) Diferenciación
- Avanzado: interpreta probabilidad como frecuencia relativa límite (intuición frequentista).

#### 14) Recursos
- StatQuest (intro de probabilidad) / cualquier texto de probabilidad básica.

#### 15) Nota docente
- Exigir siempre: “¿Cuál es `Ω`?” antes de aceptar un `P(A)`.
</details>

#### 1.2 Probabilidad Condicional

```text
P(A|B) = P(A ∩ B) / P(B)

"Probabilidad de A dado que B ocurrió"
```

**Ejemplo en ML:**
- P(spam | contiene "gratis") = ¿Qué tan probable es spam si el email dice "gratis"?

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 1.2: Probabilidad Condicional</strong></summary>

#### 1) Metadatos
- **Título:** Condicionar = restringir el universo a “B ocurrió”
- **ID (opcional):** `M04-T01_2`
- **Duración estimada:** 60–120 min
- **Nivel:** Básico–Intermedio
- **Dependencias:** 1.1

#### 2) Objetivos
- Interpretar `P(A|B)` en lenguaje natural (“probabilidad de A dado B”).
- Usar `P(A|B)=P(A∩B)/P(B)` y reconocer cuándo aplica (si `P(B)>0`).

#### 3) Relevancia
- Clasificación probabilística en ML se formula como `P(clase|datos)`.

#### 4) Mapa conceptual mínimo
- **Intersección** `A∩B`: ambos ocurren.
- **Condición** `|B`: nos quedamos solo con los casos donde B ocurre.

#### 5) Definiciones esenciales
- `P(A∩B)`: probabilidad conjunta.
- `P(A|B)`: probabilidad condicional.

#### 6) Explicación didáctica
- Intuición: al condicionar, el denominador cambia; ya no divides entre “todo”, sino entre “los casos con B”.

#### 7) Ejemplo modelado
- Si en un dataset el 10% son spam, pero si contiene “gratis” el 80% son spam, entonces `P(spam|gratis)=0.8`.

#### 8) Práctica guiada
- Construye una tabla 2×2 (spam/ham vs contiene gratis/no) y calcula `P(spam|gratis)`.

#### 9) Práctica independiente
- Da un ejemplo donde `P(A|B) > P(A)` y explica por qué no es contradictorio.

#### 10) Autoevaluación
- ¿Qué ocurre si `P(B)=0`? ¿Por qué la definición falla?

#### 11) Errores comunes
- Confundir `P(A|B)` con `P(B|A)`.
- Olvidar que `P(A∩B)` no es `P(A)P(B)` a menos que haya independencia.

#### 12) Retención
- (día 2) escribe la fórmula de `P(A|B)` y un ejemplo en una frase.

#### 13) Diferenciación
- Avanzado: conecta con “actualización de creencias” (preview a Bayes).

#### 14) Recursos
- Sección de probabilidad condicional en cualquier material de probabilidad.

#### 15) Nota docente
- Pedir al alumno que primero responda verbalmente (“¿qué significa dado B?”) antes de calcular.
</details>

#### 1.3 Independencia

```text
A y B son independientes si:
P(A ∩ B) = P(A) · P(B)

Equivalente a:
P(A|B) = P(A)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 1.3: Independencia</strong></summary>

#### 1) Metadatos
- **Título:** Independencia: “saber B no cambia A”
- **ID (opcional):** `M04-T01_3`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 1.1, 1.2

#### 2) Objetivos
- Reconocer equivalencias: `P(A∩B)=P(A)P(B)` y `P(A|B)=P(A)`.
- Evaluar con ejemplos si una suposición de independencia es razonable.

#### 3) Relevancia
- Naive Bayes se sostiene sobre una suposición fuerte de independencia condicional.

#### 4) Mapa conceptual mínimo
- **Dependencia**: información sobre B cambia tu probabilidad de A.
- **Independencia**: no cambia.

#### 5) Definiciones esenciales
- A y B independientes si `P(A|B)=P(A)` (cuando `P(B)>0`).

#### 6) Explicación didáctica
- La independencia casi nunca es exacta en datos reales; se usa como aproximación útil.

#### 7) Ejemplo modelado
- En una moneda justa: eventos “sale cara” y “sale cruz” en el mismo tiro no aplican (mutuamente excluyentes), ojo: no es independencia.

#### 8) Práctica guiada
- Da un ejemplo de eventos independientes (dos tiros de moneda) y uno claramente dependiente.

#### 9) Práctica independiente
- Explica por qué “mutuamente excluyente” no implica “independiente”.

#### 10) Autoevaluación
- ¿Qué valor debería tener `P(A∩B)` si A y B son independientes?

#### 11) Errores comunes
- Confundir independencia con exclusión mutua.
- Asumir independencia sin justificar (y luego sorprenderse por resultados malos en Naive Bayes).

#### 12) Retención
- (día 2) memoriza una equivalencia: `P(A∩B)=P(A)P(B)`.

#### 13) Diferenciación
- Avanzado: independencia condicional `P(A,B|C)=P(A|C)P(B|C)` (preview a Naive Bayes).

#### 14) Recursos
- Lecturas de independencia y diagramas de Venn.

#### 15) Nota docente
- Pedir al alumno que traduzca a lenguaje natural: “saber B no me da info sobre A”.
</details>

---

### Día 3-4: Teorema de Bayes (Crítico para ML)

#### 2.1 La Fórmula

```text
            P(B|A) · P(A)
P(A|B) = ─────────────────
               P(B)

Donde:
- P(A|B) = Posterior (lo que queremos calcular)
- P(B|A) = Likelihood (verosimilitud)
- P(A)   = Prior (conocimiento previo)
- P(B)   = Evidence (normalizador)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 2.1: Teorema de Bayes (la fórmula)</strong></summary>

#### 1) Metadatos
- **Título:** Bayes = reordenar condicionales (posterior = likelihood·prior / evidence)
- **ID (opcional):** `M04-T02_1`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 1.2 (condicional), 1.3 (independencia como contraste)

#### 2) Objetivos
- Identificar los 4 términos: posterior, likelihood, prior, evidence.
- Aplicar Bayes en un ejemplo tipo clasificación y explicar qué significa cada término.

#### 3) Relevancia
- Mucho ML supervisado puede verse como inferencia: estimar `P(clase|datos)`.

#### 4) Mapa conceptual mínimo
- **Prior**: lo que creías antes.
- **Likelihood**: qué tan compatibles son los datos con la clase.
- **Posterior**: lo que crees después de ver datos.
- **Evidence**: normalizador para que sume 1.

#### 5) Definiciones esenciales
- `P(A|B) = P(B|A)P(A) / P(B)`.

#### 6) Explicación didáctica
- Para comparar clases, muchas veces basta el numerador `P(datos|clase)P(clase)` (posterior sin normalizar).

#### 7) Ejemplo modelado
- Spam: `P(spam|palabras) ∝ P(palabras|spam)·P(spam)`.

#### 8) Práctica guiada
- Define un prior `P(spam)` y dos likelihoods y calcula qué clase gana (sin normalizar).

#### 9) Práctica independiente
- Crea un ejemplo con una enfermedad rara: prior pequeño, likelihood grande; discute el resultado.

#### 10) Autoevaluación
- ¿Qué rol cumple `P(B)`? ¿Por qué no depende de A?

#### 11) Errores comunes
- Confundir posterior con likelihood.
- Mezclar `P(A|B)` con `P(B|A)`.

#### 12) Retención
- (día 2) escribe de memoria: posterior = likelihood × prior / evidence.

#### 13) Diferenciación
- Avanzado: conecta con Naive Bayes (producto de likelihoods por feature en log).

#### 14) Recursos
- 3Blue1Brown Bayes (visual), StatQuest Bayes (intuición).

#### 15) Nota docente
- Pedir “traducción verbal” de cada término antes de hacer números.
</details>

#### 2.2 Interpretación para ML

```text
              P(datos|clase) · P(clase)
P(clase|datos) = ─────────────────────────
                      P(datos)

Ejemplo: Clasificación de spam
- P(spam|palabras) = P(palabras|spam) · P(spam) / P(palabras)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 2.2: Interpretación de Bayes para ML</strong></summary>

#### 1) Metadatos
- **Título:** Bayes como clasificador: comparar posteriors (a veces sin normalizar)
- **ID (opcional):** `M04-T02_2`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 2.1, 1.2

#### 2) Objetivos
- Reescribir un problema de clasificación como `argmax_c P(c|x)`.
- Explicar por qué `P(x)` puede omitirse al comparar clases (misma evidencia).

#### 3) Relevancia
- Este marco conecta directamente con Logistic Regression/Softmax: “salidas como probabilidades”.

#### 4) Mapa conceptual mínimo
- **Modelo generativo (tipo Bayes/Naive Bayes):** modela `P(x|c)` y `P(c)`.
- **Inferencia:** obtiene `P(c|x)`.

#### 5) Definiciones esenciales
- **Posterior sin normalizar:** `score(c) = P(x|c)·P(c)`.
- **Decisión MAP:** elegir la clase con mayor posterior.

#### 6) Explicación didáctica
- Si solo quieres la clase, no necesitas `P(x)`; si quieres probabilidades calibradas, sí.

#### 7) Ejemplo modelado
- Spam vs ham: compara `P(palabras|spam)P(spam)` contra `P(palabras|ham)P(ham)`.

#### 8) Práctica guiada
- Usa dos priors distintos (spam raro vs frecuente) y observa cómo cambia la decisión.

#### 9) Práctica independiente
- Explica un caso donde el likelihood gana pero el prior lo revierte (o viceversa).

#### 10) Autoevaluación
- ¿Cuándo te importa `P(x)`? (pista: cuando quieres una probabilidad real, no solo ranking)

#### 11) Errores comunes
- Confundir “likelihood” con “posterior”.
- Creer que omitir `P(x)` es “incorrecto” en clasificación (no lo es para argmax).

#### 12) Retención
- (día 2) memoriza: `P(c|x) ∝ P(x|c)P(c)`.

#### 13) Diferenciación
- Avanzado: en vez de multiplicar, usa logs: `log P(x|c) + log P(c)`.

#### 14) Recursos
- StatQuest: Bayes classifier / Naive Bayes.

#### 15) Nota docente
- Pedir al alumno que señale qué término es “modelo” (`P(x|c)`) y cuál es “creencia previa” (`P(c)`).
</details>

#### 2.3 Implementación en Python

```python
import numpy as np  # Importa NumPy para arrays y operaciones numéricas en el demo

def bayes_classifier(x: np.ndarray,  # features del email (placeholder en este ejemplo)
                     likelihood_spam: float,  # P(x|spam): verosimilitud de observar x si es spam
                     likelihood_ham: float,  # P(x|ham): verosimilitud de observar x si es ham
                     prior_spam: float = 0.3) -> str:  # P(spam): prior (creencia previa) de clase spam
    """
    Clasificador Bayesiano simple.

    Args:
        x: Características del email (simplificado)
        likelihood_spam: P(x|spam)
        likelihood_ham: P(x|ham)
        prior_spam: P(spam) - conocimiento previo

    Returns:
        'spam' o 'ham'
    """
    prior_ham = 1 - prior_spam  # prior complementario: P(ham)=1-P(spam)

    # Posterior (sin normalizar, solo comparamos)
    posterior_spam = likelihood_spam * prior_spam  # score proporcional a P(x|spam)P(spam)
    posterior_ham = likelihood_ham * prior_ham  # score proporcional a P(x|ham)P(ham)

    return 'spam' if posterior_spam > posterior_ham else 'ham'  # decide por argmax (sin calcular P(x))


# Ejemplo: Email con palabra "gratis"
# P("gratis"|spam) = 0.8, P("gratis"|ham) = 0.1
result = bayes_classifier(  # ejecuta la regla Bayesiana con priors y likelihoods de ejemplo
    x=None,  # simplificado
    likelihood_spam=0.8,  # probabilidad de observar la señal x si es spam
    likelihood_ham=0.1,  # probabilidad de observar la señal x si es ham
    prior_spam=0.3  # prior: probabilidad a priori de spam
)  # cierra llamada al clasificador
print(f"Clasificación: {result}")  # spam
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 2.3: Implementación de Bayes en Python</strong></summary>

#### 1) Metadatos
- **Título:** De la fórmula a código: calcular scores y decidir
- **ID (opcional):** `M04-T02_3`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 2.1, 2.2

#### 2) Objetivos
- Implementar un clasificador Bayesiano mínimo y explicar cada variable.
- Separar “cálculo de score” de “decisión final” (`argmax`).

#### 3) Relevancia
- Te entrena a convertir fórmulas en implementaciones legibles (habilidad clave para ML desde cero).

#### 4) Mapa conceptual mínimo
- **Inputs:** likelihoods + priors.
- **Procesamiento:** score por clase.
- **Output:** clase ganadora.

#### 5) Definiciones esenciales
- `posterior_spam ∝ likelihood_spam * prior_spam`.

#### 6) Explicación didáctica
- En problemas reales, multiplicar muchas probabilidades causa underflow → usar log-sum (preview).

#### 7) Ejemplo modelado
- El ejemplo usa “posterior sin normalizar” para comparar clases.

#### 8) Práctica guiada
- Extiende el código para que devuelva también el score de ambas clases.

#### 9) Práctica independiente
- Cambia priors y likelihoods y escribe 3 casos donde el resultado cambie.

#### 10) Autoevaluación
- ¿Por qué no aparece `P(datos)` en el código?

#### 11) Errores comunes
- Tratar `x` como si se usara cuando el ejemplo lo deja simplificado.
- Mezclar probabilidades con porcentajes (0.8 vs 80).

#### 12) Retención
- (día 2) escribe una función que compare 2 clases usando `score = likelihood*prior`.

#### 13) Diferenciación
- Avanzado: reescribe el clasificador en log-espacio: `log_score = log_likelihood + log_prior`.

#### 14) Recursos
- Numpy docs: `np.log`, manejo de underflow/overflow.

#### 15) Nota docente
- Pedir que el alumno comente (en voz) qué representa cada parámetro: prior vs likelihood.
</details>

#### 2.4 Naive Bayes (Conexión con Supervised Learning)

```python
def naive_bayes_predict(X: np.ndarray,  # matriz de features discretas (por muestra)
                        class_priors: np.ndarray,  # priors por clase P(c)
                        feature_probs: dict) -> np.ndarray:  # probabilidades por feature P(x_i|c)
    """
    Naive Bayes asume independencia entre features:
    P(x1, x2, ..., xn | clase) = P(x1|clase) · P(x2|clase) · ... · P(xn|clase)

    Esta "ingenuidad" simplifica mucho el cálculo.
    """
    n_samples = X.shape[0]  # número de muestras a clasificar
    n_classes = len(class_priors)  # número de clases posibles

    log_posteriors = np.zeros((n_samples, n_classes))  # matriz para acumular log-scores por clase

    for c in range(n_classes):  # recorre cada clase y computa su score logarítmico
        # Log para evitar underflow con muchas features
        log_prior = np.log(class_priors[c])  # log P(c): prior en espacio log
        log_likelihood = np.sum(np.log(feature_probs[c][X]), axis=1)  # suma de log P(x_i|c) por muestra
        log_posteriors[:, c] = log_prior + log_likelihood  # log posterior no normalizado por muestra

    return np.argmax(log_posteriors, axis=1)  # predice la clase con score máximo por muestra
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 2.4: Naive Bayes</strong></summary>

#### 1) Metadatos
- **Título:** Naive Bayes: independencia condicional para escalar a muchas features
- **ID (opcional):** `M04-T02_4`
- **Duración estimada:** 60–150 min
- **Nivel:** Intermedio
- **Dependencias:** 1.3 (independencia), 2.1 (Bayes)

#### 2) Objetivos
- Explicar la suposición: `P(x1,…,xn|c) = Π_i P(xi|c)`.
- Entender por qué se usa log: `log Π = Σ log` (evitar underflow).

#### 3) Relevancia
- Es un baseline fuerte en texto y problemas discretos; enseña buenas prácticas numéricas.

#### 4) Mapa conceptual mínimo
- **Modelo:** aprende `P(xi|c)` por feature y `P(c)`.
- **Predicción:** suma log-likelihoods + log-prior.

#### 5) Definiciones esenciales
- `log_posterior(c|x) = log P(c) + Σ_i log P(x_i|c)`.

#### 6) Explicación didáctica
- “Naive” no significa inútil: significa *suposición simplificadora* para poder multiplicar muchos términos.

#### 7) Ejemplo modelado
- En texto (bag-of-words): cada palabra aporta un término de log-likelihood.

#### 8) Práctica guiada
- Implementa una versión binaria con 2 clases y 3 features discretas y verifica con un mini dataset.

#### 9) Práctica independiente
- Discute un caso donde la independencia condicional es claramente falsa (features redundantes) y qué esperas que pase.

#### 10) Autoevaluación
- ¿Por qué `np.log` transforma multiplicaciones en sumas y por qué eso ayuda en cómputo?

#### 11) Errores comunes
- No suavizar probabilidades → `log(0)`.
- Confundir `P(x|c)` con `P(c|x)`.

#### 12) Retención
- (día 2) memoriza el score: `log_prior + sum(log_likelihoods)`.

#### 13) Diferenciación
- Avanzado: introduce Laplace smoothing (α) para evitar ceros.

#### 14) Recursos
- StatQuest Naive Bayes; notas de smoothing.

#### 15) Nota docente
- Pedir una demostración de underflow: multiplicar 100 probabilidades ~0.01 y ver que colapsa sin log.
</details>

---

## 🧩 Micro-Capítulo Maestro: Maximum Likelihood Estimation (MLE) — Nivel: Avanzado

### 1) Intuición (la metáfora del detective)

Imagina que eres un detective que llega a una escena del crimen (tus **datos** `X`).

- Tienes una lista de sospechosos (tus **modelos**).
- Cada sospechoso tiene un comportamiento ajustable por perillas (tus **parámetros** `θ`).

MLE pregunta:

> **¿Qué valores de `θ` hacen MÁS PROBABLE que estos datos específicos hayan ocurrido?**

Importante:

- No estamos diciendo “qué parámetro es más probable” (eso sería un enfoque Bayesiano).
- Estamos diciendo “qué parámetro le da la mayor probabilidad a los datos que YA vimos”.

### 2) Formalización (likelihood y log-likelihood)

Sea `X = {x1, x2, ..., xn}` un conjunto de datos i.i.d.

La **likelihood** es:

`L(θ | X) = P(X | θ) = Π_{i=1}^{n} P(x_i | θ)`

Como multiplicar muchos números pequeños causa underflow, usamos log:

`ℓ(θ) = log L(θ|X) = Σ_{i=1}^{n} log P(x_i | θ)`

Como `log` es monótona creciente, maximizar `L` y maximizar `ℓ` es equivalente:

`θ_MLE = argmax_θ ℓ(θ)`

### 3) Derivación clave: de MLE a MSE (Regresión Lineal)

La idea conceptual: cuando usas **MSE**, estás asumiendo implícitamente un modelo de ruido.

Supón que tu regresión lineal es:

`y = Xβ + ε` con `ε ~ N(0, σ² I)`

Entonces la probabilidad de observar `y` dado `β` es Gaussiana:

`P(y | X, β) ∝ exp( - (1/(2σ²)) ||y - Xβ||² )`

Tomando log-likelihood y tirando constantes que no dependen de `β`:

`ℓ(β) = const - (1/(2σ²)) ||y - Xβ||²`

Maximizar `ℓ(β)` equivale a minimizar `||y - Xβ||²`.

Conclusión:

- Minimizar **SSE/MSE** es exactamente hacer **MLE** bajo ruido Gaussiano.
- Esta conexión es el puente directo hacia **Statistical Estimation** (Línea 2).

### 4) Conexión Línea 2: estimadores, sesgo y varianza (intuición)

En Línea 2, la palabra clave es **estimador**: una regla que convierte datos en un parámetro.

- Un **estimador** es una función: `\hat{θ} = g(X)`.
- **Sesgo (bias):** si `E[\hat{θ}]` no coincide con el valor real `θ`.
- **Varianza:** cuánto cambia `\hat{θ}` si repites el muestreo.

Regla mental:

- **Más bias** suele dar **menos varianza**.
- **Menos bias** suele dar **más varianza**.

Esto reaparece en ML como *bias-variance tradeoff*.

### 5) Teoría de Estimadores (lo que te evalúan en proyectos/examen)

Aquí pasamos de la intuición a una formalización que aparece mucho en evaluación.

#### 5.1 Sesgo, varianza y MSE (descomposición clave)

Si quieres estimar un parámetro real `θ` con un estimador `\hat{θ}`, el error cuadrático medio es:

`MSE(\hat{θ}) = E[(\hat{θ} - θ)^2]`

La identidad importante es:

`MSE(\hat{θ}) = Var(\hat{θ}) + Bias(\hat{θ})^2`

Donde:

- `Bias(\hat{θ}) = E[\hat{θ}] - θ`
- `Var(\hat{θ}) = E[(\hat{θ} - E[\hat{θ}])^2]`

Lectura mental:

- Puedes reducir MSE bajando varianza, aunque suba un poco el sesgo.
- O puedes “perseguir cero sesgo” y pagar con alta varianza.

Esto es exactamente el *bias-variance trade-off* en ML (por ejemplo, regularizar o simplificar modelos).

#### 5.2 Unbiased vs consistente (2 propiedades distintas)

- **Unbiased (insesgado):** `E[\hat{θ}] = θ`.
- **Consistente:** cuando `n → ∞`, `\hat{θ} → θ` (en un sentido probabilístico).

Un estimador puede ser sesgado y aun así consistente (y a veces es preferible si reduce varianza para `n` finito).

#### 5.3 Conexión directa con regularización (puente a ML)

Ejemplo mental:

- **Ridge / L2** introduce sesgo (empuja coeficientes hacia 0).
- A cambio suele reducir varianza (solución más estable ante ruido y colinealidad).

En términos de la descomposición:

- sube `Bias^2`
- baja `Var`

Si el total baja, mejora el `MSE` esperado fuera de muestra.

<details open>
<summary><strong>📌 Complemento pedagógico — Micro-Capítulo: Maximum Likelihood Estimation (MLE)</strong></summary>

#### 1) Metadatos
- **Título:** MLE como filosofía unificadora: de “ajustar perillas” a pérdidas en ML
- **ID (opcional):** `M04-MICRO-MLE`
- **Duración estimada:** 120–180 min
- **Nivel:** Intermedio–Avanzado
- **Dependencias:** 1.1–2.4 (probabilidad + Bayes), M03 (gradiente/chain rule como preview)

#### 2) Objetivos
- Explicar qué maximiza MLE (verosimilitud de datos observados) y por qué se usa log-likelihood.
- Conectar MLE con pérdidas: MSE ↔ Gaussiana, BCE/CCE ↔ Bernoulli/Categorical.
- Interpretar sesgo/varianza/MSE como puente a regularización.

#### 3) Relevancia
- Te da el “por qué” de cross-entropy: no es un truco, es NLL.

#### 4) Mapa conceptual mínimo
- **Modelo** `P(D|θ)` → define cómo “genera” datos.
- **Likelihood** `L(θ|D)` → probabilidad de D dado θ.
- **Log-likelihood** `ℓ(θ)` → suma (estable) en vez de producto.
- **Entrenamiento** → minimizar `-ℓ(θ)`.

#### 5) Definiciones esenciales
- `θ_MLE = argmax_θ P(D|θ)`.
- `ℓ(θ)=Σ log P(x_i|θ)`.

#### 6) Explicación didáctica
- “MLE elige la perilla que hace que tus datos se vean menos sorprendentes bajo el modelo”.

#### 7) Ejemplo modelado
- Moneda Bernoulli: `p_MLE` = proporción de caras.

#### 8) Práctica guiada
- Repite el worked example cambiando la secuencia de datos y verifica que `p_MLE` cambia como frecuencia.

#### 9) Práctica independiente
- Explica por qué maximizar log-likelihood y maximizar likelihood dan el mismo argmax.

#### 10) Autoevaluación
- ¿Qué diferencia hay entre “parámetro más probable” (Bayes) y “parámetro que hace los datos más probables” (MLE)?

#### 11) Errores comunes
- Confundir `P(θ|D)` con `P(D|θ)`.
- Olvidar que log convierte producto en suma (y por qué ayuda numéricamente).

#### 12) Retención
- (día 2) escribe: `θ_MLE = argmax_θ Σ log p(x_i|θ)`.

#### 13) Diferenciación
- Avanzado: conectar con MAP (regularización como prior) (preview).

#### 14) Recursos
- StatQuest: Maximum Likelihood.

#### 15) Nota docente
- Pedir al alumno que diga “qué asume el modelo” antes de escribir `P(D|θ)`.
</details>

## 🧩 Micro-Capítulo Maestro: Introducción a Markov Chains — Nivel: Intermedio

### 1) Concepto

Una cadena de Markov es un sistema que salta entre estados.

Propiedad de Markov (“falta de memoria”):

`P(S_{t+1} | S_t, S_{t-1}, ...) = P(S_{t+1} | S_t)`

### 2) Representación matricial (puente con Álgebra Lineal)

Si tienes 3 estados (Sol, Nube, Lluvia), defines una matriz de transición `P` (3×3) donde cada fila suma 1.

Si `π_t` es un vector fila (1×3) con la distribución “hoy”, entonces:

`π_{t+1} = π_t P`

Y en `k` pasos:

`π_{t+k} = π_t P^k`

### 3) Reto mental: estacionariedad = eigenvector

Si repites multiplicaciones, muchas cadenas convergen a una distribución estacionaria `π*` tal que:

`π* = π* P`

Eso significa (en la perspectiva correcta) que `π*` es un **eigenvector** asociado al **eigenvalue 1**.

<details open>
<summary><strong>📌 Complemento pedagógico — Micro-Capítulo: Introducción a Markov Chains</strong></summary>

#### 1) Metadatos
- **Título:** Markov Chains como dinámica lineal sobre distribuciones (π_{t+1}=π_t P)
- **ID (opcional):** `M04-MICRO-MARKOV`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** M02 (multiplicación de matrices, eigenvectors), probabilidad básica (distribuciones)

#### 2) Objetivos
- Interpretar `P(S_{t+1}|S_t)` como “memoria de 1 paso”.
- Usar `π_{t+1}=π_t P` y verificar que `π_t` sigue sumando 1.
- Explicar la condición de estacionariedad `π*=π*P`.

#### 3) Relevancia
- Conecta probabilidad con álgebra lineal; reaparece en modelos secuenciales y Monte Carlo (Línea 2).

#### 4) Mapa conceptual mínimo
- **Estados** → categorías discretas.
- **Matriz P** → transiciones (filas suman 1).
- **Distribución π** → vector de probabilidades.
- **Evolución temporal** → multiplicaciones repetidas.

#### 5) Definiciones esenciales
- Matriz estocástica por filas: cada fila suma 1.
- Distribución estacionaria: `π* = π*P`.

#### 6) Explicación didáctica
- Piensa en `π` como “mezcla” de estados; multiplicar por P redistribuye masa.

#### 7) Ejemplo modelado
- 2 estados con `P=[[0.9,0.1],[0.2,0.8]]`: interpreta cada fila como “desde dónde vienes”.

#### 8) Práctica guiada
- Elige un `π_0` y calcula `π_1`, `π_2` a mano.

#### 9) Práctica independiente
- Encuentra (conceptualmente) `π*` resolviendo `π*=π*P` + suma=1.

#### 10) Autoevaluación
- ¿Por qué el eigenvalue asociado a `π*` es 1?

#### 11) Errores comunes
- Confundir si `π` es vector fila o columna (y dónde multiplicar P).
- Usar una P donde filas no suman 1.

#### 12) Retención
- (día 7) escribe `π_{t+1}=π_tP` y explica en una frase qué hace.

#### 13) Diferenciación
- Avanzado: discutir condiciones de convergencia (ergodicidad) (solo conceptual).

#### 14) Recursos
- Material introductorio de Markov Chains + conexión con eigenvectors.

#### 15) Nota docente
- Obligar “sanity check”: después de multiplicar, verificar suma=1.
</details>

---

### Día 5: Distribución Gaussiana (Normal)

#### 3.1 La Distribución Más Importante en ML

```text
                    1              (x - μ)²
f(x) = ───────────────── · exp(- ─────────)
       σ · √(2π)                   2σ²

Parámetros:
- μ (mu): Media (centro de la campana)
- σ (sigma): Desviación estándar (ancho)
- σ² (sigma²): Varianza
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 3.1: Distribución Gaussiana (definición)</strong></summary>

#### 1) Metadatos
- **Título:** PDF Gaussiana: forma, parámetros y lectura correcta
- **ID (opcional):** `M04-T03_1`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 1.1 (probabilidad), noción de función exponencial/log

#### 2) Objetivos
- Identificar qué controla `μ` (desplazamiento) y `σ`/`σ²` (dispersión).
- Distinguir “densidad” `f(x)` de “probabilidad” (área bajo la curva).

#### 3) Relevancia
- La Gaussiana es el átomo de modelos generativos (GMM) y del supuesto de ruido que conecta con MSE.

#### 4) Mapa conceptual mínimo
- **PDF** `f(x)` describe densidad.
- **Parámetros**: `μ` centra, `σ` escala.
- **Probabilidad**: integral de `f(x)` sobre un intervalo.

#### 5) Definiciones esenciales
- `X ~ N(μ, σ²)`.
- `f(x)` es densidad (puede ser >1), pero el área total integra a 1.

#### 6) Explicación didáctica
- Error clásico: interpretar `f(0.5)=0.3` como “30% de probabilidad en x=0.5” (en continuas eso es falso).

#### 7) Ejemplo modelado
- “Campana” estándar: `N(0,1)`.

#### 8) Práctica guiada
- Describe qué pasa si duplicas `σ`: el pico baja y la curva se ensancha.

#### 9) Práctica independiente
- Explica qué significa “2 desviaciones estándar” alrededor de la media en términos cualitativos.

#### 10) Autoevaluación
- ¿Por qué `P(X = x) = 0` en una variable continua aunque `f(x)` sea positiva?

#### 11) Errores comunes
- Confundir `σ` con `σ²`.
- Confundir densidad con probabilidad.

#### 12) Retención
- (día 2) escribe la forma general de la PDF y nombra sus parámetros.

#### 13) Diferenciación
- Avanzado: conecta con log-likelihood de una Gaussiana (preview a MLE).

#### 14) Recursos
- Sección “Normal distribution” (cualquier referencia de probabilidad).

#### 15) Nota docente
- Exigir que el alumno diga: “densidad ≠ probabilidad; probabilidad = área”.
</details>

#### 3.2 Por Qué es Importante

1. **Muchos fenómenos naturales** siguen esta distribución
2. **Teorema del Límite Central:** promedios de cualquier distribución → Normal
3. **GMM usa Gaussianas** para modelar clusters
4. **Inicialización de pesos** en redes neuronales

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 3.2: Por qué la Gaussiana importa en ML</strong></summary>

#### 1) Metadatos
- **Título:** Normal como “default” estadístico: TLC, ruido y modelos
- **ID (opcional):** `M04-T03_2`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 3.1

#### 2) Objetivos
- Explicar 3 usos típicos: ruido Gaussiano ↔ MSE, GMM, inicialización.
- Conectar el TLC con “promedios tienden a normal”.

#### 3) Relevancia
- Entender esto evita que la Normal se sienta como “fórmula que memorizas” sin uso.

#### 4) Mapa conceptual mínimo
- **TLC** → por qué aparece en promedios.
- **Ruido** `ε~N(0,σ²)` → por qué MSE es natural.
- **GMM** → mezcla de gaussianas para clustering.

#### 5) Definiciones esenciales
- TLC (enunciado informal): suma/promedio de muchas variables → aproximadamente normal.

#### 6) Explicación didáctica
- Muchos modelos lineales asumen ruido Gaussiano: no porque sea “verdad absoluta”, sino porque da un modelo tractable.

#### 7) Ejemplo modelado
- Regresión lineal con ruido: minimizas SSE/MSE como MLE Gaussiano (puente a Día 6).

#### 8) Práctica guiada
- Da un ejemplo cotidiano donde “muchas fuentes pequeñas de variación” sugiere normalidad.

#### 9) Práctica independiente
- Explica por qué en pesos de NN se usan gaussianas pequeñas (inicialización) y qué pasa si son muy grandes.

#### 10) Autoevaluación
- ¿Qué aspecto de la normal explica que valores extremos sean raros (colas)?

#### 11) Errores comunes
- Creer que “todo es normal” sin validar.
- Confundir “distribución de datos” con “distribución de ruido”.

#### 12) Retención
- (día 2) enumera 3 conexiones: MSE, GMM, inicialización.

#### 13) Diferenciación
- Avanzado: discusión de heavy tails y por qué a veces Laplace/Student-t es mejor.

#### 14) Recursos
- StatQuest: Normal distribution / Central Limit Theorem.

#### 15) Nota docente
- Pedir una justificación: “¿qué hipótesis hace que MSE tenga sentido?”.
</details>

#### 3.3 Implementación

```python
import numpy as np  # NumPy: arrays, operaciones vectorizadas y funciones matemáticas (exp, sqrt)

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:  # PDF univariada: f(x) de N(μ, σ²)
    """
    Probability Density Function de la Gaussiana.

    Args:
        x: Puntos donde evaluar
        mu: Media
        sigma: Desviación estándar

    Returns:
        Densidad de probabilidad en cada punto
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))  # Coeficiente de normalización: 1/(σ√(2π))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)  # Exponente: - (x-μ)² / (2σ²) (forma estándar)
    return coefficient * np.exp(exponent)  # Evaluación final: coef * exp(exponente) (vectorizado)


# Visualización
import matplotlib.pyplot as plt  # Matplotlib: gráficos para construir intuición visual

x = np.linspace(-5, 5, 1000)  # Eje 1D de evaluación (1000 puntos para curva suave)

# Diferentes Gaussianas
plt.figure(figsize=(10, 6))  # Crea un lienzo con tamaño controlado
plt.plot(x, gaussian_pdf(x, mu=0, sigma=1), label='μ=0, σ=1 (estándar)')  # Curva “campana” estándar
plt.plot(x, gaussian_pdf(x, mu=0, sigma=2), label='μ=0, σ=2 (más ancha)')  # Aumentar σ ensancha y baja el pico
plt.plot(x, gaussian_pdf(x, mu=2, sigma=1), label='μ=2, σ=1 (desplazada)')  # Cambiar μ desplaza la curva
plt.legend()  # Muestra leyenda con labels
plt.title('Distribuciones Gaussianas')  # Título descriptivo
plt.xlabel('x')  # Etiqueta del eje x
plt.ylabel('f(x)')  # Etiqueta del eje y (densidad)
plt.grid(True)  # Rejilla para lectura más fácil
plt.savefig('gaussian_distributions.png')  # Guarda imagen (útil para reportes)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 3.3: Implementación de la PDF Gaussiana (univariada)</strong></summary>

#### 1) Metadatos
- **Título:** Implementar PDF: normalización, vectorización y sanity checks
- **ID (opcional):** `M04-T03_3`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 3.1

#### 2) Objetivos
- Implementar `gaussian_pdf` sin errores de forma y con vectorización.
- Identificar el rol del coeficiente y del exponente.

#### 3) Relevancia
- Te entrena para implementar funciones de densidad y luego reutilizarlas en log-likelihood/EM.

#### 4) Mapa conceptual mínimo
- **Coeficiente** `1/(σ√(2π))` normaliza.
- **Exponente** penaliza distancia al centro.
- **Vectorización**: evaluar muchos x de una vez.

#### 5) Definiciones esenciales
- `σ>0` (si `σ<=0` el modelo no tiene sentido).

#### 6) Explicación didáctica
- Sanity check numérico: la curva debe ser no negativa y “parecer campana”.

#### 7) Ejemplo modelado
- Comparación de distintas `μ` y `σ` para construir intuición visual.

#### 8) Práctica guiada
- Añade una verificación: `assert np.all(gaussian_pdf(x,mu,sigma) >= 0)`.

#### 9) Práctica independiente
- (conceptual) ¿Qué debería pasar con el pico cuando `σ` se hace muy pequeño?

#### 10) Autoevaluación
- ¿Qué parte del código cambia si reemplazas `σ` por `σ²` como parámetro?

#### 11) Errores comunes
- Overflow/underflow en `exp` cuando `σ` es muy pequeño o `|x-μ|` grande.
- Olvidar que `sigma` es desviación estándar (no varianza).

#### 12) Retención
- (día 2) escribe la función en pseudo-código: coef × exp(exponente).

#### 13) Diferenciación
- Avanzado: implementar `log_gaussian_pdf` estable y comparar.

#### 14) Recursos
- Numpy `np.exp`, estabilidad numérica.

#### 15) Nota docente
- Pedir al alumno que explique qué controla `μ` y qué controla `σ` viendo los plots.
</details>

#### 3.4 Gaussiana Multivariada (Para GMM)

```python
def multivariate_gaussian_pdf(x: np.ndarray,  # x:(d,) vector de características (una muestra)
                               mu: np.ndarray,  # mu:(d,) vector de medias
                               cov: np.ndarray) -> float:  # cov:(d,d) matriz de covarianza
    """
    Gaussiana multivariada para vectores.

    Args:
        x: Vector de características (d,)
        mu: Vector de medias (d,)
        cov: Matriz de covarianza (d, d)

    Returns:
        Densidad de probabilidad
    """
    d = len(mu)  # d: dimensión del espacio (número de features)
    diff = x - mu  # diff:(d,) centra el punto restando la media

    # Determinante e inversa de la covarianza
    det_cov = np.linalg.det(cov)  # |Σ|: controla el “volumen” de la elipse gaussiana
    inv_cov = np.linalg.inv(cov)  # Σ^{-1}: aparece en la forma cuadrática (Mahalanobis)

    # Coeficiente de normalización
    coefficient = 1 / (np.sqrt((2 * np.pi) ** d * det_cov))  # 1 / sqrt((2π)^d |Σ|)

    # Exponente (forma cuadrática)
    exponent = -0.5 * diff.T @ inv_cov @ diff  # -(1/2)(x-μ)^T Σ^{-1} (x-μ)

    return coefficient * np.exp(exponent)  # Devuelve densidad (escala) * exp(exponente)


# Ejemplo 2D
mu = np.array([0, 0])  # μ:(2,) media en 2D
cov = np.array([[1, 0.5],  # Σ[0,0]=var(x1), Σ[0,1]=cov(x1,x2)
                [0.5, 1]])  # Correlación positiva: elipses rotadas respecto a los ejes

x = np.array([0.5, 0.5])  # Punto a evaluar (una muestra)
prob = multivariate_gaussian_pdf(x, mu, cov)  # Escalar: densidad en ese punto
print(f"P(x=[0.5, 0.5]) = {prob:.4f}")  # Imprime densidad (ojo: no es probabilidad discreta)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 3.4: Gaussiana Multivariada</strong></summary>

#### 1) Metadatos
- **Título:** Multivariada: covarianza, elipses y Mahalanobis
- **ID (opcional):** `M04-T03_4`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio–Avanzado
- **Dependencias:** M02 (det/inv, formas cuadráticas), 3.1

#### 2) Objetivos
- Interpretar el rol de `Σ` (covarianza) como escala + correlación.
- Reconocer la forma cuadrática `(x-μ)^T Σ^{-1} (x-μ)` como “distancia elíptica”.

#### 3) Relevancia
- Es el núcleo matemático de GMM y de muchas técnicas estadísticas.

#### 4) Mapa conceptual mínimo
- `μ` fija el centro.
- `Σ` fija la elipse (forma/orientación).
- `|Σ|` controla volumen.

#### 5) Definiciones esenciales
- Covarianza válida: simétrica y PSD (idealmente PD para invertir).

#### 6) Explicación didáctica
- Si `Σ` tiene covarianzas fuera de la diagonal, la elipse rota.

#### 7) Ejemplo modelado
- Caso 2D con correlación positiva (`0.5`) para ver rotación.

#### 8) Práctica guiada
- Cambia `cov` a diagonal y compara con el caso correlacionado.

#### 9) Práctica independiente
- Explica qué pasa si `det_cov` es casi 0 (covarianza casi singular).

#### 10) Autoevaluación
- ¿Por qué aparece `Σ^{-1}` en lugar de `Σ` en el exponente?

#### 11) Errores comunes
- Invertir `Σ` singular (numéricamente inestable).
- Confundir densidad con probabilidad.

#### 12) Retención
- (día 7) escribe la forma: coeficiente × exp(-0.5 * Mahalanobis).

#### 13) Diferenciación
- Avanzado: usar Cholesky para estabilidad en lugar de `inv`/`det` directos.

#### 14) Recursos
- Material de GMM / multivariate normal.

#### 15) Nota docente
- Pedir al alumno que dibuje cómo cambia la elipse al variar covarianza.
</details>

#### 3.5 GMM Just-in-Time: Mezcla de 3 gaussianas + contornos (preámbulo a Unsupervised)

**Objetivo:** que la “Gaussiana multivariada” no se quede teórica: vas a **generar datos** de una mezcla de 3 gaussianas y a **visualizar contornos** (componentes y mezcla). Esto es el puente directo a **GMM** (Módulo 06).

- **Ejecutable:**
  - `python3 scripts/gmm_3_gaussians_contours.py`
- **Entregable:**
  - una figura (pantallazo o archivo guardado con `--out`) y una explicación breve:
    - **Qué representa** el contorno negro (mezcla)
    - **Qué representan** los contornos coloreados (componentes)
    - **Qué cambia** si modificas una covarianza (rotación / elongación)

- **Preguntas (nivel maestría):**
  - **K-Means vs GMM:** ¿por qué K-Means es *hard assignment* y GMM es *soft assignment*?
  - **Covarianza:** ¿qué hace `Σ` geométricamente (orientación/forma) y por qué aparece `Σ^{-1}` en el exponente?

---

### Día 6: Maximum Likelihood Estimation (MLE)

#### 4.0 MLE → Cross-Entropy (la conexión que te piden en exámenes)

**Idea:** si un modelo produce probabilidades `P(y|x, θ)`, entrenar por MLE significa:

- maximizar `Πᵢ P(yᵢ|xᵢ, θ)`

Por estabilidad numérica y conveniencia, trabajamos con log:

- maximizar `Σᵢ log P(yᵢ|xᵢ, θ)`

Y como optimizadores minimizan, entrenamos minimizando:

- `-Σᵢ log P(yᵢ|xᵢ, θ)`  (negative log-likelihood)

Ese término es exactamente la **cross-entropy** que usas en:

- Logistic Regression (BCE) en `Módulo 05`
- clasificación multiclase (CCE) en `Módulo 07`

**Cheat sheet:**

- **MLE:** maximizar likelihood
- **Entrenamiento:** minimizar negative log-likelihood
- **En clasificación:** eso se llama cross-entropy

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.0: MLE → Cross-Entropy (la conexión que te piden en exámenes)</strong></summary>

#### 1) Metadatos
- **Título:** De maximizar likelihood a minimizar cross-entropy (NLL)
- **ID (opcional):** `M04-T04_0`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 2.1–2.3 (probabilidad condicional/Bayes), noción de logaritmo

#### 2) Objetivos
- Conectar el producto de probabilidades con suma de log-probabilidades.
- Explicar por qué optimizamos **NLL** (negative log-likelihood) en vez de maximizar likelihood.
- Reconocer que en clasificación la NLL se escribe como **cross-entropy**.

#### 3) Relevancia
- Esta equivalencia es el “puente” entre probabilidad y entrenamiento: explica por qué la loss típica en clasificación es cross-entropy.

#### 4) Mapa conceptual mínimo
- **Likelihood** `P(D|θ)` (producto)
- **Log-likelihood** `log P(D|θ)` (suma)
- **NLL** `-log P(D|θ)` (minimización)
- **Cross-entropy** (forma estándar de la NLL en clasificación)

#### 5) Definiciones esenciales
- **Likelihood:** probabilidad de observar los datos si el modelo tuviera parámetros `θ`.
- **NLL:** `-Σ log P(yᵢ|xᵢ,θ)`; es una loss no negativa (en promedio) que penaliza probabilidades pequeñas asignadas a la etiqueta correcta.

#### 6) Explicación didáctica
- El producto `Π P(yᵢ|xᵢ,θ)` se vuelve numéricamente pequeño; el log lo transforma en suma y evita underflow.
- Cambiar de “maximizar” a “minimizar” es solo conveniencia (los optimizadores típicos minimizan).

#### 7) Ejemplo modelado
- Si el modelo asigna `P(y=correcto|x)=0.01`, entonces `-log(0.01)` es grande: el entrenamiento “siente” fuerte ese error.

#### 8) Práctica guiada
- Reescribe el objetivo para un dataset de 3 muestras y verifica el paso:
  - `max Π pᵢ` → `max Σ log pᵢ` → `min -Σ log pᵢ`.

#### 9) Práctica independiente
- Describe qué pasa con la NLL si duplicas el dataset (mismas muestras dos veces). ¿Por qué se suele usar promedio `1/m`?

#### 10) Autoevaluación
- ¿Por qué `log` convierte productos en sumas y por qué eso ayuda a optimizar?

#### 11) Errores comunes
- Confundir **cross-entropy** con accuracy: una es función suave optimizable, la otra no.
- Olvidar el signo: minimizar `-log(p)` equivale a maximizar `log(p)`.

#### 12) Retención
- Regla mnemónica: **MLE ⇒ max log-likelihood ⇒ min NLL ⇒ cross-entropy (clasificación)**.

#### 13) Diferenciación
- Avanzado: compara NLL con label smoothing (cómo cambia la penalización cuando `y` no es one-hot perfecto).

#### 14) Recursos
- Función `log` y propiedades: `log(ab)=log(a)+log(b)`.

#### 15) Nota docente
- Pedir al alumno que explique “por qué el log es un truco numérico y algebraico a la vez”.
</details>

---

### Extensión Estratégica (Línea 2): Statistical Estimation

#### MLE como filosofía: “ajustar perillas”

MLE no es solo una fórmula: es una forma de pensar.

- Tienes un modelo con parámetros `θ` (las “perillas”).
- Ya viste datos `D`.
- Pregunta: ¿qué valores de `θ` hacen que `D` sea lo más probable posible?

Formalmente:

```text
θ_MLE = argmax_θ P(D | θ)
```

Como `P(D|θ)` suele ser un producto grande, usamos log:

```text
θ_MLE = argmax_θ log P(D | θ)
```

Esto es el puente directo a **Statistical Estimation** (Línea 2): estimadores, sesgo, varianza, y por qué “promedio” aparece en tantos lados.

#### Worked example: Moneda (Bernoulli) → estimador MLE

Modelo:

- `X_i ~ Bernoulli(p)` donde `p = P(cara)`.

Datos:

- `D = {x_1, ..., x_n}` con `x_i ∈ {0,1}`.

Likelihood:

```text
P(D | p) = Π_i p^{x_i} (1-p)^{(1-x_i)}
```

Log-likelihood:

```text
ℓ(p) = Σ_i [x_i log p + (1-x_i) log(1-p)]
```

Derivar y hacer 0 (intuición: el máximo ocurre cuando la “probabilidad del modelo” coincide con la frecuencia observada):

```text
dℓ/dp = Σ_i [x_i/p - (1-x_i)/(1-p)] = 0
```

Solución:

```text
p_MLE = (1/n) Σ_i x_i
```

Interpretación: el MLE de `p` es simplemente la **proporción de caras**. Este patrón (media muestral) reaparece en gaussianas y en muchos estimadores.

#### 4.1 La Idea Central

```text
MLE: Encontrar los parámetros θ que maximizan la probabilidad
     de observar los datos que tenemos.

θ_MLE = argmax P(datos | θ)
            θ
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.1: La Idea Central</strong></summary>

#### 1) Metadatos
- **Título:** Qué significa “ajustar θ para explicar los datos”
- **ID (opcional):** `M04-T04_1`
- **Duración estimada:** 30–60 min
- **Nivel:** Básico–Intermedio
- **Dependencias:** 4.0

#### 2) Objetivos
- Interpretar `argmax_θ P(datos|θ)` como “buscar el θ que hace los datos más probables”.
- Identificar qué es **dato**, qué es **parámetro** y qué es **modelo**.

#### 3) Relevancia
- Esta idea aparece en regresión logística, Naive Bayes, gaussianas, GMM y en general en modelos probabilísticos.

#### 4) Mapa conceptual mínimo
- **Modelo** `P(x|θ)` / `P(y|x,θ)`
- **Datos** `D={xᵢ,yᵢ}`
- **Parámetros** `θ`
- **Objetivo** `argmax` (o `argmin` NLL)

#### 5) Definiciones esenciales
- `argmax`: devuelve el valor del parámetro que maximiza una función.
- i.i.d. (supuesto típico): cada muestra aporta un factor multiplicativo a la likelihood.

#### 6) Explicación didáctica
- Piensa en `θ` como “perillas” del generador de datos: MLE elige las perillas que hacen “creíble” el dataset observado.

#### 7) Ejemplo modelado
- Moneda: `θ=p`; si observas muchas caras, el `p` que mejor explica el dato es alto.

#### 8) Práctica guiada
- Identifica `θ` en:
  - Bernoulli (`p`),
  - Gaussiana (`μ,σ`),
  - Softmax (`W`).

#### 9) Práctica independiente
- Escribe en una línea qué maximiza MLE para un modelo `P(y|x,θ)`.

#### 10) Autoevaluación
- ¿Qué cambia si los datos no fueran independientes?

#### 11) Errores comunes
- Mezclar `P(θ|datos)` (Bayes) con `P(datos|θ)` (MLE).

#### 12) Retención
- Frase clave: **MLE mira datos→θ (qué θ explica mejor lo observado)**.

#### 13) Diferenciación
- Avanzado: contrasta MLE con MAP (`argmax P(θ|D)`), aunque ambos suelen acabar en minimizar una loss.

#### 14) Recursos
- Repasar diferencia entre prior, likelihood y posterior.

#### 15) Nota docente
- Verbalización obligatoria: “¿qué estoy maximizando exactamente y respecto a qué variable?”
</details>

#### 4.2 Por Qué es Fundamental

- **Logistic Regression** usa MLE para encontrar los pesos
- **Cross-Entropy Loss** viene de maximizar likelihood
- **GMM** usa MLE (via EM algorithm)

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.2: Por Qué es Fundamental</strong></summary>

#### 1) Metadatos
- **Título:** Por qué MLE está “debajo” de pérdidas y modelos comunes
- **ID (opcional):** `M04-T04_2`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio
- **Dependencias:** 4.0–4.1

#### 2) Objetivos
- Identificar al menos 3 lugares del stack ML donde MLE aparece implícitamente.
- Conectar *modelado probabilístico* con *función de pérdida*.

#### 3) Relevancia
- Te permite “leer” una loss como una suposición probabilística (qué distribución estás asumiendo).

#### 4) Mapa conceptual mínimo
- **Modelo probabilístico** → **log-likelihood** → **NLL** → **gradiente/optimización**

#### 5) Definiciones esenciales
- **Estimador:** regla que produce un parámetro `\hat{θ}` desde datos.
- **Loss probabilística:** una loss que puede interpretarse como NLL bajo un modelo.

#### 6) Explicación didáctica
- Cuando eliges cross-entropy, eliges implícitamente “el dato `y` sigue una distribución categórica parametrizada por el modelo”.

#### 7) Ejemplo modelado
- Regresión:
  - Si asumes ruido Gaussiano, la NLL se parece a MSE.
  - Si asumes Bernoulli/categórica, la NLL se vuelve BCE/CCE.

#### 8) Práctica guiada
- Para cada bullet del tema (LogReg, Cross-Entropy, GMM), completa la frase:
  - “La loss es la NLL de una distribución ____”.

#### 9) Práctica independiente
- ¿Qué suposición probabilística hay detrás de usar MSE como loss?

#### 10) Autoevaluación
- ¿Por qué “maximizar likelihood” y “minimizar NLL” son el mismo objetivo?

#### 11) Errores comunes
- Creer que MLE “solo” es una técnica estadística: en ML moderno es una forma estándar de derivar losses.

#### 12) Retención
- Fórmula mental: **modelar `P(y|x)` ⇒ entrenar = maximizar `P(y|x,θ)`**.

#### 13) Diferenciación
- Avanzado: discute cuándo preferir MAP/regularización como “prior” implícito.

#### 14) Recursos
- Lectura corta: interpretación probabilística de MSE/BCE/CCE.

#### 15) Nota docente
- Mini-debate: “¿una loss define un modelo o un modelo define una loss?”
</details>

#### 4.3 MLE para Gaussiana

```python
def mle_gaussian(data: np.ndarray) -> tuple[float, float]:  # estima (mu, sigma) por MLE para una gaussiana
    """
    Estimar parámetros de Gaussiana con MLE.

    Para una Gaussiana, los estimadores MLE son:
    - μ_MLE = media muestral
    - σ²_MLE = varianza muestral (con n, no n-1)

    Args:
        data: Muestras observadas

    Returns:
        (mu_mle, sigma_mle)
    """
    n = len(data)  # número de muestras disponibles

    # MLE de la media
    mu_mle = np.mean(data)  # μ_MLE: promedio muestral

    # MLE de la varianza (dividir por n, no n-1)
    sigma_squared_mle = np.sum((data - mu_mle) ** 2) / n  # σ²_MLE: varianza muestral con divisor n
    sigma_mle = np.sqrt(sigma_squared_mle)  # σ_MLE: desviación estándar (raíz de la varianza)

    return mu_mle, sigma_mle  # retorna estimaciones (μ, σ)


# Ejemplo: Generar datos y estimar
np.random.seed(42)  # fija semilla para reproducibilidad del muestreo
true_mu, true_sigma = 5.0, 2.0  # parámetros verdaderos usados para simular datos
samples = np.random.normal(true_mu, true_sigma, size=1000)  # genera muestras N(true_mu, true_sigma^2)

estimated_mu, estimated_sigma = mle_gaussian(samples)  # estima parámetros a partir de las muestras simuladas
print(f"Parámetros reales: μ={true_mu}, σ={true_sigma}")  # muestra parámetros ground truth
print(f"MLE estimados:     μ={estimated_mu:.3f}, σ={estimated_sigma:.3f}")  # muestra estimaciones MLE
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.3: MLE para Gaussiana</strong></summary>

#### 1) Metadatos
- **Título:** Media muestral y varianza con `n` (no `n-1`) como MLE
- **ID (opcional):** `M04-T04_3`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 3.1–3.3 (Gaussiana univariada) + 4.1

#### 2) Objetivos
- Diferenciar varianza MLE (`/n`) de varianza insesgada (`/(n-1)`).
- Implementar estimadores MLE para `μ` y `σ` y validar con datos simulados.

#### 3) Relevancia
- Esta derivación aparece en EM/GMM y en cualquier modelo que use gaussianas (ruido, priors, etc.).

#### 4) Mapa conceptual mínimo
- **Asunción:** `xᵢ ~ N(μ,σ²)`
- **Objetivo:** `argmax log P(D|μ,σ)`
- **Resultado:** `μ=mean(x)` y `σ²=mean((x-μ)²)`

#### 5) Definiciones esenciales
- `σ²_MLE = (1/n) Σ (xᵢ-μ)²`.
- Estimador insesgado: usa `1/(n-1)` (otra propiedad, objetivo distinto).

#### 6) Explicación didáctica
- MLE optimiza “qué parámetros hacen más probable el dataset”, no “que el estimador sea insesgado”.

#### 7) Ejemplo modelado
- Con `n=1000`, `\hat{μ}` y `\hat{σ}` deberían acercarse a los parámetros reales por ley de los grandes números.

#### 8) Práctica guiada
- Agrega checks:
  - `assert estimated_sigma > 0`.
  - `assert abs(estimated_mu-true_mu) < 0.2` (con `n` grande).

#### 9) Práctica independiente
- Repite con `n=10` y observa la variabilidad de `\hat{σ}`.

#### 10) Autoevaluación
- ¿Por qué `/(n-1)` no sale de MLE cuando maximizas likelihood?

#### 11) Errores comunes
- Usar `np.std(data, ddof=1)` y decir que es MLE (eso es insesgado, no MLE).
- Confundir `σ` con `σ²` en el retorno.

#### 12) Retención
- Regla: **MLE de media = promedio; MLE de varianza = promedio de cuadrados centrados**.

#### 13) Diferenciación
- Avanzado: deriva la log-likelihood de la Gaussiana y ubica dónde aparece el término `log σ`.

#### 14) Recursos
- Numpy: `np.mean`, `np.sum`, `np.sqrt`.

#### 15) Nota docente
- Pregunta guiadora: “¿qué propiedad estás optimizando: likelihood o sesgo?”
</details>

#### 4.4 Conexión con Cross-Entropy Loss

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.4: Conexión con Cross-Entropy Loss</strong></summary>

#### 1) Metadatos
- **Título:** Cross-entropy como NLL: la forma “estándar” de escribir MLE en clasificación
- **ID (opcional):** `M04-T04_4`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio
- **Dependencias:** 4.0 + 2.1 (probabilidades condicionadas)

#### 2) Objetivos
- Escribir explícitamente la NLL en binario y multiclase.
- Identificar la “clase correcta” como el término que se queda en la suma cuando `y` es one-hot.

#### 3) Relevancia
- Esta conexión explica por qué la loss tiene logs y por qué penaliza con fuerza probabilidades pequeñas.

#### 4) Mapa conceptual mínimo
- `P(y|x,θ)` → `log P(y|x,θ)` → `-log P(y|x,θ)`
- One-hot `y` “selecciona” la clase correcta en `Σ y_k log(p_k)`

#### 5) Definiciones esenciales
- **Cross-entropy (multiclase):** `H(y,p)= -Σ_k y_k log(p_k)`.
- Si `y` es one-hot, entonces `H(y,p) = -log(p_clase_correcta)`.

#### 6) Explicación didáctica
- No hay “magia”: el log aparece por MLE y por estabilidad numérica.

#### 7) Ejemplo modelado
- Si `p_correcta=0.9`, pérdida ≈ `0.105`; si `p_correcta=0.01`, pérdida ≈ `4.605`.

#### 8) Práctica guiada
- Calcula `-log(p_correcta)` para `p∈{0.9,0.5,0.1,0.01}` y ordénalos.

#### 9) Práctica independiente
- Explica por qué una predicción “muy segura y equivocada” recibe mucha penalización.

#### 10) Autoevaluación
- ¿Qué pasa con la loss si el modelo siempre predice `p_correcta=1/K`?

#### 11) Errores comunes
- Calcular `np.log(softmax(z))` de forma ingenua y sufrir underflow/NaN (ver día 7).

#### 12) Retención
- Frase: **cross-entropy = costo de sorprenderte al ver la etiqueta verdadera**.

#### 13) Diferenciación
- Avanzado: conecta con KL: `H(y,p)=H(y)+KL(y||p)` (cuando `y` es distribución).

#### 14) Recursos
- Revisión: propiedades de `log` y estabilidad numérica.

#### 15) Nota docente
- Pedir que el alumno derive la forma one-hot → `-log(p_correcta)` en 3 líneas.
</details>

#### 4.5 MLE para multiclase (Softmax + Categorical Cross-Entropy)

Para `K` clases, `y` es one-hot y el modelo produce probabilidades con softmax:

- `p = softmax(z)` donde `z = XW` son logits

Likelihood (por muestra):

- `P(y|x) = Π_k p_k^{y_k}`

Log-likelihood:

- `log P(y|x) = Σ_k y_k log(p_k)`

Negative log-likelihood promedio:

- `L = -(1/m) Σᵢ Σ_k y_{ik} log(p_{ik})`

Eso es exactamente **Categorical Cross-Entropy**.

```python
def cross_entropy_from_mle():  # demuestra cross-entropy como NLL (derivada desde MLE)
    """
    Demostración de que Cross-Entropy viene de MLE.

    Para clasificación binaria con Bernoulli:
    P(y|x, θ) = p^y · (1-p)^(1-y)

    Donde p = σ(θᵀx) (predicción del modelo)

    Log-likelihood:
    log P(y|x, θ) = y·log(p) + (1-y)·log(1-p)

    Maximizar likelihood = Minimizar negative log-likelihood
    = Minimizar Cross-Entropy!
    """
    # Ejemplo numérico
    y_true = np.array([1, 0, 1, 1, 0])  # etiquetas reales (0/1)
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # probabilidades predichas p(y=1|x)

    # Cross-Entropy (negative log-likelihood promedio)
    epsilon = 1e-15  # Para evitar log(0)
    ce = -np.mean(  # NLL promedio: -E[y log(p) + (1-y) log(1-p)]
        y_true * np.log(y_pred + epsilon) +  # contribución de ejemplos positivos (y=1)
        (1 - y_true) * np.log(1 - y_pred + epsilon)  # contribución de ejemplos negativos (y=0)
    )  # cierra promedio de cross-entropy

    print(f"Cross-Entropy Loss: {ce:.4f}")  # imprime el valor de la loss para inspección
    return ce  # retorna la cross-entropy calculada

cross_entropy_from_mle()  # ejecuta la demo de MLE→cross-entropy
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 4.5: MLE para multiclase (Softmax + Categorical Cross-Entropy)</strong></summary>

#### 1) Metadatos
- **Título:** De `Π p_k^{y_k}` a `-Σ y_k log(p_k)` (y por qué eso es entrenable)
- **ID (opcional):** `M04-T04_5`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 4.0 + noción de one-hot + softmax (día 7)

#### 2) Objetivos
- Derivar la log-likelihood multiclase usando one-hot.
- Interpretar la CCE como “castigo” a la probabilidad asignada a la clase correcta.
- Reconocer el rol de `epsilon` como protección de `log(0)`.

#### 3) Relevancia
- Esta es la base de entrenamiento para redes neuronales multiclase y modelos lineales con softmax.

#### 4) Mapa conceptual mínimo
- **Logits** `z` → **Softmax** `p` → **Log-prob** `log(p)` → **CCE/NLL**

#### 5) Definiciones esenciales
- One-hot: `y_k∈{0,1}`, `Σ_k y_k = 1`.
- CCE por muestra: `L = -Σ_k y_k log(p_k)`.

#### 6) Explicación didáctica
- El producto `Π_k p_k^{y_k}` “selecciona” exactamente la probabilidad de la clase verdadera.
- El log convierte ese producto en suma (y vuelve diferenciable y más estable el entrenamiento).

#### 7) Ejemplo modelado
- Para `K=3`, si la clase verdadera es 2, la loss es `-log(p_2)`.

#### 8) Práctica guiada
- Construye un `y` one-hot y un vector `p` y verifica a mano que:
  - `-Σ y_k log(p_k)` coincide con `-log(p_clase_correcta)`.

#### 9) Práctica independiente
- Explica por qué se promedia en batch (`1/m`) y no se usa suma sin normalizar.

#### 10) Autoevaluación
- ¿Qué problema numérico aparece si `p_k` llega a 0 exacto?

#### 11) Errores comunes
- Usar softmax + log de manera ingenua y obtener `-inf/NaN`.
- Confundir `logits` (sin normalizar) con probabilidades.

#### 12) Retención
- Regla: **CCE = NLL de una categórica parametrizada por softmax**.

#### 13) Diferenciación
- Avanzado: describe por qué en práctica se prefiere “CE desde logits” con `log_softmax`.

#### 14) Recursos
- Estabilidad numérica: Log-Sum-Exp trick (día 7).

#### 15) Nota docente
- Pedir que el alumno identifique, en una implementación, dónde se aplica `max(z)` para estabilizar.
</details>

---

### Día 6.5: Teoría de la Información (Entropía + KL-Divergence)

Este bloque existe para que puedas leer “cross-entropy” como **divergencia KL + constante** y para que puedas derivar la equivalencia central:

- **Minimizar KL** (entre distribución real y modelo) es **maximizar log-likelihood**.

#### 6.5.1 Entropía

Para una distribución discreta `p`:

```text
H(p) = - Σ_x p(x) log p(x)
```

Intuición: “costo promedio de sorpresa” bajo `p`.

#### 6.5.2 Divergencia KL

Para dos distribuciones discretas `p` y `q`:

```text
KL(p||q) = Σ_x p(x) log( p(x) / q(x) )
         = Σ_x p(x) log p(x) - Σ_x p(x) log q(x)
```

#### 6.5.3 Derivación clave: minimizar KL ⇔ maximizar log-likelihood

Sea `p_data` la distribución “real” y `p_θ` tu modelo.

```text
KL(p_data || p_θ)
= E_{x~p_data}[log p_data(x)] - E_{x~p_data}[log p_θ(x)]
```

El primer término no depende de `θ`. Por lo tanto:

```text
argmin_θ KL(p_data || p_θ)  =  argmax_θ E_{x~p_data}[log p_θ(x)]
```

Y con datos i.i.d. `{x_i}`:

```text
E_{p_data}[log p_θ(x)]  ≈  (1/n) Σ_i log p_θ(x_i)
```

Así conectas KL directamente con MLE.

#### 6.5.4 Cross-Entropy como KL + constante

Cuando `y` es una distribución (por ejemplo one-hot) y `p` es la predicción:

```text
H(y,p) = H(y) + KL(y||p)
```

Como `H(y)` es constante respecto al modelo, minimizar cross-entropy equivale a minimizar `KL(y||p)`.

## 🌱 Extensión Estratégica (Línea 2): Markov Chains (intro conceptual)

> Esta sección es conceptual: no vas a implementar Markov Chains en Línea 1, pero sí necesitas que la idea te resulte familiar cuando entres al curso de **Discrete-Time Markov Chains and Monte Carlo Methods**.

### Idea central: estados y transiciones

Una cadena de Markov modela un sistema que “salta” entre **estados**.

- Hoy estás en un estado `S_t`.
- Mañana estás en `S_{t+1}`.
- Lo importante: `P(S_{t+1} | S_t)` depende solo del estado actual (memoria de 1 paso).

### Matriz de transición (conexión con Álgebra Lineal)

Definimos una matriz `P` donde:

- `P[i, j] = P(estado j | estado i)`
- Cada fila suma 1 (matriz estocástica por filas)

Si `π_t` es un vector fila con la distribución de probabilidad sobre estados en el tiempo `t`, entonces:

```text
π_{t+1} = π_t P
```

Esto conecta directamente con `Módulo 02`: es **multiplicación de matrices** aplicada a probabilidades.

### Ejemplo mínimo (2 estados)

Estados: `A` y `B`.

```text
P = [[0.9, 0.1],
     [0.2, 0.8]]
```

Interpretación:

- Si estás en `A`, te quedas en `A` con 0.9, pasas a `B` con 0.1.
- Si estás en `B`, pasas a `A` con 0.2, te quedas en `B` con 0.8.

### Estacionariedad (semilla para Línea 2)

Una distribución estacionaria `π*` satisface:

```text
π* = π* P
```

En otras palabras: es un **autovector** (eigenvector) asociado al eigenvalue `1` (visto desde la perspectiva correcta). Esto vuelve a conectar Markov Chains con eigenvalues/eigenvectors.

---

### Día 7: Softmax como Distribución de Probabilidad

#### 5.1 De Logits a Probabilidades

```text
                     exp(zᵢ)
softmax(z)ᵢ = ─────────────────
               Σⱼ exp(zⱼ)

Propiedades:
- Cada salida ∈ (0, 1)
- Suma de salidas = 1 (distribución válida)
- Preserva el orden (mayor logit → mayor probabilidad)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 5.1: De Logits a Probabilidades</strong></summary>

#### 1) Metadatos
- **Título:** Softmax como distribución: de scores a probabilidades comparables
- **ID (opcional):** `M04-T05_1`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 4.5 (CCE desde MLE), álgebra básica de exponentes

#### 2) Objetivos
- Explicar qué son **logits** y por qué no son probabilidades.
- Interpretar softmax como una normalización positiva que suma 1.
- Reconocer invariancia por desplazamiento: `softmax(z)=softmax(z+c)`.

#### 3) Relevancia
- Softmax es la salida estándar en clasificación multiclase y conecta directamente con la CCE.

#### 4) Mapa conceptual mínimo
- **Logits** `z` → `exp(z)` → **normalización** `Σ exp(z)` → **probabilidades**

#### 5) Definiciones esenciales
- **Logit:** score sin normalizar (puede ser cualquier real).
- **Distribución válida:** entradas en `(0,1)` y suma 1.

#### 6) Explicación didáctica
- `exp` asegura positividad; dividir por la suma fuerza “competencia” entre clases.

#### 7) Ejemplo modelado
- Si una clase sube su logit, su probabilidad sube y las demás bajan para mantener suma 1.

#### 8) Práctica guiada
- Verifica (a mano) que `softmax([0,0]) = [0.5,0.5]`.

#### 9) Práctica independiente
- Demuestra en 2 líneas la invariancia: `softmax(z)=softmax(z-c)` para cualquier constante `c`.

#### 10) Autoevaluación
- ¿Qué sucede si sumas 100 a todos los logits? ¿Cambia el resultado?

#### 11) Errores comunes
- Interpretar logits como probabilidades.
- Olvidar que softmax depende de las diferencias relativas entre logits.

#### 12) Retención
- Regla: **softmax convierte scores relativos en probabilidades que compiten**.

#### 13) Diferenciación
- Avanzado: explora el efecto de la temperatura `softmax(z/T)`.

#### 14) Recursos
- Relación con CCE: `L = -log p(clase correcta)`.

#### 15) Nota docente
- Pregunta rápida: “si una probabilidad sube, ¿qué debe pasar con las otras y por qué?”

</details>

#### 5.2 El Problema de Estabilidad Numérica (v3.3)

```text
⚠️ PROBLEMA: exp() puede causar overflow/underflow

Ejemplo peligroso:
    z = [1000, 1001, 1002]
    exp(z) = [inf, inf, inf]  → NaN en softmax!

Ejemplo underflow:
    z = [-1000, -1001, -1002]
    exp(z) = [0, 0, 0]  → 0/0 = NaN!
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 5.2: El Problema de Estabilidad Numérica</strong></summary>

#### 1) Metadatos
- **Título:** Por qué `exp` rompe y cómo reconocer overflow/underflow
- **ID (opcional):** `M04-T05_2`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio
- **Dependencias:** 5.1

#### 2) Objetivos
- Identificar síntomas: `inf`, `0`, `NaN` en softmax.
- Explicar por qué `inf/inf` y `0/0` aparecen.
- Justificar la necesidad de un truco algebraico (no “parche”).

#### 3) Relevancia
- Este error es común en entrenamiento real y puede arruinar gradients (loss NaN).

#### 4) Mapa conceptual mínimo
- logits grandes → `exp(z)` overflow → `inf` → `inf/inf` → `NaN`
- logits muy negativos → `exp(z)` underflow → `0` → `0/0` → `NaN`

#### 5) Definiciones esenciales
- **Overflow:** número demasiado grande para representarse (→ `inf`).
- **Underflow:** número tan pequeño que se aproxima a 0.

#### 6) Explicación didáctica
- Softmax es sensible al rango numérico por el `exp`. El objetivo es mantener exponentes en un rango seguro.

#### 7) Ejemplo modelado
- `z=[1000,1001,1002]` es un caso “conceptualmente fácil” (debería ganar la última clase) pero numéricamente peligroso.

#### 8) Práctica guiada
- ¿Cuál de estos casos produce `inf` y cuál produce `0`?
  - `exp(1000)`, `exp(-1000)`.

#### 9) Práctica independiente
- Explica por qué aunque el resultado final de softmax esté en `(0,1)`, el cálculo intermedio puede romper.

#### 10) Autoevaluación
- ¿Qué dos operaciones generan `NaN` típicamente en este contexto?

#### 11) Errores comunes
- “Solucionar” con `epsilon` dentro de `exp` (no resuelve overflow).

#### 12) Retención
- Señal roja: **si ves logits con magnitud ~1e3, softmax naive es sospechoso**.

#### 13) Diferenciación
- Avanzado: discute por qué el problema empeora con batch grande y/o modelos profundos.

#### 14) Recursos
- IEEE-754, límites de `float64/float32` (intuitivo: `exp(88)` ya es enorme en `float32`).

#### 15) Nota docente
- Pide al alumno que describa el fallo como “operación indefinida” (`inf/inf`, `0/0`).

</details>

#### 5.3 Log-Sum-Exp Trick (Estabilidad Numérica)

```text
TRUCO: softmax(z) = softmax(z - max(z))

Demostración:
    softmax(z - c)ᵢ = exp(zᵢ - c) / Σⱼ exp(zⱼ - c)
                    = exp(zᵢ)·exp(-c) / Σⱼ exp(zⱼ)·exp(-c)
                    = exp(zᵢ) / Σⱼ exp(zⱼ)
                    = softmax(z)ᵢ

Al restar max(z), todos los exponentes son ≤ 0, evitando overflow.
```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 5.3: Log-Sum-Exp Trick</strong></summary>

#### 1) Metadatos
- **Título:** Shift por `max(z)` para hacer `exp` seguro sin cambiar softmax
- **ID (opcional):** `M04-T05_3`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 5.2

#### 2) Objetivos
- Probar que restar una constante no cambia softmax.
- Entender por qué usar `max(z)` es una elección óptima simple.
- Reconocer el patrón “log-sum-exp” como herramienta general.

#### 3) Relevancia
- Es la base de implementaciones estables de softmax/log-softmax y cross-entropy desde logits.

#### 4) Mapa conceptual mínimo
- invariancia por shift → elegir `c=max(z)` → exponentes ≤ 0 → sin overflow

#### 5) Definiciones esenciales
- **Shift/centrado:** `z' = z - c`.
- **log-sum-exp:** `log(Σ exp(z))` computado de forma estable.

#### 6) Explicación didáctica
- Restar `max(z)` hace que el mayor exponente sea `exp(0)=1` y el resto `≤1`.

#### 7) Ejemplo modelado
- Si `z=[1000,1001,1002]`, entonces `z'=[-2,-1,0]` (seguro) y softmax no cambia.

#### 8) Práctica guiada
- Repite la demostración de invariancia para `softmax(z-c)` con símbolos.

#### 9) Práctica independiente
- ¿Por qué no basta con restar un número fijo como 100? ¿Qué hace especial a `max(z)`?

#### 10) Autoevaluación
- ¿Qué garantiza que `exp(z')` no overflow si `max(z')=0`?

#### 11) Errores comunes
- Restar el `max` sin `keepdims=True` y romper shapes en batch.

#### 12) Retención
- Mantra: **softmax es invariante a shift; usa `max` para estabilidad**.

#### 13) Diferenciación
- Avanzado: conecta con `log_softmax(z)=z-logsumexp(z)`.

#### 14) Recursos
- Búsqueda: “logsumexp trick” (patrón general en modelos probabilísticos).

#### 15) Nota docente
- Pide al alumno que identifique dónde aparece la misma idea en `log_softmax`.

</details>

#### 5.4 Implementación Numéricamente Estable

```python
import numpy as np  # NumPy: necesario para exp/log/max/sum en softmax estable

def softmax(z: np.ndarray) -> np.ndarray:  # convierte logits a probabilidades (softmax estable)
    """
    Softmax numéricamente estable usando Log-Sum-Exp trick.

    Truco: Restar el máximo para evitar overflow en exp()
    softmax(z) = softmax(z - max(z))

    Args:
        z: Logits (scores antes de activación)

    Returns:
        Probabilidades que suman 1
    """
    # Log-Sum-Exp trick: restar el máximo
    z_stable = z - np.max(z, axis=-1, keepdims=True)  # Shift: ancla numérica por fila (mantiene invariancia)

    exp_z = np.exp(z_stable)  # exp() seguro: valores ≤ 0 evitan overflow
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)  # Normaliza para que sumen 1 (distribución)


def log_softmax(z: np.ndarray) -> np.ndarray:  # calcula log(softmax(z)) de forma estable
    """
    Log-Softmax estable (útil para Cross-Entropy).

    log(softmax(z)) calculado de forma estable.
    Evita calcular softmax primero y luego log (pierde precisión).
    """
    z_stable = z - np.max(z, axis=-1, keepdims=True)  # Mismo shift: reduce rango numérico
    log_sum_exp = np.log(np.sum(np.exp(z_stable), axis=-1, keepdims=True))  # log(sum(exp(z_stable))) por fila
    return z_stable - log_sum_exp  # log_softmax = z - logsumexp(z)


def categorical_cross_entropy_from_logits(y_true: np.ndarray, logits: np.ndarray) -> float:  # CCE estable usando logits
    """
    Cross-entropy estable usando logits directamente.

    Evita calcular softmax explícito.
    Útil cuando entrenas modelos y quieres estabilidad.
    """
    log_probs = log_softmax(logits)  # Convierte logits a log-probabilidades estables
    return -np.mean(np.sum(y_true * log_probs, axis=1))  # NLL promedio: -E[log p(clase correcta)]


# ============================================================
# DEMOSTRACIÓN: Por qué el trick es necesario
# ============================================================

def demo_numerical_stability():  # muestra overflow/NaN en softmax ingenuo vs estable
    """Muestra por qué necesitamos el Log-Sum-Exp trick."""

    # Caso peligroso: logits muy grandes
    z_dangerous = np.array([1000.0, 1001.0, 1002.0])  # Logits extremos: exp() desborda sin protección

    # Sin el trick (INCORRECTO)
    def softmax_naive(z):  # softmax ingenuo (puede overflow con logits grandes)
        exp_z = np.exp(z)  # ¡Overflow! exp(1000) -> inf
        return exp_z / np.sum(exp_z)  # inf/inf -> NaN (resultado no es una distribución válida)

    # Con el trick (CORRECTO)
    def softmax_stable(z):  # softmax estable (resta max antes de exp)
        z_stable = z - np.max(z)  # Restar max: invariancia de softmax pero con estabilidad
        exp_z = np.exp(z_stable)  # Ahora exp() es seguro (valores ≤ 0)
        return exp_z / np.sum(exp_z)  # Normaliza a suma 1

    print("Logits peligrosos:", z_dangerous)  # muestra logits extremos que rompen exp sin protección
    print()  # línea en blanco: separa secciones en la salida

    # Naive (falla)
    import warnings  # módulo para controlar/ignorar warnings durante la demo
    with warnings.catch_warnings():  # captura warnings (overflow) para no ensuciar la salida
        warnings.simplefilter("ignore")  # Ignora warning esperado por overflow (demo)
        result_naive = softmax_naive(z_dangerous)  # Resultado ingenuo (suele contener NaN)
        print(f"Softmax NAIVE: {result_naive}")  # Imprime el vector (para ver NaN/inf)
        print(f"  → Suma: {np.sum(result_naive)} (debería ser 1.0)")  # Verifica que no normaliza bien

    # Estable (funciona)
    result_stable = softmax_stable(z_dangerous)  # Resultado estable: finito y normalizado
    print(f"\nSoftmax ESTABLE: {result_stable}")  # Imprime el vector estable
    print(f"  → Suma: {np.sum(result_stable):.6f} ✓")  # Suma ~1 confirma distribución válida

demo_numerical_stability()  # ejecuta la demo de estabilidad numérica


# Ejemplo: Clasificación multiclase (dígitos 0-9)
logits = np.array([2.0, 1.0, 0.1, -1.0, 3.0, 0.5, -0.5, 1.5, 0.0, -2.0])  # logits de ejemplo para 10 clases
probs = softmax(logits)  # convierte logits a probabilidades (softmax)

print("\nLogits → Probabilidades:")  # encabezado: muestra mapeo logit→probabilidad
for i, (l, p) in enumerate(zip(logits, probs)):  # recorre clases y sus probabilidades
    print(f"  Clase {i}: logit={l:+.1f} → prob={p:.3f}")  # imprime probabilidad por clase
print(f"\nSuma de probabilidades: {np.sum(probs):.6f}")  # sanity check: suma debe ser ~1
print(f"Clase predicha: {np.argmax(probs)}")  # predicción: clase con probabilidad máxima

```

<details open>
<summary><strong>📌 Complemento pedagógico — Tema 5.4: Implementación Numéricamente Estable</strong></summary>

#### 1) Metadatos
- **Título:** Implementar `softmax`/`log_softmax` sin NaN (y por qué funciona)
- **ID (opcional):** `M04-T05_4`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 5.2–5.3

#### 2) Objetivos
- Implementar softmax estable con `z - max(z)`.
- Entender por qué `log_softmax` es preferible a `np.log(softmax(z))`.
- Verificar propiedades: probabilidades finitas y suma 1.

#### 3) Relevancia
- Esta es una de las fuentes más comunes de `loss=NaN` en entrenamiento real (overflow/underflow en `exp`).

#### 4) Mapa conceptual mínimo
- logits `z` → shift `z-max(z)` → `exp` seguro → normalizar → softmax
- logits `z` → `log_softmax(z)=z-logsumexp(z)` → CE estable

#### 5) Definiciones esenciales
- **Shift invariante:** restar una constante a todos los logits no cambia softmax.
- **log-softmax:** log-probabilidades computadas sin pasar por probabilidades intermedias inestables.

#### 6) Explicación didáctica
- Restar `max(z)` “centra” la fila para que el mayor exponente sea `exp(0)=1` y el resto `≤1`.

#### 7) Ejemplo modelado
- El demo con logits grandes muestra que la versión naive puede producir `inf/inf → NaN`, mientras que la estable no.

#### 8) Práctica guiada
- Añade checks:
  - `assert np.all(np.isfinite(softmax(z)))`
  - `assert np.allclose(np.sum(softmax(z)), 1.0)` (vector) o por fila (batch).

#### 9) Práctica independiente
- Implementa soporte batch `(n_samples, n_classes)` y verifica que `axis=-1` es el correcto.

#### 10) Autoevaluación
- ¿Por qué `argmax(softmax(z)) == argmax(z)` aunque cambien los valores?

#### 11) Errores comunes
- Olvidar `keepdims=True` y romper broadcasting.
- Normalizar sobre el eje incorrecto.

#### 12) Retención
- Regla: **si ves `exp`, piensa en estabilidad y en restar `max`**.

#### 13) Diferenciación
- Avanzado: compara el comportamiento en `float32` vs `float64`.

#### 14) Recursos
- Patrón: “log-sum-exp trick” (idea general en modelos probabilísticos).

#### 15) Nota docente
- Pide al alumno explicar el fallo del naive como “operación indefinida” (`inf/inf`, `0/0`).
</details>

#### 5.5 Categorical Cross-Entropy (Multiclase)

```python
def categorical_cross_entropy(y_true: np.ndarray,  # labels one-hot (n_samples, n_classes)
                               y_pred: np.ndarray) -> float:  # probabilidades predichas (softmax) (n_samples, n_classes)
    """
    Loss para clasificación multiclase.

    Args:
        y_true: One-hot encoded labels (n_samples, n_classes)
        y_pred: Probabilidades softmax (n_samples, n_classes)

    Returns:
        Loss promedio
    """
    epsilon = 1e-15  # estabilidad numérica: evita log(0)
    # Solo cuenta la clase correcta (donde y_true=1)
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))  # loss promedio: -mean(sum(y*log(p)))


# Ejemplo
y_true = np.array([  # labels one-hot para 2 muestras
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Clase 4
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Clase 0
])  # cierra array de etiquetas y_true

y_pred = np.array([  # probabilidades predichas (cada fila suma 1)
    softmax(np.array([0, 0, 0, 0, 5, 0, 0, 0, 0, 0])),  # Confiado en 4
    softmax(np.array([3, 1, 0, 0, 0, 0, 0, 0, 0, 0])),  # Confiado en 0
])  # cierra array de probabilidades y_pred

loss = categorical_cross_entropy(y_true, y_pred)  # calcula la loss CCE para el ejemplo
print(f"Categorical Cross-Entropy: {loss:.4f}")  # imprime la loss para inspección

```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 5.5: Categorical Cross-Entropy (Multiclase)</strong></summary>

#### 1) Metadatos
- **Título:** Implementar CCE con one-hot y entender qué suma realmente
- **ID (opcional):** `M04-T05_5`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 4.5 + 5.4

#### 2) Objetivos
- Implementar CCE con protección numérica (`epsilon`).
- Entender por qué, con one-hot, la loss selecciona la probabilidad de la clase correcta.
- Conectar CCE con NLL/MLE: minimizar CCE ≡ maximizar likelihood categórica.

#### 3) Relevancia
- CCE es la función de pérdida estándar en clasificación multiclase con softmax.

#### 4) Mapa conceptual mínimo
- one-hot `y` → selecciona clase correcta → `-log(p_correcta)` → promedio en batch

#### 5) Definiciones esenciales
- **One-hot:** vector con un 1 en la clase correcta y 0 en las demás.
- **`epsilon`:** evita `log(0)` cuando `p` llega a 0 por redondeo.

#### 6) Explicación didáctica
- El término `np.sum(y_true * log(p), axis=1)` actúa como “selector” de la clase correcta.

#### 7) Ejemplo modelado
- Si `p_correcta` pasa de `0.9` a `0.1`, la loss sube fuertemente (penaliza confianza equivocada).

#### 8) Práctica guiada
- Calcula a mano una muestra: `L=-log(p_correcta)` y valida con el print del código.

#### 9) Práctica independiente
- Implementa la versión con índices (`y_true` como clase entera) y compara resultados.

#### 10) Autoevaluación
- ¿Por qué `epsilon` arregla `log(0)` pero no corrige overflow que ocurre antes en softmax naive?

#### 11) Errores comunes
- Pasar logits a una CE que espera probabilidades.
- No verificar que `y_pred` suma 1 por fila.

#### 12) Retención
- Fórmula: **CCE = -promedio(log(probabilidad de la clase correcta))**.

#### 13) Diferenciación
- Avanzado: discute label smoothing y cómo cambia la suma `Σ y_k log(p_k)`.

#### 14) Recursos
- Conexión directa con el tema 4.5 (NLL y MLE).

#### 15) Nota docente
- Pregunta de control: “¿qué línea hace que solo cuente la clase correcta?”
</details>

## 🎯 Ejercicios por tema (progresivos) + Soluciones

Reglas:

- **Intenta primero** sin mirar la solución.
- **Timebox sugerido:** 15–30 min por ejercicio.
- **Éxito mínimo:** tu solución debe pasar los `assert`.

---

### Ejercicio 4.1: Probabilidad condicional (P(A|B)) y consistencia

#### Enunciado

1) **Básico**

- Dado un conjunto de conteos de eventos, calcula `P(A)`, `P(B)` y `P(A ∩ B)`.

2) **Intermedio**

- Calcula `P(A|B) = P(A∩B)/P(B)` y verifica que está en `[0,1]`.

3) **Avanzado**

- Verifica que `P(A∩B) = P(A|B)·P(B)`.

#### Solución

```python
import numpy as np  # Importar librería para computación numérica

# Simulación con conteos (dataset pequeño)
n = 100  # Tamaño total del dataset
count_A = 40  # Conteo de eventos A
count_B = 50  # Conteo de eventos B
count_A_and_B = 20  # Conteo de eventos A y B simultáneamente

P_A = count_A / n  # Calcular probabilidad de A
P_B = count_B / n  # Calcular probabilidad de B
P_A_and_B = count_A_and_B / n  # Calcular probabilidad conjunta

P_A_given_B = P_A_and_B / P_B  # Calcular probabilidad condicional P(A|B)

assert 0.0 <= P_A <= 1.0  # Verificar que P_A esté en [0,1]
assert 0.0 <= P_B <= 1.0  # Verificar que P_B esté en [0,1]
assert 0.0 <= P_A_given_B <= 1.0  # Verificar que P(A|B) esté en [0,1]
assert np.isclose(P_A_and_B, P_A_given_B * P_B)  # Verificar regla del producto
```

---

### Ejercicio 4.2: Bayes en modo clasificador (posterior sin normalizar)

#### Enunciado

1) **Básico**

- Implementa el cálculo de posterior sin normalizar:
  - `score_spam = P(x|spam)·P(spam)`
  - `score_ham = P(x|ham)·P(ham)`

2) **Intermedio**

- Normaliza y obtén `P(spam|x)` y `P(ham|x)`.

3) **Avanzado**

- Verifica que las probabilidades normalizadas suman 1.

#### Solución

```python
import numpy as np  # Importar librería para computación numérica

P_spam = 0.3  # Probabilidad prior de spam
P_ham = 1.0 - P_spam  # Probabilidad prior de ham

P_x_given_spam = 0.8  # Verosimilitud P(x|spam)
P_x_given_ham = 0.1  # Verosimilitud P(x|ham)

score_spam = P_x_given_spam * P_spam  # Calcular score no normalizado para spam
score_ham = P_x_given_ham * P_ham  # Calcular score no normalizado para ham

Z = score_spam + score_ham  # Calcular constante de normalización
P_spam_given_x = score_spam / Z  # Calcular posterior P(spam|x)
P_ham_given_x = score_ham / Z  # Calcular posterior P(ham|x)

assert np.isclose(P_spam_given_x + P_ham_given_x, 1.0)  # Verificar que sumen 1
assert P_spam_given_x > P_ham_given_x  # Verificar que spam es más probable
```

---

### Ejercicio 4.3: Independencia (test empírico)

#### Enunciado

1) **Básico**

- Simula dos variables binarias independientes `A` y `B`.

2) **Intermedio**

- Estima `P(A)`, `P(B)`, `P(A∩B)` y verifica `P(A∩B) ≈ P(A)P(B)`.

3) **Avanzado**

- Simula un caso dependiente y verifica que la igualdad se rompe.

#### Solución

```python
import numpy as np  # Importar librería para computación numérica

np.random.seed(0)  # Fijar semilla para reproducibilidad
n = 20000  # Tamaño de muestra

# Independientes
A = (np.random.rand(n) < 0.4)  # Generar eventos A con P=0.4
B = (np.random.rand(n) < 0.5)  # Generar eventos B con P=0.5

P_A = A.mean()  # Calcular P(A)
P_B = B.mean()  # Calcular P(B)
P_A_and_B = (A & B).mean()  # Calcular P(A∩B)

assert abs(P_A_and_B - (P_A * P_B)) < 0.01  # Verificar independencia

# Dependientes: B es casi A
B_dep = (A | (np.random.rand(n) < 0.05))  # B depende de A
P_B_dep = B_dep.mean()  # Calcular P(B)
P_A_and_B_dep = (A & B_dep).mean()  # Calcular P(A∩B)

assert abs(P_A_and_B_dep - (P_A * P_B_dep)) > 0.02  # Verificar dependencia
```

---

### Ejercicio 4.4: MLE de Bernoulli ("fracción de heads")

#### Enunciado

1) **Básico**

- Genera muestras Bernoulli con `p_true`.

2) **Intermedio**

- Implementa el estimador MLE `p_hat = mean(x)`.

3) **Avanzado**

- Verifica que `p_hat` se aproxima a `p_true` con suficientes muestras.

#### Solución

```python
import numpy as np  # Importar librería para computación numérica

np.random.seed(1)  # Fijar semilla para reproducibilidad
p_true = 0.7  # Probabilidad verdadera
n = 5000  # Tamaño de muestra
x = (np.random.rand(n) < p_true).astype(float)  # Generar muestras Bernoulli

p_hat = float(np.mean(x))  # Estimar p mediante MLE (promedio)
assert abs(p_hat - p_true) < 0.02  # Verificar que estimación sea cercana
```

---

### Ejercicio 4.5: PDF Gaussiana univariada (sanity check)

#### Enunciado

1) **Básico**

- Implementa la PDF de una normal `N(μ,σ²)`.

2) **Intermedio**

- Verifica que para `N(0,1)` en `x=0` la densidad ≈ `0.39894228`.

3) **Avanzado**

- Verifica que `pdf(x)` es simétrica: `pdf(a) == pdf(-a)` cuando `μ=0`.

#### Solución

```python
import numpy as np  # Importar librería para computación numérica

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:  # Definir función PDF gaussiana univariada
    x = np.asarray(x, dtype=float)  # Convertir x a array numpy
    sigma = float(sigma)  # Convertir sigma a float
    assert sigma > 0  # Verificar que sigma sea positivo
    z = (x - mu) / sigma  # Calcular z-score
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * z**2)  # Calcular PDF


val0 = gaussian_pdf(np.array([0.0]), mu=0.0, sigma=1.0)[0]  # Calcular PDF en x=0
assert np.isclose(val0, 0.39894228, atol=1e-4)  # Verificar valor ~1/√(2π)

a = 1.7  # Definir valor para prueba de simetría
assert np.isclose(  # Verificar simetría del PDF
    gaussian_pdf(np.array([a]), 0.0, 1.0)[0],  # PDF en x=a
    gaussian_pdf(np.array([-a]), 0.0, 1.0)[0],  # PDF en x=-a
    rtol=1e-12,  # Tolerancia relativa
    atol=1e-12,  # Tolerancia absoluta
)  # El PDF gaussiano es simétrico
```

---

### Ejercicio 4.6: Gaussiana multivariada (2D) + covarianza válida

#### Enunciado

1) **Básico**

- Implementa la densidad `N(μ, Σ)` en 2D.

2) **Intermedio**

- Para `μ=0` y `Σ=I`, verifica que `pdf(0) = 1/(2π)`.

3) **Avanzado**

- Verifica que `Σ` es definida positiva (eigenvalores > 0) antes de invertir.

4) **Bonus (elipse de covarianza)**

- Para una matriz de covarianza no diagonal, genera puntos en la elipse de covarianza 2D para una escala `k` (por ejemplo, `k=2`) usando descomposición en eigenvalues/eigenvectors, y verifica que satisfacen `(x-μ)^T Σ^{-1} (x-μ) ≈ k^2`.

#### Solución

```python
import numpy as np  # Importar librería para computación numérica

def multivariate_gaussian_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:  # Definir función de densidad de probabilidad gaussiana multivariada
    x = np.asarray(x, dtype=float)  # Convertir x a array numpy
    mu = np.asarray(mu, dtype=float)  # Convertir mu a array numpy
    cov = np.asarray(cov, dtype=float)  # Convertir cov a array numpy
    d = x.shape[0]  # Obtener dimensión

    assert mu.shape == (d,)  # Verificar que mu tenga dimensión correcta
    assert cov.shape == (d, d)  # Verificar que cov sea matriz cuadrada
    assert np.allclose(cov, cov.T)  # Verificar que cov sea simétrica
    eigvals = np.linalg.eigvals(cov)  # Calcular eigenvalores
    assert np.all(eigvals > 0)  # Verificar que cov sea definida positiva

    diff = x - mu  # Calcular diferencia x - mu
    inv = np.linalg.inv(cov)  # Calcular inversa de cov
    det = np.linalg.det(cov)  # Calcular determinante de cov
    norm = 1.0 / (np.sqrt(((2.0 * np.pi) ** d) * det))  # Calcular factor de normalización
    expo = -0.5 * float(diff.T @ inv @ diff)  # Calcular exponente
    return float(norm * np.exp(expo))  # Devolver valor de PDF


mu = np.array([0.0, 0.0])  # Definir media
cov = np.eye(2)  # Definir covarianza (identidad)
pdf0 = multivariate_gaussian_pdf(np.array([0.0, 0.0]), mu, cov)  # Calcular PDF en origen
assert np.isclose(pdf0, 1.0 / (2.0 * np.pi), atol=1e-6)  # Verificar valor teórico
assert pdf0 > 0.0  # Verificar que sea positivo

def covariance_ellipse_points(mu: np.ndarray, cov: np.ndarray, k: float = 2.0, n: int = 200) -> np.ndarray:  # Definir función para generar puntos en elipse de covarianza
    mu = np.asarray(mu, dtype=float)  # Convertir mu a array numpy
    cov = np.asarray(cov, dtype=float)  # Convertir cov a array numpy
    assert mu.shape == (2,)  # Verificar que mu sea 2D
    assert cov.shape == (2, 2)  # Verificar que cov sea matriz 2x2
    assert np.allclose(cov, cov.T)  # Verificar que cov sea simétrica

    eigvals, eigvecs = np.linalg.eigh(cov)  # Calcular eigenvalores y eigenvectores
    assert np.all(eigvals > 0)  # Verificar que eigenvalores sean positivos

    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)  # Generar ángulos
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)  # Crear círculo unitario

    transform = eigvecs @ np.diag(np.sqrt(eigvals))  # Crear matriz de transformación
    pts = (mu.reshape(2, 1) + (k * transform @ circle)).T  # Transformar y trasladar puntos
    return pts  # Devolver puntos de la elipse


mu2 = np.array([0.0, 0.0])  # Definir media para elipse
cov2 = np.array([  # Definir covarianza no diagonal
    [2.0, 1.2],  # Varianza x=2.0, covarianza xy=1.2
    [1.2, 1.0],  # Covarianza yx=1.2, varianza y=1.0
], dtype=float)  # Matriz de covarianza 2x2
pts = covariance_ellipse_points(mu2, cov2, k=2.0, n=180)  # Generar puntos de elipse
inv2 = np.linalg.inv(cov2)  # Calcular inversa de covarianza

q = np.einsum('...i,ij,...j->...', pts - mu2, inv2, pts - mu2)  # Calcular forma cuadrática
assert np.allclose(q, 4.0, atol=1e-6)  # Verificar que puntos satisfagan (x-μ)^T Σ^{-1} (x-μ) ≈ k^2
```

---

### Ejercicio 4.6B: Visualización (Gaussiana 2D variando covarianza) (OBLIGATORIO)

#### Enunciado

Construye una visualización que haga **visible** la covarianza:

1) **Básico**

- Crea un grid 2D y grafica contornos (`contour`) de `N(μ, Σ)`.

2) **Intermedio**

- Compara al menos 3 covarianzas:
  - isotrópica (`Σ = I`)
  - elíptica (varianzas distintas)
  - correlacionada (términos fuera de la diagonal)

3) **Avanzado**

- Sobre cada plot, dibuja la **elipse de covarianza** para `k=2` y verifica que sus puntos cumplen `(x-μ)^T Σ^{-1} (x-μ) ≈ k^2`.

#### Solución

```python
import numpy as np  # NumPy: grid 2D, álgebra lineal y evaluación vectorizada
import matplotlib.pyplot as plt  # Matplotlib: contornos 2D y trazado de elipses


def multivariate_gaussian_pdf_grid(xx: np.ndarray, yy: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:  # Definir función para evaluar PDF en grid 2D
    # xx, yy: grids 2D (H,W) típicamente creados con np.meshgrid
    xx = np.asarray(xx, dtype=float)  # Asegura dtype float para evitar ints en exp/log
    yy = np.asarray(yy, dtype=float)  # Mismo contrato: (H,W)
    mu = np.asarray(mu, dtype=float)  # mu:(2,) media 2D
    cov = np.asarray(cov, dtype=float)  # cov:(2,2) covarianza

    assert mu.shape == (2,)  # Sanidad: trabajamos en 2D
    assert cov.shape == (2, 2)  # Sanidad: covarianza 2D
    assert np.allclose(cov, cov.T)  # Debe ser simétrica

    eigvals = np.linalg.eigvalsh(cov)  # Eigenvalues reales para matriz simétrica (más estable)
    assert np.all(eigvals > 0.0)  # Covarianza debe ser definida positiva (invertible)

    inv = np.linalg.inv(cov)  # Σ^{-1} para la forma cuadrática
    det = np.linalg.det(cov)  # |Σ| para el coeficiente de normalización

    pos = np.dstack([xx, yy])  # pos:(H,W,2) apila coordenadas (x,y) en el último eje
    diff = pos - mu.reshape(1, 1, 2)  # diff:(H,W,2) resta μ por broadcasting

    quad = np.einsum('...i,ij,...j->...', diff, inv, diff)  # (x-μ)^T Σ^{-1} (x-μ) para cada punto del grid
    expo = -0.5 * quad  # Exponente de la Gaussiana

    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))  # Normalización en 2D: 1 / (2π sqrt(|Σ|))
    pdf = norm * np.exp(expo)  # pdf:(H,W) densidad evaluada en el grid

    return pdf  # Devuelve matriz 2D lista para contour/contourf


def covariance_ellipse_points(mu: np.ndarray, cov: np.ndarray, k: float = 2.0, n: int = 200) -> np.ndarray:  # Definir función para generar puntos en elipse de covarianza
    # Esta función genera puntos sobre la elipse: (x-μ)^T Σ^{-1} (x-μ) = k^2
    mu = np.asarray(mu, dtype=float)  # mu:(2,) asegura float
    cov = np.asarray(cov, dtype=float)  # cov:(2,2) asegura float

    assert mu.shape == (2,)  # Solo soportamos 2D para visualización
    assert cov.shape == (2, 2)  # Covarianza 2D
    assert np.allclose(cov, cov.T)  # Simetría

    eigvals, eigvecs = np.linalg.eigh(cov)  # Descomposición simétrica: cov = Q Λ Q^T
    assert np.all(eigvals > 0.0)  # PD: eigenvalues positivos

    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)  # Parámetro angular para un círculo unitario
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)  # circle:(2,n) círculo unitario

    transform = eigvecs @ np.diag(np.sqrt(eigvals))  # Transformación que mapea círculo -> elipse base (k=1)
    pts = (mu.reshape(2, 1) + (k * transform @ circle)).T  # pts:(n,2) traslada por μ y escala por k

    return pts  # Puntos listos para plt.plot(pts[:,0], pts[:,1])


mu = np.array([0.0, 0.0], dtype=float)  # μ:(2,) centramos en el origen para comparar solo Σ

covs = [  # Definir lista de matrices de covarianza para visualizar
    np.eye(2, dtype=float),  # Σ1: isotrópica (círculo)
    np.array([[3.0, 0.0], [0.0, 1.0]], dtype=float),  # Σ2: elíptica (varianza distinta por eje)
    np.array([[2.0, 1.2], [1.2, 1.0]], dtype=float),  # Σ3: correlacionada (término fuera de diagonal)
]  # Lista de covarianzas a comparar

labels = [  # Definir etiquetas para los subgráficos
    "Σ = I (isotrópica)",  # Texto para subplot 1
    "Σ = diag(3,1) (elíptica)",  # Texto para subplot 2
    "Σ con correlación (elipse rotada)",  # Texto para subplot 3
]  # Etiquetas

grid = np.linspace(-4.0, 4.0, 250)  # Rejilla 1D para construir el grid 2D
xx, yy = np.meshgrid(grid, grid)  # xx,yy:(H,W) coordenadas del plano

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)  # 1 fila, 3 columnas

for ax, cov, title in zip(axes, covs, labels):  # Iteramos por cada Σ y su eje
    Z = multivariate_gaussian_pdf_grid(xx, yy, mu, cov)  # Z:(H,W) densidad en el plano

    ax.contour(xx, yy, Z, levels=10)  # Contornos: líneas de igual densidad

    pts = covariance_ellipse_points(mu, cov, k=2.0, n=240)  # pts:(n,2) elipse k=2
    ax.plot(pts[:, 0], pts[:, 1])  # Dibuja la elipse encima de los contornos

    inv = np.linalg.inv(cov)  # Σ^{-1} para verificar la ecuación cuadrática
    q = np.einsum('...i,ij,...j->...', pts - mu, inv, pts - mu)  # q:(n,) valor de (x-μ)^T Σ^{-1} (x-μ)
    assert np.allclose(q, 4.0, atol=1e-6)  # Debe ser ≈ k^2 = 4 si la elipse es correcta

    ax.set_title(title)  # Título por subplot
    ax.set_aspect('equal', 'box')  # Aspect ratio 1:1 para que la elipse no se distorsione
    ax.set_xlabel('x1')  # Eje x
    ax.set_ylabel('x2')  # Eje y

plt.savefig('gaussian_covariance_contours.png', dpi=160)  # Guarda la figura (útil para reportes)
```

---

### Ejercicio 4.7: Log-Sum-Exp y log-softmax estable (OBLIGATORIO)

#### Enunciado

1) **Básico**

- Implementa `logsumexp(z)` de forma estable (restando `max(z)`).

2) **Intermedio**

- Implementa `log_softmax(z) = z - logsumexp(z)`.

3) **Avanzado**

- Verifica que `sum(exp(log_softmax(z))) == 1` y que no hay `inf` con logits grandes.

#### Solución

```python
import numpy as np  # NumPy: arrays, exp/log y validación numérica

def logsumexp(z: np.ndarray) -> float:  # Definir función log-sum-exp numéricamente estable
    z = np.asarray(z, dtype=float)  # Asegura float para que exp/log sean numéricamente consistentes
    m = np.max(z)  # m = max(z) sirve como “ancla” para evitar overflow en exp
    return float(m + np.log(np.sum(np.exp(z - m))))  # Log-Sum-Exp: m + log(sum(exp(z-m)))


def log_softmax(z: np.ndarray) -> np.ndarray:  # Definir función log-softmax numéricamente estable
    z = np.asarray(z, dtype=float)  # Asegura float y copia segura
    return z - logsumexp(z)  # log_softmax(z) = z - log(sum(exp(z)))


z = np.array([1000.0, 0.0, -1000.0])  # Logits extremos para estresar estabilidad numérica
lsm = log_softmax(z)  # lsm:(3,) log-probabilidades estables
probs = np.exp(lsm)  # Convertimos a probabilidades (deben ser finitas)
assert np.isfinite(lsm).all()  # No debe haber NaN/inf en log-probabilidades
assert np.isfinite(probs).all()  # No debe haber NaN/inf en probabilidades
assert np.isclose(np.sum(probs), 1.0)  # Las probabilidades deben sumar 1
```

#### Solución (NaN trap: naive vs estable + verificación) (OBLIGATORIO)

```python
import numpy as np  # NumPy: exp/log y validación numérica
import warnings  # warnings: suprimir warnings esperados en el caso naïve (overflow)


def softmax_naive(z: np.ndarray) -> np.ndarray:  # Implementación ingenua (propensa a overflow/underflow)
    z = np.asarray(z, dtype=float)  # Asegura float para que exp opere en floats
    exp_z = np.exp(z)  # ¡Peligro! exp(1000) -> inf (overflow)
    return exp_z / np.sum(exp_z)  # Normaliza (pero si hay inf/0 puede producir NaN)


def softmax_stable(z: np.ndarray) -> np.ndarray:  # Softmax estable: aplica el Log-Sum-Exp trick
    z = np.asarray(z, dtype=float)  # Convierte a float (contrato)
    z_shift = z - np.max(z)  # Restar max(z) no cambia softmax pero evita overflow
    exp_z = np.exp(z_shift)  # Ahora exp() recibe valores <= 0 (seguro)
    return exp_z / np.sum(exp_z)  # Normaliza para que sum(p)=1


z_big = np.array([1000.0, 1001.0, 1002.0])  # Logits peligrosos (magnitudes enormes)

with warnings.catch_warnings():  # Contexto para que el notebook/terminal no se llene de warnings
    warnings.simplefilter("ignore")  # Suprimimos RuntimeWarning por overflow (esperado aquí)
    p_naive = softmax_naive(z_big)  # Resultado ingenuo (típicamente NaN)

naive_ok = np.isfinite(p_naive).all() and np.isclose(np.sum(p_naive), 1.0)  # Criterio de “distribución válida”
assert not naive_ok  # Debe fallar: aquí demostramos el NaN/inf trap

p_stable = softmax_stable(z_big)  # Softmax estable (debe funcionar)
assert np.isfinite(p_stable).all()  # No debe haber NaN/inf
assert np.isclose(np.sum(p_stable), 1.0)  # Debe sumar 1
assert np.argmax(p_stable) == np.argmax(z_big)  # Debe preservar el orden de logits
```

---

### Ejercicio 4.8: Softmax estable (invariancia a constantes)

#### Enunciado

1) **Básico**

- Implementa softmax estable: `exp(z-max)/sum(exp(z-max))`.

2) **Intermedio**

- Verifica que suma 1.

3) **Avanzado**

- Verifica invariancia: `softmax(z) == softmax(z + c)`.

#### Solución

```python
import numpy as np  # Importar librería para computación numérica

def softmax(z: np.ndarray) -> np.ndarray:  # Definir función softmax
    z = np.asarray(z, dtype=float)  # Convertir a array numpy
    z_shift = z - np.max(z)  # Restar máximo para estabilidad numérica
    expz = np.exp(z_shift)  # Calcular exponenciales
    return expz / np.sum(expz)  # Normalizar para que sume 1


z = np.array([2.0, 1.0, 0.0])  # Definir logits
p = softmax(z)  # Calcular softmax
assert np.isclose(np.sum(p), 1.0)  # Verificar que sume 1

c = 100.0  # Definir constante grande
p2 = softmax(z + c)  # Calcular softmax con constante añadida
assert np.allclose(p, p2)  # Verificar invarianza a constante
assert np.argmax(p) == np.argmax(z)  # Verificar que preserva orden
```

---

### Ejercicio 4.9: Binary Cross-Entropy estable (evitar log(0))

#### Enunciado

1) **Básico**

- Implementa BCE: `-mean(y log(p) + (1-y) log(1-p))`.

2) **Intermedio**

- Usa `clip`/`epsilon` para evitar `log(0)`.

3) **Avanzado**

- Verifica:
  - BCE cerca de 0 para predicciones casi perfectas.
  - BCE ≈ `-log(0.9)` cuando `y=1` y `p=0.9`.

#### Solución

```python
import numpy as np  # Importar librería para computación numérica

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:  # Definir función de entropía cruzada binaria
    y_true = np.asarray(y_true, dtype=float)  # Convertir y_true a array numpy
    y_pred = np.asarray(y_pred, dtype=float)  # Convertir y_pred a array numpy
    y_pred = np.clip(y_pred, eps, 1.0 - eps)  # Clipping para evitar log(0) y log(1)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))  # Calcular BCE


y_true = np.array([1.0, 0.0, 1.0, 0.0])  # Definir etiquetas verdaderas
y_pred_good = np.array([0.999, 0.001, 0.999, 0.001])  # Definir predicciones buenas
assert binary_cross_entropy(y_true, y_pred_good) < 0.01  # Verificar que pérdida sea pequeña

assert np.isclose(binary_cross_entropy(np.array([1.0]), np.array([0.9])), -np.log(0.9), atol=1e-12)  # Verificar caso simple
```

---

### Ejercicio 4.10: Categorical Cross-Entropy (multiclase) + one-hot

#### Enunciado

1) **Básico**

- Implementa CCE: `-mean(sum(y_true * log(y_pred)))`.

2) **Intermedio**

- Asegura que `y_pred` no contiene ceros (epsilon).

3) **Avanzado**

- Verifica que el loss baja cuando aumenta la probabilidad de la clase correcta.

#### Solución

```python
import numpy as np  # Importar librería para computación numérica

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:  # Definir función de entropía cruzada categórica
    y_true = np.asarray(y_true, dtype=float)  # Convertir y_true a array numpy
    y_pred = np.asarray(y_pred, dtype=float)  # Convertir y_pred a array numpy
    y_pred = np.clip(y_pred, eps, 1.0)  # Clipping para evitar log(0)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))  # Calcular pérdida promedio


y_true = np.array([[0, 1, 0], [1, 0, 0]], dtype=float)  # Definir etiquetas verdaderas (one-hot)
y_pred_bad = np.array([[0.34, 0.33, 0.33], [0.34, 0.33, 0.33]], dtype=float)  # Predicciones malas (casi uniformes)
y_pred_good = np.array([[0.05, 0.90, 0.05], [0.90, 0.05, 0.05]], dtype=float)  # Predicciones buenas (confiadas)

loss_bad = categorical_cross_entropy(y_true, y_pred_bad)  # Calcular pérdida para predicciones malas
loss_good = categorical_cross_entropy(y_true, y_pred_good)  # Calcular pérdida para predicciones buenas
assert loss_good < loss_bad  # Verificar que mejores predicciones tengan menor pérdida
```

---

### (Bonus) Ejercicio 4.11: Cadena de Markov (matriz de transición)

#### Enunciado

1) **Básico**

- Define una matriz de transición `P` (filas suman 1).

2) **Intermedio**

- Propaga una distribución `π_{t+1} = π_t P` y verifica que sigue siendo distribución.

3) **Avanzado**

- Encuentra una distribución estacionaria aproximada iterando muchas veces y verifica `π ≈ πP`.

4) **Bonus (potencias de matrices)**

- Verifica que iterar `π_{t+1} = π_t P` por `k` pasos coincide con `π_t P^k` usando `np.linalg.matrix_power`.

#### Solución

```python
import numpy as np  # Importar librería para computación numérica

P = np.array([  # Definir matriz de transición
    [0.9, 0.1],  # De estado 0: 90% queda en 0, 10% va a 1
    [0.2, 0.8],  # De estado 1: 20% va a 0, 80% queda en 1
], dtype=float)  # Matriz 2x2 de probabilidades
assert np.allclose(P.sum(axis=1), 1.0)  # Verificar que filas sumen 1

k = 50  # Número de pasos
pi0 = np.array([1.0, 0.0])  # Distribución inicial
pi = pi0.copy()  # Copiar distribución inicial
for _ in range(k):  # Iterar k pasos
    pi = pi @ P  # Actualizar distribución
    assert np.isclose(np.sum(pi), 1.0)  # Verificar que sume 1
    assert np.all(pi >= 0)  # Verificar que sea no negativa

pi_power = pi0 @ np.linalg.matrix_power(P, k)  # Calcular directamente
assert np.allclose(pi, pi_power, atol=1e-12)  # Verificar equivalencia

pi_star = pi.copy()  # Guardar distribución estacionaria
assert np.allclose(pi_star, pi_star @ P, atol=1e-6)  # Verificar estacionariedad
```

## 🔨 Entregables del Módulo

### E1: `probability.py`

```python
"""
Módulo de probabilidad esencial para ML.
Implementaciones desde cero con NumPy.
"""

import numpy as np  # Importar librería para computación numérica
from typing import Tuple  # Importar tipo para tuplas

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:  # Definir función PDF gaussiana univariada
    """Densidad de probabilidad Gaussiana univariada."""
    pass  # Implementar

def multivariate_gaussian_pdf(x: np.ndarray,  # Definir función PDF gaussiana multivariada
                               mu: np.ndarray,  # Vector de medias
                               cov: np.ndarray) -> float:  # Matriz de covarianza
    """Densidad de probabilidad Gaussiana multivariada."""
    pass  # Implementar

def mle_gaussian(data: np.ndarray) -> Tuple[float, float]:  # Definir función MLE para gaussiana
    """Estimación MLE de parámetros de Gaussiana."""
    pass  # Implementar

def softmax(z: np.ndarray) -> np.ndarray:  # Definir función softmax
    """Función softmax numéricamente estable."""
    pass  # Implementar

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # Definir función de entropía cruzada
    """Binary cross-entropy loss."""
    pass  # Implementar

def categorical_cross_entropy(y_true: np.ndarray,  # Definir función de entropía cruzada categórica
                               y_pred: np.ndarray) -> float:  # Predicciones de probabilidad
    """Categorical cross-entropy loss para multiclase."""
    pass  # Implementar
```

### E2: Tests

```python
# tests/test_probability.py
import numpy as np  # Importar librería para computación numérica
import pytest  # Importar framework de testing
from src.probability import (  # Importar funciones a probar
    gaussian_pdf, mle_gaussian, softmax,  # Funciones de probabilidad
    cross_entropy, categorical_cross_entropy  # Funciones de pérdida
)  # Cerrar importación

def test_gaussian_pdf_standard():  # Definir test para PDF gaussiano estándar
    """PDF de Gaussiana estándar en x=0 debe ser ~0.3989."""
    result = gaussian_pdf(np.array([0.0]), mu=0, sigma=1)  # Calcular PDF en x=0
    expected = 1 / np.sqrt(2 * np.pi)  # ~0.3989  # Valor esperado
    assert np.isclose(result[0], expected, rtol=1e-5)  # Verificar coincidencia

def test_softmax_sums_to_one():  # Definir test para suma de softmax
    """Softmax debe sumar 1."""
    z = np.random.randn(10)  # Generar logits aleatorios
    probs = softmax(z)  # Calcular softmax
    assert np.isclose(np.sum(probs), 1.0)  # Verificar que suma sea 1

def test_softmax_preserves_order():  # Definir test para orden de softmax
    """Mayor logit → mayor probabilidad."""
    z = np.array([1.0, 2.0, 3.0])  # Definir logits ordenados
    probs = softmax(z)  # Calcular softmax
    assert probs[2] > probs[1] > probs[0]  # Verificar orden preservado

def test_mle_gaussian_accuracy():  # Definir test para MLE gaussiano
    """MLE debe recuperar parámetros con suficientes datos."""
    np.random.seed(42)  # Fijar semilla para reproducibilidad
    true_mu, true_sigma = 10.0, 3.0  # Definir parámetros verdaderos
    data = np.random.normal(true_mu, true_sigma, size=10000)  # Generar datos

    est_mu, est_sigma = mle_gaussian(data)  # Estimar parámetros

    assert np.isclose(est_mu, true_mu, rtol=0.05)  # Verificar media estimada
    assert np.isclose(est_sigma, true_sigma, rtol=0.05)  # Verificar sigma estimado

def test_cross_entropy_perfect_prediction():  # Definir test para entropía cruzada
    """CE debe ser ~0 para predicciones perfectas."""
    y_true = np.array([1, 0, 1])  # Definir etiquetas verdaderas
    y_pred = np.array([0.999, 0.001, 0.999])  # Definir predicciones casi perfectas

    loss = cross_entropy(y_true, y_pred)  # Calcular pérdida
    assert loss < 0.01  # Verificar que pérdida sea pequeña
```

---

## 📊 Resumen Visual

```text
┌─────────────────────────────────────────────────────────────────┐
│  PROBABILIDAD PARA ML - MAPA CONCEPTUAL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TEOREMA DE BAYES                                               │
│       │                                                         │
│       ├──► Naive Bayes Classifier (Módulo 05)                   │
│       └──► Intuición de posterior vs prior                      │
│                                                                 │
│  DISTRIBUCIÓN GAUSSIANA                                         │
│       │                                                         │
│       ├──► GMM en Unsupervised (Módulo 06)                      │
│       ├──► Inicialización de pesos en DL (Módulo 07)            │
│       └──► Normalización de datos                               │
│                                                                 │
│  MAXIMUM LIKELIHOOD (MLE)                                       │
│       │                                                         │
│       ├──► Cross-Entropy Loss (Logistic Regression)             │
│       ├──► Categorical CE (Softmax + Multiclase)                │
│       └──► EM Algorithm en GMM                                  │
│                                                                 │
│  SOFTMAX                                                        │
│       │                                                         │
│       └──► Capa de salida en clasificación multiclase           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Conexiones con Otros Módulos

| Concepto | Dónde se usa |
|----------|--------------|
| Teorema de Bayes | Naive Bayes en Módulo 05 |
| Gaussiana | GMM en Módulo 06, inicialización en Módulo 07 |
| MLE | Derivación de Cross-Entropy en Módulo 05 |
| Softmax | Capa de salida en Módulo 07 |
| Cross-Entropy | Loss function principal en Módulo 05 y 07 |

---

## 🧩 Consolidación (errores comunes + debugging v5 + reto Feynman)

### Errores comunes

- **Confundir PDF con probabilidad:** en continuas, `f(x)` es densidad; la probabilidad requiere integrar en un intervalo.
- **`log(0)` en cross-entropy:** siempre usa `epsilon` o `np.clip`.
- **Overflow/underflow en `exp`:** aplica log-sum-exp / log-softmax.
- **MLE “mágico”:** si no puedes explicar por qué aparece la media, repite el worked example Bernoulli.

### Debugging / validación (v5)

- Cuando algo explote con `nan/inf`, revisa:
  - `np.log` sobre valores 0
  - `np.exp` sobre logits grandes
  - normalización incorrecta en probabilidades (que no suman 1)
- Registra hallazgos en `study_tools/DIARIO_ERRORES.md`.
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Reto Feynman (tablero blanco)

Explica en 5 líneas o menos:

1) ¿Por qué maximizar likelihood es equivalente a minimizar negative log-likelihood?
2) ¿Por qué el MLE de una moneda es “proporción de caras”?
3) ¿Qué significa `π_{t+1} = π_t P` y por qué es álgebra lineal?

## ✅ Checklist del Módulo

- [ ] Puedo explicar el Teorema de Bayes con un ejemplo
- [ ] Sé calcular la PDF de una Gaussiana a mano
- [ ] Entiendo por qué MLE da Cross-Entropy como loss
- [ ] Puedo explicar entropía y por qué `cross-entropy = H(y) + KL(y||p)`
- [ ] Puedo derivar por qué minimizar `KL(p_data||p_θ)` equivale a maximizar log-likelihood
- [ ] Implementé softmax numéricamente estable
- [ ] Puedo derivar el MLE de una Bernoulli (moneda) y explicarlo
- [ ] Puedo explicar qué es una Markov Chain y qué representa una matriz de transición
- [ ] Ejecuté `scripts/gmm_3_gaussians_contours.py` y entiendo los contornos de componentes vs mezcla
- [ ] Los tests de `probability.py` pasan

---

## 📖 Recursos Adicionales

### Videos
- [3Blue1Brown - Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- [StatQuest - Maximum Likelihood](https://www.youtube.com/watch?v=XepXtl9YKwc)
- [StatQuest - Gaussian Distribution](https://www.youtube.com/watch?v=rzFX5NWojp0)

### Lecturas
- Mathematics for ML, Cap. 6 (Probability)
- Pattern Recognition and ML (Bishop), Cap. 1-2

---

> 💡 **Nota Final:** Este módulo sigue siendo compacto comparado con un curso completo de probabilidad/estadística, pero aquí ya tienes el núcleo de Línea 1 y una “semilla” intencional para Línea 2 (estimación y Markov Chains).

---

**[← Módulo 03: Cálculo](03_CALCULO_MULTIVARIANTE.md)** | **[Módulo 05: Supervised Learning →](05_SUPERVISED_LEARNING.md)**
