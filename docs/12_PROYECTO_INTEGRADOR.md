# Módulo 10 - Proyecto Final: ML Pipeline Completo

> **🎯 Objetivo:** Construir un sistema de ML end-to-end que integre clasificación, clustering, análisis probabilístico y una red neuronal  
> **Fase:** Integración | **Demuestra dominio de los 6 cursos del Pathway**

---

## 🧠 ¿Qué Estamos Construyendo?

### El Proyecto Demuestra Dominio de las 2 Líneas del Pathway

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PROYECTO: SISTEMA DE CLASIFICACIÓN DE TEXTO CON ML COMPLETO               │
│   ─────────────────────────────────────────────────────────────             │
│                                                                             │
│   LÍNEA 1: MACHINE LEARNING (3 créditos)                                    │
│   ├── Clasificador Naive Bayes (ML Supervisado)                             │
│   ├── Clustering K-Means de documentos (ML No Supervisado)                  │
│   └── Red Neuronal MLP para clasificación (Deep Learning)                   │
│                                                                             │
│   LÍNEA 2: PROBABILIDAD Y ESTADÍSTICA (3 créditos)                          │
│   ├── Análisis Bayesiano (Fundamentos de Probabilidad)                      │
│   ├── Generador de texto con cadenas de Markov (MCMC)                       │
│   └── Evaluación estadística con intervalos de confianza (Estimación)       │
│                                                                             │
│   RESULTADO:                                                                │
│   Un pipeline que clasifica documentos usando 3 enfoques diferentes,        │
│   compara su rendimiento estadísticamente, y genera texto sintético.        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Explicación Técnica Progresiva

**Nivel 1 - Concepto:**
Un sistema que clasifica textos automáticamente usando diferentes técnicas de ML.

**Nivel 2 - Componentes:**
- **Preprocesamiento:** Tokenización, TF-IDF vectorization
- **Naive Bayes:** Clasificador probabilístico (usa Teorema de Bayes)
- **K-Means:** Agrupa documentos similares (no supervisado)
- **MLP:** Red neuronal multicapa (deep learning)
- **Markov Chain:** Genera texto sintético
- **Evaluación estadística:** Intervalos de confianza, cross-validation

**Nivel 3 - Flujo Completo:**
```
ENTRENAMIENTO:
datos → preprocesar → vectorizar → entrenar modelos → evaluar

PREDICCIÓN:
nuevo texto → vectorizar → predecir con cada modelo → comparar

GENERACIÓN:
corpus → construir cadena de Markov → generar texto nuevo

ANÁLISIS:
resultados → intervalos de confianza → comparación estadística
```

**Nivel 4 - Conexión con Pathway:**
| Componente | Curso del Pathway |
|------------|-------------------|
| Naive Bayes | Supervised Learning + Probability Foundations |
| K-Means | Unsupervised Algorithms |
| MLP | Introduction to Deep Learning |
| Markov Chain | Discrete-Time Markov Chains |
| Evaluación | Statistical Estimation |

---

## 🏗️ Arquitectura Detallada

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ML TEXT CLASSIFICATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                           │
│  │   DATOS      │                                                           │
│  │  (X, y)      │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        PREPROCESAMIENTO                              │   │
│  │  Tokenizer → TF-IDF Vectorizer → Train/Test Split                    │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                           │
│         ┌───────────────────────┼───────────────────────┐                   │
│         ▼                       ▼                       ▼                   │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐             │
│  │ NAIVE BAYES  │       │   K-MEANS    │       │     MLP      │             │
│  │ (Supervisado)│       │(No Supervis.)│       │(Deep Learn.) │             │
│  │              │       │              │       │              │             │
│  │ P(y|x)=      │       │ Clustering   │       │ Backprop     │             │
│  │ P(x|y)P(y)   │       │ + Asignar    │       │ + SGD        │             │
│  └──────┬───────┘       └──────┬───────┘       └──────┬───────┘             │
│         │                       │                     │                     │
│         └───────────────────────┼─────────────────────┘                     │
│                                 ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     EVALUACIÓN ESTADÍSTICA                           │   │
│  │  Accuracy, Precision, Recall, F1, Cross-Validation                   │   │
│  │  Intervalos de Confianza, Comparación de Modelos                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     GENERADOR MARKOV (Bonus)                         │   │
│  │  Corpus → Matriz de Transición → Generar Texto Sintético             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura de Archivos del Proyecto

```
ml-text-classifier/
├── src/
│   ├── __init__.py
│   │
│   ├── # PREPROCESAMIENTO (Módulos 04-06, 10-11)
│   ├── preprocessing.py         # Tokenizer, TF-IDF Vectorizer
│   ├── data_utils.py            # Train/test split, data loading
│   │
│   ├── # PROBABILIDAD (Módulos 19-21)
│   ├── probability.py           # Distribuciones, Bayes
│   ├── statistics.py            # MLE, intervalos de confianza
│   ├── markov.py                # Cadenas de Markov, generador de texto
│   │
│   ├── # MACHINE LEARNING (Módulos 22-23)
│   ├── naive_bayes.py           # Clasificador Naive Bayes
│   ├── kmeans.py                # Clustering K-Means
│   ├── evaluation.py            # Métricas, cross-validation
│   │
│   ├── # DEEP LEARNING (Módulo 24)
│   ├── neural_network.py        # MLP desde cero
│   ├── activations.py           # Sigmoid, ReLU, Softmax
│   ├── optimizers.py            # SGD, Adam
│   │
│   └── # INTEGRACIÓN
│   └── pipeline.py              # Pipeline completo que usa todo
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_probability.py
│   ├── test_naive_bayes.py
│   ├── test_kmeans.py
│   ├── test_neural_network.py
│   ├── test_markov.py
│   └── test_pipeline.py         # Tests de integración
│
├── data/
│   └── sample_dataset/          # Dataset de clasificación de texto
│       ├── train.csv
│       └── test.csv
│
├── notebooks/
│   └── demo.ipynb               # Demo interactivo del pipeline
│
├── docs/
│   ├── MODEL_COMPARISON.md      # Comparación estadística de modelos
│   └── PATHWAY_ALIGNMENT.md     # Cómo el proyecto cubre el Pathway
│
├── README.md                    # Documentación principal (inglés)
├── pyproject.toml
└── requirements-dev.txt
```

---

## 💻 Implementación Guiada: ML Pipeline

### Paso 1: Pipeline Principal

```python
# src/pipeline.py
"""ML Pipeline integrating all components for the Pathway project."""

from typing import Dict, List, Tuple
from .preprocessing import TFIDFVectorizer, train_test_split
from .naive_bayes import NaiveBayesClassifier
from .kmeans import KMeans
from .neural_network import NeuralNetwork
from .evaluation import accuracy, precision, recall, f1_score, cross_validate
from .statistics import confidence_interval
from .markov import MarkovTextGenerator


class MLPipeline:
    """Complete ML pipeline demonstrating Pathway competencies.
    
    This class integrates:
    - Naive Bayes (Supervised Learning + Probability)
    - K-Means (Unsupervised Learning)
    - Neural Network (Deep Learning)
    - Statistical evaluation (Statistical Estimation)
    - Markov text generation (Markov Chains)
    
    Example:
        >>> pipeline = MLPipeline()
        >>> pipeline.load_data(texts, labels)
        >>> pipeline.train_all_models()
        >>> results = pipeline.compare_models()
        >>> print(results)
        {'naive_bayes': 0.85, 'kmeans': 0.72, 'neural_net': 0.88}
    """
    
    def __init__(self, n_classes: int = 2) -> None:
        """Initialize pipeline with empty models."""
        self.vectorizer = TFIDFVectorizer()
        self.naive_bayes = NaiveBayesClassifier()
        self.kmeans = KMeans(n_clusters=n_classes)
        self.neural_net: NeuralNetwork = None  # Initialized after vectorization
        self.markov_generator = MarkovTextGenerator()
        
        self.n_classes = n_classes
        self.X_train: List[List[float]] = []
        self.X_test: List[List[float]] = []
        self.y_train: List[int] = []
        self.y_test: List[int] = []
    
    def load_data(
        self, 
        texts: List[str], 
        labels: List[int],
        test_size: float = 0.2
    ) -> None:
        """Load and preprocess data.
        
        Args:
            texts: List of text documents.
            labels: List of class labels.
            test_size: Fraction for test set.
        """
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, labels, test_size=test_size
        )
        
        # Initialize neural network with correct input size
        n_features = len(X[0]) if X else 0
        self.neural_net = NeuralNetwork(
            layer_sizes=[n_features, 64, 32, self.n_classes],
            activations=['relu', 'relu', 'softmax']
        )
        
        # Train Markov generator on all texts
        self.markov_generator.fit(texts)
    
    def train_all_models(self, verbose: bool = True) -> Dict[str, float]:
        """Train all three models and return training metrics.
        
        Returns:
            Dictionary with training accuracy for each model.
        """
        results = {}
        
        # 1. Train Naive Bayes (Módulo 19 + 22)
        if verbose:
            print("Training Naive Bayes...")
        self.naive_bayes.fit(self.X_train, self.y_train)
        nb_pred = self.naive_bayes.predict(self.X_train)
        results['naive_bayes_train'] = accuracy(self.y_train, nb_pred)
        
        # 2. Train K-Means (Módulo 23)
        if verbose:
            print("Training K-Means...")
        self.kmeans.fit(self.X_train)
        # Assign cluster labels to classes (majority voting)
        km_pred = self._kmeans_predict_with_labels(self.X_train, self.y_train)
        results['kmeans_train'] = accuracy(self.y_train, km_pred)
        
        # 3. Train Neural Network (Módulo 24)
        if verbose:
            print("Training Neural Network...")
        self.neural_net.fit(self.X_train, self.y_train, epochs=100)
        nn_pred = self.neural_net.predict(self.X_train)
        results['neural_net_train'] = accuracy(self.y_train, nn_pred)
        
        return results
    
    def evaluate_all_models(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all models on test set.
        
        Returns metrics and confidence intervals (Módulo 20).
        """
        results = {}
        
        for name, model in [
            ('naive_bayes', self.naive_bayes),
            ('neural_net', self.neural_net)
        ]:
            y_pred = model.predict(self.X_test)
            
            acc = accuracy(self.y_test, y_pred)
            prec = precision(self.y_test, y_pred)
            rec = recall(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Confidence interval for accuracy (Módulo 20)
            ci_low, ci_high = confidence_interval(
                successes=int(acc * len(self.y_test)),
                trials=len(self.y_test),
                confidence=0.95
            )
            
            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'accuracy_ci_95': (ci_low, ci_high)
            }
        
        return results
    
    def cross_validate_models(self, k: int = 5) -> Dict[str, List[float]]:
        """K-fold cross-validation for all models (Módulo 20)."""
        X_all = self.X_train + self.X_test
        y_all = self.y_train + self.y_test
        
        return {
            'naive_bayes': cross_validate(NaiveBayesClassifier, X_all, y_all, k),
            'neural_net': cross_validate(NeuralNetwork, X_all, y_all, k)
        }
    
    def generate_text(self, seed: str = None, length: int = 50) -> str:
        """Generate synthetic text using Markov chain (Módulo 21)."""
        return self.markov_generator.generate(seed=seed, length=length)
    
    def _kmeans_predict_with_labels(
        self, 
        X: List[List[float]], 
        y: List[int]
    ) -> List[int]:
        """Assign class labels to K-Means clusters via majority voting."""
        cluster_labels = self.kmeans.predict(X)
        
        # Map cluster -> most common class
        from collections import Counter
        cluster_to_class = {}
        for k in range(self.n_classes):
            cluster_points = [y[i] for i in range(len(y)) if cluster_labels[i] == k]
            if cluster_points:
                cluster_to_class[k] = Counter(cluster_points).most_common(1)[0][0]
            else:
                cluster_to_class[k] = 0
        
        return [cluster_to_class[c] for c in cluster_labels]
```

### Paso 2: Generador de Texto Markov (Módulo 21)

```python
# src/markov.py
"""Markov Chain text generator - demonstrates Discrete-Time Markov Chains."""

from typing import Dict, List, Optional
import random
from collections import defaultdict


class MarkovTextGenerator:
    """Text generator using Markov Chains.
    
    Demonstrates:
    - Discrete-time Markov Chains (Pathway course)
    - Transition probability matrices
    - Random sampling from distributions
    
    Example:
        >>> generator = MarkovTextGenerator(order=2)
        >>> generator.fit(["The quick brown fox", "The quick dog"])
        >>> generator.generate(seed="The quick", length=10)
        "The quick brown fox jumps..."
    """
    
    def __init__(self, order: int = 2) -> None:
        """Initialize with n-gram order."""
        self.order = order
        self.transitions: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.start_states: List[tuple] = []
    
    def fit(self, texts: List[str]) -> 'MarkovTextGenerator':
        """Build transition matrix from corpus.
        
        For each n-gram, count how often each word follows it.
        """
        for text in texts:
            words = text.split()
            if len(words) < self.order + 1:
                continue
            
            # Store starting states
            self.start_states.append(tuple(words[:self.order]))
            
            # Build transitions
            for i in range(len(words) - self.order):
                state = tuple(words[i:i + self.order])
                next_word = words[i + self.order]
                self.transitions[state][next_word] += 1
        
        return self
    
    def _sample_next(self, state: tuple) -> Optional[str]:
        """Sample next word given current state.
        
        Uses counts as unnormalized probabilities.
        """
        if state not in self.transitions:
            return None
        
        choices = self.transitions[state]
        total = sum(choices.values())
        
        r = random.random() * total
        cumulative = 0
        
        for word, count in choices.items():
            cumulative += count
            if r <= cumulative:
                return word
        
        return list(choices.keys())[-1]
    
    def generate(self, seed: str = None, length: int = 50) -> str:
        """Generate text using the Markov chain.
        
        Args:
            seed: Starting words (must match order)
            length: Number of words to generate
            
        Returns:
            Generated text string
        """
        if seed:
            words = seed.split()
            if len(words) < self.order:
                # Pad with random start
                start = random.choice(self.start_states) if self.start_states else ()
                words = list(start)[:self.order - len(words)] + words
            state = tuple(words[-self.order:])
        else:
            if not self.start_states:
                return ""
            state = random.choice(self.start_states)
            words = list(state)
        
        for _ in range(length):
            next_word = self._sample_next(state)
            if next_word is None:
                break
            words.append(next_word)
            state = tuple(words[-self.order:])
        
        return " ".join(words)
```

---

## 📊 Análisis de Complejidad Completo

### Template COMPLEXITY_ANALYSIS.md

```markdown
# Complexity Analysis - ML Text Classification Pipeline

## Overview

This document analyzes the time and space complexity of all models
in the ML Pipeline for the MS in AI Pathway project.

## Notation

- N = number of documents
- T = average tokens per document
- V = vocabulary size (unique terms)
- Q = query length (tokens)
- R = number of results

## Component Analysis

### 1. Tokenizer.tokenize(text)

**Time:** O(T)
- Split text: O(T)
- Lowercase: O(T)
- Filter stop words: O(T) with set lookup

**Space:** O(T) for output list

### 2. InvertedIndex.add_document(doc_id, tokens)

**Time:** O(T)
- For each token: O(1) dict access + O(1) set add
- Total: O(T)

**Space:** O(V) for index + O(N) doc_ids per term

### 3. InvertedIndex.search_or(terms)

**Time:** O(Q × avg_docs_per_term)
- For each query term: O(1) lookup
- Union of sets: O(total matching docs)

### 4. TFIDFVectorizer.fit_transform(corpus)

**Time:** O(N × T + V)
- Build vocabulary: O(N × T)
- Compute IDF: O(V)
- Transform each doc: O(N × V)

**Space:** O(N × V) for document vectors

### 5. cosine_similarity(v1, v2)

**Time:** O(V)
- Dot product: O(V)
- Magnitudes: O(V) each
- Division: O(1)

### 6. quicksort(results)

**Time:** O(R log R) average, O(R²) worst case
**Space:** O(log R) for recursion stack

### 7. SearchEngine.build_index()

**Time:** O(N × T + N × V)
- Tokenize all docs: O(N × T)
- Build inverted index: O(N × T)
- Build TF-IDF vectors: O(N × T + N × V)

**Space:** O(V + N × V)
- Inverted index: O(V)
- Document vectors: O(N × V)

### 8. SearchEngine.search(query)

**Time:** O(Q + R × V + R log R)
- Tokenize query: O(Q)
- Find candidates: O(Q)
- Transform query: O(V)
- Calculate similarities: O(R × V)
- Sort results: O(R log R)

**Space:** O(V + R)
- Query vector: O(V)
- Results list: O(R)

## Summary Table

| Operation | Time | Space |
|-----------|------|-------|
| add_document | O(1) | O(T) |
| build_index | O(N×T + N×V) | O(V + N×V) |
| search | O(Q + R×V + R log R) | O(V + R) |

## Bottlenecks and Optimizations

1. **TF-IDF vectors are dense** → Could use sparse representation
2. **Similarity calculated for all candidates** → Could use inverted index scores
3. **QuickSort worst case** → Using random pivot mitigates this
```

---

## ⚠️ Errores Comunes y Soluciones

### Error 1: Olvidar llamar build_index()

```python
# ❌ Error: RuntimeError
engine = SearchEngine()
engine.add_document(1, "Title", "Content")
results = engine.search("query")  # ¡No se indexó!

# ✅ Correcto
engine = SearchEngine()
engine.add_document(1, "Title", "Content")
engine.build_index()  # ¡Importante!
results = engine.search("query")
```

### Error 2: No manejar queries vacías

```python
# ❌ Puede causar errores
def search(self, query):
    tokens = self.tokenizer.tokenize(query)
    # Si query="", tokens=[] y query_vector tiene problemas

# ✅ Manejar caso vacío
def search(self, query):
    tokens = self.tokenizer.tokenize(query)
    if not tokens:
        return []  # Retornar lista vacía
```

### Error 3: Modificar documento después de indexar

```python
# ❌ El índice queda desactualizado
engine.add_document(1, "Title", "Python tutorial")
engine.build_index()
engine.corpus.get(1).content = "Java tutorial"  # ¡Índice no actualizado!

# ✅ Reconstruir índice después de modificaciones
engine.add_document(2, "Title2", "New content")
engine.build_index()  # Reconstruir
```

### Error 4: No normalizar texto consistentemente

```python
# ❌ "Python" vs "python" son diferentes
index.search("Python")  # Encuentra
index.search("python")  # No encuentra

# ✅ Normalizar siempre en tokenizer
def tokenize(self, text):
    return text.lower().split()  # Siempre minúsculas
```

---

## 💡 Recomendaciones Profesionales

### 1. Testing
```python
# Mínimo: tests unitarios para cada componente
pytest tests/ -v --cov=src --cov-report=term-missing
# Objetivo: >80% coverage
```

### 2. Type Hints
```python
# Todas las funciones deben tener type hints
def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
```

### 3. Docstrings
```python
# Google style docstrings para todas las funciones públicas
def function(param: Type) -> ReturnType:
    """One-line description.
    
    Longer description if needed.
    
    Args:
        param: Description of parameter.
    
    Returns:
        Description of return value.
    
    Raises:
        ErrorType: When this error occurs.
    
    Example:
        >>> function(value)
        expected_result
    """
```

### 4. Configuración de Herramientas

```toml
# pyproject.toml
[tool.mypy]
strict = true
python_version = "3.11"

[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src"
```

---

## 📋 Checklist de Entrega (100 puntos)

### Línea 1: Machine Learning (40 pts)
- [ ] `NaiveBayesClassifier` desde cero (10 pts) - Módulo 19 + 22
- [ ] `KMeans` clustering desde cero (10 pts) - Módulo 23
- [ ] `NeuralNetwork` MLP con backprop (15 pts) - Módulo 24
- [ ] Preprocesamiento TF-IDF (5 pts) - Módulo 11

### Línea 2: Probabilidad y Estadística (30 pts)
- [ ] Análisis Bayesiano en Naive Bayes (10 pts) - Módulo 19
- [ ] `MarkovTextGenerator` funcional (10 pts) - Módulo 21
- [ ] Evaluación con intervalos de confianza (5 pts) - Módulo 20
- [ ] Cross-validation implementado (5 pts) - Módulo 20

### Testing y Documentación (20 pts)
- [ ] Tests unitarios para cada modelo (10 pts)
- [ ] README.md profesional en inglés (5 pts)
- [ ] `MODEL_COMPARISON.md` con análisis estadístico (5 pts)

### Funcionalidad (10 pts)
- [ ] Pipeline entrena y predice correctamente (5 pts)
- [ ] Demo interactivo funcional (5 pts)

---

## 🏗️ Arquitectura Final

```
MLPipeline (Proyecto Integrador)
    │
    ├── Preprocesamiento
    │   ├── Tokenizer (text → tokens)
    │   └── TFIDFVectorizer (tokens → vectors)
    │
    ├── LÍNEA 1: Machine Learning
    │   ├── NaiveBayesClassifier (Supervisado + Bayes)
    │   ├── KMeans (No Supervisado)
    │   └── NeuralNetwork (Deep Learning)
    │
    ├── LÍNEA 2: Probabilidad/Estadística
    │   ├── MarkovTextGenerator (Cadenas de Markov)
    │   ├── confidence_interval() (Estimación)
    │   └── cross_validate() (Evaluación)
    │
    └── Evaluación
        ├── accuracy, precision, recall, f1
        └── Comparación estadística de modelos
```

---

## 📊 Análisis de Complejidad Requerido

Documenta en `MODEL_COMPARISON.md`:

| Modelo | Train | Predict | Space |
|--------|-------|---------|-------|
| `NaiveBayes.fit()` | O(N × V) | O(V × C) | O(V × C) |
| `KMeans.fit()` | O(N × K × V × I) | O(K × V) | O(K × V) |
| `NeuralNetwork.fit()` | O(E × N × L) | O(N × L) | O(L) |
| `MarkovGenerator.fit()` | O(T) | O(L) | O(V²) |

Donde: N=samples, V=features, C=classes, K=clusters, I=iterations, E=epochs, L=layers, T=tokens

---

## 📝 Template README.md

```markdown
# ML Text Classification Pipeline

A complete ML pipeline built from scratch in pure Python for the MS in AI Pathway.

## Pathway Alignment

This project demonstrates competency in both Pathway lines:

### Machine Learning (3 credits)
- ✅ Naive Bayes Classifier (Supervised Learning)
- ✅ K-Means Clustering (Unsupervised Learning)
- ✅ Neural Network with Backpropagation (Deep Learning)

### Probability & Statistics (3 credits)
- ✅ Bayesian Analysis in Naive Bayes (Probability Foundations)
- ✅ Markov Chain Text Generator (Discrete-Time Markov Chains)
- ✅ Confidence Intervals & Cross-Validation (Statistical Estimation)

## Features
- All models implemented from scratch (no sklearn, no pytorch)
- TF-IDF text vectorization
- Statistical model comparison with confidence intervals
- Markov chain text generation

## Installation
\`\`\`bash
git clone <repo>
cd ml-text-classifier
python -m venv venv
source venv/bin/activate
\`\`\`

## Usage
\`\`\`python
from src.pipeline import MLPipeline

pipeline = MLPipeline(n_classes=2)
pipeline.load_data(texts, labels)
pipeline.train_all_models()
results = pipeline.evaluate_all_models()
print(results)
# {'naive_bayes': {'accuracy': 0.85, 'accuracy_ci_95': (0.78, 0.92)}, ...}

# Generate synthetic text
generated = pipeline.generate_text(seed="The model", length=20)
\`\`\`

## Model Comparison
See [MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md)

## Testing
\`\`\`bash
python -m pytest tests/ -v --cov=src
\`\`\`
```

---

## ✅ Criterios de Aprobación

| Puntuación | Nivel | Significado |
|------------|-------|-------------|
| 90-100 | 🏆 Listo para Pathway | Dominas ambas líneas |
| 75-89 | ✅ Buen nivel | Reforzar gaps menores |
| 60-74 | ⚠️ Necesita trabajo | Revisar módulos 19-24 |
| <60 | ❌ Insuficiente | Volver a estudiar fundamentos |

---

## 🎯 Verificación de Competencias del Pathway

| Curso del Pathway | ¿Cubierto? | Evidencia en el Proyecto |
|-------------------|------------|--------------------------|
| **ML: Supervised Learning** | ✅ | NaiveBayesClassifier, evaluación |
| **ML: Unsupervised Algorithms** | ✅ | KMeans clustering |
| **ML: Deep Learning** | ✅ | NeuralNetwork con backprop |
| **Prob: Foundations** | ✅ | Bayes en clasificador, distribuciones |
| **Prob: Markov Chains** | ✅ | MarkovTextGenerator |
| **Prob: Statistical Estimation** | ✅ | Intervalos de confianza, cross-val |

---

## 🔗 Navegación

| ← Anterior | Índice |
|------------|--------|
| [24_INTRO_DEEP_LEARNING](24_INTRO_DEEP_LEARNING.md) | [00_INDICE](00_INDICE.md) |
