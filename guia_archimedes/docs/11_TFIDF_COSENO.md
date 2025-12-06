# 11 - TF-IDF y Similitud de Coseno

> **🎯 Objetivo:** Implementar el sistema de ranking por relevancia del motor de búsqueda usando TF-IDF y similitud de coseno.

---

## 🧠 Analogía: Encontrar el Libro Correcto en una Biblioteca

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PROBLEMA: Buscar "python machine learning" en 1000 documentos             │
│                                                                             │
│   OPCIÓN 1: Solo contar palabras                                            │
│   ─────────────────────────────────                                         │
│   Doc A: "python python python" → 3 menciones de "python"                   │
│   Doc B: "python machine learning tutorial" → 1 mención                     │
│   ¿Doc A es mejor? ¡No! Solo repite la palabra.                             │
│                                                                             │
│   OPCIÓN 2: TF-IDF                                                          │
│   ────────────────                                                          │
│   • TF (Term Frequency): ¿Cuánto aparece la palabra en ESTE documento?      │
│   • IDF (Inverse Document Frequency): ¿Qué tan RARA es en TODOS los docs?   │
│                                                                             │
│   Palabras como "the", "is" → aparecen en todos → IDF bajo → poco útiles    │
│   Palabras como "tensorflow" → aparece en pocos → IDF alto → muy útiles     │
│                                                                             │
│   TF-IDF = TF × IDF → Balance entre frecuencia y rareza                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Term Frequency (TF)](#1-tf)
2. [Inverse Document Frequency (IDF)](#2-idf)
3. [TF-IDF Combinado](#3-tfidf)
4. [Vectorización de Documentos](#4-vectorizacion)
5. [Ranking con Similitud de Coseno](#5-ranking)
6. [Integración en Archimedes](#6-integracion)

---

## 1. Term Frequency (TF) {#1-tf}

### 1.1 Concepto

```
┌─────────────────────────────────────────────────────────────────┐
│  TF = ¿Cuántas veces aparece el término en este documento?      │
│                                                                 │
│  Documento: "the cat sat on the mat"                            │
│  Tokens: ["the", "cat", "sat", "on", "the", "mat"]              │
│                                                                 │
│  TF("the") = 2/6 = 0.333                                        │
│  TF("cat") = 1/6 = 0.167                                        │
│  TF("dog") = 0/6 = 0.000                                        │
│                                                                 │
│  FÓRMULA:                                                       │
│                count(term, document)                            │
│  TF(t, d) = ─────────────────────────                           │
│              total_terms(document)                              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Implementación

```python
def compute_tf(term: str, document: list[str]) -> float:
    """Compute Term Frequency for a term in a document.
    
    TF measures how often a term appears in a document,
    normalized by document length.
    
    Args:
        term: The word to compute TF for.
        document: List of tokens in the document.
    
    Returns:
        TF value between 0 and 1.
    
    Example:
        >>> compute_tf("the", ["the", "cat", "sat", "on", "the", "mat"])
        0.3333333333333333
    """
    if not document:
        return 0.0
    
    count = document.count(term)
    return count / len(document)


def compute_tf_vector(document: list[str], vocabulary: list[str]) -> list[float]:
    """Compute TF for all terms in vocabulary.
    
    Args:
        document: List of tokens in document.
        vocabulary: List of all unique terms across corpus.
    
    Returns:
        Vector where each position is TF of corresponding vocabulary term.
    
    Example:
        >>> vocab = ["cat", "dog", "mat", "sat", "the"]
        >>> doc = ["the", "cat", "sat", "on", "the", "mat"]
        >>> compute_tf_vector(doc, vocab)
        [0.167, 0.0, 0.167, 0.167, 0.333]  # approximately
    """
    return [compute_tf(term, document) for term in vocabulary]
```

### 1.3 Variantes de TF

```python
def compute_tf_raw(term: str, document: list[str]) -> int:
    """Raw count (no normalization)."""
    return document.count(term)


def compute_tf_log(term: str, document: list[str]) -> float:
    """Logarithmic TF: reduces impact of high frequency.
    
    Formula: 1 + log(count) if count > 0, else 0
    """
    import math
    count = document.count(term)
    if count == 0:
        return 0.0
    return 1 + math.log(count)


def compute_tf_binary(term: str, document: list[str]) -> int:
    """Binary TF: 1 if present, 0 otherwise."""
    return 1 if term in document else 0
```

---

## 2. Inverse Document Frequency (IDF) {#2-idf}

### 2.1 Concepto

```
┌─────────────────────────────────────────────────────────────────┐
│  IDF = ¿Qué tan rara es esta palabra en todo el corpus?         │
│                                                                 │
│  Corpus de 1000 documentos:                                     │
│  • "the" aparece en 950 docs → muy común → IDF bajo             │
│  • "python" aparece en 50 docs → algo rara → IDF medio          │
│  • "tensorflow" aparece en 5 docs → muy rara → IDF alto         │
│                                                                 │
│  FÓRMULA:                                                       │
│                    total_documents                              │
│  IDF(t) = log( ───────────────────────── )                      │
│                 documents_containing(t)                         │
│                                                                 │
│  El logaritmo suaviza los valores extremos                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementación

```python
import math


def compute_idf(term: str, corpus: list[list[str]]) -> float:
    """Compute Inverse Document Frequency for a term.
    
    IDF measures how rare a term is across the corpus.
    Rare terms get higher IDF.
    
    Args:
        term: The word to compute IDF for.
        corpus: List of documents (each document is list of tokens).
    
    Returns:
        IDF value (higher = more rare/important).
    
    Example:
        >>> corpus = [["cat", "dog"], ["cat", "mouse"], ["dog", "bird"]]
        >>> compute_idf("cat", corpus)
        0.405...  # appears in 2 of 3 docs
        >>> compute_idf("bird", corpus)
        1.098...  # appears in 1 of 3 docs (more rare)
    """
    if not corpus:
        return 0.0
    
    total_docs = len(corpus)
    docs_with_term = sum(1 for doc in corpus if term in doc)
    
    if docs_with_term == 0:
        return 0.0
    
    return math.log(total_docs / docs_with_term)


def compute_idf_smooth(term: str, corpus: list[list[str]]) -> float:
    """IDF with smoothing to avoid division by zero and extremes.
    
    Formula: log(1 + (N / (1 + df)))
    
    This variant:
    - Adds 1 to denominator to handle terms not in corpus
    - Adds 1 inside log to avoid negative values
    """
    if not corpus:
        return 0.0
    
    total_docs = len(corpus)
    docs_with_term = sum(1 for doc in corpus if term in doc)
    
    return math.log(1 + (total_docs / (1 + docs_with_term)))
```

### 2.3 Pre-computar IDF para Vocabulario

```python
def compute_idf_dict(
    vocabulary: list[str],
    corpus: list[list[str]]
) -> dict[str, float]:
    """Pre-compute IDF for all terms in vocabulary.
    
    More efficient than computing IDF repeatedly.
    
    Returns:
        Dictionary mapping term to its IDF value.
    """
    return {term: compute_idf(term, corpus) for term in vocabulary}
```

---

## 3. TF-IDF Combinado {#3-tfidf}

### 3.1 La Fórmula

```
┌─────────────────────────────────────────────────────────────────┐
│  TF-IDF = TF × IDF                                              │
│                                                                 │
│  Combina:                                                       │
│  • TF: Importancia local (en este documento)                    │
│  • IDF: Importancia global (en todo el corpus)                  │
│                                                                 │
│  RESULTADO:                                                     │
│  • Palabra común en este doc pero rara globalmente → ALTO       │
│  • Palabra rara en este doc y común globalmente → BAJO          │
│  • Palabra muy común en todos lados ("the") → MUY BAJO          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementación

```python
def compute_tfidf(
    term: str,
    document: list[str],
    corpus: list[list[str]]
) -> float:
    """Compute TF-IDF score for a term in a document.
    
    Args:
        term: Word to score.
        document: The specific document (list of tokens).
        corpus: All documents in the collection.
    
    Returns:
        TF-IDF score (higher = more important).
    
    Example:
        >>> corpus = [["python", "code"], ["java", "code"], ["python", "ml"]]
        >>> compute_tfidf("python", ["python", "code"], corpus)
        0.203...  # "python" is somewhat distinctive
        >>> compute_tfidf("code", ["python", "code"], corpus)
        0.0  # "code" appears in too many docs (if 2/3)
    """
    tf = compute_tf(term, document)
    idf = compute_idf(term, corpus)
    return tf * idf
```

### 3.3 Vector TF-IDF Completo

```python
def compute_tfidf_vector(
    document: list[str],
    vocabulary: list[str],
    idf_dict: dict[str, float]
) -> list[float]:
    """Compute TF-IDF vector for a document.
    
    Args:
        document: Document as list of tokens.
        vocabulary: Ordered list of all terms.
        idf_dict: Pre-computed IDF values.
    
    Returns:
        Vector of TF-IDF values, one per vocabulary term.
    """
    tfidf_vector = []
    
    for term in vocabulary:
        tf = compute_tf(term, document)
        idf = idf_dict.get(term, 0.0)
        tfidf_vector.append(tf * idf)
    
    return tfidf_vector
```

---

## 4. Vectorización de Documentos {#4-vectorizacion}

### 4.1 Clase TFIDFVectorizer

```python
import math
from collections import Counter


class TFIDFVectorizer:
    """Transform documents into TF-IDF vectors.
    
    Similar to sklearn's TfidfVectorizer but from scratch.
    
    Attributes:
        vocabulary_: List of terms (ordered).
        idf_: Dictionary of IDF values.
    """
    
    def __init__(self) -> None:
        """Initialize empty vectorizer."""
        self.vocabulary_: list[str] = []
        self.idf_: dict[str, float] = {}
        self._fitted: bool = False
    
    def fit(self, corpus: list[list[str]]) -> "TFIDFVectorizer":
        """Learn vocabulary and IDF from corpus.
        
        Args:
            corpus: List of documents (each is list of tokens).
        
        Returns:
            self (for method chaining).
        """
        # Build vocabulary (all unique terms)
        all_terms: set[str] = set()
        for doc in corpus:
            all_terms.update(doc)
        self.vocabulary_ = sorted(all_terms)  # Sorted for consistency
        
        # Compute IDF for each term
        total_docs = len(corpus)
        for term in self.vocabulary_:
            docs_with_term = sum(1 for doc in corpus if term in doc)
            if docs_with_term > 0:
                self.idf_[term] = math.log(total_docs / docs_with_term)
            else:
                self.idf_[term] = 0.0
        
        self._fitted = True
        return self
    
    def transform(self, documents: list[list[str]]) -> list[list[float]]:
        """Transform documents to TF-IDF vectors.
        
        Args:
            documents: Documents to transform.
        
        Returns:
            List of TF-IDF vectors.
        
        Raises:
            RuntimeError: If vectorizer hasn't been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted before transform")
        
        vectors = []
        for doc in documents:
            vector = self._transform_single(doc)
            vectors.append(vector)
        return vectors
    
    def _transform_single(self, document: list[str]) -> list[float]:
        """Transform single document to TF-IDF vector."""
        vector = []
        doc_length = len(document) if document else 1
        term_counts = Counter(document)
        
        for term in self.vocabulary_:
            tf = term_counts.get(term, 0) / doc_length
            idf = self.idf_.get(term, 0.0)
            vector.append(tf * idf)
        
        return vector
    
    def fit_transform(self, corpus: list[list[str]]) -> list[list[float]]:
        """Fit and transform in one step."""
        self.fit(corpus)
        return self.transform(corpus)
    
    def transform_query(self, query_tokens: list[str]) -> list[float]:
        """Transform a search query to TF-IDF vector.
        
        Uses same vocabulary and IDF as corpus.
        """
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted first")
        return self._transform_single(query_tokens)
    
    def get_feature_names(self) -> list[str]:
        """Return vocabulary terms in order."""
        return self.vocabulary_.copy()
    
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"TFIDFVectorizer({len(self.vocabulary_)} terms, {status})"
```

### 4.2 Uso del Vectorizer

```python
# Corpus de ejemplo
corpus = [
    ["python", "machine", "learning", "tutorial"],
    ["java", "programming", "tutorial"],
    ["python", "data", "science"],
    ["machine", "learning", "deep", "learning"],
]

# Crear y ajustar vectorizer
vectorizer = TFIDFVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Ver vocabulario
print(vectorizer.get_feature_names())
# ['data', 'deep', 'java', 'learning', 'machine', 'programming', 
#  'python', 'science', 'tutorial']

# Ver vector del primer documento
print(tfidf_matrix[0])
# [0.0, 0.0, 0.0, 0.173, 0.346, 0.0, 0.173, 0.0, 0.173]
# "machine" y "learning" tienen valores, "python" también

# Transformar query
query = ["python", "learning"]
query_vector = vectorizer.transform_query(query)
```

---

## 5. Ranking con Similitud de Coseno {#5-ranking}

### 5.1 Función de Similitud

```python
import math


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors.
    
    Returns value between 0 and 1 for TF-IDF vectors.
    Higher = more similar.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same dimension")
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in v1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in v2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)
```

### 5.2 Ranking de Documentos

```python
from typing import NamedTuple


class SearchResult(NamedTuple):
    """A search result with document ID and relevance score."""
    doc_id: int
    score: float


def rank_documents(
    query_vector: list[float],
    document_vectors: list[list[float]],
    top_k: int = 10
) -> list[SearchResult]:
    """Rank documents by similarity to query.
    
    Args:
        query_vector: TF-IDF vector of search query.
        document_vectors: TF-IDF vectors of all documents.
        top_k: Number of top results to return.
    
    Returns:
        List of SearchResult sorted by score (descending).
    """
    results = []
    
    for doc_id, doc_vector in enumerate(document_vectors):
        score = cosine_similarity(query_vector, doc_vector)
        if score > 0:  # Only include non-zero matches
            results.append(SearchResult(doc_id=doc_id, score=score))
    
    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    
    return results[:top_k]
```

### 5.3 Integración: Búsqueda Completa

```python
class TFIDFSearchEngine:
    """Simple search engine using TF-IDF and cosine similarity.
    
    Example:
        >>> engine = TFIDFSearchEngine()
        >>> engine.index([
        ...     (0, ["python", "tutorial"]),
        ...     (1, ["java", "tutorial"]),
        ...     (2, ["python", "machine", "learning"]),
        ... ])
        >>> engine.search("python learning")
        [SearchResult(doc_id=2, score=0.8), SearchResult(doc_id=0, score=0.3)]
    """
    
    def __init__(self) -> None:
        self.vectorizer = TFIDFVectorizer()
        self.document_vectors: list[list[float]] = []
        self.doc_ids: list[int] = []
    
    def index(self, documents: list[tuple[int, list[str]]]) -> None:
        """Index documents for searching.
        
        Args:
            documents: List of (doc_id, tokens) tuples.
        """
        self.doc_ids = [doc_id for doc_id, _ in documents]
        corpus = [tokens for _, tokens in documents]
        self.document_vectors = self.vectorizer.fit_transform(corpus)
    
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search for documents matching query.
        
        Args:
            query: Search query string.
            top_k: Number of results to return.
        
        Returns:
            List of SearchResult with doc_id and score.
        """
        # Tokenize query (simple split for now)
        query_tokens = query.lower().split()
        
        # Transform to TF-IDF vector
        query_vector = self.vectorizer.transform_query(query_tokens)
        
        # Rank documents
        results = []
        for i, doc_vector in enumerate(self.document_vectors):
            score = cosine_similarity(query_vector, doc_vector)
            if score > 0:
                results.append(SearchResult(
                    doc_id=self.doc_ids[i],
                    score=score
                ))
        
        # Sort and return top k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
```

---

## 6. Integración en Archimedes {#6-integracion}

### 6.1 Módulo Completo

```python
# src/vectorizer.py

import math
from collections import Counter
from typing import NamedTuple


class TFIDFVectorizer:
    """TF-IDF Vectorizer for Archimedes Indexer."""
    
    def __init__(self) -> None:
        self.vocabulary_: list[str] = []
        self.idf_: dict[str, float] = {}
        self._fitted: bool = False
    
    def fit(self, corpus: list[list[str]]) -> "TFIDFVectorizer":
        all_terms: set[str] = set()
        for doc in corpus:
            all_terms.update(doc)
        self.vocabulary_ = sorted(all_terms)
        
        total_docs = len(corpus)
        for term in self.vocabulary_:
            docs_with_term = sum(1 for doc in corpus if term in doc)
            self.idf_[term] = math.log(total_docs / docs_with_term) if docs_with_term else 0
        
        self._fitted = True
        return self
    
    def transform(self, documents: list[list[str]]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("Must fit before transform")
        return [self._transform_single(doc) for doc in documents]
    
    def _transform_single(self, document: list[str]) -> list[float]:
        doc_len = len(document) if document else 1
        counts = Counter(document)
        return [
            (counts.get(term, 0) / doc_len) * self.idf_.get(term, 0)
            for term in self.vocabulary_
        ]
    
    def fit_transform(self, corpus: list[list[str]]) -> list[list[float]]:
        return self.fit(corpus).transform(corpus)


# src/similarity.py

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


class ScoredResult(NamedTuple):
    doc_id: int
    score: float


def rank_by_similarity(
    query_vector: list[float],
    doc_vectors: list[list[float]],
    doc_ids: list[int],
    top_k: int = 10
) -> list[ScoredResult]:
    """Rank documents by cosine similarity to query."""
    results = [
        ScoredResult(doc_id, cosine_similarity(query_vector, doc_vec))
        for doc_id, doc_vec in zip(doc_ids, doc_vectors)
    ]
    # Filter zero scores and sort
    results = [r for r in results if r.score > 0]
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]
```

---

## ⚠️ Errores Comunes

### Error 1: No normalizar por longitud de documento

```python
# ❌ Documentos largos siempre ganan
def bad_tf(term, doc):
    return doc.count(term)  # Raw count

# ✅ Normalizar
def good_tf(term, doc):
    return doc.count(term) / len(doc)
```

### Error 2: Log de cero en IDF

```python
# ❌ Error si término no está en ningún documento
def bad_idf(term, corpus):
    df = sum(1 for d in corpus if term in d)
    return math.log(len(corpus) / df)  # ZeroDivisionError!

# ✅ Manejar caso especial
def good_idf(term, corpus):
    df = sum(1 for d in corpus if term in d)
    if df == 0:
        return 0.0
    return math.log(len(corpus) / df)
```

### Error 3: Vocabulario inconsistente

```python
# ❌ Query tiene términos no vistos
query_terms = ["quantum", "computing"]  # No estaban en corpus
# Vector query tendrá dimension incorrecta

# ✅ Usar solo términos del vocabulario
def transform_query(self, query_tokens):
    # Solo considera términos en self.vocabulary_
    ...
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 11.1: Implementar TF
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-111)

### Ejercicio 11.2: Implementar IDF
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-112)

### Ejercicio 11.3: Sistema de Ranking
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-113)

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [TF-IDF Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) | Lectura | 🔴 Obligatorio |
| [Stanford IR Book Ch.6](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) | Libro | 🟡 Recomendado |
| [Sklearn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) | Docs | 🟢 Complementario (para comparar) |

---

## 🔗 Referencias del Glosario

- [TF-IDF](GLOSARIO.md#tf-idf)
- [Term Frequency](GLOSARIO.md#term-frequency)
- [Inverse Document Frequency](GLOSARIO.md#idf)
- [Similitud de Coseno](GLOSARIO.md#similitud-coseno)
- [Vectorización](GLOSARIO.md#vectorizacion)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [10_ALGEBRA_LINEAL](10_ALGEBRA_LINEAL.md) | [00_INDICE](00_INDICE.md) | [12_PROYECTO_INTEGRADOR](12_PROYECTO_INTEGRADOR.md) |
