# 12 - Proyecto Integrador: Archimedes Indexer

> **🎯 Objetivo:** Ensamblar todos los componentes en un motor de búsqueda funcional con análisis Big O.

---

## 🧠 Metodología Feynman: ¿Qué Estamos Construyendo?

### Explicación para un Niño de 10 Años

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   IMAGINA UNA BIBLIOTECA MÁGICA                                             │
│   ─────────────────────────────                                             │
│                                                                             │
│   Tienes 1000 libros y quieres encontrar todos los que hablan de "dragones" │ 
│                                                                             │
│   SIN MAGIA (búsqueda lineal):                                              │
│   📚 Abrir libro 1, leer todo, ¿tiene "dragones"? No.                       │
│   📚 Abrir libro 2, leer todo, ¿tiene "dragones"? No.                       │
│   📚 ... repetir 1000 veces ... ¡MUY LENTO!                                 │
│                                                                             │
│   CON MAGIA (índice invertido):                                             │
│   📋 El bibliotecario tiene una lista secreta:                              │
│      "dragones" → libros 23, 156, 789                                       │
│   📚 ¡Vas directo a esos 3 libros! ¡INSTANTÁNEO!                            │
│                                                                             │
│   ESO ES ARCHIMEDES INDEXER:                                                │
│   Un bibliotecario mágico para documentos de texto.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Explicación Técnica Progresiva

**Nivel 1 - Concepto:**
Un motor de búsqueda que encuentra documentos relevantes para una consulta.

**Nivel 2 - Componentes:**
- **Tokenizer:** Divide texto en palabras
- **Índice Invertido:** Mapea palabra → documentos
- **TF-IDF:** Calcula importancia de palabras
- **Cosine Similarity:** Mide qué tan similar es query a cada documento

**Nivel 3 - Flujo Completo:**
```
INDEXACIÓN (una vez):
documento → tokenizar → actualizar índice → calcular TF-IDF

BÚSQUEDA (cada query):
query → tokenizar → buscar en índice → calcular similitud → ordenar → resultados
```

**Nivel 4 - Complejidad:**
- Indexación: O(N × T) donde N=docs, T=tokens promedio
- Búsqueda: O(V + R × V) donde V=vocabulario, R=resultados

---

## 🏗️ Arquitectura Detallada

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARCHIMEDES INDEXER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                           │
│  │   ENTRADA    │                                                           │
│  │  Documentos  │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐       ┌──────────────┐      ┌──────────────┐              │
│  │  Tokenizer   │─────▶│   Corpus     │─────▶│  Inverted    │              │
│  │              │       │  (Document)  │      │   Index      │              │
│  └──────────────┘       └──────────────┘      └──────┬───────┘              │
│                                                      │                      │
│                                                      ▼                      │
│                                               ┌──────────────┐              │
│                                               │    TF-IDF    │              │
│                                               │  Vectorizer  │              │
│                                               └──────┬───────┘              │
│                                                      │                      │
│  ┌──────────────┐       ┌──────────────┐              │                     │
│  │    Query     │─────▶│  Similarity  │◀────────────┘                      │
│  │              │       │   (Cosine)   │                                    │
│  └──────────────┘       └──────┬───────┘                                    │
│                               │                                             │
│                               ▼                                             │
│                        ┌──────────────┐       ┌──────────────┐              │
│                        │   QuickSort  │─────▶│  Resultados  │              │
│                        │   (ranking)  │       │   Ordenados  │              │
│                        └──────────────┘       └──────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura de Archivos del Proyecto

```
archimedes-indexer/
├── src/
│   ├── __init__.py              # Package marker
│   ├── document.py              # Módulos 01-02: Document, Corpus
│   ├── tokenizer.py             # Módulos 03-04: Tokenizer
│   ├── inverted_index.py        # Módulos 05-06: InvertedIndex
│   ├── sorting.py               # Módulos 07-08: quicksort, mergesort
│   ├── searching.py             # Módulo 09: binary_search
│   ├── linear_algebra.py        # Módulo 10: Vector operations
│   ├── vectorizer.py            # Módulo 11: TFIDFVectorizer
│   ├── similarity.py            # Módulo 11: cosine_similarity
│   └── search_engine.py         # Módulo 12: SearchEngine (integración)
├── tests/
│   ├── __init__.py
│   ├── test_document.py
│   ├── test_tokenizer.py
│   ├── test_inverted_index.py
│   ├── test_sorting.py
│   ├── test_searching.py
│   ├── test_linear_algebra.py
│   ├── test_vectorizer.py
│   ├── test_similarity.py
│   └── test_search_engine.py    # Tests de integración
├── data/
│   └── sample_corpus/
│       ├── doc_001.txt
│       ├── doc_002.txt
│       └── ...
├── docs/
│   ├── COMPLEXITY_ANALYSIS.md   # Análisis Big O de todo el sistema
│   └── API_REFERENCE.md         # Documentación de la API
├── README.md                    # Documentación principal (inglés)
├── pyproject.toml               # Configuración del proyecto
└── requirements-dev.txt         # pytest, mypy, ruff
```

---

## 💻 Implementación Guiada: SearchEngine

### Paso 1: La Clase Principal

```python
# src/search_engine.py
"""Main search engine orchestrating all components."""

from typing import NamedTuple
from .document import Document, Corpus
from .tokenizer import Tokenizer
from .inverted_index import InvertedIndex
from .vectorizer import TFIDFVectorizer
from .similarity import cosine_similarity
from .sorting import quicksort


class SearchResult(NamedTuple):
    """A search result with document info and relevance score.
    
    Attributes:
        doc_id: Unique document identifier.
        title: Document title.
        score: Relevance score (0.0 to 1.0).
        snippet: Preview of document content.
    """
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchEngine:
    """Full-text search engine using TF-IDF and cosine similarity.
    
    This class integrates all components:
    - Tokenizer for text processing
    - InvertedIndex for fast term lookup
    - TFIDFVectorizer for document representation
    - Cosine similarity for ranking
    
    Example:
        >>> engine = SearchEngine()
        >>> engine.add_document(1, "Python Guide", "Learn Python programming...")
        >>> engine.add_document(2, "Java Tutorial", "Java programming basics...")
        >>> engine.build_index()
        >>> results = engine.search("python programming")
        >>> for r in results:
        ...     print(f"{r.title}: {r.score:.3f}")
        Python Guide: 0.847
        Java Tutorial: 0.213
    
    Complexity:
        - add_document: O(1)
        - build_index: O(N × T) where N=docs, T=avg tokens
        - search: O(V + R × V + R log R) where V=vocab, R=results
    """
    
    def __init__(self) -> None:
        """Initialize search engine with empty corpus."""
        self.corpus = Corpus()
        self.tokenizer = Tokenizer()
        self.index = InvertedIndex()
        self.vectorizer = TFIDFVectorizer()
        
        self._document_vectors: list[list[float]] = []
        self._indexed: bool = False
    
    def add_document(self, doc_id: int, title: str, content: str) -> None:
        """Add a document to the corpus.
        
        Args:
            doc_id: Unique identifier for the document.
            title: Document title for display.
            content: Full text content to index.
        
        Raises:
            ValueError: If doc_id already exists.
        
        Note:
            Documents are not searchable until build_index() is called.
        """
        if self.corpus.contains(doc_id):
            raise ValueError(f"Document {doc_id} already exists")
        
        doc = Document(doc_id=doc_id, title=title, content=content)
        self.corpus.add(doc)
        self._indexed = False  # Mark index as stale
    
    def build_index(self) -> None:
        """Build inverted index and TF-IDF vectors.
        
        Must be called after adding documents and before searching.
        Can be called again to rebuild after adding more documents.
        
        Complexity: O(N × T) where N=documents, T=avg tokens per doc
        """
        # Reset index
        self.index = InvertedIndex()
        tokenized_docs: list[list[str]] = []
        
        # Process each document
        for doc in self.corpus:
            tokens = self.tokenizer.tokenize(doc.content)
            doc.tokens = tokens
            tokenized_docs.append(tokens)
            self.index.add_document(doc.doc_id, tokens)
        
        # Build TF-IDF vectors
        self._document_vectors = self.vectorizer.fit_transform(tokenized_docs)
        self._indexed = True
    
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search for documents matching the query.
        
        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
        
        Returns:
            List of SearchResult sorted by relevance (descending).
        
        Raises:
            RuntimeError: If build_index() hasn't been called.
        
        Complexity: O(V + R × V + R log R)
            - O(V): Transform query to vector
            - O(R × V): Calculate similarity for R candidate docs
            - O(R log R): Sort results
        """
        if not self._indexed:
            raise RuntimeError("Must call build_index() before searching")
        
        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []
        
        # Get candidate documents (those containing at least one query term)
        candidates = self.index.search_or(query_tokens)
        if not candidates:
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform_query(query_tokens)
        
        # Calculate similarity for each candidate
        results: list[tuple[int, float]] = []
        for doc_idx, doc in enumerate(self.corpus):
            if doc.doc_id in candidates:
                score = cosine_similarity(query_vector, self._document_vectors[doc_idx])
                if score > 0:
                    results.append((doc_idx, score))
        
        # Sort by score (descending) using our quicksort
        results = quicksort(results, key=lambda x: -x[1])
        
        # Convert to SearchResult objects
        search_results: list[SearchResult] = []
        for doc_idx, score in results[:top_k]:
            doc = self.corpus.get_by_index(doc_idx)
            snippet = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            search_results.append(SearchResult(
                doc_id=doc.doc_id,
                title=doc.title,
                score=round(score, 4),
                snippet=snippet
            ))
        
        return search_results
    
    def get_stats(self) -> dict:
        """Get statistics about the search engine.
        
        Returns:
            Dictionary with corpus and index statistics.
        """
        return {
            "documents": len(self.corpus),
            "vocabulary_size": self.vectorizer.vocabulary_size if self._indexed else 0,
            "indexed": self._indexed,
        }
```

### Paso 2: Clases de Soporte

```python
# src/document.py
"""Document and Corpus classes."""

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Document:
    """A searchable document.
    
    Attributes:
        doc_id: Unique identifier.
        title: Document title.
        content: Full text content.
        tokens: Tokenized content (populated by SearchEngine).
    """
    doc_id: int
    title: str
    content: str
    tokens: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if self.doc_id < 0:
            raise ValueError("doc_id must be non-negative")
        if not self.content.strip():
            raise ValueError("content cannot be empty")


class Corpus:
    """Collection of documents."""
    
    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._id_to_index: dict[int, int] = {}
    
    def add(self, doc: Document) -> None:
        """Add document to corpus."""
        if doc.doc_id in self._id_to_index:
            raise ValueError(f"Document {doc.doc_id} already exists")
        self._id_to_index[doc.doc_id] = len(self._documents)
        self._documents.append(doc)
    
    def get(self, doc_id: int) -> Document:
        """Get document by ID."""
        idx = self._id_to_index.get(doc_id)
        if idx is None:
            raise KeyError(f"Document {doc_id} not found")
        return self._documents[idx]
    
    def get_by_index(self, index: int) -> Document:
        """Get document by internal index."""
        return self._documents[index]
    
    def contains(self, doc_id: int) -> bool:
        """Check if document exists."""
        return doc_id in self._id_to_index
    
    def __len__(self) -> int:
        return len(self._documents)
    
    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents)
```

---

## 📊 Análisis de Complejidad Completo

### Template COMPLEXITY_ANALYSIS.md

```markdown
# Complexity Analysis - Archimedes Indexer

## Overview

This document analyzes the time and space complexity of all operations
in the Archimedes Indexer search engine.

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

### Estructura y Código (40 pts)
- [ ] Clase `Document` y `Corpus` (5 pts)
- [ ] `Tokenizer` con stop words (5 pts)
- [ ] `InvertedIndex` con AND/OR (10 pts)
- [ ] `quicksort()` implementado (5 pts)
- [ ] `binary_search()` implementado (5 pts)
- [ ] `TFIDFVectorizer` desde cero (5 pts)
- [ ] `cosine_similarity()` implementado (5 pts)

### Testing (20 pts)
- [ ] Tests unitarios para cada módulo (10 pts)
- [ ] Tests de integración (5 pts)
- [ ] Coverage > 80% (5 pts)

### Documentación (20 pts)
- [ ] README.md profesional en inglés (10 pts)
- [ ] `COMPLEXITY_ANALYSIS.md` con Big O (10 pts)

### Funcionalidad (20 pts)
- [ ] Motor busca y retorna resultados (10 pts)
- [ ] Resultados ordenados por relevancia (5 pts)
- [ ] Demo funcional (5 pts)

---

## 🏗️ Arquitectura

```
SearchEngine
    ├── Corpus (Document collection)
    ├── Tokenizer (text → tokens)
    ├── InvertedIndex (term → doc_ids)
    ├── TFIDFVectorizer (docs → vectors)
    └── Ranker (cosine similarity + quicksort)
```

---

## 📊 Análisis de Complejidad Requerido

Documenta en `COMPLEXITY_ANALYSIS.md`:

| Operación | Tu Análisis |
|-----------|-------------|
| `add_document()` | O(?) |
| `build_index()` | O(?) |
| `search(query)` | O(?) |
| `quicksort()` | O(?) promedio, O(?) peor |
| `cosine_similarity()` | O(?) |

---

## 📝 Template README.md

```markdown
# Archimedes Indexer

A search engine built from scratch in pure Python.

## Features
- Inverted index for fast term lookup
- TF-IDF vectorization
- Cosine similarity ranking
- No external dependencies (no numpy, pandas, sklearn)

## Installation
\`\`\`bash
git clone <repo>
cd archimedes-indexer
python -m venv venv
source venv/bin/activate
\`\`\`

## Usage
\`\`\`python
from src.search_engine import SearchEngine

engine = SearchEngine()
engine.add_document(1, "Python Tutorial", "Learn Python...")
engine.build_index()
results = engine.search("python programming")
\`\`\`

## Complexity Analysis
See [COMPLEXITY_ANALYSIS.md](docs/COMPLEXITY_ANALYSIS.md)

## Testing
\`\`\`bash
python -m pytest tests/ -v --cov=src
\`\`\`
```

---

## ✅ Criterios de Aprobación

| Puntuación | Nivel |
|------------|-------|
| 90-100 | Listo para Pathway |
| 75-89 | Reforzar áreas débiles |
| 60-74 | Más práctica necesaria |
| <60 | Revisar módulos |

---

## 🔗 Navegación

| ← Anterior | Índice |
|------------|--------|
| [11_TFIDF_COSENO](11_TFIDF_COSENO.md) | [00_INDICE](00_INDICE.md) |
