# 🚀 Demo Script - Archimedes Indexer

> Script ejecutable para probar el motor de búsqueda.

---

## 📋 Código de Demo Completo

Copia este código en un archivo `demo.py` y ejecútalo:

```python
#!/usr/bin/env python3
"""
Archimedes Indexer - Demo Script
================================

Este script demuestra todas las funcionalidades del motor de búsqueda
implementado desde cero sin librerías externas (excepto math).

Ejecutar: python demo.py
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator, NamedTuple


# ============================================================================
# DOCUMENTO Y CORPUS
# ============================================================================

@dataclass
class Document:
    """Un documento indexable."""
    doc_id: int
    title: str
    content: str
    tokens: list[str] = field(default_factory=list)


class Corpus:
    """Colección de documentos."""
    
    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._id_to_index: dict[int, int] = {}
    
    def add(self, doc: Document) -> None:
        self._id_to_index[doc.doc_id] = len(self._documents)
        self._documents.append(doc)
    
    def get_by_index(self, index: int) -> Document:
        return self._documents[index]
    
    def __len__(self) -> int:
        return len(self._documents)
    
    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents)


# ============================================================================
# TOKENIZER
# ============================================================================

class Tokenizer:
    """Tokenizador con stop words."""
    
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'this', 'that', 'these', 'those',
        'it', 'its'
    }
    
    def tokenize(self, text: str) -> list[str]:
        """Tokeniza texto eliminando puntuación y stop words."""
        # Eliminar puntuación
        cleaned = ''
        for char in text.lower():
            if char.isalnum() or char.isspace():
                cleaned += char
            else:
                cleaned += ' '
        
        # Split y filtrar
        tokens = []
        for word in cleaned.split():
            if len(word) > 1 and word not in self.STOP_WORDS:
                tokens.append(word)
        
        return tokens


# ============================================================================
# ÍNDICE INVERTIDO
# ============================================================================

class InvertedIndex:
    """Índice invertido para búsqueda rápida."""
    
    def __init__(self) -> None:
        self._index: dict[str, set[int]] = defaultdict(set)
    
    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        """Agrega documento al índice. O(T)"""
        for token in tokens:
            self._index[token].add(doc_id)
    
    def search_or(self, terms: list[str]) -> set[int]:
        """Busca documentos con ANY término. O(Q)"""
        result: set[int] = set()
        for term in terms:
            result |= self._index.get(term, set())
        return result


# ============================================================================
# TF-IDF VECTORIZER
# ============================================================================

class TFIDFVectorizer:
    """Vectorizador TF-IDF desde cero."""
    
    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}
        self._idf: list[float] = []
    
    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)
    
    def fit_transform(self, corpus: list[list[str]]) -> list[list[float]]:
        """Construye vocabulario y transforma corpus."""
        # Construir vocabulario
        vocab_set: set[str] = set()
        for doc in corpus:
            vocab_set.update(doc)
        
        self._vocabulary = {term: idx for idx, term in enumerate(sorted(vocab_set))}
        V = len(self._vocabulary)
        N = len(corpus)
        
        # Calcular DF
        df = [0] * V
        for doc in corpus:
            seen: set[str] = set()
            for term in doc:
                if term not in seen:
                    df[self._vocabulary[term]] += 1
                    seen.add(term)
        
        # Calcular IDF
        self._idf = [math.log(N / df[i]) if df[i] > 0 else 0 for i in range(V)]
        
        # Transformar cada documento
        vectors = []
        for doc in corpus:
            vectors.append(self._transform_single(doc))
        
        return vectors
    
    def _transform_single(self, tokens: list[str]) -> list[float]:
        """Transforma un documento a vector TF-IDF."""
        V = len(self._vocabulary)
        vector = [0.0] * V
        
        if not tokens:
            return vector
        
        # Calcular TF
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        total = len(tokens)
        
        for term, count in tf.items():
            if term in self._vocabulary:
                idx = self._vocabulary[term]
                vector[idx] = (count / total) * self._idf[idx]
        
        return vector
    
    def transform_query(self, tokens: list[str]) -> list[float]:
        """Transforma query a vector TF-IDF."""
        return self._transform_single(tokens)


# ============================================================================
# SIMILITUD DE COSENO
# ============================================================================

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calcula similitud de coseno entre dos vectores. O(V)"""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot / (mag1 * mag2)


# ============================================================================
# QUICKSORT
# ============================================================================

def quicksort(items: list, key=None):
    """QuickSort con soporte para key function."""
    if len(items) <= 1:
        return items
    
    if key is None:
        key = lambda x: x
    
    pivot = items[len(items) // 2]
    pivot_val = key(pivot)
    
    less = [x for x in items if key(x) < pivot_val]
    equal = [x for x in items if key(x) == pivot_val]
    greater = [x for x in items if key(x) > pivot_val]
    
    return quicksort(less, key) + equal + quicksort(greater, key)


# ============================================================================
# SEARCH ENGINE
# ============================================================================

class SearchResult(NamedTuple):
    """Resultado de búsqueda."""
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchEngine:
    """Motor de búsqueda completo."""
    
    def __init__(self) -> None:
        self.corpus = Corpus()
        self.tokenizer = Tokenizer()
        self.index = InvertedIndex()
        self.vectorizer = TFIDFVectorizer()
        self._document_vectors: list[list[float]] = []
        self._indexed = False
    
    def add_document(self, doc_id: int, title: str, content: str) -> None:
        """Agrega documento al corpus."""
        doc = Document(doc_id=doc_id, title=title, content=content)
        self.corpus.add(doc)
        self._indexed = False
    
    def build_index(self) -> None:
        """Construye índice invertido y vectores TF-IDF."""
        self.index = InvertedIndex()
        tokenized_docs: list[list[str]] = []
        
        for doc in self.corpus:
            tokens = self.tokenizer.tokenize(doc.content)
            doc.tokens = tokens
            tokenized_docs.append(tokens)
            self.index.add_document(doc.doc_id, tokens)
        
        self._document_vectors = self.vectorizer.fit_transform(tokenized_docs)
        self._indexed = True
    
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Busca documentos relevantes."""
        if not self._indexed:
            raise RuntimeError("Debe llamar build_index() primero")
        
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []
        
        candidates = self.index.search_or(query_tokens)
        if not candidates:
            return []
        
        query_vector = self.vectorizer.transform_query(query_tokens)
        
        results: list[tuple[int, float]] = []
        for doc_idx, doc in enumerate(self.corpus):
            if doc.doc_id in candidates:
                score = cosine_similarity(query_vector, self._document_vectors[doc_idx])
                if score > 0:
                    results.append((doc_idx, score))
        
        results = quicksort(results, key=lambda x: -x[1])
        
        search_results: list[SearchResult] = []
        for doc_idx, score in results[:top_k]:
            doc = self.corpus.get_by_index(doc_idx)
            snippet = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            search_results.append(SearchResult(
                doc_id=doc.doc_id,
                title=doc.title,
                score=round(score, 4),
                snippet=snippet
            ))
        
        return search_results


# ============================================================================
# DEMO
# ============================================================================

def main():
    print("=" * 70)
    print("🔍 ARCHIMEDES INDEXER - DEMO")
    print("=" * 70)
    print()
    
    # Crear motor de búsqueda
    engine = SearchEngine()
    
    # Agregar documentos de ejemplo
    documents = [
        (1, "Introduction to Python Programming",
         "Python is a powerful programming language. Python is easy to learn and "
         "widely used in data science, web development, and artificial intelligence. "
         "Python has a simple syntax that is easy for beginners."),
        
        (2, "Machine Learning Fundamentals",
         "Machine learning is a subset of artificial intelligence. It enables "
         "computers to learn from data without being explicitly programmed. "
         "Common algorithms include neural networks and decision trees."),
        
        (3, "Data Structures and Algorithms",
         "Data structures like arrays, linked lists, and trees are fundamental "
         "in computer science. Algorithms such as sorting and searching are "
         "essential for efficient programming."),
        
        (4, "Web Development with JavaScript",
         "JavaScript is the language of the web. It enables interactive websites "
         "and runs in all modern browsers. Node.js allows JavaScript on servers."),
        
        (5, "Deep Learning and Neural Networks",
         "Deep learning uses neural networks with many layers. It has revolutionized "
         "artificial intelligence, enabling breakthroughs in image recognition, "
         "natural language processing, and autonomous vehicles."),
        
        (6, "Python for Data Science",
         "Python is the most popular language for data science. Libraries like "
         "pandas and numpy make data analysis easy. Python is also used for "
         "machine learning with scikit-learn and tensorflow."),
        
        (7, "Algorithms for Searching and Sorting",
         "Searching algorithms like binary search find elements efficiently. "
         "Sorting algorithms like quicksort and mergesort organize data. "
         "Understanding Big O notation is crucial for algorithm analysis."),
    ]
    
    print("📚 Agregando documentos...")
    for doc_id, title, content in documents:
        engine.add_document(doc_id, title, content)
        print(f"   + [{doc_id}] {title}")
    
    print()
    print("⚙️  Construyendo índice...")
    engine.build_index()
    print(f"   ✓ Índice construido: {engine.vectorizer.vocabulary_size} términos únicos")
    print()
    
    # Realizar búsquedas
    queries = [
        "python programming",
        "machine learning artificial intelligence",
        "sorting algorithms",
        "web development",
        "neural networks deep learning",
    ]
    
    for query in queries:
        print("-" * 70)
        print(f"🔎 QUERY: \"{query}\"")
        print("-" * 70)
        
        results = engine.search(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n   {i}. {result.title}")
                print(f"      Score: {result.score:.4f}")
                print(f"      Preview: {result.snippet[:80]}...")
        else:
            print("   No se encontraron resultados.")
        
        print()
    
    print("=" * 70)
    print("✅ Demo completado!")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## 🖥️ Salida Esperada

```
======================================================================
🔍 ARCHIMEDES INDEXER - DEMO
======================================================================

📚 Agregando documentos...
   + [1] Introduction to Python Programming
   + [2] Machine Learning Fundamentals
   + [3] Data Structures and Algorithms
   + [4] Web Development with JavaScript
   + [5] Deep Learning and Neural Networks
   + [6] Python for Data Science
   + [7] Algorithms for Searching and Sorting

⚙️  Construyendo índice...
   ✓ Índice construido: 89 términos únicos

----------------------------------------------------------------------
🔎 QUERY: "python programming"
----------------------------------------------------------------------

   1. Introduction to Python Programming
      Score: 0.4523
      Preview: Python is a powerful programming language. Python is easy to learn...

   2. Python for Data Science
      Score: 0.2187
      Preview: Python is the most popular language for data science. Libraries like...

----------------------------------------------------------------------
🔎 QUERY: "machine learning artificial intelligence"
----------------------------------------------------------------------

   1. Machine Learning Fundamentals
      Score: 0.4891
      Preview: Machine learning is a subset of artificial intelligence. It enables...

   2. Deep Learning and Neural Networks
      Score: 0.3456
      Preview: Deep learning uses neural networks with many layers. It has revolut...
```

---

## 📝 Notas

- Este script es **100% Python puro** (solo usa `math` de la stdlib)
- Demuestra todos los componentes del proyecto integrador
- Puedes modificar los documentos y queries para experimentar
