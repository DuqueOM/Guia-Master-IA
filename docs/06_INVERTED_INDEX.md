# 06 - Índice Invertido

> **🎯 Objetivo:** Construir el núcleo del motor de búsqueda: un índice invertido que mapea palabras a documentos.

---

## 🧠 Analogía: El Índice de un Libro de Texto

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   LIBRO DE TEXTO: Índice al final                                           │
│   ───────────────────────────────                                           │
│                                                                             │
│   "algoritmo" .......... páginas 12, 45, 78, 134                            │
│   "array" .............. páginas 23, 56                                     │
│   "búsqueda binaria" ... páginas 89, 90, 91                                 │
│   "recursión" .......... páginas 67, 68, 150                                │
│                                                                             │
│   Sin este índice: leer TODO el libro para encontrar "recursión"            │
│   Con el índice: ir directo a las páginas 67, 68, 150                       │
│                                                                             │
│   ÍNDICE INVERTIDO = Lo mismo, pero para TODOS los documentos               │
│   ───────────────────────────────────────────────────────────               │
│                                                                             │
│   "python" → [doc_1, doc_3, doc_7]                                          │
│   "search" → [doc_2, doc_3]                                                 │
│   "engine" → [doc_1, doc_2, doc_3]                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [¿Qué es un Índice Invertido?](#1-que-es)
2. [Estructura de Datos](#2-estructura)
3. [Implementación Básica](#3-implementacion)
4. [Búsqueda con AND/OR](#4-busqueda)
5. [Índice con Frecuencias](#5-frecuencias)
6. [Análisis de Complejidad](#6-analisis)

---

## 1. ¿Qué es un Índice Invertido? {#1-que-es}

### 1.1 Forward Index vs Inverted Index

```
┌─────────────────────────────────────────────────────────────────┐
│  FORWARD INDEX (índice directo)                                 │
│  ─────────────────────────────                                  │
│  doc_1 → ["python", "code", "example"]                          │
│  doc_2 → ["java", "code", "tutorial"]                           │
│  doc_3 → ["python", "tutorial", "search"]                       │
│                                                                 │
│  Para buscar "python": revisar TODOS los documentos → O(n×m)    │
│                                                                 │
│  INVERTED INDEX (índice invertido)                              │
│  ─────────────────────────────────                              │
│  "python"   → [doc_1, doc_3]                                    │
│  "code"     → [doc_1, doc_2]                                    │
│  "tutorial" → [doc_2, doc_3]                                    │
│  "example"  → [doc_1]                                           │
│  "java"     → [doc_2]                                           │
│  "search"   → [doc_3]                                           │
│                                                                 │
│  Para buscar "python": lookup directo → O(1)                    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Por Qué Todos los Buscadores lo Usan

- **Google, Bing, DuckDuckGo:** Índices invertidos masivos
- **Elasticsearch, Solr:** Bases de datos de búsqueda basadas en índices invertidos
- **Bases de datos SQL:** Índices B-tree para columnas buscables

```
Sin índice: buscar en 1 billón de documentos → 1 billón de comparaciones
Con índice: buscar en 1 billón de documentos → 1 lookup + leer lista de docs
```

---

## 2. Estructura de Datos {#2-estructura}

### 2.1 Representación Básica

```python
# Estructura más simple: palabra → lista de doc_ids
InvertedIndex = dict[str, list[int]]

# Ejemplo:
index: InvertedIndex = {
    "python": [1, 3, 5],
    "java": [2, 4],
    "code": [1, 2, 3, 4, 5]
}
```

### 2.2 Representación con Sets (Mejor para AND/OR)

```python
# Con sets: operaciones de conjuntos más eficientes
InvertedIndex = dict[str, set[int]]

index: InvertedIndex = {
    "python": {1, 3, 5},
    "java": {2, 4},
    "code": {1, 2, 3, 4, 5}
}

# Búsqueda AND: documentos con "python" Y "code"
result = index["python"] & index["code"]  # {1, 3, 5}

# Búsqueda OR: documentos con "python" O "java"
result = index["python"] | index["java"]  # {1, 2, 3, 4, 5}
```

### 2.3 Representación con Frecuencias

```python
# Para ranking: guardar cuántas veces aparece cada palabra
# palabra → {doc_id: frecuencia}
InvertedIndexWithFreq = dict[str, dict[int, int]]

index: InvertedIndexWithFreq = {
    "python": {1: 3, 3: 1, 5: 2},  # doc_1 tiene "python" 3 veces
    "java": {2: 5, 4: 1},
    "code": {1: 1, 2: 1, 3: 2, 4: 1, 5: 1}
}
```

---

## 3. Implementación Básica {#3-implementacion}

### 3.1 Clase InvertedIndex

```python
from collections import defaultdict
from typing import Iterator


class InvertedIndex:
    """Inverted index for text search.
    
    Maps terms to the set of document IDs containing them.
    
    Attributes:
        _index: Internal dictionary mapping terms to doc_id sets.
        _doc_count: Number of documents indexed.
    
    Example:
        >>> idx = InvertedIndex()
        >>> idx.add_document(1, ["hello", "world"])
        >>> idx.add_document(2, ["hello", "python"])
        >>> idx.search("hello")
        {1, 2}
    """
    
    def __init__(self) -> None:
        """Initialize empty inverted index."""
        self._index: defaultdict[str, set[int]] = defaultdict(set)
        self._doc_count: int = 0
        self._doc_ids: set[int] = set()
    
    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        """Add a document to the index.
        
        Args:
            doc_id: Unique identifier for the document.
            tokens: List of tokens (words) in the document.
        
        Raises:
            ValueError: If doc_id already exists in index.
        
        Complexity:
            O(t) where t = len(tokens)
        """
        if doc_id in self._doc_ids:
            raise ValueError(f"Document {doc_id} already indexed")
        
        self._doc_ids.add(doc_id)
        self._doc_count += 1
        
        for token in tokens:
            self._index[token].add(doc_id)
    
    def search(self, term: str) -> set[int]:
        """Find all documents containing a term.
        
        Args:
            term: Word to search for.
        
        Returns:
            Set of document IDs containing the term.
        
        Complexity:
            O(1) for lookup (returns reference to existing set)
        """
        return self._index.get(term, set()).copy()
    
    def get_term_count(self) -> int:
        """Return number of unique terms in index."""
        return len(self._index)
    
    def get_document_count(self) -> int:
        """Return number of indexed documents."""
        return self._doc_count
    
    def contains_term(self, term: str) -> bool:
        """Check if term exists in index."""
        return term in self._index
    
    def get_document_frequency(self, term: str) -> int:
        """Return number of documents containing term.
        
        Also known as DF (Document Frequency).
        """
        return len(self._index.get(term, set()))
    
    def __repr__(self) -> str:
        return (
            f"InvertedIndex(terms={self.get_term_count()}, "
            f"documents={self._doc_count})"
        )
    
    def __contains__(self, term: str) -> bool:
        """Allow 'term in index' syntax."""
        return self.contains_term(term)
    
    def __len__(self) -> int:
        """Return number of terms."""
        return self.get_term_count()
```

### 3.2 Uso Básico

```python
# Crear índice
index = InvertedIndex()

# Agregar documentos (ya tokenizados)
index.add_document(1, ["python", "programming", "tutorial"])
index.add_document(2, ["java", "programming", "guide"])
index.add_document(3, ["python", "data", "science"])

# Buscar
print(index.search("python"))       # {1, 3}
print(index.search("programming"))  # {1, 2}
print(index.search("missing"))      # set()

# Información del índice
print(index.get_term_count())       # 7 (términos únicos)
print(index.get_document_count())   # 3
print(index.get_document_frequency("python"))  # 2
```

---

## 4. Búsqueda con AND/OR {#4-busqueda}

### 4.1 Implementación de Búsqueda Multi-Término

```python
class InvertedIndex:
    # ... (métodos anteriores) ...
    
    def search_and(self, terms: list[str]) -> set[int]:
        """Find documents containing ALL terms.
        
        Args:
            terms: List of terms to search for.
        
        Returns:
            Set of doc IDs containing all terms.
        
        Example:
            >>> idx.search_and(["python", "data"])
            {3}  # Only doc 3 has both
        
        Complexity:
            O(t × min_set_size) where t = len(terms)
        """
        if not terms:
            return set()
        
        # Start with docs containing first term
        result = self.search(terms[0])
        
        # Intersect with docs containing each subsequent term
        for term in terms[1:]:
            result &= self._index.get(term, set())
            
            # Early exit if no matches
            if not result:
                return set()
        
        return result
    
    def search_or(self, terms: list[str]) -> set[int]:
        """Find documents containing ANY term.
        
        Args:
            terms: List of terms to search for.
        
        Returns:
            Set of doc IDs containing at least one term.
        
        Example:
            >>> idx.search_or(["python", "java"])
            {1, 2, 3}  # All docs with either
        
        Complexity:
            O(t × avg_set_size) where t = len(terms)
        """
        result: set[int] = set()
        
        for term in terms:
            result |= self._index.get(term, set())
        
        return result
    
    def search_phrase(self, query: str) -> set[int]:
        """Search for documents matching query.
        
        Tokenizes query and performs AND search.
        
        Args:
            query: Search query string.
        
        Returns:
            Set of matching document IDs.
        """
        # Simple tokenization (should use proper tokenizer)
        terms = query.lower().split()
        return self.search_and(terms)
```

### 4.2 Ejemplo de Búsqueda

```python
index = InvertedIndex()
index.add_document(1, ["python", "web", "flask"])
index.add_document(2, ["python", "data", "pandas"])
index.add_document(3, ["java", "web", "spring"])
index.add_document(4, ["python", "web", "django"])

# AND: documentos con python Y web
result = index.search_and(["python", "web"])
print(result)  # {1, 4}

# OR: documentos con flask O django
result = index.search_or(["flask", "django"])
print(result)  # {1, 4}

# Combinado: (python AND web) OR java
python_web = index.search_and(["python", "web"])
java_docs = index.search("java")
result = python_web | java_docs
print(result)  # {1, 3, 4}
```

---

## 5. Índice con Frecuencias {#5-frecuencias}

### 5.1 Para TF-IDF Necesitamos Frecuencias

```python
from collections import defaultdict
from typing import NamedTuple


class TermInfo(NamedTuple):
    """Information about a term in a document."""
    doc_id: int
    frequency: int


class InvertedIndexWithFreq:
    """Inverted index that stores term frequencies.
    
    Needed for TF-IDF ranking.
    """
    
    def __init__(self) -> None:
        # term → {doc_id: frequency}
        self._index: defaultdict[str, dict[int, int]] = defaultdict(dict)
        self._doc_lengths: dict[int, int] = {}  # doc_id → total tokens
        self._doc_count: int = 0
    
    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        """Add document with frequency tracking.
        
        Args:
            doc_id: Unique document identifier.
            tokens: List of tokens in document.
        """
        if doc_id in self._doc_lengths:
            raise ValueError(f"Document {doc_id} already indexed")
        
        # Count frequencies
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Add to index
        for token, count in token_counts.items():
            self._index[token][doc_id] = count
        
        self._doc_lengths[doc_id] = len(tokens)
        self._doc_count += 1
    
    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """Get frequency of term in specific document.
        
        Returns 0 if term not in document.
        """
        return self._index.get(term, {}).get(doc_id, 0)
    
    def get_document_frequency(self, term: str) -> int:
        """Get number of documents containing term."""
        return len(self._index.get(term, {}))
    
    def get_documents_for_term(self, term: str) -> dict[int, int]:
        """Get all documents containing term with frequencies.
        
        Returns:
            Dict mapping doc_id to term frequency.
        """
        return self._index.get(term, {}).copy()
    
    def get_document_length(self, doc_id: int) -> int:
        """Get total token count for document."""
        return self._doc_lengths.get(doc_id, 0)
    
    def get_all_doc_ids(self) -> set[int]:
        """Get set of all indexed document IDs."""
        return set(self._doc_lengths.keys())
    
    @property
    def total_documents(self) -> int:
        """Total number of indexed documents."""
        return self._doc_count
```

### 5.2 Uso del Índice con Frecuencias

```python
index = InvertedIndexWithFreq()

# Documento 1: "python" aparece 3 veces
index.add_document(1, ["python", "python", "code", "python", "tutorial"])

# Documento 2: "python" aparece 1 vez
index.add_document(2, ["java", "code", "python"])

# Obtener frecuencias
print(index.get_term_frequency("python", 1))  # 3
print(index.get_term_frequency("python", 2))  # 1
print(index.get_term_frequency("python", 3))  # 0 (doc no existe)

# Document frequency (en cuántos docs aparece)
print(index.get_document_frequency("python"))  # 2
print(index.get_document_frequency("java"))    # 1

# Para TF-IDF
print(index.get_document_length(1))  # 5 tokens totales
print(index.total_documents)         # 2
```

---

## 6. Análisis de Complejidad {#6-analisis}

### 6.1 Complejidad de Operaciones

```
┌─────────────────────────────────────────────────────────────────┐
│  OPERACIÓN                      │ COMPLEJIDAD                   │
│  ────────────────────────────── │ ───────────                   │
│  add_document(doc_id, tokens)   │ O(t) donde t = len(tokens)    │
│  search(term)                   │ O(1) lookup + O(k) copia      │
│  search_and([terms])            │ O(t × s) t=terms, s=set size  │
│  search_or([terms])             │ O(t × s)                      │
│  get_document_frequency(term)   │ O(1)                          │
│  contains_term(term)            │ O(1)                          │
└─────────────────────────────────────────────────────────────────┘

Donde:
- t = número de tokens
- k = número de documentos que contienen el término
- s = tamaño promedio de los sets de documentos
```

### 6.2 Complejidad de Espacio

```
┌─────────────────────────────────────────────────────────────────┐
│  ESPACIO DEL ÍNDICE                                             │
│                                                                 │
│  Si tenemos:                                                    │
│  - D documentos                                                 │
│  - V términos únicos (vocabulario)                              │
│  - T tokens totales                                             │
│                                                                 │
│  Índice básico (sin frecuencias):                               │
│  - Diccionario: O(V) entradas                                   │
│  - Sets: O(T) referencias a doc_ids en total                    │
│  - Total: O(V + T)                                              │
│                                                                 │
│  En la práctica:                                                │
│  - El índice es MUCHO más pequeño que los documentos            │
│  - Solo guardamos doc_ids, no el texto completo                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Ejemplo de Análisis

```python
"""
ANÁLISIS DE COMPLEJIDAD: add_document()

def add_document(self, doc_id: int, tokens: list[str]) -> None:
    if doc_id in self._doc_ids:          # O(1) - set lookup
        raise ValueError(...)
    
    self._doc_ids.add(doc_id)            # O(1) - set add
    self._doc_count += 1                  # O(1)
    
    for token in tokens:                  # O(t) iteraciones
        self._index[token].add(doc_id)    # O(1) dict + set

TOTAL: O(1) + O(1) + O(1) + O(t × 1) = O(t)

Donde t = len(tokens)
"""
```

---

## ⚠️ Errores Comunes

### Error 1: Retornar referencia al set interno

```python
# ❌ Peligroso: permite modificar el índice externamente
def search(self, term: str) -> set[int]:
    return self._index.get(term, set())  # Retorna referencia

result = index.search("python")
result.add(999)  # ¡Modifica el índice!

# ✅ Seguro: retornar copia
def search(self, term: str) -> set[int]:
    return self._index.get(term, set()).copy()
```

### Error 2: No manejar términos no encontrados

```python
# ❌ KeyError si el término no existe
def search(self, term: str) -> set[int]:
    return self._index[term]

# ✅ Retornar set vacío
def search(self, term: str) -> set[int]:
    return self._index.get(term, set()).copy()
```

### Error 3: Indexar documento duplicado

```python
# ❌ Silenciosamente duplica
def add_document(self, doc_id: int, tokens: list[str]) -> None:
    for token in tokens:
        self._index[token].add(doc_id)  # doc_id ya podría estar

# ✅ Verificar y lanzar error
def add_document(self, doc_id: int, tokens: list[str]) -> None:
    if doc_id in self._doc_ids:
        raise ValueError(f"Document {doc_id} already indexed")
    # ...
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 6.1: Índice Básico
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-61)

### Ejercicio 6.2: Búsqueda AND/OR
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-62)

### Ejercicio 6.3: Índice con Frecuencias
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-63)

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Inverted Index - Wikipedia](https://en.wikipedia.org/wiki/Inverted_index) | Lectura | 🔴 Obligatorio |
| [How Search Engines Work](https://www.youtube.com/watch?v=JZBhBaznk0k) | Video | 🟡 Recomendado |
| [Elasticsearch Internals](https://www.elastic.co/blog/found-elasticsearch-from-the-bottom-up) | Blog | 🟢 Complementario |

---

## 🔗 Referencias del Glosario

- [Índice Invertido](GLOSARIO.md#indice-invertido)
- [Document Frequency](GLOSARIO.md#document-frequency)
- [Term Frequency](GLOSARIO.md#term-frequency)
- [Posting List](GLOSARIO.md#posting-list)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [05_HASHMAPS_SETS](05_HASHMAPS_SETS.md) | [00_INDICE](00_INDICE.md) | [07_RECURSION](07_RECURSION.md) |
