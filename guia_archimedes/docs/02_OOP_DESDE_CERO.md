# Módulo 02 - OOP desde Cero

> **🎯 Objetivo:** Diseñar clases profesionales que representen documentos y colecciones, aplicando principios SOLID básicos  
> **Fase:** Fundamentos | **Prerrequisito para:** Todos los módulos siguientes

---

## 🧠 Analogía: La Fábrica de Documentos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   CLASE = PLANO DE FÁBRICA                                                  │
│   ─────────────────────────                                                 │
│   Document (plano)  ──────►  doc1, doc2, doc3 (productos)                   │
│                                                                             │
│   El plano define:                                                          │
│   • Qué propiedades tiene cada documento (id, contenido, tokens)            │
│   • Qué puede hacer cada documento (tokenizar, contar palabras)             │
│                                                                             │
│   CORPUS = ALMACÉN                                                          │
│   ─────────────────                                                         │
│   Corpus (almacén)  ──────►  Contiene múltiples documentos                  │
│                              Sabe agregar, buscar, iterar                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Clases y Objetos Básicos](#1-clases-basicas)
2. [Métodos Mágicos](#2-metodos-magicos)
3. [Properties y Encapsulamiento](#3-properties)
4. [Composición vs Herencia](#4-composicion)
5. [Principios SOLID Básicos](#5-solid)
6. [Dataclasses](#6-dataclasses)

---

## 1. Clases y Objetos Básicos {#1-clases-basicas}

### 1.1 Anatomía de una Clase

```python
class Document:
    """Represents a single document in the corpus."""
    
    # Atributo de clase (compartido por todas las instancias)
    document_count: int = 0
    
    def __init__(self, doc_id: int, content: str) -> None:
        """Initialize a new Document.
        
        Args:
            doc_id: Unique identifier for this document.
            content: Raw text content of the document.
        """
        # Atributos de instancia (únicos para cada objeto)
        self.doc_id: int = doc_id
        self.content: str = content
        self.tokens: list[str] = []
        
        # Incrementar contador de clase
        Document.document_count += 1
    
    def tokenize(self) -> list[str]:
        """Split content into lowercase tokens.
        
        Returns:
            List of tokens extracted from content.
        """
        self.tokens = self.content.lower().split()
        return self.tokens
    
    def word_count(self) -> int:
        """Return the number of tokens.
        
        Note:
            Must call tokenize() first, or returns 0.
        """
        return len(self.tokens)
```

### 1.2 Creando y Usando Objetos

```python
# Crear instancias (objetos)
doc1 = Document(1, "Hello World")
doc2 = Document(2, "Goodbye World")

# Llamar métodos
doc1.tokenize()
print(doc1.tokens)  # ['hello', 'world']
print(doc1.word_count())  # 2

# Acceder al atributo de clase
print(Document.document_count)  # 2
```

### 1.3 Self: La Referencia al Objeto Actual

```
┌─────────────────────────────────────────────────────────────────┐
│  self = "yo mismo"                                              │
│                                                                 │
│  Cuando llamas doc1.tokenize(), Python traduce a:               │
│  Document.tokenize(doc1)                                        │
│                                                                 │
│  self es simplemente el objeto sobre el que se llama el método  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Métodos Mágicos (Dunder Methods) {#2-metodos-magicos}

### 2.1 Los Más Importantes

| Método | Cuándo se llama | Propósito |
|--------|-----------------|-----------|
| `__init__` | Al crear objeto | Inicializar atributos |
| `__repr__` | `repr(obj)`, debugger | Representación técnica |
| `__str__` | `str(obj)`, `print(obj)` | Representación legible |
| `__eq__` | `obj1 == obj2` | Comparar igualdad |
| `__len__` | `len(obj)` | Retornar "longitud" |
| `__iter__` | `for x in obj` | Hacer iterable |

### 2.2 Implementación Completa

```python
class Document:
    def __init__(self, doc_id: int, content: str) -> None:
        self.doc_id = doc_id
        self.content = content
        self.tokens: list[str] = []
    
    def __repr__(self) -> str:
        """Technical representation for debugging.
        
        Example:
            >>> doc = Document(1, "Hello World")
            >>> repr(doc)
            "Document(doc_id=1, content='Hello World')"
        """
        return f"Document(doc_id={self.doc_id}, content='{self.content[:20]}...')"
    
    def __str__(self) -> str:
        """Human-readable representation.
        
        Example:
            >>> print(doc)
            Document #1: Hello World (2 words)
        """
        word_count = len(self.tokens) if self.tokens else "not tokenized"
        return f"Document #{self.doc_id}: {self.content[:30]}... ({word_count} words)"
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on doc_id.
        
        Two documents are equal if they have the same doc_id.
        """
        if not isinstance(other, Document):
            return NotImplemented
        return self.doc_id == other.doc_id
    
    def __len__(self) -> int:
        """Return number of tokens (after tokenization)."""
        return len(self.tokens)
    
    def __hash__(self) -> int:
        """Make Document hashable (usable in sets/dicts)."""
        return hash(self.doc_id)
```

### 2.3 Uso de Métodos Mágicos

```python
doc = Document(1, "Hello World from Archimedes")
doc.tokenize()

# __repr__ (en debugger o consola)
>>> doc
Document(doc_id=1, content='Hello World from Arc...')

# __str__ (con print)
>>> print(doc)
Document #1: Hello World from Archimedes... (4 words)

# __len__
>>> len(doc)
4

# __eq__
doc2 = Document(1, "Different content")
>>> doc == doc2
True  # Mismo doc_id

# __hash__ permite usar en sets
>>> docs_set = {doc, doc2}
>>> len(docs_set)
1  # Son "iguales" por doc_id
```

---

## 3. Properties y Encapsulamiento {#3-properties}

### 3.1 ¿Por Qué Encapsular?

```
┌─────────────────────────────────────────────────────────────────┐
│  PROBLEMA: Acceso directo sin validación                        │
│                                                                 │
│  doc.doc_id = -5     # ¿ID negativo? ¡Inválido!                 │
│  doc.content = None  # ¿Contenido None? ¡Error futuro!          │
│                                                                 │
│  SOLUCIÓN: Properties con validación                            │
│                                                                 │
│  doc.doc_id = -5     # Lanza ValueError                         │
│  doc.content = None  # Lanza TypeError                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementando Properties

```python
class Document:
    def __init__(self, doc_id: int, content: str) -> None:
        # Usar los setters para validar desde el inicio
        self._doc_id: int = 0  # Atributo "privado" (convención)
        self._content: str = ""
        
        # Estos llaman a los setters
        self.doc_id = doc_id
        self.content = content
        self.tokens: list[str] = []
    
    @property
    def doc_id(self) -> int:
        """Get document ID."""
        return self._doc_id
    
    @doc_id.setter
    def doc_id(self, value: int) -> None:
        """Set document ID with validation."""
        if not isinstance(value, int):
            raise TypeError(f"doc_id must be int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"doc_id must be non-negative, got {value}")
        self._doc_id = value
    
    @property
    def content(self) -> str:
        """Get document content."""
        return self._content
    
    @content.setter
    def content(self, value: str) -> None:
        """Set content with validation."""
        if not isinstance(value, str):
            raise TypeError(f"content must be str, got {type(value).__name__}")
        if not value.strip():
            raise ValueError("content cannot be empty or whitespace only")
        self._content = value
    
    @property
    def is_tokenized(self) -> bool:
        """Check if document has been tokenized (read-only)."""
        return len(self.tokens) > 0
```

### 3.3 Uso de Properties

```python
doc = Document(1, "Hello World")

# Lectura transparente (parece atributo normal)
print(doc.doc_id)  # 1

# Escritura con validación automática
doc.doc_id = 5     # OK
doc.doc_id = -1    # ValueError: doc_id must be non-negative

# Property de solo lectura
print(doc.is_tokenized)  # False
doc.tokenize()
print(doc.is_tokenized)  # True
# doc.is_tokenized = True  # AttributeError: can't set attribute
```

---

## 4. Composición vs Herencia {#4-composicion}

### 4.1 La Regla de Oro

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "Favor composition over inheritance"                          │
│   (Prefiere composición sobre herencia)                         │
│                                                                 │
│   HERENCIA: "ES UN" (is-a)                                      │
│   ─────────────────────────                                     │
│   Un Perro ES UN Animal                                         │
│   ✅ Tiene sentido                                              │
│                                                                 │
│   COMPOSICIÓN: "TIENE UN" (has-a)                               │
│   ───────────────────────────────                               │
│   Un Corpus TIENE Documentos                                    │
│   ✅ Más flexible                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Composición: Corpus Contiene Documents

```python
class Corpus:
    """A collection of documents."""
    
    def __init__(self, name: str) -> None:
        """Initialize an empty corpus.
        
        Args:
            name: Name of this corpus.
        """
        self.name: str = name
        self._documents: dict[int, Document] = {}  # Composición: contiene Documents
    
    def add_document(self, doc: Document) -> None:
        """Add a document to the corpus.
        
        Args:
            doc: Document to add.
        
        Raises:
            ValueError: If document with same ID already exists.
        """
        if doc.doc_id in self._documents:
            raise ValueError(f"Document with id {doc.doc_id} already exists")
        self._documents[doc.doc_id] = doc
    
    def get_document(self, doc_id: int) -> Document | None:
        """Retrieve a document by ID.
        
        Args:
            doc_id: ID of document to retrieve.
        
        Returns:
            The Document if found, None otherwise.
        """
        return self._documents.get(doc_id)
    
    def remove_document(self, doc_id: int) -> bool:
        """Remove a document by ID.
        
        Returns:
            True if document was removed, False if not found.
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False
    
    def __len__(self) -> int:
        """Return number of documents in corpus."""
        return len(self._documents)
    
    def __iter__(self):
        """Iterate over documents."""
        return iter(self._documents.values())
    
    def __contains__(self, doc_id: int) -> bool:
        """Check if document ID exists."""
        return doc_id in self._documents
```

### 4.3 Cuándo Usar Herencia

La herencia es apropiada cuando hay una relación "es un" clara:

```python
from abc import ABC, abstractmethod

class Tokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        pass

class SimpleTokenizer(Tokenizer):
    """Basic whitespace tokenizer."""
    
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()

class AdvancedTokenizer(Tokenizer):
    """Tokenizer that also removes punctuation."""
    
    def __init__(self, min_length: int = 2) -> None:
        self.min_length = min_length
    
    def tokenize(self, text: str) -> list[str]:
        # Remove punctuation
        cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        words = cleaned.lower().split()
        return [w for w in words if len(w) >= self.min_length]
```

---

## 5. Principios SOLID Básicos {#5-solid}

### 5.1 S - Single Responsibility Principle

```
┌─────────────────────────────────────────────────────────────────┐
│  PRINCIPIO: Una clase debe tener una sola razón para cambiar    │
│                                                                 │
│  ❌ MAL: Document que hace todo                                 │
│  ───────────────────────────                                    │
│  class Document:                                                │
│      def tokenize(self): ...                                    │
│      def save_to_file(self): ...      # Persistencia            │
│      def compute_tfidf(self): ...     # Cálculo ML              │
│      def render_html(self): ...       # Presentación            │
│                                                                 │
│  ✅ BIEN: Responsabilidades separadas                           │
│  ──────────────────────────────────                             │
│  class Document:          # Solo datos del documento            │
│  class Tokenizer:         # Solo tokenización                   │
│  class DocumentStorage:   # Solo persistencia                   │
│  class TFIDFCalculator:   # Solo cálculos                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 O - Open/Closed Principle

```python
# ✅ Abierto para extensión, cerrado para modificación

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass

# Extender sin modificar la clase base
class SpanishTokenizer(Tokenizer):
    """Tokenizer with Spanish stop words."""
    
    STOP_WORDS = {"el", "la", "los", "las", "de", "en"}
    
    def tokenize(self, text: str) -> list[str]:
        words = text.lower().split()
        return [w for w in words if w not in self.STOP_WORDS]
```

### 5.3 Aplicación en el Proyecto

```python
# Cada clase tiene una responsabilidad clara:

class Document:
    """Solo almacena datos de un documento."""
    pass

class Corpus:
    """Solo administra una colección de documentos."""
    pass

class Tokenizer:
    """Solo convierte texto en tokens."""
    pass

class InvertedIndex:
    """Solo indexa documentos para búsqueda."""
    pass

class SearchEngine:
    """Orquesta los demás componentes."""
    pass
```

---

## 6. Dataclasses {#6-dataclasses}

### 6.1 Simplificando Clases de Datos

```python
from dataclasses import dataclass, field

# ❌ Mucho boilerplate
class DocumentOld:
    def __init__(self, doc_id: int, content: str, title: str = "") -> None:
        self.doc_id = doc_id
        self.content = content
        self.title = title
    
    def __repr__(self) -> str:
        return f"Document(doc_id={self.doc_id}, content='{self.content[:20]}...', title='{self.title}')"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocumentOld):
            return NotImplemented
        return self.doc_id == other.doc_id and self.content == other.content

# ✅ Dataclass: automático
@dataclass
class Document:
    doc_id: int
    content: str
    title: str = ""
    tokens: list[str] = field(default_factory=list)
    
    # Puedes agregar métodos normalmente
    def tokenize(self) -> list[str]:
        self.tokens = self.content.lower().split()
        return self.tokens
```

### 6.2 Opciones de Dataclass

```python
@dataclass(frozen=True)  # Inmutable (no se puede modificar)
class ImmutableDocument:
    doc_id: int
    content: str

@dataclass(order=True)  # Permite comparar <, >, etc.
class RankedDocument:
    score: float  # Primer campo = criterio de ordenamiento
    doc_id: int
    content: str

# Uso
docs = [RankedDocument(0.8, 1, "doc1"), RankedDocument(0.9, 2, "doc2")]
sorted_docs = sorted(docs, reverse=True)  # Ordenar por score
```

### 6.3 Cuándo Usar Dataclass

| Usa Dataclass cuando... | Usa Clase normal cuando... |
|------------------------|---------------------------|
| Principalmente almacena datos | Lógica compleja de validación |
| __init__, __repr__, __eq__ estándar | Necesitas control total |
| Quieres código conciso | Properties con setters |

---

## ⚠️ Errores Comunes y Cómo Evitarlos

### Error 1: Olvidar self

```python
# ❌ Error: NameError: name 'doc_id' is not defined
class Document:
    def __init__(self, doc_id: int) -> None:
        doc_id = doc_id  # ¡No guarda nada!

# ✅ Correcto
class Document:
    def __init__(self, doc_id: int) -> None:
        self.doc_id = doc_id
```

### Error 2: Mutar lista compartida

```python
# ❌ Bug: todos los documentos comparten la misma lista
class Document:
    tokens: list[str] = []  # ¡Atributo de clase!

# ✅ Correcto: inicializar en __init__
class Document:
    def __init__(self) -> None:
        self.tokens: list[str] = []  # Atributo de instancia
```

### Error 3: __eq__ sin __hash__

```python
# ❌ Si defines __eq__, Python elimina __hash__ por defecto
class Document:
    def __eq__(self, other): ...
    # No se puede usar en sets/dicts

# ✅ Definir ambos
class Document:
    def __eq__(self, other): ...
    def __hash__(self): return hash(self.doc_id)
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 2.1: Clase Document Básica
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-21)

### Ejercicio 2.2: Métodos Mágicos
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-22)

### Ejercicio 2.3: Properties con Validación
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-23)

### Ejercicio 2.4: Clase Corpus
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-24)

### Ejercicio 2.5: Refactorizar a SOLID
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-25)

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Real Python: OOP](https://realpython.com/python3-object-oriented-programming/) | Tutorial | 🔴 Obligatorio |
| [Dataclasses Documentation](https://docs.python.org/3/library/dataclasses.html) | Docs | 🟡 Recomendado |
| [SOLID Principles](https://realpython.com/solid-principles-python/) | Tutorial | 🟡 Recomendado |

---

## 🔗 Referencias del Glosario

- [Clase](GLOSARIO.md#clase)
- [Instancia](GLOSARIO.md#instancia)
- [Método Mágico](GLOSARIO.md#metodo-magico)
- [Property](GLOSARIO.md#property)
- [Composición](GLOSARIO.md#composicion)
- [SOLID](GLOSARIO.md#solid)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [01_PYTHON_PROFESIONAL](01_PYTHON_PROFESIONAL.md) | [00_INDICE](00_INDICE.md) | [03_LOGICA_DISCRETA](03_LOGICA_DISCRETA.md) |
