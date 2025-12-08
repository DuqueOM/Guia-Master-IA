# Módulo 01 - Python Profesional

> **🎯 Objetivo:** Transformar código Python funcional en código profesional con type hints, funciones puras y estándares de la industria  
> **Fase:** Fundamentos | **Prerrequisito para:** Todos los módulos siguientes

---

## 🧠 Analogía: El Arquitecto vs El Albañil

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                │
│   ALBAÑIL                              ARQUITECTO                              │
│   ────────                             ──────────                              │
│   "Pon ladrillos aquí"                 "Plano estructural con medidas"         │
│   Funciona, pero no escala             Cualquiera puede construirlo            │
│   Solo él sabe cómo                    Verificable y mantenible                │
│                                                                                │
│   def process(x):                      def process(data: list[int]) -> int:    │
│       return x + 1                         """Sum all positive numbers."""     │
│                                            return sum(n for n in data if n>0)  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

El código profesional es como un plano arquitectónico: cualquier ingeniero puede leerlo, entenderlo y construir a partir de él.

---

## 📋 Contenido

1. [Type Hints: Documentación Ejecutable](#1-type-hints)
2. [Funciones Puras vs Impuras](#2-funciones-puras)
3. [PEP8 y Estilo Consistente](#3-pep8)
4. [Docstrings Profesionales](#4-docstrings)
5. [Configuración de Herramientas](#5-configuracion)

---

## 1. Type Hints: Documentación Ejecutable {#1-type-hints}

### 1.1 ¿Por qué Type Hints?

```
┌─────────────────────────────────────────────────────────────────┐
│  SIN TYPE HINTS                                                 │
│  ───────────────                                                │
│  def tokenize(text):     # ¿text es str? ¿bytes? ¿list?         │
│      return text.split() # ¿Retorna list? ¿set? ¿generator?     │
│                                                                 │
│  CON TYPE HINTS                                                 │
│  ──────────────                                                 │
│  def tokenize(text: str) -> list[str]:                          │
│      return text.split()  # Claro: recibe str, retorna list     │
└─────────────────────────────────────────────────────────────────┘
```

**Beneficios:**
- 📖 Documentación que no se desactualiza
- 🐛 Errores detectados antes de ejecutar (con `mypy`)
- 🤖 Autocompletado inteligente en el IDE
- 🔍 Código más fácil de leer y mantener

### 1.2 Tipos Básicos

```python
# Tipos primitivos
name: str = "Archimedes"
count: int = 42
ratio: float = 3.14159
is_active: bool = True
nothing: None = None

# Colecciones (Python 3.9+)
words: list[str] = ["hello", "world"]
scores: dict[str, float] = {"doc1": 0.85, "doc2": 0.92}
unique_words: set[str] = {"the", "and", "or"}
coordinates: tuple[float, float] = (10.5, 20.3)
```

### 1.3 Tipos en Funciones

```python
# ❌ ANTES: ¿Qué recibe? ¿Qué retorna?
def clean(text):
    return text.lower().strip()

# ✅ DESPUÉS: Claro y verificable
def clean(text: str) -> str:
    """Remove whitespace and convert to lowercase."""
    return text.lower().strip()
```

### 1.4 Tipos Avanzados

```python
from typing import Optional, Union

# Optional: puede ser el tipo o None
def find_document(doc_id: int) -> Optional[str]:
    """Return document content or None if not found."""
    if doc_id in documents:
        return documents[doc_id]
    return None

# Union: puede ser uno de varios tipos (Python 3.10+ usa |)
def process(data: Union[str, list[str]]) -> list[str]:
    """Accept string or list of strings."""
    if isinstance(data, str):
        return [data]
    return data

# Python 3.10+ syntax
def process_modern(data: str | list[str]) -> list[str]:
    if isinstance(data, str):
        return [data]
    return data
```

### 1.5 Type Hints para Clases

```python
class Document:
    def __init__(self, doc_id: int, content: str) -> None:
        self.doc_id: int = doc_id
        self.content: str = content
        self.tokens: list[str] = []
    
    def tokenize(self) -> list[str]:
        """Split content into tokens."""
        self.tokens = self.content.lower().split()
        return self.tokens
    
    def word_count(self) -> int:
        """Return number of tokens."""
        return len(self.tokens)
```

### 1.6 Verificación con mypy

```bash
# Instalar mypy
pip install mypy

# Verificar un archivo
mypy src/document.py

# Verificar todo el proyecto
mypy src/

# Configuración en pyproject.toml
```

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
```

---

## 2. Funciones Puras vs Impuras {#2-funciones-puras}

### 2.1 ¿Qué es una Función Pura?

```
┌─────────────────────────────────────────────────────────────────┐
│  FUNCIÓN PURA                                                   │
│  ─────────────                                                  │
│  1. Mismo input → siempre mismo output                          │
│  2. Sin efectos secundarios (no modifica estado externo)        │
│                                                                 │
│  VENTAJAS:                                                      │
│  ✅ Fácil de testear                                            │
│  ✅ Fácil de entender                                           │
│  ✅ Paralelizable                                               │
│  ✅ Cacheable (memoization)                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Ejemplos Comparativos

```python
# ❌ IMPURA: modifica estado externo
results = []

def add_result_impure(value):
    results.append(value)  # Modifica lista externa
    return len(results)

# ✅ PURA: retorna nuevo valor sin modificar nada
def add_result_pure(results: list[int], value: int) -> list[int]:
    return results + [value]  # Retorna nueva lista


# ❌ IMPURA: depende de estado externo
multiplier = 2

def multiply_impure(x):
    return x * multiplier  # Depende de variable externa

# ✅ PURA: todo lo necesario viene como parámetro
def multiply_pure(x: int, multiplier: int) -> int:
    return x * multiplier
```

### 2.3 Evitar Mutación de Argumentos

```python
# ❌ PELIGROSO: modifica el argumento original
def remove_stopwords_bad(tokens: list[str], stopwords: set[str]) -> list[str]:
    for word in list(tokens):  # Itera sobre copia para poder modificar
        if word in stopwords:
            tokens.remove(word)  # ¡Modifica la lista original!
    return tokens

# ✅ SEGURO: crea nueva lista
def remove_stopwords_good(tokens: list[str], stopwords: set[str]) -> list[str]:
    return [word for word in tokens if word not in stopwords]
```

### 2.4 Cuándo las Funciones Impuras Son Necesarias

Algunas operaciones requieren efectos secundarios:
- Escribir a disco
- Imprimir a consola
- Conectar a base de datos
- Generar números aleatorios

**Estrategia:** Aislar las funciones impuras y mantener la lógica de negocio pura.

```python
# Lógica pura (testeable)
def prepare_document(content: str) -> dict[str, any]:
    tokens = content.lower().split()
    return {
        "tokens": tokens,
        "word_count": len(tokens),
        "char_count": len(content)
    }

# Función impura aislada
def save_document(doc_data: dict[str, any], filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(doc_data, f)
```

---

## 3. PEP8 y Estilo Consistente {#3-pep8}

### 3.1 Reglas Esenciales

| Regla | Ejemplo Correcto |
|-------|------------------|
| Indentación: 4 espacios | `def func():⏎····code` |
| Línea máxima: 88-100 caracteres | Configurar en linter |
| Espacios alrededor de operadores | `x = 1 + 2` (no `x=1+2`) |
| Nombres de variables: snake_case | `word_count`, `doc_id` |
| Nombres de clases: PascalCase | `Document`, `InvertedIndex` |
| Constantes: UPPER_CASE | `MAX_TOKENS = 1000` |

### 3.2 Nombres Descriptivos

```python
# ❌ Nombres crípticos
def proc(d):
    r = []
    for i in d:
        if len(i) > 3:
            r.append(i)
    return r

# ✅ Nombres descriptivos
def filter_short_words(tokens: list[str], min_length: int = 3) -> list[str]:
    """Remove tokens shorter than min_length."""
    return [token for token in tokens if len(token) > min_length]
```

### 3.3 Configurar Linter (ruff)

```bash
# Instalar ruff (rápido y moderno)
pip install ruff

# Verificar código
ruff check src/

# Corregir automáticamente
ruff check --fix src/
```

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "UP"]

[tool.ruff.per-file-ignores]
"tests/*" = ["S101"]  # Permitir assert en tests
```

---

## 4. Docstrings Profesionales {#4-docstrings}

### 4.1 Formato Google Style

```python
def compute_tf(term: str, document: list[str]) -> float:
    """Compute Term Frequency for a term in a document.
    
    Term Frequency measures how often a term appears in a document,
    normalized by the total number of terms.
    
    Args:
        term: The word to search for.
        document: List of tokens in the document.
    
    Returns:
        The term frequency as a float between 0 and 1.
    
    Raises:
        ValueError: If document is empty.
    
    Example:
        >>> compute_tf("hello", ["hello", "world", "hello"])
        0.6666666666666666
    """
    if not document:
        raise ValueError("Document cannot be empty")
    
    count = document.count(term)
    return count / len(document)
```

### 4.2 Docstrings para Clases

```python
class Document:
    """Represents a text document with metadata.
    
    A Document holds the original content along with processed
    tokens and provides methods for text analysis.
    
    Attributes:
        doc_id: Unique identifier for the document.
        content: Original text content.
        tokens: List of processed tokens (populated after tokenize()).
    
    Example:
        >>> doc = Document(1, "Hello World")
        >>> doc.tokenize()
        ['hello', 'world']
    """
    
    def __init__(self, doc_id: int, content: str) -> None:
        """Initialize a Document.
        
        Args:
            doc_id: Unique identifier.
            content: Raw text content.
        """
        self.doc_id = doc_id
        self.content = content
        self.tokens: list[str] = []
```

---

## 5. Configuración de Herramientas {#5-configuracion}

### 5.1 pyproject.toml Completo

```toml
[project]
name = "archimedes-indexer"
version = "0.1.0"
description = "A search engine built from scratch"
requires-python = ">=3.11"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
ignore_missing_imports = true

[tool.ruff]
line-length = 88
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "W",   # pycodestyle warnings
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

### 5.2 Comandos de Verificación

```bash
# Verificar tipos
mypy src/

# Verificar estilo
ruff check src/

# Corregir estilo automáticamente
ruff check --fix src/

# Todo junto (crear en Makefile)
make check
```

### 5.3 Makefile Básico

```makefile
.PHONY: check lint type-check test

check: lint type-check test

lint:
	ruff check src/ tests/

type-check:
	mypy src/

test:
	python -m pytest tests/ -v

fix:
	ruff check --fix src/ tests/
```

---

## ⚠️ Errores Comunes y Cómo Evitarlos

### Error 1: Type hints incorrectos

```python
# ❌ Error: list sin tipo genérico
def get_words(text: str) -> list:  # mypy warning
    return text.split()

# ✅ Correcto
def get_words(text: str) -> list[str]:
    return text.split()
```

### Error 2: Mutar argumentos por defecto

```python
# ❌ Bug clásico: lista mutable como default
def add_word(word: str, words: list[str] = []) -> list[str]:
    words.append(word)  # ¡Se acumula entre llamadas!
    return words

# ✅ Correcto: usar None como default
def add_word(word: str, words: list[str] | None = None) -> list[str]:
    if words is None:
        words = []
    return words + [word]
```

### Error 3: Olvidar el return type en __init__

```python
# ❌ Incompleto
def __init__(self, doc_id: int):
    self.doc_id = doc_id

# ✅ Completo (siempre -> None)
def __init__(self, doc_id: int) -> None:
    self.doc_id = doc_id
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 1.1: Tipar una Función
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-11) - Agregar type hints a función de tokenización.

### Ejercicio 1.2: Convertir a Función Pura
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-12) - Refactorizar función impura.

### Ejercicio 1.3: Configurar Linters
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-13) - Crear pyproject.toml completo.

### Ejercicio 1.4: Escribir Docstrings
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-14) - Documentar módulo completo.

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Real Python: Type Checking](https://realpython.com/python-type-checking/) | Tutorial | 🔴 Obligatorio |
| [PEP 8](https://peps.python.org/pep-0008/) | Documentación | 🔴 Obligatorio |
| [mypy Documentation](https://mypy.readthedocs.io/) | Documentación | 🟡 Recomendado |
| [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) | Guía | 🟢 Complementario |

---

## 🔗 Referencias del Glosario

- [Type Hint](GLOSARIO.md#type-hint)
- [Función Pura](GLOSARIO.md#funcion-pura)
- [PEP8](GLOSARIO.md#pep8)
- [Docstring](GLOSARIO.md#docstring)
- [Linter](GLOSARIO.md#linter)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| - | [00_INDICE](00_INDICE.md) | [02_OOP_DESDE_CERO](02_OOP_DESDE_CERO.md) |
