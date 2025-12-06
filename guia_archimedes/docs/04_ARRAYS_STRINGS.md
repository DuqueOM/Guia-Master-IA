# 04 - Arrays, Strings y Memoria

> **🎯 Objetivo:** Dominar la manipulación de listas y strings en Python, entendiendo su complejidad y construyendo un tokenizador básico.

---

## 🧠 Analogía: El Estante de Libros

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   LISTA = ESTANTE DE LIBROS NUMERADO                                        │
│   ──────────────────────────────────                                        │
│                                                                             │
│   Posición:  [0]     [1]     [2]     [3]     [4]                            │
│              ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐                          │
│              │ A │   │ B │   │ C │   │ D │   │ E │                          │
│              └───┘   └───┘   └───┘   └───┘   └───┘                          │
│                                                                             │
│   • Acceder a [2] → Inmediato (O(1)): "Voy al estante 2"                    │
│   • Insertar al final → Rápido: solo añadir al final                        │
│   • Insertar al inicio → Lento: mover todos los demás                       │
│                                                                             │
│   STRING = COLLAR DE CUENTAS (no puedes cambiar una cuenta)                 │
│   ─────────────────────────────────────────────────────────                 │
│   "HELLO" → Si quieres cambiar 'E' por 'A', debes hacer nuevo collar        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Listas en Python: Bajo Nivel](#1-listas)
2. [Slicing y Copias](#2-slicing)
3. [Complejidad de Operaciones](#3-complejidad)
4. [Strings: Inmutabilidad](#4-strings)
5. [Tokenización: Tu Primer Componente](#5-tokenizacion)

---

## 1. Listas en Python: Bajo Nivel {#1-listas}

### 1.1 Cómo Funciona una Lista

```
┌─────────────────────────────────────────────────────────────────┐
│  INTERNAMENTE: Array dinámico                                   │
│                                                                 │
│  Memoria:   [ptr0][ptr1][ptr2][ptr3][____][____]                │
│              ↓     ↓     ↓     ↓                                │
│            "hi" "world"  42   3.14                              │
│                                                                 │
│  La lista guarda PUNTEROS a los objetos, no los objetos         │
│  Tiene espacio extra para crecer sin reasignar                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Creación y Acceso

```python
# Crear listas
words: list[str] = ["hello", "world", "python"]
numbers: list[int] = [1, 2, 3, 4, 5]
mixed: list = [1, "two", 3.0, None]  # Evitar en código tipado

# Acceso por índice: O(1)
first = words[0]      # "hello"
last = words[-1]      # "python" (desde el final)

# Longitud: O(1) (Python guarda el tamaño)
length = len(words)   # 3

# Modificación: O(1)
words[0] = "hi"       # ["hi", "world", "python"]
```

### 1.3 Agregar y Eliminar

```python
words = ["a", "b", "c"]

# Agregar al final: O(1) amortizado
words.append("d")     # ["a", "b", "c", "d"]

# Agregar al inicio: O(n) - ¡LENTO!
words.insert(0, "z")  # ["z", "a", "b", "c", "d"]
# Todos los elementos deben moverse

# Extender con otra lista: O(k) donde k = len(otra_lista)
words.extend(["e", "f"])  # ["z", "a", "b", "c", "d", "e", "f"]

# Eliminar del final: O(1)
last = words.pop()    # Retorna "f", words = ["z", "a", "b", "c", "d", "e"]

# Eliminar del inicio: O(n) - ¡LENTO!
first = words.pop(0)  # Retorna "z", todos deben moverse

# Eliminar por valor: O(n) - busca y luego mueve
words.remove("c")     # Busca "c" y lo elimina
```

---

## 2. Slicing y Copias {#2-slicing}

### 2.1 Slicing Básico

```python
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Sintaxis: list[start:stop:step]
nums[2:5]      # [2, 3, 4]      - desde índice 2 hasta 5 (no incluido)
nums[:3]       # [0, 1, 2]      - desde inicio hasta 3
nums[7:]       # [7, 8, 9]      - desde 7 hasta el final
nums[::2]      # [0, 2, 4, 6, 8] - cada 2 elementos
nums[::-1]     # [9, 8, ..., 0]  - reverso

# Índices negativos
nums[-3:]      # [7, 8, 9]      - últimos 3
nums[:-2]      # [0, 1, ..., 7] - todos menos últimos 2
```

### 2.2 Copia Superficial vs Profunda

```python
# ⚠️ ASIGNACIÓN: NO ES COPIA, es alias
original = [1, 2, 3]
alias = original
alias[0] = 99
print(original)  # [99, 2, 3] ¡Original modificado!

# ✅ COPIA SUPERFICIAL: nueva lista, mismos objetos internos
original = [1, 2, 3]
copy1 = original[:]       # Slicing
copy2 = original.copy()   # Método copy
copy3 = list(original)    # Constructor

copy1[0] = 99
print(original)  # [1, 2, 3] ¡Original intacto!

# ⚠️ Con objetos anidados, copia superficial NO es suficiente
nested = [[1, 2], [3, 4]]
shallow = nested.copy()
shallow[0][0] = 99        # Modifica el objeto interno
print(nested)             # [[99, 2], [3, 4]] ¡Modificado!

# ✅ COPIA PROFUNDA: copia todo recursivamente
import copy
nested = [[1, 2], [3, 4]]
deep = copy.deepcopy(nested)
deep[0][0] = 99
print(nested)             # [[1, 2], [3, 4]] ¡Intacto!
```

### 2.3 Cuándo Importa

```python
# ❌ Bug común: modificar lista mientras se itera
def remove_short_words_bad(words: list[str]) -> list[str]:
    for word in words:  # Itera sobre la misma lista
        if len(word) < 3:
            words.remove(word)  # ¡Modifica durante iteración!
    return words

# ✅ Solución 1: crear nueva lista
def remove_short_words_good(words: list[str]) -> list[str]:
    return [w for w in words if len(w) >= 3]

# ✅ Solución 2: iterar sobre copia
def remove_short_words_alt(words: list[str]) -> list[str]:
    for word in words[:]:  # Copia con [:]
        if len(word) < 3:
            words.remove(word)
    return words
```

---

## 3. Complejidad de Operaciones {#3-complejidad}

### 3.1 Tabla Completa

| Operación | Complejidad | Ejemplo |
|-----------|-------------|---------|
| Acceso `list[i]` | O(1) | `words[5]` |
| Asignar `list[i] = x` | O(1) | `words[5] = "new"` |
| `len(list)` | O(1) | `len(words)` |
| `list.append(x)` | O(1)* | `words.append("x")` |
| `list.pop()` | O(1) | `words.pop()` |
| `list.insert(0, x)` | O(n) | `words.insert(0, "x")` |
| `list.pop(0)` | O(n) | `words.pop(0)` |
| `x in list` | O(n) | `"hello" in words` |
| `list.index(x)` | O(n) | `words.index("hello")` |
| `list.count(x)` | O(n) | `words.count("the")` |
| `list.remove(x)` | O(n) | `words.remove("hello")` |
| `list.sort()` | O(n log n) | `words.sort()` |
| Slice `list[a:b]` | O(b-a) | `words[5:10]` |
| `list.extend(k)` | O(k) | `words.extend(["a","b"])` |

*Amortizado: ocasionalmente O(n) cuando se reasigna memoria.

### 3.2 Implicaciones Prácticas

```python
# ❌ Ineficiente: insertar al inicio muchas veces → O(n²) total
def build_reversed_bad(items: list[str]) -> list[str]:
    result = []
    for item in items:
        result.insert(0, item)  # O(n) cada vez
    return result

# ✅ Eficiente: append y luego revertir → O(n) total
def build_reversed_good(items: list[str]) -> list[str]:
    result = []
    for item in items:
        result.append(item)  # O(1) cada vez
    result.reverse()  # O(n) una vez
    return result

# ✅ Más pythonic
def build_reversed_best(items: list[str]) -> list[str]:
    return items[::-1]
```

---

## 4. Strings: Inmutabilidad {#4-strings}

### 4.1 Strings Son Inmutables

```python
text = "Hello"

# ❌ No puedes modificar un carácter
text[0] = "J"  # TypeError: 'str' object does not support item assignment

# ✅ Debes crear un nuevo string
text = "J" + text[1:]  # "Jello"

# Cada operación crea un NUEVO string
text = "Hello"
text = text + " World"  # Nuevo objeto, no modificación
text = text.lower()     # Nuevo objeto
text = text.strip()     # Nuevo objeto
```

### 4.2 Concatenación Eficiente

```python
# ❌ Ineficiente: muchas concatenaciones → O(n²)
def build_string_bad(words: list[str]) -> str:
    result = ""
    for word in words:
        result = result + word + " "  # Crea nuevo string cada vez
    return result.strip()

# ✅ Eficiente: join → O(n)
def build_string_good(words: list[str]) -> str:
    return " ".join(words)

# Benchmark con 10,000 palabras:
# build_string_bad:  ~0.1s
# build_string_good: ~0.001s (100x más rápido)
```

### 4.3 Métodos de String Útiles

```python
text = "  Hello, World! How are you?  "

# Limpieza
text.strip()      # "Hello, World! How are you?"
text.lower()      # "  hello, world! how are you?  "
text.upper()      # "  HELLO, WORLD! HOW ARE YOU?  "

# Búsqueda
text.find("World")     # 9 (índice) o -1 si no existe
text.count("o")        # 3
"Hello" in text        # True
text.startswith("  H") # True
text.endswith("?  ")   # True

# División
text.split()           # ["Hello,", "World!", "How", "are", "you?"]
text.split(",")        # ["  Hello", " World! How are you?  "]

# Reemplazo
text.replace("!", "")  # Sin signos de exclamación
text.replace(" ", "_") # Espacios por guiones bajos

# Verificación
"hello".isalpha()      # True (solo letras)
"hello123".isalnum()   # True (letras y números)
"123".isdigit()        # True (solo dígitos)
"   ".isspace()        # True (solo espacios)
```

---

## 5. Tokenización: Tu Primer Componente {#5-tokenizacion}

### 5.1 ¿Qué es Tokenización?

```
┌─────────────────────────────────────────────────────────────────┐
│  TOKENIZACIÓN = Convertir texto en unidades procesables         │
│                                                                 │
│  Entrada:  "Hello, World! How are you?"                         │
│  Salida:   ["hello", "world", "how", "are", "you"]              │
│                                                                 │
│  Pasos típicos:                                                 │
│  1. Convertir a minúsculas                                      │
│  2. Eliminar puntuación                                         │
│  3. Dividir por espacios                                        │
│  4. Filtrar palabras vacías (stop words)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Tokenizador Básico

```python
def tokenize_basic(text: str) -> list[str]:
    """Split text into lowercase words.
    
    Args:
        text: Input text to tokenize.
    
    Returns:
        List of lowercase tokens.
    
    Example:
        >>> tokenize_basic("Hello, World!")
        ['hello,', 'world!']
    """
    return text.lower().split()
```

### 5.3 Tokenizador con Limpieza de Puntuación

```python
def remove_punctuation(text: str) -> str:
    """Remove all punctuation from text.
    
    Uses character-by-character filtering.
    
    Args:
        text: Text potentially containing punctuation.
    
    Returns:
        Text with punctuation replaced by spaces.
    """
    result = []
    for char in text:
        if char.isalnum() or char.isspace():
            result.append(char)
        else:
            result.append(' ')  # Reemplazar puntuación por espacio
    return ''.join(result)


def tokenize_clean(text: str) -> list[str]:
    """Tokenize text with punctuation removal.
    
    Args:
        text: Input text.
    
    Returns:
        List of clean, lowercase tokens.
    
    Example:
        >>> tokenize_clean("Hello, World! How are you?")
        ['hello', 'world', 'how', 'are', 'you']
    """
    cleaned = remove_punctuation(text)
    return cleaned.lower().split()
```

### 5.4 Tokenizador con Stop Words

```python
# Stop words comunes en inglés
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "in", "on", "at", "by", "for", "with", "about", "to", "from",
    "of", "as", "if", "then", "than", "so", "no", "not", "only"
})


def tokenize(
    text: str,
    remove_stopwords: bool = True,
    min_length: int = 2
) -> list[str]:
    """Full tokenization pipeline.
    
    Args:
        text: Input text to tokenize.
        remove_stopwords: Whether to filter out stop words.
        min_length: Minimum token length to keep.
    
    Returns:
        List of processed tokens.
    
    Example:
        >>> tokenize("The quick brown fox jumps over the lazy dog.")
        ['quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
    """
    # 1. Remove punctuation
    cleaned = remove_punctuation(text)
    
    # 2. Lowercase and split
    tokens = cleaned.lower().split()
    
    # 3. Filter by length
    tokens = [t for t in tokens if len(t) >= min_length]
    
    # 4. Remove stop words
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    
    return tokens
```

### 5.5 Clase Tokenizer (Aplicando OOP)

```python
class Tokenizer:
    """Configurable text tokenizer.
    
    Attributes:
        stop_words: Set of words to filter out.
        min_length: Minimum token length.
    
    Example:
        >>> tokenizer = Tokenizer()
        >>> tokenizer.tokenize("Hello, World!")
        ['hello', 'world']
    """
    
    DEFAULT_STOP_WORDS: frozenset[str] = STOP_WORDS
    
    def __init__(
        self,
        stop_words: set[str] | None = None,
        min_length: int = 2
    ) -> None:
        """Initialize tokenizer with configuration.
        
        Args:
            stop_words: Custom stop words (None uses defaults).
            min_length: Minimum token length to keep.
        """
        self.stop_words: frozenset[str] = (
            frozenset(stop_words) if stop_words is not None
            else self.DEFAULT_STOP_WORDS
        )
        self.min_length: int = min_length
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        return ''.join(
            c if c.isalnum() or c.isspace() else ' '
            for c in text
        )
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into clean tokens.
        
        Args:
            text: Input text.
        
        Returns:
            List of processed tokens.
        """
        cleaned = self._remove_punctuation(text)
        tokens = cleaned.lower().split()
        
        return [
            token for token in tokens
            if len(token) >= self.min_length
            and token not in self.stop_words
        ]
    
    def __repr__(self) -> str:
        return (
            f"Tokenizer(stop_words={len(self.stop_words)} words, "
            f"min_length={self.min_length})"
        )
```

### 5.6 Análisis de Complejidad

```
┌─────────────────────────────────────────────────────────────────┐
│  COMPLEJIDAD DE tokenize(text)                                  │
│                                                                 │
│  Sea n = len(text), m = número de tokens                        │
│                                                                 │
│  1. remove_punctuation: O(n) - recorre cada carácter            │
│  2. lower(): O(n) - recorre cada carácter                       │
│  3. split(): O(n) - recorre buscando espacios                   │
│  4. Filtrar por longitud: O(m) - recorre tokens                 │
│  5. Filtrar stop words: O(m) - lookup O(1) por token            │
│                                                                 │
│  TOTAL: O(n + m) ≈ O(n) ya que m ≤ n                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Errores Comunes

### Error 1: Modificar lista durante iteración

```python
# ❌ Bug: resultado impredecible
words = ["a", "the", "b", "an", "c"]
for word in words:
    if word in {"the", "an"}:
        words.remove(word)
# Resultado: ["a", "b", "c"] pero puede fallar

# ✅ Correcto: list comprehension
words = [w for w in words if w not in {"the", "an"}]
```

### Error 2: Concatenar strings en loop

```python
# ❌ O(n²) - crea nuevo string cada vez
result = ""
for word in words:
    result += word + " "

# ✅ O(n) - usa join
result = " ".join(words)
```

### Error 3: Olvidar que strings son inmutables

```python
# ❌ No hace nada
text = "hello"
text.upper()  # Retorna nuevo string, no modifica
print(text)   # "hello" (sin cambios)

# ✅ Asignar resultado
text = text.upper()
print(text)   # "HELLO"
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 4.1: Manipulación de Listas
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-41)

### Ejercicio 4.2: Tokenizador Básico
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-42)

### Ejercicio 4.3: Análisis de Complejidad
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-43)

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Python Lists](https://docs.python.org/3/tutorial/datastructures.html) | Docs | 🔴 Obligatorio |
| [String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods) | Docs | 🔴 Obligatorio |
| [Time Complexity](https://wiki.python.org/moin/TimeComplexity) | Wiki | 🟡 Recomendado |

---

## 🔗 Referencias del Glosario

- [Array](GLOSARIO.md#array)
- [String](GLOSARIO.md#string)
- [Inmutabilidad](GLOSARIO.md#inmutabilidad)
- [Tokenización](GLOSARIO.md#tokenizacion)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [03_LOGICA_DISCRETA](03_LOGICA_DISCRETA.md) | [00_INDICE](00_INDICE.md) | [05_HASHMAPS_SETS](05_HASHMAPS_SETS.md) |
