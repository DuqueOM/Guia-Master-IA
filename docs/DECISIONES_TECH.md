# 🔧 Decisiones Técnicas (ADRs)

> Architecture Decision Records para el proyecto Archimedes Indexer.

---

## Índice de Decisiones

| # | Decisión | Estado |
|---|----------|--------|
| 1 | Python puro sin librerías | ✅ Aceptada |
| 2 | Versión de Python 3.11+ | ✅ Aceptada |
| 3 | Estructura src/ layout | ✅ Aceptada |
| 4 | Set vs List para posting lists | ✅ Aceptada |
| 5 | QuickSort con random pivot | ✅ Aceptada |
| 6 | TF-IDF normalizado | ✅ Aceptada |
| 7 | pytest para testing | ✅ Aceptada |
| 8 | ruff para linting | ✅ Aceptada |

---

## ADR-001: Python Puro sin Librerías

### Contexto
El objetivo del proyecto es aprender fundamentos de CS, no usar herramientas.

### Decisión
**Prohibir** numpy, pandas, sklearn, y cualquier librería de ML/data science.

### Consecuencias
- ✅ Fuerza entendimiento profundo de algoritmos
- ✅ Código más simple de debuggear
- ✅ Demuestra habilidad, no uso de herramientas
- ❌ Menos eficiente que librerías optimizadas
- ❌ Más código para escribir

---

## ADR-002: Python 3.11+

### Contexto
Necesitamos decidir versión mínima de Python.

### Decisión
Usar **Python 3.11** como mínimo.

### Justificación
- Sintaxis `list[str]` sin `from __future__ import annotations`
- Union types con `|` (ej: `str | None`)
- Mejor performance
- Mensajes de error más claros

### Consecuencias
- ✅ Código más limpio y moderno
- ✅ Mejor experiencia de desarrollo
- ❌ No compatible con Python 3.9/3.10

---

## ADR-003: Estructura src/ Layout

### Contexto
Hay dos layouts comunes: flat (módulos en raíz) y src/ (módulos en carpeta src/).

### Decisión
Usar **src/ layout**:

```
project/
├── src/
│   ├── __init__.py
│   └── module.py
└── tests/
```

### Justificación
- Evita importar accidentalmente código no instalado
- Estándar en proyectos profesionales
- Compatible con empaquetado moderno

---

## ADR-004: Set vs List para Posting Lists

### Contexto
Posting lists mapean término → documentos. ¿Usar list o set?

### Decisión
Usar **set[int]** para doc_ids.

### Justificación
- O(1) para verificar si documento contiene término
- Intersección/unión nativas para AND/OR
- No importa el orden en la mayoría de casos

### Trade-offs
- ✅ Operaciones de conjuntos eficientes
- ❌ No mantiene orden de inserción
- ❌ Necesita convertir a list para ordenar por score

---

## ADR-005: QuickSort con Random Pivot

### Contexto
QuickSort puede ser O(n²) con pivot malo.

### Decisión
Usar **random pivot selection**.

### Justificación
- Evita peor caso en datos ya ordenados
- O(n log n) esperado
- Fácil de implementar

### Alternativas Consideradas
- Pivot fijo (primero/último): Rechazado, vulnerable a datos ordenados
- Median of three: Válido pero más complejo
- Cambiar a MergeSort: Usa más memoria

---

## ADR-006: TF-IDF Normalizado

### Contexto
Hay variantes de TF-IDF. ¿Cuál usar?

### Decisión
Usar fórmula estándar:
- TF = count(term, doc) / total_terms(doc)
- IDF = log(N / df(term))
- TF-IDF = TF × IDF

### Justificación
- Fácil de entender y explicar
- Documentos largos no dominan
- Consistente con literatura

---

## ADR-007: pytest para Testing

### Contexto
Python tiene varias opciones de testing: unittest, pytest, nose.

### Decisión
Usar **pytest**.

### Justificación
- Sintaxis más simple (assert nativo)
- Fixtures potentes
- Mejor output de errores
- pytest-cov para coverage

### Ejemplo
```python
# pytest style (simple)
def test_tokenize():
    assert tokenize("Hello") == ["hello"]

# unittest style (verbose)
class TestTokenize(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Hello"), ["hello"])
```

---

## ADR-008: ruff para Linting

### Contexto
Opciones de linting: flake8, pylint, ruff.

### Decisión
Usar **ruff**.

### Justificación
- 10-100x más rápido que alternativas
- Combina múltiples herramientas (flake8, isort, pyupgrade)
- Corrección automática (`--fix`)
- Desarrollo activo

### Configuración
```toml
[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "UP"]
```

---

## Matriz de Decisiones

| Área | Herramienta/Enfoque | Por Qué |
|------|---------------------|---------|
| Lenguaje | Python 3.11+ | Sintaxis moderna |
| Librerías | Ninguna (solo stdlib) | Aprendizaje |
| Layout | src/ | Profesional |
| Posting Lists | set | O(1) lookup |
| Sorting | QuickSort random | O(n log n) esperado |
| TF-IDF | Normalizado | Estándar |
| Testing | pytest | Simple, potente |
| Linting | ruff | Rápido, moderno |
| Type checking | mypy | Estándar |
