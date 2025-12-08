# 📊 Rúbrica de Evaluación

> Criterios para evaluar el proyecto Archimedes Indexer.

---

## Escala de Puntuación

| Puntuación | Nivel | Significado |
|------------|-------|-------------|
| 90-100 | 🏆 Excelente | Listo para Pathway y entrevistas |
| 75-89 | ✅ Bueno | Reforzar áreas débiles |
| 60-74 | ⚠️ Suficiente | Más práctica antes de Pathway |
| <60 | ❌ Insuficiente | Revisar módulos fundamentales |

---

## Desglose por Categoría (100 puntos)

### 1. Funcionalidad (30 pts)

| Criterio | Pts | Descripción |
|----------|-----|-------------|
| Motor funcional | 10 | Indexa documentos y retorna resultados |
| Ranking correcto | 10 | Resultados ordenados por relevancia |
| Búsqueda AND/OR | 5 | Soporta ambos tipos de consulta |
| Edge cases | 5 | Maneja queries vacías, docs vacíos, etc. |

### 2. Calidad de Código (25 pts)

| Criterio | Pts | Descripción |
|----------|-----|-------------|
| Type hints | 5 | Todos los parámetros y retornos tipados |
| Docstrings | 5 | Todas las funciones públicas documentadas |
| PEP8 | 5 | Código pasa linters sin warnings |
| Estructura | 5 | Módulos separados, imports limpios |
| SOLID básico | 5 | Cada clase una responsabilidad |

### 3. Testing (20 pts)

| Criterio | Pts | Descripción |
|----------|-----|-------------|
| Tests unitarios | 8 | Tests para cada módulo |
| Tests integración | 4 | Test del flujo completo |
| Coverage > 80% | 4 | Cobertura de código |
| Edge cases testeados | 4 | Casos límite cubiertos |

### 4. Análisis Big O (15 pts)

| Criterio | Pts | Descripción |
|----------|-----|-------------|
| Documento completo | 5 | Análisis de todas las operaciones |
| Correctitud | 5 | Análisis matemáticamente correcto |
| Justificación | 5 | Explica el razonamiento |

### 5. Documentación (10 pts)

| Criterio | Pts | Descripción |
|----------|-----|-------------|
| README.md | 5 | Profesional, en inglés, con ejemplos |
| Instrucciones uso | 3 | Cómo instalar y ejecutar |
| Demo/ejemplo | 2 | Código de ejemplo funcional |

---

## Checklist Rápido

### ✅ Funcionalidad
- [ ] `SearchEngine.add_document()` funciona
- [ ] `SearchEngine.build_index()` funciona
- [ ] `SearchEngine.search()` retorna resultados ordenados
- [ ] Resultados tienen score entre 0 y 1

### ✅ Código
- [ ] `mypy src/` pasa sin errores
- [ ] `ruff check src/` pasa sin errores
- [ ] Todas las funciones tienen docstrings
- [ ] No hay código duplicado

### ✅ Tests
- [ ] `pytest tests/` pasa
- [ ] Coverage > 80%
- [ ] Tests para cada módulo

### ✅ Documentación
- [ ] README.md existe y está completo
- [ ] COMPLEXITY_ANALYSIS.md existe
- [ ] Ejemplos de uso incluidos

---

## Ejemplos de Evaluación

### Ejemplo: Análisis Big O (15/15 pts)

```markdown
# COMPLEXITY_ANALYSIS.md

## add_document(doc_id, tokens)
- Complejidad: O(t) donde t = len(tokens)
- Justificación: Iteramos una vez sobre los tokens para agregarlos al índice.
  Cada operación de agregar al set es O(1) amortizado.

## search(query)
- Complejidad: O(q + V + N × V + N log N)
  - O(q): Tokenizar query
  - O(V): Crear vector query (V = vocabulario)
  - O(N × V): Calcular similitud con cada documento
  - O(N log N): Ordenar resultados
- Simplificado: O(N × V) domina para corpus grandes

## quicksort(items)
- Promedio: O(n log n)
- Peor caso: O(n²) cuando el pivote es siempre el mínimo/máximo
- Espacio: O(log n) para el call stack
```

### Ejemplo: Test Unitario Bien Escrito

```python
# test_similarity.py
import pytest
from src.similarity import cosine_similarity

class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)
    
    def test_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        v1 = [0.0, 0.0]
        v2 = [1.0, 1.0]
        assert cosine_similarity(v1, v2) == 0.0
```

---

## Comando de Verificación Final

```bash
# Verificar todo antes de entregar
mypy src/
ruff check src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Objetivo:** Todos los comandos deben pasar sin errores.
