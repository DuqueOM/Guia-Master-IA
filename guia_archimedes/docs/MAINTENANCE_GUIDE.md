# 🔄 Guía de Mantenimiento

> Cómo mantener y actualizar la guía Archimedes Indexer.

---

## 📅 Calendario de Mantenimiento

### Mensual
- [ ] Verificar que todos los links externos funcionan
- [ ] Revisar si hay nuevos recursos relevantes
- [ ] Actualizar RECURSOS.md si hay cursos nuevos

### Trimestral
- [ ] Revisar que el código de ejemplo sigue funcionando
- [ ] Actualizar versiones de Python si hay nueva LTS
- [ ] Revisar feedback de usuarios (si hay)

### Semestral
- [ ] Revisar cambios en Pathway de CU Boulder
- [ ] Actualizar SIMULACRO_ENTREVISTA.md con nuevas preguntas
- [ ] Verificar que herramientas recomendadas siguen activas

---

## 🔍 Verificación de la Guía

### Script de Verificación de Links

```bash
#!/bin/bash
# check_links.sh

echo "Checking internal links..."
grep -r "\[.*\](.*\.md)" guia_archimedes/*.md | while read line; do
    file=$(echo "$line" | cut -d: -f1)
    link=$(echo "$line" | grep -oP '\(.*?\.md\)' | tr -d '()')
    if [[ ! -z "$link" && ! -f "guia_archimedes/$link" ]]; then
        echo "BROKEN: $file -> $link"
    fi
done

echo "Done!"
```

### Verificación de Estructura

```bash
# Verificar que todos los módulos existen
for i in {01..12}; do
    if [[ ! -f "guia_archimedes/${i}_*.md" ]]; then
        echo "MISSING: Módulo $i"
    fi
done

# Verificar documentos auxiliares
for doc in EJERCICIOS EJERCICIOS_SOLUCIONES GLOSARIO RUBRICA_EVALUACION CHECKLIST RECURSOS SIMULACRO_ENTREVISTA; do
    if [[ ! -f "guia_archimedes/${doc}.md" ]]; then
        echo "MISSING: $doc.md"
    fi
done
```

---

## 📝 Estructura de un Módulo

Cada módulo debe seguir esta estructura:

```markdown
# XX - Título del Módulo

> **🎯 Objetivo:** [Descripción en una línea]

---

## 🧠 Analogía: [Nombre]

[Diagrama ASCII y explicación]

---

## 📋 Contenido

1. [Sección 1](#1-seccion)
2. [Sección 2](#2-seccion)
...

---

## 1. Sección {#1-seccion}

### 1.1 Subsección

[Contenido con código, tablas, diagramas]

---

## ⚠️ Errores Comunes

[Lista de errores típicos]

---

## 🔧 Ejercicios Prácticos

### Ejercicio X.1
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-x1)

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [...] | ... | 🔴/🟡/🟢 |

---

## 🔗 Referencias del Glosario

- [Término](GLOSARIO.md#termino)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [XX_ANTERIOR](XX_ANTERIOR.md) | [00_INDICE](00_INDICE.md) | [XX_SIGUIENTE](XX_SIGUIENTE.md) |
```

---

## 🆕 Agregar Nuevo Contenido

### Nuevo Ejercicio

1. Agregar en `EJERCICIOS.md` en la sección del módulo correspondiente
2. Agregar solución en `EJERCICIOS_SOLUCIONES.md`
3. Actualizar el índice al inicio de ambos archivos

### Nuevo Término en Glosario

1. Agregar en orden alfabético en `GLOSARIO.md`
2. Seguir formato:
   ```markdown
   ### Término
   **Definición:** [Definición técnica]
   **Analogía:** [Explicación simple]
   **Ejemplo:** [Código o caso de uso]
   ```

### Nueva Pregunta en Simulacro

1. Agregar en sección correspondiente de `SIMULACRO_ENTREVISTA.md`
2. Actualizar conteo total en encabezado
3. Incluir respuesta detallada

---

## 🎨 Convenciones de Estilo

### Iconos

| Icono | Uso |
|-------|-----|
| 🎯 | Objetivo |
| 🧠 | Analogía/Concepto |
| 📋 | Índice/Lista |
| ⚠️ | Advertencia |
| 💡 | Tip |
| ✅ | Correcto/Buena práctica |
| ❌ | Incorrecto/Anti-patrón |
| 🔧 | Ejercicio práctico |
| 📚 | Recursos externos |
| 🔗 | Referencia cruzada |
| 🧭 | Navegación |

### Código

- Usar Python 3.11+ syntax
- Incluir type hints siempre
- Agregar docstrings en ejemplos largos
- Marcar código malo con `# ❌` y bueno con `# ✅`

### Tablas

- Usar para comparaciones, índices, referencias
- Mantener columnas alineadas
- Primera columna descriptiva

---

## 📊 Métricas de Calidad

### Completitud
- [ ] 12 módulos (01-12)
- [ ] Índice principal (00_INDICE.md)
- [ ] SYLLABUS y PLAN_ESTUDIOS
- [ ] Documentos auxiliares completos

### Consistencia
- [ ] Todos los módulos siguen la estructura
- [ ] Links internos funcionan
- [ ] Numeración correcta

### Claridad
- [ ] Cada módulo tiene objetivo claro
- [ ] Analogías ayudan a entender
- [ ] Código es ejecutable

---

## 🐛 Reporte de Errores

Si encuentras un error:

1. Identifica el archivo y línea
2. Describe el problema
3. Propón corrección si es posible
4. Actualiza el archivo directamente

---

## 📁 Estructura de Archivos

```
guia_archimedes/
├── index.md                           # Landing page
├── 00_INDICE.md                       # Índice principal
├── SYLLABUS.md                        # Programa del curso
├── PLAN_ESTUDIOS.md                   # Cronograma día a día
│
├── # MÓDULOS FUNDAMENTALES (01-06)
├── 01_PYTHON_PROFESIONAL.md           # Type hints, PEP8
├── 02_OOP_DESDE_CERO.md               # Clases, SOLID
├── 03_LOGICA_DISCRETA.md              # Big O, conjuntos
├── 04_ARRAYS_STRINGS.md               # Listas, slicing
├── 05_HASHMAPS_SETS.md                # Diccionarios, hashing
├── 06_INVERTED_INDEX.md               # Índice invertido
│
├── # MÓDULOS DSA AVANZADO (13-15) ⭐ PATHWAY
├── 13_LINKED_LISTS_STACKS_QUEUES.md   # Estructuras lineales
├── 14_TREES.md                        # BST, traversals
├── 15_GRAPHS.md                       # BFS, DFS
│
├── # MÓDULOS ALGORITMOS (07-09, 16-18) ⭐ PATHWAY
├── 07_RECURSION.md                    # Divide & conquer
├── 08_SORTING.md                      # QuickSort, MergeSort
├── 09_BINARY_SEARCH.md                # Búsqueda binaria
├── 16_DYNAMIC_PROGRAMMING.md          # DP, memoization
├── 17_GREEDY.md                       # Greedy algorithms
├── 18_HEAPS.md                        # Priority queues
│
├── # MÓDULOS MATEMÁTICAS (10-11)
├── 10_ALGEBRA_LINEAL.md               # Vectores, matrices
├── 11_TFIDF_COSENO.md                 # TF-IDF, coseno
│
├── # PROYECTO INTEGRADOR
├── 12_PROYECTO_INTEGRADOR.md          # Motor de búsqueda
│
├── # DOCUMENTOS AUXILIARES
├── EJERCICIOS.md                      # 55+ ejercicios
├── EJERCICIOS_SOLUCIONES.md           # Soluciones
├── GLOSARIO.md                        # 80+ términos A-Z
├── RUBRICA_EVALUACION.md              # Criterios (100 pts)
├── CHECKLIST.md                       # Verificación final
├── RECURSOS.md                        # Cursos, libros
├── SIMULACRO_ENTREVISTA.md            # 80 preguntas Pathway
├── DECISIONES_TECH.md                 # ADRs del proyecto
├── REFERENCIAS_CRUZADAS.md            # Mapa de navegación
├── EVALUACION_GUIA.md                 # Autoevaluación
└── MAINTENANCE_GUIDE.md               # Esta guía
```

**Total: 33 archivos**

---

**Última actualización:** Diciembre 2025
