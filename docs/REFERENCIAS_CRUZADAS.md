# 🔗 Mapa de Referencias Cruzadas

> Navegación entre los 10 módulos obligatorios de la guía.

---

## 📊 Flujo de los 10 Módulos Obligatorios

```
┌──────────────────────────────────────────┐
│ FASE 1: FUNDAMENTOS (Semanas 1-6)        │
├──────────────────────────────────────────┤
│ Módulo 01 - Python Profesional           │
│  ↓                                       │
│ Módulo 02 - OOP desde Cero               │
│  ↓                                       │
│ Módulo 03 - Álgebra Lineal para ML       │
└──────────────────────────────────────────┘
 ↓
┌──────────────────────────────────────────┐
│ FASE 2: PROBABILIDAD ⭐ PATHWAY L2       │
│ (Semanas 7-14)                           │
├──────────────────────────────────────────┤
│ Módulo 04 - Fundamentos de Probabilidad  │
│  ↓                                       │
│ Módulo 05 - Estadística Inferencial      │
│  ↓                                       │
│ Módulo 06 - Markov y Monte Carlo         │
└──────────────────────────────────────────┘
 ↓
┌──────────────────────────────────────────┐
│ FASE 3: MACHINE LEARNING ⭐ PATHWAY L1   │
│ (Semanas 15-22)                          │
├──────────────────────────────────────────┤
│ Módulo 07 - ML Supervisado               │
│  ↓                                       │
│ Módulo 08 - ML No Supervisado            │
│  ↓                                       │
│ Módulo 09 - Deep Learning                │
└──────────────────────────────────────────┘
 ↓
┌──────────────────────────────────────────┐
│ FASE 4: PROYECTO FINAL (Semanas 23-26)   │
├──────────────────────────────────────────┤
│ Módulo 10 - ML Pipeline Completo         │
│    (integra módulos 04-09)               │
└──────────────────────────────────────────┘
```

---

## ⚠️ Anexos DSA (OPCIONALES - Solo para entrevistas)

```
┌──────────────────────────────────────────┐
│ ANEXOS DSA - NO requeridos para Pathway  │
├──────────────────────────────────────────┤
│ Arrays y Strings                         │
│ Hash Maps y Sets                         │
│ Recursión                                │
│ Sorting                                  │
│ Trees y BST                              │
│ Graphs, BFS, DFS                         │
│ Dynamic Programming                      │
└──────────────────────────────────────────┘
```

Estos módulos son útiles si quieres prepararte para **entrevistas técnicas de código**, pero **NO son necesarios** para:
- Completar el proyecto de la guía
- Aprobar las materias del Pathway de CU Boulder

---

## 📖 Referencias de los 10 Módulos Obligatorios

### Módulo 01 - Python Profesional
| Prerrequisito | Siguiente |
|---------------|-----------|
| Ninguno | Módulo 02 |

### Módulo 02 - OOP desde Cero
| Prerrequisito | Siguiente |
|---------------|-----------|
| Módulo 01 | Módulo 03 |

### Módulo 03 - Álgebra Lineal para ML
| Prerrequisito | Siguiente |
|---------------|-----------|
| Módulos 01-02 | Módulo 04 |

### Módulo 04 - Fundamentos de Probabilidad ⭐
| Prerrequisito | Siguiente | Curso Pathway |
|---------------|-----------|---------------|
| Módulo 03 | Módulo 05 | Probability Fundamentals |

### Módulo 05 - Estadística Inferencial ⭐
| Prerrequisito | Siguiente | Curso Pathway |
|---------------|-----------|---------------|
| Módulo 04 | Módulo 06 | Statistical Estimation |

### Módulo 06 - Markov y Monte Carlo ⭐
| Prerrequisito | Siguiente | Curso Pathway |
|---------------|-----------|---------------|
| Módulos 04-05 | Módulo 07 | Markov Chains & MC |

### Módulo 07 - ML Supervisado ⭐
| Prerrequisito | Siguiente | Curso Pathway |
|---------------|-----------|---------------|
| Módulos 03-06 | Módulo 08 | Supervised Learning |

### Módulo 08 - ML No Supervisado ⭐
| Prerrequisito | Siguiente | Curso Pathway |
|---------------|-----------|---------------|
| Módulo 07 | Módulo 09 | Unsupervised Algorithms |

### Módulo 09 - Deep Learning ⭐
| Prerrequisito | Siguiente | Curso Pathway |
|---------------|-----------|---------------|
| Módulos 07-08 | Módulo 10 | Intro to Deep Learning |

### Módulo 10 - Proyecto Final
| Prerrequisito | Entregable |
|---------------|------------|
| Módulos 01-09 | Pipeline ML completo |

---

## 📚 Referencias en Documentos Auxiliares

### EJERCICIOS.md
- Enlaza a cada módulo para los ejercicios correspondientes
- Enlaza a EJERCICIOS_SOLUCIONES.md para respuestas

### EJERCICIOS_SOLUCIONES.md
- Enlaza de vuelta a EJERCICIOS.md
- Referencias a módulos para contexto

### GLOSARIO.md
- Referenciado desde todos los módulos
- Organizado A-Z con siglas al final

### SIMULACRO_ENTREVISTA.md
- Referencias a módulos por sección temática
- Enlaza a RECURSOS.md para más práctica

### RECURSOS.md
- Organizado por tema (Matemáticas, DSA, Python)
- URLs a cursos del Pathway

### CHECKLIST.md
- Referencias a todos los componentes del proyecto
- Enlaza a RUBRICA_EVALUACION.md

### RUBRICA_EVALUACION.md
- Criterios que mapean a módulos específicos

---

## ✅ Verificación de Enlaces

### Comandos para Verificar

```bash
# Verificar enlaces internos rotos (ejecutar desde la raíz del repo)
grep -r "\[.*\](.*\.md)" docs/*.md | \
  grep -v "http" | \
  while read line; do
    file=$(echo "$line" | cut -d: -f1)
    link=$(echo "$line" | grep -oP '\\(.*?\\.md\\)' | tr -d '()')
    if [[ ! -z "$link" && ! -f "docs/$link" ]]; then
      echo "BROKEN: $file -> $link"
    fi
  done
```

---

## 🗺️ Flujo de Navegación: 10 Módulos Obligatorios

```
index.md → 00_INDICE.md

     Módulo 01 (Python)
           ↓
     Módulo 02 (OOP)
           ↓
     Módulo 03 (Álgebra Lineal)
           ↓
     Módulo 04 (Probabilidad) ⭐
           ↓
     Módulo 05 (Estadística) ⭐
           ↓
     Módulo 06 (Markov/MC) ⭐
           ↓
     Módulo 07 (ML Supervisado) ⭐
           ↓
     Módulo 08 (ML No Supervisado) ⭐
           ↓
     Módulo 09 (Deep Learning) ⭐
           ↓
     Módulo 10 (Proyecto Final)
```

**Tiempo total: 26 semanas = 6 meses**
