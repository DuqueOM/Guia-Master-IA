# 📋 Plan de Acción Mejorado v4.0 - Guía de Integración

> Este documento explica cómo integrar las mejoras estratégicas con el plan de estudios existente.

---

## 🎯 Resumen de Mejoras

Este plan no cambia **qué** estudias (el PDF ya es excelente), sino **cómo** ejecutas el estudio para garantizar que no abandones y superes el nivel requerido.

---

## 📁 Nuevos Recursos Creados

### Configuración del Entorno
| Archivo | Descripción |
|---------|-------------|
| `pyproject.toml` | Configuración de proyecto Python con dependencias |
| `.pre-commit-config.yaml` | Hooks automáticos para código limpio |
| `setup_env.sh` | Script de instalación rápida |

### Herramientas de Estudio (`study_tools/`)
| Archivo | Propósito | Cuándo Usar |
|---------|-----------|-------------|
| `DIARIO_ERRORES.md` | Registro de errores | Diariamente |
| `DRILL_DIMENSIONES_NUMPY.md` | Ejercicios de shapes | Semanas 1-2 |
| `SIMULACRO_EXAMEN_TEORICO.md` | Preguntas tipo examen | Sábados |
| `VISUALIZACION_GRADIENT_DESCENT.md` | Código de visualización | Semanas 6-7 |
| `DRYRUN_BACKPROPAGATION.md` | Backprop en papel | Semana 18 |
| `PUENTE_NUMPY_PYTORCH.md` | Traducción a PyTorch | Semana 24 |

### AI Code Reviewer (`prompts/`)
| Archivo | Descripción |
|---------|-------------|
| `AI_CODE_REVIEWER.md` | Prompt para ChatGPT/Claude como revisor de código |

### Tests (`tests/`)
| Archivo | Descripción |
|---------|-------------|
| `test_dimension_assertions.py` | Tests de dimensiones para validar código ML |

---

## 📅 Integración con el Cronograma Existente

### Semana 0: Preparación del Laboratorio

```bash
# 1. Ejecutar setup
cd "/home/duque_om/projects/Guia Master"
bash setup_env.sh

# 2. Activar entorno
source venv/bin/activate

# 3. Configurar AI Code Reviewer
# Copiar contenido de prompts/AI_CODE_REVIEWER.md a ChatGPT/Claude
```

### Semanas 1-2: Añadir Drill de Dimensiones

**Ajuste al estudio diario:**
```
Antes de codificar (5 min):
  → Abrir study_tools/DRILL_DIMENSIONES_NUMPY.md
  → Completar 5 ejercicios de predicción de shape
  → Verificar en Python
```

### Semanas 6-7: Añadir Visualización 3D

**Ajuste semanal:**
```
Durante estudio de Gradient Descent:
  → Ejecutar código de study_tools/VISUALIZACION_GRADIENT_DESCENT.md
  → Experimentar con diferentes learning rates
  → Usar GeoGebra para exploración interactiva
```

### Semana 18: Dry Run Obligatorio

**Antes de codificar Backpropagation:**
```
1. Abrir study_tools/DRYRUN_BACKPROPAGATION.md
2. Completar ejercicio en papel (30 min)
3. Verificar con código de verificación
4. SOLO ENTONCES empezar tu implementación
```

### Semana 24: Traducción a PyTorch

**Día extra al final del proyecto MNIST:**
```
1. Abrir study_tools/PUENTE_NUMPY_PYTORCH.md
2. Tomar tu clase NeuralNetwork de NumPy
3. Reescribir en PyTorch (15 líneas)
4. Comparar resultados
5. Responder checklist de "iluminación"
```

### Cada Sábado: Simulacro de Examen

**Protocolo de 1 hora:**
```
1. Sin IDE, sin internet
2. Solo lápiz, papel, calculadora básica
3. Abrir study_tools/SIMULACRO_EXAMEN_TEORICO.md
4. Completar simulacro correspondiente a la fase
5. Auto-evaluar con criterios de puntuación
6. Registrar temas débiles
```

---

## 🔄 Protocolo Diario "Sandwich"

```
┌─────────────────────────────────────────────────────────────┐
│ MAÑANA (Teoría - Input)                                     │
│ • Ver videos / leer documentación                           │
│ • NO tomar notas lineales (ineficiente)                     │
│ • Enfocarse en ENTENDER, no memorizar                       │
├─────────────────────────────────────────────────────────────┤
│ MEDIODÍA (Implementación - Output)                          │
│ • Escribir código                                           │
│ • Usar pre-commit para validación automática                │
│ • Consultar AI Code Reviewer para estilo/vectorización      │
├─────────────────────────────────────────────────────────────┤
│ CIERRE (Validación Feynman)                                 │
│ • Explicar el concepto como si enseñaras a alguien          │
│ • Registrar TODOS los errores en DIARIO_ERRORES.md          │
│ • Identificar: ¿Qué no entendí completamente hoy?           │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Checklist de Configuración Inicial

- [ ] Ejecutar `bash setup_env.sh`
- [ ] Verificar que `pre-commit run --all-files` funciona
- [ ] Configurar AI Code Reviewer en ChatGPT/Claude
- [ ] Leer `study_tools/README.md` completo
- [ ] Crear primera entrada en `DIARIO_ERRORES.md`
- [ ] Completar Nivel 1 de `DRILL_DIMENSIONES_NUMPY.md`

---

## 📊 Tabla de Ajustes por Fase

| Fase | Semanas | Ajuste Estratégico | Recurso |
|------|---------|-------------------|---------|
| Fundamentos | 1-2 | Drill de Dimensiones | `DRILL_DIMENSIONES_NUMPY.md` |
| Fundamentos | 4 | Simulacro Álgebra Lineal | `SIMULACRO_EXAMEN_TEORICO.md` |
| Fundamentos | 6-7 | Visualización 3D | `VISUALIZACION_GRADIENT_DESCENT.md` |
| Fundamentos | 8 | Simulacro Cálculo | `SIMULACRO_EXAMEN_TEORICO.md` |
| Probabilidad | 12 | Simulacro Probabilidad | `SIMULACRO_EXAMEN_TEORICO.md` |
| ML | 16 | Simulacro Supervised Learning | `SIMULACRO_EXAMEN_TEORICO.md` |
| DL | 18 | Dry Run Backprop | `DRYRUN_BACKPROPAGATION.md` |
| DL | 22 | Simulacro Deep Learning | `SIMULACRO_EXAMEN_TEORICO.md` |
| Proyecto | 24 | Traducción PyTorch | `PUENTE_NUMPY_PYTORCH.md` |

---

## 🎯 Criterios de Éxito

### Por Checkpoint
- Simulacro correspondiente ≥ 75 puntos
- Diario de Errores actualizado
- Código pasa pre-commit

### Señales de Progreso
- Predices `.shape` sin ejecutar código
- Errores del Diario no se repiten
- Simulacros < 60 minutos
- Puedes explicar conceptos sin notas

---

## 🚨 Señales de Alarma

| Señal | Acción |
|-------|--------|
| Mismo error 3+ veces | Revisar tema desde cero |
| Simulacro < 60 pts | Repetir fase antes de avanzar |
| No puedes explicar sin código | Más teoría, menos implementación |
| Pre-commit falla siempre | Revisar estilo en `AI_CODE_REVIEWER.md` |

---

## 🔗 Referencias Rápidas

- **Guía principal**: [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md)
- **Índice de contenidos**: [00_INDICE.md](00_INDICE.md)
- **Checklist de progreso**: [CHECKLIST.md](CHECKLIST.md)
- **Herramientas de estudio**: `study_tools/README.md`
