# 🧰 Herramientas de Estudio - Plan v4.0 Estratégico

Este directorio contiene los materiales del **Plan de Acción Mejorado v4.0** para maximizar tu preparación para el Master en IA de CU Boulder.

---

## 📁 Contenido

| Archivo | Propósito | Cuándo usar |
|---------|-----------|-------------|
| `DIARIO_ERRORES.md` | Registro de errores matemáticos y de código | Diariamente, al final del estudio |
| `DIARIO_METACOGNITIVO.md` | Metacognición (qué entendí / qué no / patrón de error) | 5 min diarios |
| `DRILL_DIMENSIONES_NUMPY.md` | Ejercicios de predicción de `.shape` | Semanas 1-2, 5 min diarios |
| `TEORIA_CODIGO_BRIDGE.md` | Traducción matemática ↔ implementación (shapes + sanity checks) | 1 vez por semana (20–30 min) |
| `SIMULACRO_EXAMEN_TEORICO.md` | Banco de preguntas tipo examen | Sábados, 1 hora |
| `SIMULACRO_PERFORMANCE_BASED.md` | Simulacros 50% teoría / 50% pseudocódigo-código (PB) | Semanas 8, 16 y 23 |
| `CIERRE_SEMANAL.md` | Consolidación + rescate cognitivo (mapa mental, Feynman, autoevaluación) | Sábados, 1 hora |
| `RUBRICA_v1.md` | Rúbrica de evaluación (criterios, pesos, condiciones duras) | Semana 0 (crear/calibrar) + checkpoints (PB-8/16/23 y fin de módulos) |
| `VISUALIZACION_GRADIENT_DESCENT.md` | Código para visualizar optimización | Semanas 6-7 |
| `DRYRUN_BACKPROPAGATION.md` | Plantilla para backprop en papel | Semana 18, antes de codificar |
| `PUENTE_NUMPY_PYTORCH.md` | Traducción NumPy → PyTorch | Semana 24 |
| `BADGES_CHECKPOINTS.md` | Badges (mini-victorias verificables por módulo) | Al cerrar cada módulo |

---

## 📅 Integración con el Cronograma

### Semana 0 (Preparación)
- [ ] Instalar pre-commit hooks: `pip install pre-commit && pre-commit install`
- [ ] Configurar AI Code Reviewer (ver `../prompts/AI_CODE_REVIEWER.md`)
- [ ] Leer este README completo
- [ ] Crear y calibrar rúbrica (ver `RUBRICA_v1.md` + `../rubrica.csv`)
- [ ] Test rápido: evaluar 1 entregable pequeño (p.ej. `DRILL_DIMENSIONES_NUMPY.md`) y ajustar descriptores/pesos

### Semanas 1-2
- [ ] Completar `DRILL_DIMENSIONES_NUMPY.md` (1 hora extra)
- [ ] Iniciar `DIARIO_ERRORES.md`

### Semanas 6-7
- [ ] Ejecutar código de `VISUALIZACION_GRADIENT_DESCENT.md`
- [ ] Usar GeoGebra para explorar superficies 3D

### Semana 18
- [ ] Completar ejercicios de `DRYRUN_BACKPROPAGATION.md` en papel
- [ ] Verificar con código DESPUÉS de hacer a mano

### Semana 24
- [ ] Traducir tu red neuronal a PyTorch usando `PUENTE_NUMPY_PYTORCH.md`
- [ ] Comparar resultados NumPy vs PyTorch

### Cada Sábado
- [ ] Simulacro de 1 hora usando `SIMULACRO_EXAMEN_TEORICO.md`
- [ ] Sin IDE, sin internet, solo lápiz y papel
- [ ] Cierre semanal usando `CIERRE_SEMANAL.md`
- [ ] Autoevaluación rápida con la rúbrica (ver `RUBRICA_v1.md`) y registrar brechas

### Cada día (5 min)
- [ ] Diario metacognitivo en `DIARIO_METACOGNITIVO.md`

### Una vez por semana (20–30 min)
- [ ] Puente Teoría ↔ Código en `TEORIA_CODIGO_BRIDGE.md`

---

## 🔄 Protocolo Diario "Sandwich"

```
┌─────────────────────────────────────────────────┐
│  MAÑANA (Input)                                 │
│  • Ver videos / leer teoría                     │
│  • NO tomar notas lineales                      │
├─────────────────────────────────────────────────┤
│  MEDIODÍA (Output)                              │
│  • Implementar en código                        │
│  • Usar AI Code Reviewer para validar estilo    │
├─────────────────────────────────────────────────┤
│  CIERRE (Validación Feynman)                    │
│  • Explicar el concepto como si enseñaras       │
│  • Registrar errores en DIARIO_ERRORES.md       │
└─────────────────────────────────────────────────┘
```

---

## ⚙️ Configuración Inicial

### 1. Pre-commit hooks
```bash
cd /home/duque_om/projects/Guia\ Master
pip install pre-commit ruff mypy
pre-commit install
```

### 2. Dependencias del proyecto
```bash
pip install -e ".[dev]"  # Instala dependencias de desarrollo
pip install -e ".[pytorch]"  # Para Semana 24
```

### 3. Verificar instalación
```bash
pre-commit run --all-files
```

---

## 📊 Tracking de Progreso

Usa esta tabla para monitorear tu avance:

| Semana | Drill Dimensiones | Simulacro | Diario Errores | Visualizaciones |
|--------|-------------------|-----------|----------------|-----------------|
| 1 | ⬜ | - | ⬜ | - |
| 2 | ⬜ | - | ⬜ | - |
| 3 | - | - | ⬜ | - |
| 4 | - | ⬜ 1A | ⬜ | - |
| 5 | - | - | ⬜ | - |
| 6 | - | - | ⬜ | ⬜ |
| 7 | - | - | ⬜ | ⬜ |
| 8 | - | ⬜ 1B | ⬜ | - |
| ... | | | | |

Leyenda: ⬜ Pendiente | ✅ Completado | ❌ Saltado

---

## 🎯 Métricas de Éxito

### Antes de cada Checkpoint del PDF:
1. **Simulacro correspondiente** ≥ 75 puntos
2. **Diario de Errores** actualizado
3. **Drill de Dimensiones** completado (Semanas 1-2)

### Señales de que vas bien:
- Puedes predecir `.shape` sin ejecutar código
- Tu código pasa pre-commit al primer intento
- Los errores del Diario no se repiten
- Resuelves simulacros en < 60 minutos
