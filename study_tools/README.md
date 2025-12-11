# 🧰 Herramientas de Estudio - Plan v4.0 Estratégico

Este directorio contiene los materiales del **Plan de Acción Mejorado v4.0** para maximizar tu preparación para el Master en IA de CU Boulder.

---

## 📁 Contenido

| Archivo | Propósito | Cuándo usar |
|---------|-----------|-------------|
| `DIARIO_ERRORES.md` | Registro de errores matemáticos y de código | Diariamente, al final del estudio |
| `DRILL_DIMENSIONES_NUMPY.md` | Ejercicios de predicción de `.shape` | Semanas 1-2, 5 min diarios |
| `SIMULACRO_EXAMEN_TEORICO.md` | Banco de preguntas tipo examen | Sábados, 1 hora |
| `VISUALIZACION_GRADIENT_DESCENT.md` | Código para visualizar optimización | Semanas 6-7 |
| `DRYRUN_BACKPROPAGATION.md` | Plantilla para backprop en papel | Semana 18, antes de codificar |
| `PUENTE_NUMPY_PYTORCH.md` | Traducción NumPy → PyTorch | Semana 24 |

---

## 📅 Integración con el Cronograma

### Semana 0 (Preparación)
- [ ] Instalar pre-commit hooks: `pip install pre-commit && pre-commit install`
- [ ] Configurar AI Code Reviewer (ver `../prompts/AI_CODE_REVIEWER.md`)
- [ ] Leer este README completo

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
│  • Usar AI Code Reviewer para validar estilo   │
├─────────────────────────────────────────────────┤
│  CIERRE (Validación Feynman)                    │
│  • Explicar el concepto como si enseñaras      │
│  • Registrar errores en DIARIO_ERRORES.md      │
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
