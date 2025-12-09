# 📂 Material Adicional en el Repositorio

> **Este documento lista recursos que NO están incluidos en el PDF/audio principal, pero que puedes consultar en el repositorio cuando los necesites.**

---

## 🔧 Documentos Internos y de Mantenimiento

Estos archivos son para mantenimiento del proyecto, no para estudio:

| Archivo | Descripción | ¿Cuándo consultarlo? |
|---------|-------------|----------------------|
| `MAINTENANCE_GUIDE.md` | Guía para mantener y actualizar la guía | Solo si vas a contribuir al repo |
| `DECISIONES_TECH.md` | Registro de decisiones técnicas del proyecto | Curiosidad sobre el "por qué" de ciertas elecciones |
| `REFERENCIAS_CRUZADAS.md` | Mapa de dependencias entre módulos | Si quieres entender cómo se conectan los temas |

**Ubicación:** `docs/`

---

## 🧩 Módulos DSA Adicionales (No requeridos para el Pathway)

Estos módulos cubren estructuras de datos y algoritmos que **NO son necesarios para aprobar el Pathway**, pero pueden ser útiles si:
- Te preparas para entrevistas técnicas de software
- Quieres profundizar en fundamentos de CS

| Archivo | Tema | ¿Cuándo estudiarlo? |
|---------|------|---------------------|
| `03_LOGICA_DISCRETA.md` | Lógica proposicional, conjuntos, relaciones | Antes de probabilidad si necesitas refuerzo matemático |
| `06_INVERTED_INDEX.md` | Índices invertidos para búsqueda | Después del proyecto integrador, si quieres optimizar búsqueda |
| `09_BINARY_SEARCH.md` | Búsqueda binaria y variantes | Preparación para entrevistas técnicas |
| `11_TFIDF_COSENO.md` | TF-IDF y similitud coseno | Complemento a ML supervisado para NLP |
| `13_LINKED_LISTS_STACKS_QUEUES.md` | Listas enlazadas, pilas, colas | Preparación para entrevistas técnicas |
| `17_GREEDY.md` | Algoritmos greedy | Preparación para entrevistas técnicas |
| `18_HEAPS.md` | Heaps y colas de prioridad | Preparación para entrevistas técnicas |

**Ubicación:** `docs/`

**Recomendación:** Completa primero los 10 módulos obligatorios. Luego, si tienes tiempo antes de entrevistas, estudia estos en orden: `09` → `13` → `14` → `15` → `16` → `17` → `18`.

---

## 📝 Soluciones y Scripts

| Archivo | Descripción | ¿Cuándo usarlo? |
|---------|-------------|-----------------|
| `EJERCICIOS_SOLUCIONES.md` | Soluciones a los ejercicios prácticos | **Después** de intentar resolver los ejercicios tú mismo |
| `DEMO_SCRIPT.md` | Script para demostrar el proyecto final | Cuando prepares tu presentación del Módulo 10 |

**Ubicación:** `docs/`

---

## 🎯 Cómo Usar Este Material

### Durante el programa (6 meses)
1. **Sigue el PDF/audio** como tu guía principal
2. **No consultes las soluciones** hasta haber intentado los ejercicios
3. **Ignora los módulos DSA adicionales** - no son necesarios para el Pathway

### Después de completar el programa
1. Si buscas trabajo en tech, estudia los módulos DSA adicionales
2. Usa `EJERCICIOS_SOLUCIONES.md` para verificar tu trabajo
3. Usa `DEMO_SCRIPT.md` para preparar presentaciones

### Si quieres contribuir
1. Lee `MAINTENANCE_GUIDE.md`
2. Revisa `DECISIONES_TECH.md` para entender el contexto
3. Usa `REFERENCIAS_CRUZADAS.md` para no romper dependencias

---

## 📁 Estructura del Repositorio

```
Guia Science in AI/
├── docs/                    # Todos los .md de la guía
│   ├── 00-24_*.md          # Módulos numerados
│   ├── *.md                # Material complementario
│   ├── generate_audio.py   # Genera audios MP3
│   └── generate_pdfs_pro.py # Genera el PDF
├── audios/                  # MP3 generados (mismo orden que el PDF)
├── pdf/                     # PDF generado
└── README.md               # Instrucciones del repo
```

---

> 💡 **Recuerda:** El PDF y los audios contienen **todo lo necesario** para completar el programa. Este documento solo lista material extra que puedes ignorar hasta que lo necesites.
