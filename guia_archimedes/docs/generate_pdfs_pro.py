#!/usr/bin/env python3
"""Generador de PDF para la guía Archimedes Indexer.

- Usa WeasyPrint + markdown para generar un único PDF completo.
- Entrada: todos los .md de `guia_archimedes` en un orden definido.
- Salida: `pdf/GUIA_ARCHIMEDES_COMPLETA_v1.pdf`.

Requisitos (pip): markdown, weasyprint

Ejecutar desde la carpeta `guia_archimedes`:

    python3 generate_pdfs_pro.py
"""

import re
from pathlib import Path
from typing import List

import markdown
from weasyprint import CSS, HTML

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "pdf"
FINAL_FILENAME = "GUIA_ARCHIMEDES_COMPLETA_v1.pdf"

# Orden lógico de la guía Archimedes
ORDERED_FILES: List[str] = [
    "index.md",
    "00_INDICE.md",
    "SYLLABUS.md",
    "PLAN_ESTUDIOS.md",
    # Bloque Fundamentos
    "01_PYTHON_PROFESIONAL.md",
    "02_OOP_DESDE_CERO.md",
    "03_LOGICA_DISCRETA.md",
    "04_ARRAYS_STRINGS.md",
    "05_HASHMAPS_SETS.md",
    "06_INVERTED_INDEX.md",
    # Bloque Estructuras de Datos
    "13_LINKED_LISTS_STACKS_QUEUES.md",
    "14_TREES.md",
    "15_GRAPHS.md",
    # Bloque Algoritmos
    "07_RECURSION.md",
    "08_SORTING.md",
    "09_BINARY_SEARCH.md",
    "16_DYNAMIC_PROGRAMMING.md",
    "17_GREEDY.md",
    "18_HEAPS.md",
    # Bloque Matemáticas y Proyecto
    "10_ALGEBRA_LINEAL.md",
    "11_TFIDF_COSENO.md",
    "12_PROYECTO_INTEGRADOR.md",
    # Transversales
    "EJERCICIOS.md",
    "EJERCICIOS_SOLUCIONES.md",
    "GLOSARIO.md",
    "CHECKLIST.md",
    "RECURSOS.md",
    "SIMULACRO_ENTREVISTA.md",
    "DECISIONES_TECH.md",
    "RUBRICA_EVALUACION.md",
    "REFERENCIAS_CRUZADAS.md",
    "EVALUACION_GUIA.md",
    # Anexos
    "DEMO_SCRIPT.md",
    "MAINTENANCE_GUIDE.md",
]

FILE_TITLES = {
    "index.md": "Landing Page",
    "00_INDICE.md": "Índice Principal",
    "SYLLABUS.md": "Syllabus Archimedes",
    "PLAN_ESTUDIOS.md": "Plan de Estudios (6 meses)",
    "01_PYTHON_PROFESIONAL.md": "01 - Python Profesional",
    "02_OOP_DESDE_CERO.md": "02 - OOP desde Cero",
    "03_LOGICA_DISCRETA.md": "03 - Lógica y Big O",
    "04_ARRAYS_STRINGS.md": "04 - Arrays y Strings",
    "05_HASHMAPS_SETS.md": "05 - Hash Maps y Sets",
    "06_INVERTED_INDEX.md": "06 - Índice Invertido",
    "07_RECURSION.md": "07 - Recursión",
    "08_SORTING.md": "08 - Algoritmos de Ordenamiento",
    "09_BINARY_SEARCH.md": "09 - Búsqueda Binaria",
    "10_ALGEBRA_LINEAL.md": "10 - Álgebra Lineal",
    "11_TFIDF_COSENO.md": "11 - TF-IDF y Similitud de Coseno",
    "12_PROYECTO_INTEGRADOR.md": "12 - Proyecto Integrador",
    "13_LINKED_LISTS_STACKS_QUEUES.md": "13 - Linked Lists, Stacks, Queues",
    "14_TREES.md": "14 - Trees y BST",
    "15_GRAPHS.md": "15 - Graphs, BFS, DFS",
    "16_DYNAMIC_PROGRAMMING.md": "16 - Dynamic Programming",
    "17_GREEDY.md": "17 - Greedy Algorithms",
    "18_HEAPS.md": "18 - Heaps y Priority Queues",
    "EJERCICIOS.md": "Ejercicios",
    "EJERCICIOS_SOLUCIONES.md": "Soluciones de Ejercicios",
    "GLOSARIO.md": "Glosario",
    "CHECKLIST.md": "Checklist Final",
    "RECURSOS.md": "Recursos Recomendados",
    "SIMULACRO_ENTREVISTA.md": "Simulacro de Entrevista",
    "DECISIONES_TECH.md": "Decisiones Técnicas",
    "RUBRICA_EVALUACION.md": "Rúbrica de Evaluación",
    "REFERENCIAS_CRUZADAS.md": "Referencias Cruzadas",
    "EVALUACION_GUIA.md": "Evaluación de la Guía",
    "DEMO_SCRIPT.md": "Demo Script - Archimedes Indexer",
    "MAINTENANCE_GUIDE.md": "Guía de Mantenimiento",
}

# CSS sencillo pero legible
CONTENT_CSS = """
@page {
    size: A4;
    margin: 15mm 18mm 15mm 18mm;
}

body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 10pt;
    line-height: 1.4;
    color: #111827;
}

h1 {
    font-size: 16pt;
    color: #111827;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
    margin-top: 16px;
}

h2 {
    font-size: 13pt;
    color: #1f2937;
    border-bottom: 1px solid #e5e7eb;
    margin-top: 12px;
}

h3 {
    font-size: 11pt;
    color: #111827;
    margin-top: 8px;
}

p, li {
    font-size: 10pt;
}

code {
    font-family: "JetBrains Mono", monospace;
    background: #f3f4f6;
    padding: 0 2px;
}

pre {
    font-family: "JetBrains Mono", monospace;
    font-size: 8.5pt;
    background: #111827;
    color: #e5e7eb;
    padding: 6px 8px;
    border-radius: 4px;
    white-space: pre-wrap;
}

pre code {
    background: transparent;
    color: inherit;
    padding: 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
}

th, td {
    border: 1px solid #e5e7eb;
    padding: 4px 6px;
}

th {
    background: #f3f4f6;
}

.cover-page {
    page-break-after: always;
    text-align: center;
    /* Portada centrada ocupando casi toda la hoja A4 (restando márgenes @page) */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 260mm;  /* ~297mm - márgenes superiores/inferiores */
    margin: 0;
}

.cover-title {
    font-size: 24pt;
    font-weight: 700;
    margin-bottom: 8px;
}

.cover-subtitle {
    font-size: 12pt;
    color: #4b5563;
    margin-bottom: 20px;
}

.cover-meta {
    font-size: 9pt;
    color: #6b7280;
}
"""


def clean_markdown(text: str) -> str:
    """Limpieza ligera de Markdown para evitar artefactos en PDF.

    - Respeta bloques de código
    - Elimina líneas de solo guiones/asteriscos usadas como separadores excesivos
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    in_code = False

    for line in lines:
        s = line.strip()

        if s.startswith("```"):
            in_code = not in_code
            cleaned.append(line)
            continue

        if in_code:
            cleaned.append(line)
            continue

        # Separadores muy largos
        if len(s) > 3 and (set(s) <= {"-"} or set(s) <= {"="} or set(s) <= {"_"}):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def md_file_title(filename: str) -> str:
    """Devuelve título amigable para un archivo markdown."""
    return FILE_TITLES.get(
        filename, filename.replace(".md", "").replace("_", " ").title()
    )


def transform_internal_links(text: str) -> str:
    """Convierte enlaces Markdown internos a anclas dentro del PDF.

    [Texto](archivo.md)      -> [Texto](#mod_archivo)
    [Texto](ruta/archivo.md) -> [Texto](#mod_archivo)
    [Texto](archivo.md#sec)  -> [Texto](#mod_archivo)
    """

    def replace_link(match: "re.Match[str]") -> str:  # type: ignore[type-arg]
        link_text = match.group(1)
        path = match.group(2)

        # Enlaces web: se dejan intactos
        if path.startswith("http"):
            return match.group(0)

        # Enlaces a archivos .md (con o sin #fragmento)
        if ".md" in path:
            file_part = path.split("#", 1)[0]
            filename = Path(file_part).name
            base_name = filename.replace(".md", "")
            clean_id = "mod_" + re.sub(r"[^\w]+", "_", base_name)
            return f"[{link_text}](#{clean_id})"

        return match.group(0)

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, text)


def build_html() -> str:
    """Construye el HTML completo de la guía."""
    parts: list[str] = []

    # Portada
    parts.append(
        """
        <div class="cover-page">
            <div class="cover-title">Archimedes Indexer</div>
            <div class="cover-subtitle">De Python Básico a Pathway MS AI (CU Boulder)</div>
            <div class="cover-meta">DUQUEOM · 2025 · Versión 1.0</div>
        </div>
        """
    )

    # Contenido (cada archivo con portada propia + contenido)
    for fname in ORDERED_FILES:
        path = BASE_DIR / fname
        if not path.exists():
            continue

        md_raw = path.read_text(encoding="utf-8")
        md_clean = clean_markdown(md_raw)
        linked_md = transform_internal_links(md_clean)
        html_body = markdown.markdown(
            linked_md,
            extensions=["tables", "fenced_code", "nl2br"],
        )

        title = md_file_title(fname)
        base_name = fname.replace(".md", "")
        module_id = "mod_" + re.sub(r"[^\w]+", "_", base_name)

        section = f"""
        <div id="{module_id}" class="cover-page">
            <a name="{module_id}"></a>
            <div class="cover-title">{title}</div>
            <div class="cover-subtitle">Guía Archimedes Indexer</div>
            <div class="cover-meta">DUQUEOM · 2025</div>
        </div>
        <section>
            {html_body}
        </section>
        """
        parts.append(section)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Archimedes Indexer - Guía Completa</title>
</head>
<body>
    {''.join(parts)}
</body>
</html>
"""

    # Guardar HTML de depuración para inspeccionar enlaces/portadas
    OUTPUT_DIR.mkdir(exist_ok=True)
    debug_path = OUTPUT_DIR / "_DEBUG_FULL.html"
    debug_path.write_text(full_html, encoding="utf-8")

    return full_html


def main() -> None:
    print("=" * 60)
    print("GENERADOR PDF - ARCHIMEDES INDEXER")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    html = build_html()
    css = CSS(string=CONTENT_CSS)

    output_path = OUTPUT_DIR / FINAL_FILENAME
    print(f"[1] Generando PDF en: {output_path}")

    HTML(string=html, base_url=str(BASE_DIR)).write_pdf(
        target=output_path, stylesheets=[css]
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"[OK] PDF generado: {output_path.name} ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
