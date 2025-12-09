#!/usr/bin/env python3
"""Generador de PDF para la guía MS in AI Pathway.

- Usa WeasyPrint + markdown para generar un único PDF completo.
- Entrada: todos los .md de la carpeta `docs/` en un orden definido.
- Salida: `pdf/GUIA_MS_AI_PATHWAY_v2.pdf`.

Requisitos (pip): markdown, weasyprint

Ejecutar desde la raíz del repo:

    python3 docs/generate_pdfs_pro.py
"""

import re
from pathlib import Path
from typing import List

import markdown
from weasyprint import CSS, HTML

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "pdf"
FINAL_FILENAME = "GUIA_MS_AI_PATHWAY_v2.pdf"

# Carpeta donde están los archivos markdown (este script vive en docs/)
DOCS_DIR = BASE_DIR

# Orden lógico de la guía: Libro completo y autocontenido
# 100% enfocado en las 6 materias del Pathway + material de soporte
ORDERED_FILES: List[str] = [
    # === PARTE 1: INTRODUCCIÓN Y NAVEGACIÓN ===
    "index.md",
    "00_INDICE.md",
    "SYLLABUS.md",
    "PLAN_ESTUDIOS.md",
    # === PARTE 2: LOS 10 MÓDULOS OBLIGATORIOS ===
    # FASE 1: FUNDAMENTOS (Módulos 01-03)
    "01_PYTHON_PROFESIONAL.md",  # Módulo 01
    "02_OOP_DESDE_CERO.md",  # Módulo 02
    "10_ALGEBRA_LINEAL.md",  # Módulo 03 (Álgebra Lineal para ML)
    # FASE 2: PROBABILIDAD Y ESTADÍSTICA - PATHWAY LÍNEA 2 (Módulos 04-06)
    "19_PROBABILIDAD_FUNDAMENTOS.md",  # Módulo 04
    "20_ESTADISTICA_INFERENCIAL.md",  # Módulo 05
    "21_CADENAS_MARKOV_MONTECARLO.md",  # Módulo 06
    # FASE 3: MACHINE LEARNING - PATHWAY LÍNEA 1 (Módulos 07-09)
    "22_ML_SUPERVISADO.md",  # Módulo 07
    "23_ML_NO_SUPERVISADO.md",  # Módulo 08
    "24_INTRO_DEEP_LEARNING.md",  # Módulo 09
    # FASE 4: PROYECTO FINAL (Módulo 10)
    "12_PROYECTO_INTEGRADOR.md",  # Módulo 10
    # === PARTE 3: SOPORTE DEL PROGRAMA ===
    "CHECKLIST.md",
    "RUBRICA_EVALUACION.md",
    "EVALUACION_GUIA.md",
    # === PARTE 4: MATERIAL COMPLEMENTARIO (Recomendado) ===
    "EJERCICIOS.md",
    "GLOSARIO.md",
    "SIMULACRO_ENTREVISTA.md",
    "RECURSOS.md",
    # === PARTE 5: ANEXOS DSA (Solo para entrevistas técnicas) ===
    "04_ARRAYS_STRINGS.md",
    "05_HASHMAPS_SETS.md",
    "07_RECURSION.md",
    "08_SORTING.md",
    "14_TREES.md",
    "15_GRAPHS.md",
    "16_DYNAMIC_PROGRAMMING.md",
    # === PARTE 6: REFERENCIA AL REPOSITORIO ===
    "99_MATERIAL_REPO.md",
]

FILE_TITLES = {
    # Parte 1: Introducción y navegación
    "index.md": "Guía MS in AI Pathway",
    "00_INDICE.md": "Índice de Módulos",
    "SYLLABUS.md": "Syllabus del Programa",
    "PLAN_ESTUDIOS.md": "Plan de Estudios (26 semanas)",
    # Parte 2: Los 10 módulos obligatorios
    "01_PYTHON_PROFESIONAL.md": "Módulo 01 - Python Profesional",
    "02_OOP_DESDE_CERO.md": "Módulo 02 - OOP desde Cero",
    "10_ALGEBRA_LINEAL.md": "Módulo 03 - Álgebra Lineal para ML",
    "19_PROBABILIDAD_FUNDAMENTOS.md": "Módulo 04 - Fundamentos de Probabilidad ⭐",
    "20_ESTADISTICA_INFERENCIAL.md": "Módulo 05 - Estadística Inferencial ⭐",
    "21_CADENAS_MARKOV_MONTECARLO.md": "Módulo 06 - Markov y Monte Carlo ⭐",
    "22_ML_SUPERVISADO.md": "Módulo 07 - ML Supervisado ⭐",
    "23_ML_NO_SUPERVISADO.md": "Módulo 08 - ML No Supervisado ⭐",
    "24_INTRO_DEEP_LEARNING.md": "Módulo 09 - Deep Learning ⭐",
    "12_PROYECTO_INTEGRADOR.md": "Módulo 10 - Proyecto Final",
    # Parte 3: Soporte del programa
    "CHECKLIST.md": "Checklist de Finalización",
    "RUBRICA_EVALUACION.md": "Rúbrica de Evaluación",
    "EVALUACION_GUIA.md": "Guía de Evaluación",
    # Parte 4: Material complementario
    "EJERCICIOS.md": "Ejercicios Prácticos",
    "GLOSARIO.md": "Glosario Técnico",
    "SIMULACRO_ENTREVISTA.md": "Simulacro de Entrevista",
    "RECURSOS.md": "Recursos Recomendados",
    # Parte 5: Anexos DSA
    "04_ARRAYS_STRINGS.md": "Anexo DSA - Arrays y Strings",
    "05_HASHMAPS_SETS.md": "Anexo DSA - Hash Maps y Sets",
    "07_RECURSION.md": "Anexo DSA - Recursión",
    "08_SORTING.md": "Anexo DSA - Ordenamiento",
    "14_TREES.md": "Anexo DSA - Trees y BST",
    "15_GRAPHS.md": "Anexo DSA - Graphs, BFS, DFS",
    "16_DYNAMIC_PROGRAMMING.md": "Anexo DSA - Dynamic Programming",
    # Parte 6: Referencia al repositorio
    "99_MATERIAL_REPO.md": "Material Adicional en el Repositorio",
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
    padding-top: 30mm;
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
            <div class="cover-title">Guía 0→100: MS in AI Pathway</div>
            <div class="cover-subtitle">De Python Básico a Machine Learning y Deep Learning<br/>
            Preparación para CU Boulder MS in Artificial Intelligence</div>
            <div class="cover-meta">DUQUEOM · 2025 · Versión 2.0<br/>
            ⭐ Enfoque: Probabilidad, Estadística, ML, Deep Learning</div>
        </div>
        """
    )

    # Contenido (cada archivo con portada propia + contenido)
    for fname in ORDERED_FILES:
        path = DOCS_DIR / fname
        if not path.exists():
            print(f"  [WARN] No encontrado: {fname}")
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
            <div class="cover-title">{title}</div>
            <div class="cover-subtitle">Guía MS in AI Pathway</div>
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
    <title>Guía MS in AI Pathway - De Python a ML/DL</title>
</head>
<body>
    {''.join(parts)}
</body>
</html>
"""
    return full_html


def main() -> None:
    print("=" * 60)
    print("GENERADOR PDF - GUÍA MS IN AI PATHWAY")
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
