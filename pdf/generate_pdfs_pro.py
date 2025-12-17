#!/usr/bin/env python3
# flake8: noqa
"""
GENERADOR PDF PRO v9.0 - WeasyPrint Edition
- Generación nativa de PDF (texto seleccionable garantizado)
- Control estricto de huérfanos y viudas
- Portadas y contenido integrados
"""

import argparse
import csv
import re
from dataclasses import dataclass
from html import escape as html_escape
from pathlib import Path
from typing import List, Optional

import markdown
from PyPDF2 import PdfMerger
from weasyprint import CSS, HTML
from weasyprint.text.fonts import FontConfiguration

try:
    import markdown_katex  # noqa: F401

    HAS_KATEX = True
except Exception:
    HAS_KATEX = False

try:
    from pygments.formatters import HtmlFormatter

    PYGMENTS_CSS = HtmlFormatter(style="monokai").get_style_defs(".codehilite")
    HAS_PYGMENTS = True
except Exception:
    PYGMENTS_CSS = ""
    HAS_PYGMENTS = False


def get_markdown_extensions():
    extensions = ["tables", "fenced_code", "admonition", "nl2br"]
    extension_configs = {}
    if HAS_KATEX:
        extensions.insert(0, "markdown_katex")
        extension_configs["markdown_katex"] = {
            "no_inline_svg": True,
            "insert_fonts_css": True,
        }
    if HAS_PYGMENTS:
        extensions.insert(2, "codehilite")
        extension_configs["codehilite"] = {
            "guess_lang": False,
            "noclasses": False,
        }
    return extensions, extension_configs


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUT_DIR = PROJECT_DIR / "pdf"
FINAL_FILENAME = "GUIA_MS_AI_ML_SPECIALIST_v3.3.pdf"

ORDERED_FILES = [
    "index.md",
    "00_INDICE.md",
    "PLAN_ESTUDIOS.md",
    "01_PYTHON_CIENTIFICO.md",
    "02_ALGEBRA_LINEAL_ML.md",
    "03_CALCULO_MULTIVARIANTE.md",
    "04_PROBABILIDAD_ML.md",
    "05_SUPERVISED_LEARNING.md",
    "06_UNSUPERVISED_LEARNING.md",
    "07_DEEP_LEARNING.md",
    "08_PROYECTO_MNIST.md",
    "CHECKLIST.md",
    "PLAN_V4_ESTRATEGICO.md",
    "PLAN_V5_ESTRATEGICO.md",
    "RECURSOS.md",
    "GLOSARIO.md",
]


@dataclass(frozen=True)
class AppendixItem:
    title: str
    path: Path
    language: str


APPENDIX_ITEMS: List[AppendixItem] = [
    AppendixItem(
        "study_tools/README.md", PROJECT_DIR / "study_tools" / "README.md", "markdown"
    ),
    AppendixItem(
        "study_tools/CIERRE_SEMANAL.md",
        PROJECT_DIR / "study_tools" / "CIERRE_SEMANAL.md",
        "markdown",
    ),
    AppendixItem(
        "study_tools/DIARIO_METACOGNITIVO.md",
        PROJECT_DIR / "study_tools" / "DIARIO_METACOGNITIVO.md",
        "markdown",
    ),
    AppendixItem(
        "study_tools/TEORIA_CODIGO_BRIDGE.md",
        PROJECT_DIR / "study_tools" / "TEORIA_CODIGO_BRIDGE.md",
        "markdown",
    ),
    AppendixItem(
        "study_tools/BADGES_CHECKPOINTS.md",
        PROJECT_DIR / "study_tools" / "BADGES_CHECKPOINTS.md",
        "markdown",
    ),
    AppendixItem(
        "study_tools/SIMULACRO_PERFORMANCE_BASED.md",
        PROJECT_DIR / "study_tools" / "SIMULACRO_PERFORMANCE_BASED.md",
        "markdown",
    ),
    AppendixItem(
        "visualizations/viz_transformations.py",
        PROJECT_DIR / "visualizations" / "viz_transformations.py",
        "python",
    ),
    AppendixItem(
        "visualizations/viz_convolution.py",
        PROJECT_DIR / "visualizations" / "viz_convolution.py",
        "python",
    ),
    AppendixItem(
        "visualizations/viz_gradient_3d.py",
        PROJECT_DIR / "visualizations" / "viz_gradient_3d.py",
        "python",
    ),
    AppendixItem("requirements.txt", PROJECT_DIR / "requirements.txt", "text"),
    AppendixItem("pyproject.toml", PROJECT_DIR / "pyproject.toml", "toml"),
    AppendixItem("setup_env.sh", PROJECT_DIR / "setup_env.sh", "bash"),
]

# Mapa de Emojis a Símbolos Unicode
EMOJI_TO_SYMBOL = {
    "📚": "▣",
    "📖": "▤",
    "📄": "□",
    "📑": "▥",
    "📘": "▦",
    "🎯": "◎",
    "💡": "★",
    "⚡": "★",
    "🔥": "★",
    "✨": "★",
    "🔴": "●",
    "🟡": "◐",
    "🟢": "○",
    "🏷️": "▪",
    "🎬": "▶",
    "🧪": "◆",
    "🔬": "◆",
    "⚗️": "◆",
    "🐳": "▶",
    "🐍": "▷",
    "🚀": "►",
    "📊": "▣",
    "📈": "▲",
    "📉": "▼",
    "🔧": "●",
    "⚙️": "●",
    "🛠️": "●",
    "🔩": "●",
    "✅": "✓",
    "✔️": "✓",
    "☑️": "✓",
    "❌": "✗",
    "⛔": "✗",
    "⚠️": "▲",
    "🚨": "▲",
    "💥": "▲",
    "📦": "■",
    "📁": "■",
    "📂": "■",
    "🔗": "→",
    "➡️": "→",
    "👉": "→",
    "⬅️": "←",
    "📐": "◇",
    "🏗️": "◇",
    "🏛️": "◇",
    "🧠": "◎",
    "💭": "◎",
    "🤔": "◎",
    "📝": "▪",
    "✏️": "▪",
    "🖊️": "▪",
    "🎨": "◐",
    "🖼️": "◐",
    "🔒": "◑",
    "🔑": "◑",
    "🛡️": "◑",
    "⏰": "◒",
    "🕐": "◒",
    "⏱️": "◒",
    "💻": "▢",
    "🖥️": "▢",
    "📱": "▢",
    "🌐": "◯",
    "🌍": "◯",
    "🌎": "◯",
    "📺": "▣",
    "🎥": "▣",
    "🧭": "◈",
    "🗺️": "◈",
    "👤": "●",
    "👥": "●",
    "🙋": "●",
    "💬": "◆",
    "🗣️": "◆",
    "🏆": "★",
    "🥇": "★",
    "🎖️": "★",
    "📋": "▤",
    "🗒️": "▤",
    "📃": "▤",
}

FILE_TITLES = {
    "index.md": "PORTADA",
    "00_INDICE.md": "ÍNDICE GENERAL",
    "PLAN_ESTUDIOS.md": "PLAN DE ESTUDIOS",
    "01_PYTHON_CIENTIFICO.md": "MÓDULO 01 - PYTHON + PANDAS + NUMPY",
    "02_ALGEBRA_LINEAL_ML.md": "MÓDULO 02 - ÁLGEBRA LINEAL PARA ML",
    "03_CALCULO_MULTIVARIANTE.md": "MÓDULO 03 - CÁLCULO MULTIVARIANTE",
    "04_PROBABILIDAD_ML.md": "MÓDULO 04 - PROBABILIDAD PARA ML",
    "05_SUPERVISED_LEARNING.md": "MÓDULO 05 - SUPERVISED LEARNING",
    "06_UNSUPERVISED_LEARNING.md": "MÓDULO 06 - UNSUPERVISED LEARNING",
    "07_DEEP_LEARNING.md": "MÓDULO 07 - DEEP LEARNING + CNNs",
    "08_PROYECTO_MNIST.md": "MÓDULO 08 - PROYECTO MNIST ANALYST",
    "CHECKLIST.md": "CHECKLIST FINAL",
    "PLAN_V4_ESTRATEGICO.md": "PLAN DE ACCIÓN MEJORADO v4.0",
    "PLAN_V5_ESTRATEGICO.md": "PLAN DE ACCIÓN PERFECCIONADO v5.0",
    "RECURSOS.md": "RECURSOS DE APRENDIZAJE",
    "GLOSARIO.md": "GLOSARIO TÉCNICO",
}

# CSS Optimizado para WeasyPrint (Estilo Compacto y Profesional)
CONTENT_CSS = (
    """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=STIX+Two+Text:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
/* Intentar cargar fuente de emojis si está disponible localmente */
@font-face {
  font-family: 'Noto Color Emoji';
  src: local('Noto Color Emoji'), local('Apple Color Emoji'), local('Segoe UI Emoji');
}

@page {
    size: Letter;
    margin: 10mm 15mm 10mm 15mm; /* Márgenes de página reducidos (1.5cm laterales, 1cm verticales) */
    @bottom-right {
        content: counter(page);
        font-family: 'Inter', sans-serif;
        font-size: 9pt;
        color: #64748b;
        margin-bottom: 5mm;
        margin-right: 5mm;
    }
}

@page :first {
    margin: 0;
    @bottom-right { content: none; }
}

@page cover {
    margin: 0;
    size: Letter;
    @bottom-right { content: none; }
}

* { box-sizing: border-box; }

body {
    font-family: 'STIX Two Text', 'Times New Roman', serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #1e293b;
    margin: 0;
    padding: 0; /* SIN padding para que las portadas funcionen */
    max-width: 100%;
}

.katex { font-size: 1em; }
.katex-display { text-align: center; margin: 8px 0; }

/* Contenido con margen de seguridad */
.content {
    padding: 0 15px 0 5px; /* Padding solo en el contenido, no en portadas */
}

/* Headers */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif;
    line-height: 1.2;
    max-width: 100%;
    word-wrap: break-word;
    page-break-after: avoid;
}

h1 {
    font-size: 14pt;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 4px;
    margin: 0 0 6px 0;
}

h2 {
    font-size: 12pt;
    font-weight: 600;
    color: #1e40af;
    margin: 8px 0 4px 0;
    padding-bottom: 2px;
    border-bottom: 1px solid #e2e8f0;
}

h3 {
    font-size: 11pt;
    font-weight: 600;
    color: #2563eb;
    margin: 6px 0 3px 0;
}

h4 {
    font-size: 8.5pt; /* Mismo tamaño que body pero bold */
    font-weight: 600;
    color: #3b82f6;
    margin: 4px 0 2px 0;
}

/* Evitar partir tablas, imágenes y CÓDIGO */
img, figure, .admonition {
    page-break-inside: avoid;
    break-inside: avoid; /* Refuerzo para navegadores modernos/WeasyPrint */
}

/* Text Blocks */
p, li {
    margin: 0 0 6px 0;
    text-align: justify;
    orphans: 2;
    widows: 2;
    overflow-wrap: anywhere;
    word-break: normal;
    max-width: 100%;
}

ul, ol { margin: 0 0 8px 0; padding-left: 16px; max-width: 100%; }
li { margin-bottom: 2px; text-align: left; }
blockquote p { margin: 0; text-align: left; } /* Forzar izquierda dentro del blockquote */

blockquote {
    margin: 8px 0;
    padding: 8px 10px;
    border-left: 4px solid #3b82f6;
    background: #eff6ff;
    color: #0f172a;
    border-radius: 6px;
}

/* Admonitions (Python-Markdown extension) */
.admonition {
    margin: 8px 0;
    padding: 8px 10px;
    border-left: 4px solid #0ea5e9;
    background: #ecfeff;
    color: #0f172a;
    border-radius: 6px;
}
.admonition > .admonition-title {
    margin: 0 0 6px 0;
    font-weight: 700;
    color: #075985;
}

details {
    display: block;
    margin: 8px 0;
    padding: 8px 10px;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #94a3b8;
    background: #f8fafc;
    color: #0f172a;
    border-radius: 6px;
    break-inside: auto;
}

details > summary {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 6px 0;
    list-style: none;
    break-after: avoid-page;
}

details > summary::-webkit-details-marker { display: none; }

details:not([open]) > :not(summary) { display: block; }

details > summary + * { break-before: avoid-page; }

/* Links - IMPORTANTE para PDF */
a {
    color: #2563eb;
    text-decoration: underline;
    cursor: pointer;
}
a:hover {
    color: #1d4ed8;
}
/* Links en tablas */
td a, th a {
    color: #2563eb;
    text-decoration: underline;
}

/* Code */
code {
    background: #f1f5f9;
    color: #be185d;
    padding: 0px 2px;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9em;
    overflow-wrap: break-word;
    word-break: break-word; /* Evitar romper tokens en exceso */
}

pre {
    background: #0f172a;
    color: #f1f5f9;
    border-radius: 4px;
    padding: 8px;

    /* FIX ANCHO */
    width: auto;
    margin: 8px 10px 8px 0;

    font-family: 'JetBrains Mono', monospace;
    font-size: 7.5pt;
    line-height: 1.3;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-wrap: break-word;
    word-break: normal;
    text-align: left;
    page-break-inside: avoid;
    break-inside: avoid;
    border: 1px solid #1e293b;
}
pre code { background: none; color: inherit; padding: 0; word-break: normal; }

/* Pygments wrapper */
.codehilite {
    margin: 8px 0;
}
.codehilite pre {
    margin: 0;
}

/* Imágenes */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 8px auto;
}

/* Tables */
table {
    width: 100%;
    max-width: 100%;
    table-layout: auto;
    border-collapse: collapse;
    border-spacing: 0;
    margin: 10px 0;
    font-size: 8.2pt;
    break-inside: auto;
    border: 1px solid #e2e8f0;
}

thead { display: table-header-group; }
tfoot { display: table-footer-group; }
tr { break-inside: avoid; page-break-inside: avoid; }

th {
    background: #f1f5f9;
    color: #1e293b;
    font-weight: 600;
    text-align: left;
    padding: 6px 6px;
    border-bottom: 2px solid #e2e8f0;
    overflow-wrap: anywhere;
    word-break: normal;
    hyphens: auto;
}

td {
    border-bottom: 1px solid #e2e8f0;
    border-left: 1px solid #e2e8f0;
    padding: 5px 6px;
    vertical-align: top;
    text-align: left; /* Texto en tablas siempre a la izquierda */
    overflow-wrap: anywhere;
    word-wrap: break-word;
    word-break: normal;
    hyphens: auto;
}

tr:nth-child(even) { background: #f8fafc; }

th { border-left: 1px solid #e2e8f0; }
th:first-child, td:first-child { border-left: none; }

/* Portadas - FULL PAGE sin márgenes */
.cover-page {
    page: cover;
    break-after: page;
    width: 100%;
    height: 279.4mm; /* Altura exacta Letter */
    background: linear-gradient(160deg, #1e3a8a 0%, #172554 100%);
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    margin: 0;
    padding: 0;
    position: relative;
    box-sizing: border-box;
}

.cover-content {
    width: 80%;
    padding: 40px 20px;     /* Más padding vertical */
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    background: rgba(255,255,255,0.03);
}

.cover-title {
    font-size: 24pt;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    line-height: 1.2;
    color: #ffffff;
}

.cover-subtitle {
    font-size: 12pt;
    font-weight: 300;
    color: #bfdbfe;
    letter-spacing: 2px;
    margin-bottom: 30px;
    text-transform: uppercase;
}

.cover-badge {
    display: inline-block;
    background: #3b82f6;
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 10pt;
    font-weight: 600;
    letter-spacing: 1px;
}

.global-cover {
    background: #0f172a;
    background-image: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
}

.global-title {
    font-size: 42pt;
    font-weight: 900;
    color: #60a5fa;
    letter-spacing: 3px;
    margin-bottom: 10px;
    line-height: 1;
}

/* Anti-huérfanos CSS puro (sin wrappers) */
h2, h3, h4 {
    break-after: avoid-page;
}
h2 + *, h3 + *, h4 + * {
    break-before: avoid-page;
}
"""
    + "\n"
    + PYGMENTS_CSS
    + "\n"
)


class PDFGenerator:
    def __init__(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        self.font_config = FontConfiguration()
        self.css = CSS(string=CONTENT_CSS, font_config=self.font_config)

    def _module_id_from_path(self, path: str) -> str:
        filename = Path(path).name
        base_name = filename.replace(".md", "")
        return "mod_" + re.sub(r"[^\w]+", "_", base_name)

    def _read_csv_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(r) for r in reader]

    def _rubrica_csv_to_reference_markdown(self, path: Path) -> str:
        rows = self._read_csv_rows(path)
        scopes: list[str] = []
        by_scope: dict[str, list[dict[str, str]]] = {}
        for r in rows:
            scope = (r.get("scope") or "").strip()
            if not scope:
                continue
            if scope not in by_scope:
                by_scope[scope] = []
                scopes.append(scope)
            by_scope[scope].append(r)

        lines: list[str] = []
        lines.append("### rubrica.csv (vista legible)")
        lines.append("")
        lines.append(
            "Este archivo es la **fuente estructurada** de la rúbrica (pesos/criterios/gates). Para editarlo, hazlo en el repositorio; aquí se muestra en formato tabla."
        )
        lines.append("")

        columns = [
            "criterion_id",
            "weight_points",
            "hard_gate",
            "category",
            "criterion",
            "evidence_required",
        ]
        header = "| " + " | ".join(columns) + " |"
        sep = "|" + "|".join(["---"] * len(columns)) + "|"

        for scope in scopes:
            lines.append(f"#### Scope: {scope}")
            lines.append("")
            lines.append(header)
            lines.append(sep)
            for r in by_scope[scope]:
                vals = [(r.get(c) or "").replace("\n", " ").strip() for c in columns]
                lines.append("| " + " | ".join(vals) + " |")
            lines.append("")

        return "\n".join(lines)

    def build_rubric_scoring_sheet_markdown(self) -> str:
        csv_path = PROJECT_DIR / "rubrica.csv"
        if not csv_path.exists():
            return "## Hoja de scoring\n\nNo se encontró `rubrica.csv`.\n"

        rows = self._read_csv_rows(csv_path)
        scopes: list[str] = []
        by_scope: dict[str, list[dict[str, str]]] = {}
        for r in rows:
            scope = (r.get("scope") or "").strip()
            if not scope:
                continue
            if scope not in by_scope:
                by_scope[scope] = []
                scopes.append(scope)
            by_scope[scope].append(r)

        lines: list[str] = []
        lines.append("## Hoja de scoring (para imprimir / registrar)")
        lines.append("")
        lines.append(
            "Marca un nivel por criterio y agrega evidencia. Luego calcula el total ponderado (0–100)."
        )
        lines.append("")
        lines.append("Niveles:")
        lines.append("")
        lines.append("- Exceeds = 1.0")
        lines.append("- Meets = 0.8")
        lines.append("- Approaching = 0.5")
        lines.append("- Not met = 0.0")
        lines.append("")
        lines.append("Registro rápido de simulacros:")
        lines.append("")
        lines.append("- PB-8: ________/100")
        lines.append("- PB-16: ________/100")
        lines.append("- PB-23: ________/100  (condición dura: **PB-23 ≥ 80**) ")
        lines.append("")

        table_cols = [
            "criterion_id",
            "weight_points",
            "nivel (E/M/A/N)",
            "evidencia / notas",
        ]
        header = "| " + " | ".join(table_cols) + " |"
        sep = "|" + "|".join(["---"] * len(table_cols)) + "|"

        for scope in scopes:
            lines.append(f"### {scope}")
            lines.append("")
            lines.append(header)
            lines.append(sep)
            for r in by_scope[scope]:
                cid = (r.get("criterion_id") or "").strip()
                pts = (r.get("weight_points") or "").strip()
                gate = (r.get("hard_gate") or "").strip().lower() == "true"
                if gate:
                    cid = f"{cid} (GATE)"
                lines.append(f"| {cid} | {pts} | ____ | __________________________ |")
            lines.append("")

        lines.append("### Total")
        lines.append("")
        lines.append("- TOTAL: ________/100")
        lines.append("- Estado: __________ (Listo / Aún no listo)")
        lines.append("")
        return "\n".join(lines)

    def _demote_headings(self, md: str, level_offset: int = 1) -> str:
        parts = self._split_by_fenced_codeblocks(md)
        out: List[str] = []

        def demote_line(line: str) -> str:
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if not m:
                return line
            hashes = m.group(1)
            title = m.group(2)
            new_level = min(6, len(hashes) + level_offset)
            return ("#" * new_level) + " " + title

        for is_code, seg in parts:
            if is_code:
                out.append(seg)
                continue
            out.append("\n".join(demote_line(ln) for ln in seg.splitlines()))

        return "\n".join(out)

    def _split_by_fenced_codeblocks(self, text: str) -> List[tuple[bool, str]]:
        parts: List[tuple[bool, str]] = []
        fence_re = re.compile(r"(^```[\s\S]*?^```\s*$)", flags=re.MULTILINE)
        last = 0
        for m in fence_re.finditer(text):
            if m.start() > last:
                parts.append((False, text[last : m.start()]))
            parts.append((True, m.group(0)))
            last = m.end()
        if last < len(text):
            parts.append((False, text[last:]))
        return parts

    def _convert_latex_math_to_katex_syntax(self, text: str) -> str:
        def convert_segment(seg: str) -> str:
            seg = re.sub(
                r"\$\$(.+?)\$\$",
                lambda m: "```math\n" + m.group(1).strip() + "\n```",
                seg,
                flags=re.DOTALL,
            )
            seg = re.sub(
                r"(?<!\\)\$(?!`)([^\n$]+?)(?<!\\)\$",
                lambda m: "$`" + m.group(1).strip() + "`$",
                seg,
            )
            return seg

        out = []
        for is_code, seg in self._split_by_fenced_codeblocks(text):
            out.append(seg if is_code else convert_segment(seg))
        return "".join(out)

    def _fence_code_block(self, content: str, language: str) -> str:
        content = content.rstrip("\n")
        fence = "```"
        while fence in content:
            fence += "`"
        return f"{fence}{language}\n{content}\n{fence}\n"

    def build_appendix_markdown(self) -> str:
        lines = ["# APÉNDICE: Archivos del Repositorio", ""]
        items: List[AppendixItem] = []

        if (PROJECT_DIR / "study_tools").exists():
            for p in sorted((PROJECT_DIR / "study_tools").glob("*.md")):
                if p.name == "RUBRICA_v1.md":
                    continue
                items.append(AppendixItem(f"study_tools/{p.name}", p, "markdown"))

        if (PROJECT_DIR / "prompts").exists():
            for p in sorted((PROJECT_DIR / "prompts").glob("*.md")):
                items.append(AppendixItem(f"prompts/{p.name}", p, "markdown"))

        if (PROJECT_DIR / "scripts").exists():
            for p in sorted((PROJECT_DIR / "scripts").glob("*.py")):
                items.append(AppendixItem(f"scripts/{p.name}", p, "python"))

        if (PROJECT_DIR / "visualizations").exists():
            for p in sorted((PROJECT_DIR / "visualizations").iterdir()):
                if not p.is_file():
                    continue
                if p.suffix == ".py":
                    lang = "python"
                elif p.suffix == ".ipynb":
                    lang = "json"
                else:
                    lang = "text"
                items.append(AppendixItem(f"visualizations/{p.name}", p, lang))

        rubrica_csv = PROJECT_DIR / "rubrica.csv"
        if rubrica_csv.exists():
            items.append(AppendixItem("rubrica.csv", rubrica_csv, "csv"))

        for item in items:
            raw = item.path.read_text(encoding="utf-8", errors="ignore")

            filename = Path(item.title).name
            base_name = filename.replace(".md", "")
            clean_id = "mod_" + re.sub(r"[^\w]+", "_", base_name)
            lines.append(f'<a id="{clean_id}"></a>')
            lines.append(f"## {html_escape(item.title)}")
            lines.append("")
            if item.language == "markdown":
                lines.append(self._demote_headings(raw, level_offset=1))
            elif item.language == "csv":
                lines.append(self._rubrica_csv_to_reference_markdown(item.path))
            else:
                lines.append(self._fence_code_block(raw, item.language))
            lines.append("")

        return "\n".join(lines)

    def clean_markdown(self, text: str) -> str:
        if HAS_KATEX:
            text = self._convert_latex_math_to_katex_syntax(text)
        lines = text.split("\n")
        cleaned = []
        in_code = False
        code_fence_prefix: str = ""

        for line in lines:
            s = line.strip()

            # Manejo de bloques de código
            if s.startswith("```"):
                leading = len(line) - len(line.lstrip(" \t"))
                prefix = line[:leading]
                in_code = not in_code
                if in_code:
                    code_fence_prefix = prefix
                else:
                    code_fence_prefix = ""

                if leading <= 3:
                    cleaned.append(line.lstrip(" \t"))
                else:
                    cleaned.append(line)
                continue

            if in_code:
                if code_fence_prefix and line.startswith(code_fence_prefix):
                    cleaned.append(line[len(code_fence_prefix) :])
                else:
                    cleaned.append(line)
                continue

            # Limpieza de regex (headers, nav, etc)
            if re.match(r"^[-_*]{3,}\s*$", s):
                continue
            if re.match(r"^═+\s*$", s) or "════" in s:
                continue
            if "[←" in s or "Volver al Índice" in s or "[Siguiente:" in s:
                continue

            # Eliminar divs HTML que rompen tablas Markdown
            if "<div" in s or "</div>" in s:
                continue

            # Limpieza de artefactos '# #' que aparecen antes de títulos
            if re.match(r"^#+\s+#+\s+", s):
                s = re.sub(r"^#+\s+#+\s+", "", s)

            # Limpieza específica para '# # ' suelto
            s = re.sub(r"^#\s+#\s+", "", s)

            # Para headers: usar la versión limpia (s) para que Markdown los reconozca
            # Los espacios iniciales rompen el parseo de headers
            if s.startswith("#"):
                # Asegurar línea en blanco antes de headers
                if cleaned and cleaned[-1].strip() != "":
                    cleaned.append("")
                cleaned.append(s)  # Usar versión sin espacios iniciales
                continue

            # Lógica de Tablas:
            if s.startswith("|"):
                if cleaned:
                    prev_s = cleaned[-1].strip()
                    if prev_s != "" and not prev_s.startswith("|"):
                        cleaned.append("")

            cleaned.append(line)

        return "\n".join(cleaned)

    def transform_internal_links(
        self,
        text: str,
        debug: bool = False,
        allowed_internal_ids: set[str] | None = None,
    ) -> str:
        """
        Convierte enlaces Markdown [Texto](archivo.md) a enlaces internos HTML <a href="#archivo">Texto</a>.
        """
        transformed_count = 0

        def replace_link(match):
            nonlocal transformed_count
            link_text = match.group(1)
            path = match.group(2)

            # Links web: no tocar
            if path.startswith("http"):
                return match.group(0)

            # Links a archivos .md: convertir a anclas internas
            if path.endswith(".md"):
                filename = Path(path).name
                # ID: mod_FILENAME (sin extensión, limpio)
                base_name = filename.replace(".md", "")
                clean_id = "mod_" + re.sub(r"[^\w]+", "_", base_name)
                if (
                    allowed_internal_ids is not None
                    and clean_id not in allowed_internal_ids
                ):
                    return match.group(0)
                transformed_count += 1
                return f"[{link_text}](#{clean_id})"

            return match.group(0)

        result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, text)

        if debug and transformed_count > 0:
            print(
                f"    [DEBUG] {transformed_count} links transformados a anclas internas"
            )

        return result

    def get_cover_title(self, filename: str) -> str:
        return FILE_TITLES.get(
            filename, filename.replace(".md", "").replace("_", " ").upper()
        )

    def generate_module_html(self, title: str, md_content: str, filename: str) -> str:
        # 1. Limpiar Markdown
        clean_md = self.clean_markdown(md_content)

        # 2. Transformar enlaces a internos (con debug)
        linked_md = self.transform_internal_links(clean_md, debug=True)

        md_extensions, md_extension_configs = get_markdown_extensions()
        content_html = markdown.markdown(
            linked_md,
            extensions=md_extensions,
            extension_configs=md_extension_configs,
        )

        # ID para el ancla: mod_FILENAME (mismo formato que transform_internal_links)
        base_name = filename.replace(".md", "")
        module_id = "mod_" + re.sub(r"[^\w]+", "_", base_name)

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"></head>
        <body>
            <!-- PORTADA CON ANCLA INTEGRADA -->
            <div id="{module_id}" class="cover-page">
                <a name="{module_id}"></a>
                <div class="cover-content">
                    <div style="font-size:40pt;margin-bottom:20px;">💎</div>
                    <div class="cover-title">{title}</div>
                    <div class="cover-subtitle">MS in AI Pathway - ML Specialist v3.3</div>
                    <div class="cover-badge">MATEMÁTICAS APLICADAS A CÓDIGO</div>
                </div>
                <div style="position:absolute;bottom:30px;font-size:10pt;opacity:0.6;">DUQUEOM | 2025</div>
            </div>

            <!-- CONTENIDO -->
            <div class="content">
                {content_html}
            </div>
        </body>
        </html>
        """
        return full_html

    def generate_global_cover_html(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"></head>
        <body>
            <div class="cover-page global-cover">
                <div class="cover-content" style="border-color:rgba(96,165,250,0.3);background:rgba(15,23,42,0.6);">
                    <div style="font-size:60pt;margin-bottom:30px;">🚀</div>
                    <div class="global-title">MS AI PATHWAY</div>
                    <div class="cover-subtitle" style="color:#94a3b8;margin-bottom:40px;">ML Specialist v3.3 | 24 Semanas</div>
                    <div class="cover-badge" style="background:linear-gradient(90deg,#3b82f6,#8b5cf6);border:none;padding:10px 30px;">LÍNEA 1: MACHINE LEARNING</div>
                </div>
                <div style="position:absolute;bottom:30px;font-size:11pt;color:#64748b;">DUQUEOM | 2025</div>
            </div>
        </body>
        </html>
        """

    def process_module(self, md_file: str, debug_html: bool = False) -> Optional[Path]:
        path = DOCS_DIR / md_file
        if not path.exists():
            return None

        print(f"  > {md_file}")
        raw = path.read_text(encoding="utf-8")
        clean = self.clean_markdown(raw)
        title = self.get_cover_title(md_file)

        html_content = self.generate_module_html(title, clean, md_file)

        # DEBUG: Guardar HTML del índice para inspección
        if debug_html and md_file == "00_INDICE.md":
            debug_path = OUTPUT_DIR / "_DEBUG_00_INDICE.html"
            debug_path.write_text(html_content, encoding="utf-8")
            print(f"    [DEBUG] HTML guardado en {debug_path}")

        stem = md_file.replace(".md", "")
        pdf_path = OUTPUT_DIR / f"{stem}.pdf"

        HTML(string=html_content, base_url=str(DOCS_DIR)).write_pdf(
            target=pdf_path, stylesheets=[self.css]
        )
        return pdf_path

    def generate_global_cover(self) -> Path:
        html = self.generate_global_cover_html()
        pdf_path = OUTPUT_DIR / "_000_COVER.pdf"
        HTML(string=html, base_url=str(DOCS_DIR)).write_pdf(
            target=pdf_path, stylesheets=[self.css]
        )
        return pdf_path

    def merge_all(self, pdfs: List[Path]):
        final = OUTPUT_DIR / FINAL_FILENAME
        print(f"\n[*] Combinando {len(pdfs)} PDFs...")
        m = PdfMerger()
        for p in pdfs:
            if p.exists():
                m.append(str(p))
        m.write(str(final))
        m.close()
        print(f"[OK] {final.name} ({final.stat().st_size / 1e6:.1f} MB)")


def main():
    print("=" * 50)
    print("  GENERADOR PDF PRO v9.1 (WeasyPrint - Single Document)")
    print("=" * 50)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--mode",
        choices=["lean", "full", "both"],
        default="both",
    )
    args = parser.parse_args()

    gen = PDFGenerator()

    def collect_allowed_internal_ids(include_appendix: bool) -> set[str]:
        ids: set[str] = set()
        for f in ORDERED_FILES:
            ids.add(gen._module_id_from_path(f))
        ids.add(gen._module_id_from_path("RUBRICA_v1.md"))
        ids.add("mod_RUBRICA_SHEET")
        if include_appendix:
            ids.add("mod_APPENDIX")
            if (PROJECT_DIR / "study_tools").exists():
                for p in (PROJECT_DIR / "study_tools").glob("*.md"):
                    ids.add(gen._module_id_from_path(p.name))
            if (PROJECT_DIR / "prompts").exists():
                for p in (PROJECT_DIR / "prompts").glob("*.md"):
                    ids.add(gen._module_id_from_path(p.name))
        return ids

    def render_section(
        title: str,
        module_id: str,
        md_text: str,
        allowed_ids: set[str],
        icon: str = "💎",
        subtitle: str = "MS in AI Pathway - ML Specialist v3.0",
    ) -> str:
        clean = gen.clean_markdown(md_text)
        linked_md = gen.transform_internal_links(
            clean, debug=True, allowed_internal_ids=allowed_ids
        )
        md_extensions, md_extension_configs = get_markdown_extensions()
        content_html = markdown.markdown(
            linked_md,
            extensions=md_extensions,
            extension_configs=md_extension_configs,
        )
        return f"""
            <div class=\"cover-page\">
                <div class=\"cover-content\">
                    <div style=\"font-size:40pt;margin-bottom:20px;\">{icon}</div>
                    <div id=\"{module_id}\" class=\"cover-title\">{title}</div>
                    <div class=\"cover-subtitle\">{subtitle}</div>
                    <div class=\"cover-badge\">MATEMÁTICAS APLICADAS A CÓDIGO</div>
                </div>
                <div style=\"position:absolute;bottom:30px;font-size:10pt;opacity:0.6;\">DUQUEOM | 2025</div>
            </div>
            <div class=\"content\">
                {content_html}
            </div>
        """

    def build_html(include_appendix: bool) -> str:
        allowed_ids = collect_allowed_internal_ids(include_appendix=include_appendix)

        parts: list[str] = []
        parts.append(gen.generate_global_cover_html())

        for f in ORDERED_FILES:
            path = DOCS_DIR / f
            if not path.exists():
                continue
            raw = path.read_text(encoding="utf-8")
            title = gen.get_cover_title(f)
            module_id = gen._module_id_from_path(f)
            parts.append(render_section(title, module_id, raw, allowed_ids))

        rubric_path = PROJECT_DIR / "study_tools" / "RUBRICA_v1.md"
        if rubric_path.exists():
            rubric_raw = rubric_path.read_text(encoding="utf-8")
            rubric_md = gen._demote_headings(rubric_raw, level_offset=1)
            parts.append(
                render_section(
                    "RÚBRICA (Referencia)",
                    gen._module_id_from_path("RUBRICA_v1.md"),
                    rubric_md,
                    allowed_ids,
                    icon="📋",
                )
            )

        sheet_md = gen.build_rubric_scoring_sheet_markdown()
        parts.append(
            render_section(
                "RÚBRICA: Hoja de scoring",
                "mod_RUBRICA_SHEET",
                sheet_md,
                allowed_ids,
                icon="🧾",
            )
        )

        if include_appendix:
            appendix_md = gen.build_appendix_markdown()
            appendix_md = gen.transform_internal_links(
                appendix_md, debug=True, allowed_internal_ids=allowed_ids
            )
            md_extensions, md_extension_configs = get_markdown_extensions()
            appendix_html = markdown.markdown(
                appendix_md,
                extensions=md_extensions,
                extension_configs=md_extension_configs,
            )
            appendix_id = "mod_APPENDIX"
            appendix_block = f"""
                <div class=\"cover-page\">
                    <div class=\"cover-content\">
                        <div style=\"font-size:40pt;margin-bottom:20px;\">📎</div>
                        <div id=\"{appendix_id}\" class=\"cover-title\">APÉNDICE</div>
                        <div class=\"cover-subtitle\">Archivos del repositorio (referencia)</div>
                        <div class=\"cover-badge\">RECURSOS + CONFIG + SCRIPTS</div>
                    </div>
                    <div style=\"position:absolute;bottom:30px;font-size:10pt;opacity:0.6;\">DUQUEOM | 2025</div>
                </div>
                <div class=\"content\">{appendix_html}</div>
            """
            parts.append(appendix_block)

        global_cover_body = gen.generate_global_cover_html()
        body_match = re.search(r"<body>(.*?)</body>", global_cover_body, re.DOTALL)
        global_cover_content = body_match.group(1) if body_match else ""
        return f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset=\"UTF-8\"></head>
        <body>
            {global_cover_content}
            {"".join(parts[1:])}
        </body>
        </html>
        """

    outputs: list[tuple[str, str, bool]] = []
    if args.mode in {"lean", "both"}:
        outputs.append(("LEAN", "GUIA_MS_AI_ML_SPECIALIST_LEAN.pdf", False))
    if args.mode in {"full", "both"}:
        outputs.append(("FULL", "GUIA_MS_AI_ML_SPECIALIST_FULL.pdf", True))

    for tag, filename, include_appendix in outputs:
        print("\n[1] Generando documento único...")
        html = build_html(include_appendix=include_appendix)
        debug_path = OUTPUT_DIR / f"_DEBUG_{tag}.html"
        debug_path.write_text(html, encoding="utf-8")
        print(f"  [DEBUG] HTML completo guardado en {debug_path}")
        print("\n[4] Generando PDF final...")
        final_path = OUTPUT_DIR / filename
        HTML(string=html, base_url=str(PROJECT_DIR)).write_pdf(
            target=final_path, stylesheets=[gen.css]
        )
        print(f"\n[OK] {final_path.name} ({final_path.stat().st_size / 1e6:.1f} MB)")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
