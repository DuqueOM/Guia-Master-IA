#!/bin/bash
# Setup script para el Plan de Estudio v4.0
# Ejecutar: bash setup_env.sh

set -e

echo "🚀 Configurando entorno de desarrollo para Master en IA..."

# Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
else
    echo "✅ Entorno virtual ya existe"
fi

# Activar entorno
source venv/bin/activate

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

echo "📄 Instalando dependencias para generación de PDF (markdown, PyPDF2, weasyprint)..."
pip install markdown markdown-katex PyPDF2 weasyprint pygments

echo "🧪 Instalando dependencias para laboratorios interactivos (Streamlit/Manim)..."
pip install -r requirements-visual.txt

# Instalar herramientas de desarrollo
echo "🔧 Instalando herramientas de desarrollo..."
pip install ruff mypy pre-commit pytest pytest-cov

# Configurar pre-commit
echo "⚙️ Configurando pre-commit hooks..."
pre-commit install

# Verificar instalación
echo "✅ Verificando instalación..."
pre-commit run --all-files || true

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ ¡Configuración completada!"
echo ""
echo "Para activar el entorno en futuras sesiones:"
echo "  source venv/bin/activate"
echo ""
echo "Para instalar PyTorch (M07 / Semana 20 y proyecto final):"
echo "  pip install torch torchvision"

echo "Para instalar SOLO dependencias visuales (labs):"
echo "  pip install -r requirements-visual.txt"
echo ""
echo "Para ejecutar tests:"
echo "  pytest tests/ -v"
echo "════════════════════════════════════════════════════════"
