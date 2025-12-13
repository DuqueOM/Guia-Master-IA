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
pip install numpy matplotlib scipy scikit-learn pandas seaborn jupyter jupyterlab ipython plotly ipywidgets

echo "📄 Instalando dependencias para generación de PDF (markdown, PyPDF2, weasyprint)..."
pip install markdown markdown-katex PyPDF2 weasyprint pygments

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
echo ""
echo "Para ejecutar tests:"
echo "  pytest tests/ -v"
echo "════════════════════════════════════════════════════════"
