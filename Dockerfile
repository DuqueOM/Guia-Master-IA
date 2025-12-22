# =============================================================================
# Guía Master IA - Docker Development Environment
# =============================================================================
# Imagen para reproducibilidad del entorno de desarrollo
#
# Build:
#   docker build -t guia-master-ia .
#
# Run:
#   docker run -it --rm -p 8888:8888 -v $(pwd):/workspace guia-master-ia
#
# =============================================================================

FROM python:3.11-slim

# Metadatos
LABEL maintainer="DuqueOM"
LABEL description="Entorno de desarrollo para Guía Master IA - MS in AI CU Boulder"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo
WORKDIR /workspace

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Instalar herramientas de desarrollo adicionales
RUN pip install \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    nbval>=0.11.0 \
    ruff>=0.1.0 \
    mypy>=1.7.0

# Descargar datos de NLTK necesarios
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copiar el resto del proyecto
COPY . .

# Exponer puerto para JupyterLab
EXPOSE 8888

# Comando por defecto: JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
