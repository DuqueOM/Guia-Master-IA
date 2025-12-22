# Módulo 08 - Proyecto Final: NLP Disaster Tweets Pipeline

> **🎯 Objetivo:** Pipeline end-to-end de NLP que demuestra competencia en las 3 áreas del Pathway
> **Fase:** 3 - Proyecto Integrador | **Semanas 21-24** (4 semanas)
> **Dataset:** **Kaggle NLP with Disaster Tweets** (7,613 tweets etiquetados, clasificación binaria)
> **Nivel:** Avanzado (requiere dominio de M05, M06, M07)

---

## 🧠 ¿Qué Estamos Construyendo?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PROYECTO: NLP DISASTER TWEET CLASSIFICATION PIPELINE                      │
│   ─────────────────────────────────────────────────────                     │
│                                                                             │
│   LÍNEA 1: SUPERVISED LEARNING (CSCA 5622) - Semanas 21-22                  │
│   ├── Semana 21: EDA + Preprocessing + Vectorización (TF-IDF)               │
│   └── Semana 22: Baselines (Logistic Regression, Naive Bayes, SVM)          │
│                                                                             │
│   LÍNEA 2: UNSUPERVISED/REPRESENTATIONS (CSCA 5632) - Implícito             │
│   └── Word Embeddings (GloVe), Representaciones Latentes                    │
│                                                                             │
│   LÍNEA 3: DEEP LEARNING (CSCA 5642) - Semanas 23-24                        │
│   ├── Semana 23: Bidirectional LSTM + GloVe Embeddings                      │
│   └── Semana 24: Transfer Learning (BERT) + Reporte Final                   │
│                                                                             │
│   RESULTADO:                                                                │
│   Un pipeline que clasifica tweets como desastres reales o metafóricos      │
│   usando técnicas desde TF-IDF hasta Transformers.                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> 💡 **Nota:** Este proyecto es MÁS COMPLEJO que MNIST porque el texto tiene ambigüedad semántica, ruido, y requiere preprocesamiento especializado.

## 🎯 Benchmark principal: F1-Score (no Accuracy)

La guía enfatiza **F1-Score** sobre Accuracy porque:

- **Desbalance de clases:** ~57% No-Disaster, ~43% Disaster
- **Costo asimétrico:** Un falso negativo (no detectar desastre real) es más grave que un falso positivo
- **Estándar de la industria:** En detección de eventos, F1 es la métrica de referencia

Checklist de diagnóstico (mínimo):

- **Datos**: distribución de clases, longitud de tweets, palabras frecuentes por clase
- **Preprocesamiento**: URLs eliminadas, menciones procesadas, tokenización consistente
- **Vectorización**: coverage de vocabulario, dimensionalidad de TF-IDF
- **Evaluación**: F1-Score, Precision, Recall, Matriz de Confusión

---

## 📚 Estructura del Proyecto

### Cronograma (4 Semanas)

| Semana | Fase | Materia Demostrada | Entregable |
|--------|------|-------------------|------------|
| 21 | EDA + Preprocessing | Supervised Learning (prep) | Pipeline de limpieza + EDA notebook |
| 22 | Baselines ML | Supervised Learning | LogReg + NB + métricas |
| 23 | Deep Learning | Deep Learning | Bi-LSTM + GloVe funcionando |
| 24 | Transfer Learning + Reporte | Integración | BERT + REPORT.md + comparación |

### Estructura de Archivos

```
nlp-disaster-tweets/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Pipeline de limpieza de texto (Módulo 01+05)
│   ├── features.py            # TF-IDF, embeddings (Módulo 06)
│   ├── models.py              # Definiciones de modelos (Módulo 05+07)
│   ├── evaluation.py          # Métricas y visualización (Módulo 05)
│   └── utils.py               # Funciones auxiliares
│
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Baseline_Models.ipynb
│   ├── 03_Deep_Learning_LSTM.ipynb
│   └── 04_Transfer_Learning_BERT.ipynb
│
├── models/
│   ├── tfidf_logreg.pkl
│   ├── lstm_best.h5
│   └── bert_finetuned/
│
├── reports/
│   ├── REPORT.md              # Reporte académico final
│   └── MODEL_COMPARISON.md    # Benchmark de modelos
│
├── datasets/
│   └── README_DATASETS.md     # Instrucciones de descarga
│
├── README.md
└── requirements.txt
```

---

## 💻 Parte 1: Carga de Datos y EDA (Semana 21)

### 1.1 Data Loader para Disaster Tweets

```python
"""SEMANA 21: Carga y Exploración del Dataset de Disaster Tweets

El dataset contiene:
- 7,613 tweets etiquetados para entrenamiento
- 3,263 tweets sin etiquetar para test (Kaggle submission)
- Cada tweet: texto + keyword (opcional) + location (opcional)
- 2 clases: 0 = No disaster, 1 = Real disaster

Columnas:
- id: identificador único del tweet
- keyword: palabra clave relacionada con desastres (puede ser NaN)
- location: ubicación del usuario (puede ser NaN, muy ruidosa)
- text: contenido del tweet (máximo 280 caracteres)
- target: 1 = desastre real, 0 = no desastre (solo en train)
"""  # Cierra docstring del módulo; si faltara, imports quedarían dentro del string

import pandas as pd  # Importa pandas para manipulación de DataFrames y lectura de CSV
import numpy as np  # Importa NumPy para operaciones numéricas y estadísticas
import matplotlib.pyplot as plt  # Importa matplotlib para visualización de datos
import seaborn as sns  # Importa seaborn para gráficos estadísticos más elegantes
from collections import Counter  # Importa Counter para conteo eficiente de frecuencias
from typing import Tuple, List, Dict  # Importa tipos para anotaciones (no afecta runtime)
import re  # Importa re para expresiones regulares en limpieza de texto


def load_disaster_tweets(train_path: str, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el dataset de Disaster Tweets desde archivos CSV.

    Args:
        train_path: ruta al archivo train.csv
        test_path: ruta al archivo test.csv (opcional, para submission)

    Returns:
        df_train: DataFrame con datos de entrenamiento (incluye 'target')
        df_test: DataFrame con datos de test (sin 'target') o None
    """  # Cierra docstring; el código siguiente ejecuta la carga real
    df_train = pd.read_csv(train_path)  # Lee CSV de entrenamiento; pandas infiere dtypes automáticamente
    df_test = pd.read_csv(test_path) if test_path else None  # Lee test CSV si se proporciona ruta
    return df_train, df_test  # Retorna tupla (train, test); test puede ser None


def explore_dataset(df: pd.DataFrame) -> Dict:
    """Realiza EDA básico del dataset y retorna estadísticas.

    Esta función es crítica para entender:
    1. Balance de clases (¿necesitamos class_weight?)
    2. Valores faltantes (¿keyword/location útiles?)
    3. Distribución de longitudes (¿max_length para padding?)
    """  # Cierra docstring; código de exploración sigue
    stats = {}  # Diccionario para almacenar estadísticas; se irá poblando

    # Estadísticas básicas
    stats['n_samples'] = len(df)  # Número total de muestras (filas)
    stats['n_features'] = len(df.columns)  # Número de columnas/features

    # Balance de clases (solo si existe 'target')
    if 'target' in df.columns:  # Verifica que sea conjunto de entrenamiento
        class_counts = df['target'].value_counts()  # Cuenta ocurrencias por clase
        stats['class_distribution'] = class_counts.to_dict()  # Convierte a dict {0: n0, 1: n1}
        stats['class_balance'] = class_counts[1] / class_counts[0]  # Ratio clase_1 / clase_0
        print(f"\n📊 Distribución de Clases:")  # Header informativo
        print(f"   No Disaster (0): {class_counts[0]:,} ({class_counts[0]/len(df):.1%})")  # Cuenta clase 0
        print(f"   Disaster (1):    {class_counts[1]:,} ({class_counts[1]/len(df):.1%})")  # Cuenta clase 1

    # Valores faltantes
    missing = df.isnull().sum()  # Cuenta NaN por columna
    stats['missing_values'] = missing.to_dict()  # Almacena como dict
    print(f"\n📊 Valores Faltantes:")  # Header
    for col, count in missing.items():  # Itera columnas con valores faltantes
        if count > 0:  # Solo muestra columnas con NaN
            print(f"   {col}: {count:,} ({count/len(df):.1%})")  # Imprime columna y porcentaje

    # Longitud de tweets
    df['text_length'] = df['text'].apply(len)  # Calcula longitud en caracteres por tweet
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))  # Cuenta palabras por tweet
    stats['avg_text_length'] = df['text_length'].mean()  # Longitud promedio en caracteres
    stats['avg_word_count'] = df['word_count'].mean()  # Conteo promedio de palabras
    stats['max_text_length'] = df['text_length'].max()  # Longitud máxima (para padding)

    print(f"\n📊 Estadísticas de Texto:")  # Header
    print(f"   Longitud promedio: {stats['avg_text_length']:.1f} caracteres")  # Promedio chars
    print(f"   Palabras promedio: {stats['avg_word_count']:.1f} palabras")  # Promedio words
    print(f"   Longitud máxima: {stats['max_text_length']} caracteres")  # Máximo chars

    return stats  # Retorna diccionario con todas las estadísticas


def visualize_class_distribution(df: pd.DataFrame, save_path: str = None):
    """Visualiza distribución de clases con gráfico de barras.

    Importante para detectar desbalance antes de entrenar.
    """  # Cierra docstring
    fig, ax = plt.subplots(figsize=(8, 5))  # Crea figura de 8x5 pulgadas

    class_counts = df['target'].value_counts()  # Cuenta por clase
    colors = ['#3498db', '#e74c3c']  # Azul para 0, rojo para 1 (convención: rojo=alerta)

    bars = ax.bar(['No Disaster (0)', 'Disaster (1)'], class_counts.values, color=colors)  # Barras

    # Añadir valores encima de las barras
    for bar, count in zip(bars, class_counts.values):  # Itera barras y conteos
        height = bar.get_height()  # Altura de la barra
        ax.annotate(f'{count:,}\n({count/len(df):.1%})',  # Texto con conteo y porcentaje
                   xy=(bar.get_x() + bar.get_width()/2, height),  # Posición centrada
                   ha='center', va='bottom', fontsize=12)  # Alineación y tamaño

    ax.set_ylabel('Número de Tweets', fontsize=12)  # Etiqueta eje Y
    ax.set_title('Distribución de Clases - Disaster Tweets', fontsize=14)  # Título
    ax.set_ylim(0, max(class_counts.values) * 1.15)  # Espacio para anotaciones

    plt.tight_layout()  # Ajusta espaciado
    if save_path:  # Guarda si se proporciona ruta
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Alta resolución
    plt.show()  # Muestra figura


def visualize_text_lengths(df: pd.DataFrame, save_path: str = None):
    """Visualiza distribución de longitudes de texto por clase.

    Útil para decidir max_length en padding y detectar outliers.
    """  # Cierra docstring
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Dos subplots lado a lado

    # Longitud en caracteres
    for label, color in [(0, '#3498db'), (1, '#e74c3c')]:  # Itera clases con colores
        subset = df[df['target'] == label]['text_length']  # Filtra por clase
        axes[0].hist(subset, bins=50, alpha=0.6, label=f'Class {label}', color=color)  # Histograma
    axes[0].set_xlabel('Longitud (caracteres)', fontsize=11)  # Etiqueta X
    axes[0].set_ylabel('Frecuencia', fontsize=11)  # Etiqueta Y
    axes[0].set_title('Distribución de Longitud por Clase', fontsize=12)  # Título
    axes[0].legend()  # Leyenda

    # Conteo de palabras
    for label, color in [(0, '#3498db'), (1, '#e74c3c')]:  # Itera clases
        subset = df[df['target'] == label]['word_count']  # Filtra por clase
        axes[1].hist(subset, bins=30, alpha=0.6, label=f'Class {label}', color=color)  # Histograma
    axes[1].set_xlabel('Número de Palabras', fontsize=11)  # Etiqueta X
    axes[1].set_ylabel('Frecuencia', fontsize=11)  # Etiqueta Y
    axes[1].set_title('Distribución de Palabras por Clase', fontsize=12)  # Título
    axes[1].legend()  # Leyenda

    plt.tight_layout()  # Ajusta espaciado
    if save_path:  # Guarda si se proporciona ruta
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Alta resolución
    plt.show()  # Muestra figura


def get_top_keywords(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Obtiene las keywords más frecuentes por clase.

    Las keywords son proporcionadas por Kaggle y pueden ser útiles como features.
    """  # Cierra docstring
    results = []  # Lista para almacenar resultados

    for label in [0, 1]:  # Itera por cada clase
        subset = df[df['target'] == label]['keyword'].dropna()  # Filtra y elimina NaN
        top_kw = Counter(subset).most_common(n)  # Top n keywords más frecuentes
        for kw, count in top_kw:  # Itera keywords y conteos
            results.append({'class': label, 'keyword': kw, 'count': count})  # Añade a resultados

    return pd.DataFrame(results)  # Retorna como DataFrame para análisis


# === DEMO: Cómo usar el Data Loader ===
if __name__ == "__main__":
    # Cargar datos (ajustar rutas según ubicación)
    df_train, _ = load_disaster_tweets('data/train.csv')

    # Explorar
    stats = explore_dataset(df_train)

    # Visualizar
    visualize_class_distribution(df_train)
    visualize_text_lengths(df_train)

    # Keywords
    top_kw = get_top_keywords(df_train)
    print("\n📊 Top Keywords por Clase:")
    print(top_kw.head(20))
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 1.1: Data Loader (Disaster Tweets)</strong></summary>

#### 1) Metadatos
- **Título:** Data loader robusto: EDA sistemático para texto y detección de desbalance
- **ID (opcional):** `M08-NLP-01_1`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio-Avanzado
- **Dependencias:** Pandas básico, matplotlib, comprensión de clasificación binaria

#### 2) Objetivos
- Cargar correctamente el dataset y entender la estructura de columnas.
- Identificar desbalance de clases y decidir estrategias (class_weight, oversampling).
- Analizar distribución de longitudes para decidir max_length en tokenización.
- Detectar valores faltantes y decidir si keyword/location son útiles.

#### 3) Relevancia
- Si no entiendes tus datos, entrenarás modelos con sesgos ocultos.
- El desbalance (57/43) requiere usar F1-Score en lugar de Accuracy.
- La longitud de tweets afecta directamente el padding y la arquitectura LSTM/BERT.

#### 4) Mapa conceptual mínimo
- **CSV** → DataFrame → **EDA** (clases, longitudes, missing) → Decisiones de diseño.
- **Desbalance** → `class_weight='balanced'` o F1 como métrica.
- **Longitud** → `max_length` para padding.

#### 5) Definiciones esenciales
- **Target:** Variable objetivo binaria (0/1).
- **Keyword:** Palabra clave pre-asignada por Kaggle (puede ser NaN).
- **Desbalance:** Cuando una clase tiene significativamente más muestras.

#### 6) Explicación didáctica
- El ratio 57/43 es desbalance "leve" pero suficiente para que Accuracy engañe.
- Tweets cortos (< 10 palabras) pueden ser más ambiguos; tweets largos dan más contexto.

#### 7) Ejemplo modelado
- Si `class_balance ≈ 0.75`, significa que hay ~3 tweets de clase 0 por cada 4 de clase 1.

#### 8) Práctica guiada
- Imprime 5 ejemplos de cada clase y analiza manualmente si son fáciles/difíciles de clasificar.

#### 9) Práctica independiente
- Crea un histograma de longitudes separado por clase y analiza si hay diferencias sistemáticas.

#### 10) Autoevaluación
- ¿Por qué no usamos `location` como feature directamente?
- ¿Qué pasa si ignoramos el desbalance y usamos Accuracy?

#### 11) Errores comunes
- Asumir que el dataset está balanceado sin verificar.
- No revisar valores faltantes en keyword/location.
- Usar max_length demasiado pequeño y truncar información útil.

#### 12) Retención
- Checklist EDA: `shape`, `dtypes`, `class_counts`, `missing`, `length_distribution`.

#### 13) Diferenciación
- Avanzado: Analizar si hay correlación entre `keyword` y `target` (podría ser leakage).

#### 14) Recursos
- Pandas documentation, Kaggle competition page, papers sobre text classification.

#### 15) Nota docente
- Pide que el alumno identifique 3 tweets "difíciles" y explique por qué son ambiguos.
</details>

---

### 1.2 Pipeline de Preprocesamiento de Texto

```python
"""SEMANA 21: Preprocesamiento de Texto para NLP

El preprocesamiento es CRÍTICO en NLP porque:
1. Tweets tienen ruido único: URLs, menciones, hashtags, emojis
2. La misma información puede expresarse de muchas formas
3. Errores aquí se propagan a todo el pipeline

Pipeline estándar:
1. Normalización (lowercase)
2. Limpieza (URLs, menciones, caracteres especiales)
3. Tokenización (dividir en palabras/tokens)
4. Normalización léxica (stemming/lemmatization)
5. Filtrado (stopwords, tokens cortos)
"""  # Cierra docstring del módulo

import re  # Importa re para expresiones regulares; core de limpieza de texto
import string  # Importa string para constantes como punctuation
from typing import List, Optional  # Tipos para anotaciones
import numpy as np  # NumPy para operaciones vectorizadas


def download_nltk_resources():
    """Descarga recursos NLTK necesarios (solo primera vez).

    NLTK requiere datos externos que no vienen con el paquete.
    """  # Cierra docstring
    import nltk  # Import local para evitar dependencia si no se usa
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']  # Recursos necesarios
    for resource in resources:  # Itera recursos
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt'  # Ruta varía por tipo
                          else f'corpora/{resource}' if resource in ['stopwords', 'wordnet']
                          else f'taggers/{resource}')  # Intenta encontrar recurso
        except LookupError:  # Si no existe
            print(f"Descargando {resource}...")  # Informa al usuario
            nltk.download(resource, quiet=True)  # Descarga silenciosamente


class TextPreprocessor:
    """Pipeline de preprocesamiento para tweets.

    Diseñado específicamente para texto de redes sociales:
    - Maneja URLs, menciones, hashtags
    - Normaliza elongaciones ("loooove" → "love")
    - Opcionalmente remueve stopwords y lematiza

    Attributes:
        remove_stopwords: Si True, elimina stopwords comunes
        lemmatize: Si True, aplica lematización
        min_word_length: Longitud mínima de palabras a conservar
        stop_words: Set de stopwords en inglés
        lemmatizer: Instancia de WordNetLemmatizer
    """  # Cierra docstring de clase

    # Patrones regex compilados (más eficiente que compilar cada vez)
    URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+', re.MULTILINE)  # Patrón para URLs
    MENTION_PATTERN = re.compile(r'@\w+')  # Patrón para menciones (@usuario)
    HASHTAG_PATTERN = re.compile(r'#(\w+)')  # Patrón para hashtags; captura palabra sin #
    HTML_PATTERN = re.compile(r'<[^>]+>')  # Patrón para tags HTML
    SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z\s]')  # Todo excepto letras y espacios
    WHITESPACE_PATTERN = re.compile(r'\s+')  # Múltiples espacios
    ELONGATION_PATTERN = re.compile(r'(.)\1{2,}')  # Caracteres repetidos 3+ veces

    def __init__(  # Constructor: configura el pipeline
        self,
        remove_stopwords: bool = True,  # Si eliminar stopwords
        lemmatize: bool = True,  # Si aplicar lematización
        min_word_length: int = 2,  # Longitud mínima de tokens
        lowercase: bool = True  # Si convertir a minúsculas
    ):
        """Inicializa el preprocesador con opciones configurables."""  # Docstring breve
        self.remove_stopwords = remove_stopwords  # Guarda flag de stopwords
        self.lemmatize = lemmatize  # Guarda flag de lematización
        self.min_word_length = min_word_length  # Guarda longitud mínima
        self.lowercase = lowercase  # Guarda flag de lowercase

        # Inicializar recursos NLTK
        download_nltk_resources()  # Asegura que recursos estén disponibles

        from nltk.corpus import stopwords  # Import local para evitar error si NLTK no está
        from nltk.stem import WordNetLemmatizer  # Lematizador basado en WordNet
        from nltk.tokenize import TweetTokenizer  # Tokenizador especializado para tweets

        self.stop_words = set(stopwords.words('english'))  # Set de stopwords (búsqueda O(1))
        self.lemmatizer = WordNetLemmatizer()  # Instancia de lematizador
        self.tokenizer = TweetTokenizer(  # Tokenizador para tweets
            preserve_case=False,  # Convierte a minúsculas
            reduce_len=True,  # Reduce elongaciones ("loooove" → "loove")
            strip_handles=True  # Elimina menciones @usuario
        )

    def clean_text(self, text: str) -> str:
        """Limpieza básica de texto: URLs, menciones, caracteres especiales.

        Args:
            text: Texto crudo del tweet

        Returns:
            Texto limpio sin URLs, menciones, ni caracteres especiales
        """  # Cierra docstring
        if not isinstance(text, str):  # Maneja NaN o tipos no-string
            return ""  # Retorna string vacío para valores inválidos

        # 1. Lowercase (opcional)
        if self.lowercase:  # Si flag está activo
            text = text.lower()  # Convierte todo a minúsculas

        # 2. Eliminar URLs
        text = self.URL_PATTERN.sub('', text)  # Reemplaza URLs con string vacío

        # 3. Eliminar menciones (@usuario)
        text = self.MENTION_PATTERN.sub('', text)  # Reemplaza menciones con vacío

        # 4. Procesar hashtags (conservar palabra, eliminar #)
        text = self.HASHTAG_PATTERN.sub(r'\1', text)  # Captura grupo 1 (palabra sin #)

        # 5. Eliminar HTML tags
        text = self.HTML_PATTERN.sub('', text)  # Elimina <tags>

        # 6. Reducir elongaciones ("loooove" → "loo")
        text = self.ELONGATION_PATTERN.sub(r'\1\1', text)  # Máximo 2 repeticiones

        # 7. Eliminar caracteres especiales y números
        text = self.SPECIAL_CHAR_PATTERN.sub(' ', text)  # Reemplaza con espacio

        # 8. Normalizar espacios
        text = self.WHITESPACE_PATTERN.sub(' ', text).strip()  # Un solo espacio, sin trailing

        return text  # Retorna texto limpio

    def tokenize(self, text: str) -> List[str]:
        """Tokeniza texto en lista de palabras.

        Usa TweetTokenizer de NLTK optimizado para redes sociales.
        """  # Cierra docstring
        return self.tokenizer.tokenize(text)  # Tokeniza usando NLTK TweetTokenizer

    def remove_stops(self, tokens: List[str]) -> List[str]:
        """Elimina stopwords de la lista de tokens.

        Stopwords son palabras muy frecuentes que aportan poco significado:
        "the", "is", "at", "which", "on", etc.
        """  # Cierra docstring
        return [t for t in tokens if t not in self.stop_words]  # Filtra stopwords

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Aplica lematización: reduce palabras a su forma base.

        Ejemplos:
        - "running" → "run"
        - "better" → "good" (con POS tag correcto)
        - "studies" → "study"
        """  # Cierra docstring
        return [self.lemmatizer.lemmatize(t) for t in tokens]  # Lematiza cada token

    def filter_by_length(self, tokens: List[str]) -> List[str]:
        """Filtra tokens por longitud mínima.

        Tokens muy cortos (1 carácter) suelen ser ruido.
        """  # Cierra docstring
        return [t for t in tokens if len(t) >= self.min_word_length]  # Filtra por longitud

    def preprocess(self, text: str, return_tokens: bool = False):
        """Pipeline completo de preprocesamiento.

        Ejecuta todos los pasos en orden:
        1. clean_text() - Limpieza básica
        2. tokenize() - Dividir en tokens
        3. remove_stops() - Eliminar stopwords (opcional)
        4. lemmatize_tokens() - Lematizar (opcional)
        5. filter_by_length() - Filtrar tokens cortos

        Args:
            text: Texto crudo a preprocesar
            return_tokens: Si True, retorna lista de tokens; si False, string

        Returns:
            Texto preprocesado como string o lista de tokens
        """  # Cierra docstring
        # Paso 1: Limpieza
        text = self.clean_text(text)  # Aplica limpieza básica

        # Paso 2: Tokenización
        tokens = self.tokenize(text)  # Divide en tokens

        # Paso 3: Eliminar stopwords (opcional)
        if self.remove_stopwords:  # Si flag activo
            tokens = self.remove_stops(tokens)  # Filtra stopwords

        # Paso 4: Lematización (opcional)
        if self.lemmatize:  # Si flag activo
            tokens = self.lemmatize_tokens(tokens)  # Lematiza tokens

        # Paso 5: Filtrar por longitud
        tokens = self.filter_by_length(tokens)  # Elimina tokens cortos

        # Retornar en formato solicitado
        if return_tokens:  # Si se piden tokens
            return tokens  # Lista de strings
        return ' '.join(tokens)  # String con tokens separados por espacio

    def preprocess_batch(self, texts: List[str], return_tokens: bool = False) -> List:
        """Preprocesa múltiples textos.

        Útil para procesar todo el DataFrame de una vez.
        """  # Cierra docstring
        return [self.preprocess(text, return_tokens) for text in texts]  # Aplica a cada texto


# === DEMO: Cómo usar el Preprocesador ===
if __name__ == "__main__":
    # Ejemplos de tweets
    tweets = [
        "BREAKING: Massive earthquake hits California! Stay safe! http://t.co/xyz @CNN #earthquake",
        "My mixtape is so fire it's causing earthquakes 🔥🔥🔥 @DJ_Fire",
        "Prayers for the victims of the flooding in Houston. #HoustonStrong",
        "I'm DYINGGGG of laughter at this video 😂😂😂 #dead #funny"
    ]

    # Crear preprocesador
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)

    # Procesar tweets
    print("=" * 60)
    print("DEMO: TextPreprocessor")
    print("=" * 60)

    for tweet in tweets:
        clean = preprocessor.preprocess(tweet)
        print(f"\nOriginal: {tweet}")
        print(f"Limpio:   {clean}")
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 1.2: Pipeline de Preprocesamiento</strong></summary>

#### 1) Metadatos
- **Título:** Preprocesamiento de texto: de tweet crudo a tokens limpios
- **ID (opcional):** `M08-NLP-01_2`
- **Duración estimada:** 120–180 min
- **Nivel:** Intermedio
- **Dependencias:** Regex básico, NLTK, comprensión de tokenización

#### 2) Objetivos
- Implementar un pipeline de limpieza robusto para tweets.
- Entender cada paso del preprocesamiento y su justificación.
- Saber cuándo usar/no usar stopwords y lematización.
- Manejar casos edge: NaN, texto vacío, caracteres especiales.

#### 3) Relevancia
- El 80% del éxito en NLP depende del preprocesamiento.
- Errores aquí (ej: eliminar negaciones "not") destruyen el modelo.
- Tweets tienen ruido único que pipelines genéricos no manejan.

#### 4) Mapa conceptual mínimo
- **Texto crudo** → limpieza → tokenización → normalización → **tokens limpios**.
- **Regex** es la herramienta core para patrones de texto.
- **Lematización** reduce vocabulario sin perder semántica.

#### 5) Definiciones esenciales
- **Token:** Unidad atómica de texto (palabra, símbolo).
- **Stopword:** Palabra muy frecuente con poco contenido semántico.
- **Lematización:** Reducción a forma canónica ("running" → "run").
- **Stemming:** Reducción heurística que puede generar no-palabras.

#### 6) Explicación didáctica
- Cada regex tiene un propósito específico; documéntalos.
- El orden de operaciones importa: lowercase antes de regex case-sensitive.
- Elongaciones ("loooove") son comunes en redes sociales y deben normalizarse.

#### 7) Ejemplo modelado
- Tweet: "BREAKING: Fire in LA! http://t.co/x @LAFD #LAFire"
- Después de clean_text: "breaking fire la lafire"
- Después de lematización: "break fire la lafire"

#### 8) Práctica guiada
- Procesa 10 tweets manualmente y verifica que el output tiene sentido.
- Identifica un caso donde lematización cambia el significado incorrectamente.

#### 9) Práctica independiente
- Añade manejo de emojis: ¿convertirlos a texto o eliminarlos?
- Implementa detección de negaciones para no eliminar "not" como stopword.

#### 10) Autoevaluación
- ¿Por qué conservamos la palabra del hashtag pero eliminamos el #?
- ¿Qué problemas causa eliminar stopwords para frases como "this is not good"?

#### 11) Errores comunes
- Eliminar "not", "no", "never" como stopwords (destruye negaciones).
- No manejar NaN/None correctamente (crash del pipeline).
- Regex demasiado agresivos que eliminan información útil.
- Olvidar normalizar espacios múltiples.

#### 12) Retención
- Mantra: "Preprocesa conservadoramente; es más fácil limpiar más que recuperar información."

#### 13) Diferenciación
- Avanzado: Usar SpaCy para NER y conservar entidades nombradas.
- Avanzado: Implementar spell correction para errores tipográficos.

#### 14) Recursos
- NLTK Book capítulo 3, regex101.com para probar patrones, SpaCy docs.

#### 15) Nota docente
- Pide que el alumno encuentre un tweet donde el preprocesamiento falla y proponga una solución.
</details>

---

## � Parte 2: Vectorización y Modelos Baseline (Semana 22)

### 2.1 TF-IDF Vectorization

```python
"""SEMANA 22: Vectorización de Texto con TF-IDF

TF-IDF (Term Frequency - Inverse Document Frequency) convierte texto a vectores numéricos.
Es el estándar para modelos de ML clásicos (LogReg, SVM, Naive Bayes).

Fórmula matemática:
    TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

    donde:
    - TF(t, d) = frecuencia del término t en documento d (o 1 + log(tf) si sublinear)
    - IDF(t, D) = log(N / df(t)) + 1  (sklearn añade 1 para suavizar)
    - N = número total de documentos
    - df(t) = número de documentos que contienen t

Intuición: Palabras frecuentes en un documento pero raras en el corpus son más informativas.
Ejemplo: "earthquake" es raro globalmente pero frecuente en tweets de desastres → alto TF-IDF.
"""  # Cierra docstring del módulo; código ejecutable sigue

import numpy as np  # NumPy para operaciones numéricas y manejo de arrays sparse
import pandas as pd  # Pandas para manejo de DataFrames
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # Vectorizadores de sklearn
from sklearn.model_selection import train_test_split  # Split estratificado train/test
from typing import Tuple, List, Dict  # Tipos para anotaciones de funciones


class TextVectorizer:
    """Wrapper para vectorización de texto con TF-IDF o Bag of Words.

    Esta clase encapsula TfidfVectorizer de sklearn con configuración
    optimizada para clasificación de texto en redes sociales.

    Parámetros clave y su efecto:
    - max_features: Limita vocabulario para evitar overfitting y reducir memoria
    - ngram_range: (1,2) captura unigramas Y bigramas ("new york" como feature)
    - min_df: Ignora palabras que aparecen en < N documentos (ruido)
    - max_df: Ignora palabras que aparecen en > X% de documentos (stopwords implícitas)
    - sublinear_tf: Usa 1+log(tf) para reducir impacto de alta frecuencia

    Attributes:
        vectorizer: Instancia de TfidfVectorizer o CountVectorizer
        method: 'tfidf' o 'bow' según método elegido
    """  # Cierra docstring de clase; atributos de clase siguen

    def __init__(  # Constructor: configura el vectorizador con hiperparámetros
        self,
        method: str = 'tfidf',  # Método: 'tfidf' (recomendado) o 'bow'
        max_features: int = 5000,  # Tamaño máximo del vocabulario (5000 es buen balance)
        ngram_range: Tuple[int, int] = (1, 2),  # Rango de n-gramas: unigramas + bigramas
        min_df: int = 2,  # Frecuencia mínima de documento (elimina palabras muy raras)
        max_df: float = 0.95,  # Frecuencia máxima (proporción; elimina palabras muy comunes)
        sublinear_tf: bool = True  # Usa 1 + log(tf) en lugar de tf crudo
    ):
        """Inicializa el vectorizador con parámetros configurables."""  # Docstring breve
        self.method = method  # Guarda método para referencia posterior
        self.max_features = max_features  # Guarda para reporting
        self.ngram_range = ngram_range  # Guarda para reporting

        if method == 'tfidf':  # Si se elige TF-IDF (recomendado para clasificación)
            self.vectorizer = TfidfVectorizer(  # Crea instancia de TfidfVectorizer
                max_features=max_features,  # Limita vocabulario; evita curse of dimensionality
                ngram_range=ngram_range,  # (1,2) = unigramas + bigramas
                min_df=min_df,  # Ignora palabras muy raras (aparecen en < 2 docs)
                max_df=max_df,  # Ignora palabras muy comunes (aparecen en > 95% docs)
                sublinear_tf=sublinear_tf,  # Escala logarítmica para TF (mejor para texto)
                strip_accents='unicode',  # Normaliza acentos (café → cafe)
                lowercase=True,  # Convierte a minúsculas (Ya lo hace preprocesador, pero por seguridad)
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Solo palabras de 2+ letras (ignora números sueltos)
            )
        else:  # Si se elige Bag of Words (conteos crudos)
            self.vectorizer = CountVectorizer(  # Crea instancia de CountVectorizer
                max_features=max_features,  # Limita vocabulario
                ngram_range=ngram_range,  # Rango de n-gramas
                min_df=min_df,  # Frecuencia mínima
                max_df=max_df,  # Frecuencia máxima
                strip_accents='unicode',  # Normaliza acentos
                lowercase=True,  # Convierte a minúsculas
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Solo palabras de 2+ letras
            )

    def fit(self, texts: List[str]) -> 'TextVectorizer':
        """Ajusta el vectorizador al corpus (aprende vocabulario).

        CRÍTICO: Solo hacer fit en datos de ENTRENAMIENTO.
        Hacer fit en test causa data leakage (el modelo "ve" palabras del futuro).

        Args:
            texts: Lista de textos preprocesados

        Returns:
            self para permitir chaining: vectorizer.fit(X).transform(X)
        """  # Cierra docstring; código de fit sigue
        self.vectorizer.fit(texts)  # Aprende vocabulario del corpus de entrenamiento
        return self  # Retorna self para chaining

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforma textos a vectores usando vocabulario ya aprendido.

        Los textos de test se transforman con el vocabulario de train.
        Palabras nuevas (OOV - Out of Vocabulary) se ignoran.

        Args:
            texts: Lista de textos a transformar

        Returns:
            Matriz sparse de shape (n_samples, n_features)
        """  # Cierra docstring
        return self.vectorizer.transform(texts)  # Aplica transformación con vocab existente

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit y transform en un solo paso (SOLO para datos de train).

        Equivalente a: vectorizer.fit(texts).transform(texts)
        Más eficiente porque no recorre los datos dos veces.
        """  # Cierra docstring
        return self.vectorizer.fit_transform(texts)  # Fit + transform combinados

    def get_feature_names(self) -> np.ndarray:
        """Retorna nombres de las features (vocabulario aprendido).

        Útil para:
        1. Inspeccionar qué palabras/n-gramas están en el vocabulario
        2. Interpretar coeficientes de modelos lineales
        3. Debug: verificar que el vocabulario tiene sentido
        """  # Cierra docstring
        return self.vectorizer.get_feature_names_out()  # Retorna array de palabras/n-gramas

    def get_vocabulary_stats(self) -> Dict:
        """Retorna estadísticas del vocabulario aprendido."""  # Docstring breve
        vocab = self.get_feature_names()  # Obtiene vocabulario
        unigrams = [w for w in vocab if ' ' not in w]  # Palabras sin espacio = unigramas
        bigrams = [w for w in vocab if ' ' in w]  # Palabras con espacio = bigramas
        return {  # Dict con estadísticas
            'total_features': len(vocab),  # Total de features
            'unigrams': len(unigrams),  # Número de unigramas
            'bigrams': len(bigrams),  # Número de bigramas
            'sample_unigrams': unigrams[:10],  # Muestra de unigramas
            'sample_bigrams': bigrams[:10] if bigrams else []  # Muestra de bigramas
        }


def prepare_train_test_split(  # Función de utilidad para preparar datos completos
    df: pd.DataFrame,
    text_column: str = 'text_clean',  # Columna con texto preprocesado
    target_column: str = 'target',  # Columna con labels
    test_size: float = 0.2,  # 20% para test (estándar)
    random_state: int = 42  # Semilla para reproducibilidad
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TextVectorizer]:
    """Prepara datos para entrenamiento: split + vectorización.

    Pipeline completo:
    1. Split estratificado (mantiene proporción de clases)
    2. Fit vectorizador SOLO en train
    3. Transform ambos conjuntos con el mismo vocabulario

    IMPORTANTE: Siempre fit en train, transform en test.

    Args:
        df: DataFrame con columnas de texto y target
        text_column: Nombre de columna con texto preprocesado
        target_column: Nombre de columna con labels (0/1)
        test_size: Proporción para test (0.2 = 20%)
        random_state: Semilla para reproducibilidad

    Returns:
        Tuple de (X_train, X_test, y_train, y_test, vectorizer)
    """  # Cierra docstring; código de preparación sigue
    # 1. Split estratificado (mantiene proporción 57/43 en ambos conjuntos)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df[text_column],  # Textos preprocesados
        df[target_column],  # Labels binarios
        test_size=test_size,  # Proporción de test
        random_state=random_state,  # Reproducibilidad
        stratify=df[target_column]  # CRÍTICO: estratificar por clase para mantener balance
    )

    # 2. Crear y ajustar vectorizador SOLO en train
    vectorizer = TextVectorizer(  # Instancia con configuración optimizada
        method='tfidf',  # TF-IDF es mejor que BoW para clasificación
        max_features=5000,  # 5000 features es buen balance precisión/eficiencia
        ngram_range=(1, 2)  # Unigramas + bigramas
    )

    # 3. Fit en train, transform ambos
    X_train = vectorizer.fit_transform(X_train_text.tolist())  # FIT + transform en train
    X_test = vectorizer.transform(X_test_text.tolist())  # Solo transform en test (usa vocab de train)

    # 4. Reportar estadísticas
    print(f"\n📊 Datos Preparados para ML:")  # Header informativo
    print(f"   X_train: {X_train.shape} (samples × features)")  # Shape de matriz train
    print(f"   X_test:  {X_test.shape}")  # Shape de matriz test
    print(f"   y_train: {len(y_train)} ({y_train.mean():.1%} positivos)")  # Balance en train
    print(f"   y_test:  {len(y_test)} ({y_test.mean():.1%} positivos)")  # Balance en test

    vocab_stats = vectorizer.get_vocabulary_stats()  # Estadísticas de vocabulario
    print(f"\n📊 Vocabulario:")  # Header
    print(f"   Total features: {vocab_stats['total_features']:,}")  # Total
    print(f"   Unigramas: {vocab_stats['unigrams']:,}")  # Unigramas
    print(f"   Bigramas: {vocab_stats['bigrams']:,}")  # Bigramas

    return X_train, X_test, y_train.values, y_test.values, vectorizer  # Retorna todo
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 2.1: TF-IDF Vectorization</strong></summary>

#### 1) Metadatos
- **Título:** TF-IDF: transformando texto a vectores numéricos para ML
- **ID (opcional):** `M08-NLP-02_1`
- **Duración estimada:** 90–120 min
- **Nivel:** Intermedio
- **Dependencias:** Preprocesamiento de texto (1.2), conceptos de matrices sparse

#### 2) Objetivos
- Entender la intuición matemática detrás de TF-IDF.
- Configurar TfidfVectorizer con parámetros apropiados para el problema.
- Evitar data leakage: fit SOLO en train, transform en test.
- Analizar el vocabulario aprendido e interpretar features.

#### 3) Relevancia
- TF-IDF es el baseline estándar en NLP clásico; si falla, revisa preprocesamiento.
- Es completamente interpretable: puedes ver exactamente qué palabras importan.
- Es extremadamente rápido: segundos de entrenamiento vs horas para BERT.
- Funciona sorprendentemente bien para muchos problemas de clasificación.

#### 4) Mapa conceptual mínimo
```
Texto → Tokenización → Conteo (TF) → Ponderación (IDF) → Vector TF-IDF
                                          ↓
                       Palabras raras tienen mayor peso
```

#### 5) Definiciones esenciales
- **TF (Term Frequency):** Frecuencia del término en el documento. Mide relevancia local.
- **IDF (Inverse Document Frequency):** log(N/df). Penaliza términos que aparecen en muchos documentos.
- **N-grama:** Secuencia de n tokens. Bigrama "new york" captura contexto que unigramas pierden.
- **Sparse matrix:** Matriz donde mayoría de elementos son 0. Eficiente en memoria (CSR format).

#### 6) Explicación didáctica
- Si "earthquake" aparece en 5% de documentos pero 10 veces en un tweet específico → alto TF-IDF.
- Si "the" aparece en 99% de documentos → IDF ≈ 0 → contribución mínima.
- Bigramas como "breaking news" o "stay safe" pueden ser más discriminativos que unigramas.

#### 7) Ejemplo modelado
- Tweet: "earthquake in california, buildings collapsed"
- Unigramas con alto TF-IDF: "earthquake", "collapsed", "buildings"
- Bigramas con alto TF-IDF: "buildings collapsed"
- Palabras con bajo TF-IDF: "in" (muy común)

#### 8) Práctica guiada
- Vectoriza el corpus de train y examina `vectorizer.get_feature_names()`.
- Encuentra los 10 bigramas más frecuentes y analiza si son útiles.

#### 9) Práctica independiente
- Experimenta con `ngram_range=(1,3)` (trigramas) y compara el tamaño del vocabulario.
- Prueba diferentes valores de `max_features` (1000, 5000, 10000) y mide impacto en F1.

#### 10) Autoevaluación
- ¿Por qué hacemos `fit` solo en train y no en todo el dataset?
- ¿Qué pasa si `max_df=0.5`? ¿Qué palabras se eliminan?
- ¿Por qué `sublinear_tf=True` es mejor para texto?

#### 11) Errores comunes
- **Data leakage:** Hacer fit en todo el dataset (train + test). El modelo "ve el futuro".
- **Vocabulario gigante:** No limitar `max_features`. Causa overfitting y lentitud.
- **Ignorar sparse:** Convertir a dense con `.toarray()` innecesariamente. Explota la memoria.
- **No verificar vocabulario:** Asumir que tiene sentido sin inspeccionarlo.

#### 12) Retención
- Mantra: "Fit on train, transform on both. Never fit on test."
- Regla: TF-IDF > BoW para clasificación (casi siempre).

#### 13) Diferenciación
- Avanzado: Comparar TF-IDF con BM25 (usado en motores de búsqueda).
- Avanzado: Implementar TF-IDF desde cero para entender la matemática.

#### 14) Recursos
- Sklearn TfidfVectorizer docs: parámetros y ejemplos
- Paper original: "A Statistical Interpretation of Term Specificity" (Sparck Jones, 1972)

#### 15) Nota docente
- Pide que el alumno explique en sus palabras por qué IDF penaliza palabras comunes.
- Ejercicio: Calcular TF-IDF a mano para un documento de 3 palabras.
</details>

---

### 2.2 Modelos Baseline: Logistic Regression

```python
"""SEMANA 22: Logistic Regression para Clasificación de Texto

Logistic Regression es el modelo baseline por excelencia para NLP porque:
1. Es rápido de entrenar (segundos incluso con millones de features)
2. Es interpretable (coeficientes = importancia de palabras)
3. Funciona bien con datos sparse (TF-IDF)
4. Sirve como baseline sólido para comparar con deep learning

Modelo matemático:
    P(y=1|x) = σ(w·x + b) = 1 / (1 + exp(-(w·x + b)))

    donde:
    - x: vector TF-IDF del documento (sparse, ~5000 dims)
    - w: pesos aprendidos (uno por feature/palabra)
    - b: bias (intercepto)
    - σ: función sigmoide que mapea a [0,1]

Interpretación de coeficientes:
    - w_i > 0: palabra i incrementa P(disaster)
    - w_i < 0: palabra i decrementa P(disaster)
    - |w_i| grande: palabra i es muy discriminativa
"""  # Cierra docstring del módulo

import numpy as np  # NumPy para operaciones numéricas
import time  # Para medir tiempos de entrenamiento
from sklearn.linear_model import LogisticRegression  # Modelo lineal para clasificación
from sklearn.metrics import (  # Métricas de evaluación
    classification_report,  # Reporte detallado por clase
    confusion_matrix,  # Matriz de confusión
    f1_score,  # F1-Score (métrica principal)
    precision_score,  # Precision
    recall_score,  # Recall
    roc_auc_score,  # Área bajo curva ROC
    precision_recall_curve,  # Curva precision-recall
    roc_curve  # Curva ROC
)
from typing import Dict, Tuple, List  # Tipos para anotaciones
import matplotlib.pyplot as plt  # Para visualizaciones


def train_logistic_regression(  # Función principal de entrenamiento
    X_train: np.ndarray,  # Matriz TF-IDF de train (sparse)
    y_train: np.ndarray,  # Labels de train (0/1)
    C: float = 1.0,  # Inverso de regularización (menor = más regularización)
    class_weight: str = 'balanced',  # Manejo de desbalance de clases
    max_iter: int = 1000  # Iteraciones máximas del solver
) -> LogisticRegression:
    """Entrena Logistic Regression para clasificación binaria de tweets.

    Parámetros importantes:
    - C: Controla regularización L2. C=1 es default, C<1 más regularización.
    - class_weight='balanced': Ajusta pesos inversamente proporcionales a frecuencia.
      Si clase 0 tiene 57% y clase 1 tiene 43%, los pesos son ~1.75 y ~2.33.
    - solver='lbfgs': Algoritmo de optimización eficiente para L2.

    Args:
        X_train: Matriz de features (n_samples, n_features), típicamente sparse
        y_train: Vector de labels (n_samples,), valores 0 o 1
        C: Parámetro de regularización inverso
        class_weight: 'balanced' ajusta por frecuencia, None ignora desbalance
        max_iter: Máximo de iteraciones (aumentar si no converge)

    Returns:
        Modelo LogisticRegression entrenado
    """  # Cierra docstring; código de entrenamiento sigue
    print("\n🔬 Entrenando Logistic Regression...")  # Status informativo
    print(f"   Parámetros: C={C}, class_weight='{class_weight}'")  # Hiperparámetros

    start_time = time.time()  # Marca tiempo inicial para medir duración

    model = LogisticRegression(  # Crea instancia del modelo
        C=C,  # Regularización: C grande = menos regularización, más riesgo de overfitting
        class_weight=class_weight,  # 'balanced' compensa desbalance automáticamente
        max_iter=max_iter,  # Iteraciones del solver (aumentar si warning de convergencia)
        solver='lbfgs',  # L-BFGS: eficiente para L2, maneja sparse matrices
        random_state=42,  # Reproducibilidad
        n_jobs=-1  # Usar todos los cores disponibles
    )

    model.fit(X_train, y_train)  # Entrena el modelo (optimiza w y b)

    train_time = time.time() - start_time  # Calcula duración
    print(f"   ✅ Entrenamiento completado en {train_time:.2f} segundos")  # Reporta tiempo

    return model  # Retorna modelo entrenado


def evaluate_model(  # Función de evaluación completa
    model,  # Modelo entrenado (LogReg, NB, etc.)
    X_test: np.ndarray,  # Features de test
    y_test: np.ndarray,  # Labels de test
    model_name: str = "Model"  # Nombre para reportes
) -> Dict[str, float]:
    """Evalúa modelo y retorna métricas completas.

    Métricas calculadas:
    - Accuracy: (TP+TN)/(TP+TN+FP+FN) - NO usar como métrica principal con desbalance
    - Precision: TP/(TP+FP) - De los predichos positivos, ¿cuántos son correctos?
    - Recall: TP/(TP+FN) - De los positivos reales, ¿cuántos detectamos?
    - F1-Score: 2×P×R/(P+R) - Media armónica de Precision y Recall
    - ROC-AUC: Área bajo curva ROC - Mide separabilidad de clases

    En clasificación de desastres:
    - Alto Recall es crítico (no queremos perder desastres reales)
    - Precision también importa (muchos falsos positivos causan fatiga)
    - F1 balancea ambas

    Args:
        model: Modelo con métodos predict() y predict_proba()
        X_test: Features de test
        y_test: Labels verdaderos de test
        model_name: Nombre para los reportes

    Returns:
        Dict con todas las métricas
    """  # Cierra docstring; código de evaluación sigue
    # Obtener predicciones
    y_pred = model.predict(X_test)  # Predicciones binarias (0/1)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades de clase 1

    # Calcular métricas
    metrics = {  # Diccionario con todas las métricas
        'accuracy': (y_pred == y_test).mean(),  # Accuracy (solo referencia)
        'precision': precision_score(y_test, y_pred),  # Precision para clase 1
        'recall': recall_score(y_test, y_pred),  # Recall para clase 1
        'f1_score': f1_score(y_test, y_pred),  # F1-Score (métrica principal)
        'roc_auc': roc_auc_score(y_test, y_proba)  # ROC-AUC
    }

    # Imprimir resultados
    print(f"\n📊 Resultados de {model_name}:")  # Header
    print(f"   Accuracy:  {metrics['accuracy']:.4f}  (⚠️ no usar como métrica principal)")  # Accuracy con warning
    print(f"   Precision: {metrics['precision']:.4f}")  # Precision
    print(f"   Recall:    {metrics['recall']:.4f}")  # Recall
    print(f"   F1-Score:  {metrics['f1_score']:.4f}  ⭐ (métrica principal)")  # F1 destacado
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")  # AUC

    # Classification report detallado
    print(f"\n   Classification Report:")  # Header
    print(classification_report(  # Reporte de sklearn
        y_test, y_pred,
        target_names=['No Disaster', 'Disaster'],  # Nombres de clases
        digits=4  # 4 decimales
    ))

    return metrics  # Retorna diccionario de métricas


def get_feature_importance(  # Función para interpretar el modelo
    model: LogisticRegression,
    feature_names: np.ndarray,
    top_n: int = 20
) -> Tuple[List, List]:
    """Obtiene las palabras más importantes para cada clase.

    En LogReg binaria, los coeficientes indican importancia:
    - Coeficiente positivo grande → incrementa P(disaster)
    - Coeficiente negativo grande → incrementa P(no disaster)

    Esta interpretabilidad es una GRAN ventaja sobre deep learning.

    Args:
        model: LogisticRegression entrenado
        feature_names: Array con nombres de features (palabras/n-gramas)
        top_n: Número de features a retornar por clase

    Returns:
        Tuple de (top_disaster_words, top_no_disaster_words)
    """  # Cierra docstring
    coefs = model.coef_[0]  # Coeficientes del modelo (1 fila para binario)
    sorted_idx = np.argsort(coefs)  # Índices ordenados de menor a mayor

    # Top palabras para clase "Disaster" (coeficientes más positivos)
    top_disaster_idx = sorted_idx[-top_n:][::-1]  # Últimos N, invertidos
    top_disaster = [(feature_names[i], coefs[i]) for i in top_disaster_idx]

    # Top palabras para clase "No Disaster" (coeficientes más negativos)
    top_no_disaster_idx = sorted_idx[:top_n]  # Primeros N
    top_no_disaster = [(feature_names[i], coefs[i]) for i in top_no_disaster_idx]

    return top_disaster, top_no_disaster  # Retorna ambas listas


def plot_confusion_matrix(  # Visualización de matriz de confusión
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
):
    """Visualiza matriz de confusión con anotaciones.

    La matriz de confusión muestra:
    - True Negatives (TN): Predicho 0, Real 0 ✓
    - False Positives (FP): Predicho 1, Real 0 ✗
    - False Negatives (FN): Predicho 0, Real 1 ✗ (crítico en desastres)
    - True Positives (TP): Predicho 1, Real 1 ✓
    """  # Cierra docstring
    cm = confusion_matrix(y_true, y_pred)  # Calcula matriz de confusión

    fig, ax = plt.subplots(figsize=(8, 6))  # Crea figura
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Heatmap
    ax.figure.colorbar(im, ax=ax)  # Colorbar

    # Etiquetas
    classes = ['No Disaster', 'Disaster']  # Nombres de clases
    ax.set(  # Configura ejes
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=f'Confusion Matrix - {model_name}',
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    # Anotaciones en cada celda
    thresh = cm.max() / 2  # Umbral para color de texto
    for i in range(len(classes)):  # Itera filas
        for j in range(len(classes)):  # Itera columnas
            ax.text(j, i, format(cm[i, j], 'd'),  # Número en celda
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")  # Color según fondo

    plt.tight_layout()  # Ajusta layout
    plt.show()  # Muestra figura
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 2.2: Logistic Regression</strong></summary>

#### 1) Metadatos
- **Título:** Logistic Regression: el workhorse de clasificación de texto
- **ID (opcional):** `M08-NLP-02_2`
- **Duración estimada:** 90–120 min
- **Nivel:** Intermedio
- **Dependencias:** TF-IDF (2.1), conceptos de clasificación binaria, regularización

#### 2) Objetivos
- Entrenar y evaluar LogReg para clasificación de tweets.
- Entender el rol de regularización (C) y class_weight.
- Interpretar coeficientes para entender qué palabras importan.
- Usar F1-Score como métrica principal (no Accuracy).

#### 3) Relevancia
- LogReg es el baseline obligatorio antes de cualquier modelo complejo.
- Si LogReg+TF-IDF da F1=0.78 y BERT da F1=0.80, ¿vale la pena BERT?
- La interpretabilidad de LogReg es invaluable para debugging y confianza.

#### 4) Mapa conceptual mínimo
```
x (TF-IDF vector) → w·x + b → sigmoid → P(disaster) → threshold → 0/1
                      ↓
            coeficientes interpretables
```

#### 5) Definiciones esenciales
- **Regularización L2:** Penaliza ||w||² para evitar coeficientes extremos (overfitting).
- **C:** Inverso de fuerza de regularización. C pequeño = más regularización.
- **class_weight='balanced':** Peso de clase = n_samples / (n_classes × n_samples_per_class).
- **Threshold:** Por default 0.5. Se puede ajustar para balancear precision/recall.

#### 6) Explicación didáctica
- LogReg aprende un hiperplano en el espacio TF-IDF de ~5000 dimensiones.
- Cada palabra tiene un "voto" (coeficiente) a favor o en contra de "disaster".
- Si un tweet tiene muchas palabras con coeficientes positivos → P(disaster) alta.

#### 7) Ejemplo modelado
- Palabras con coef > 0: "earthquake", "flood", "emergency", "victims"
- Palabras con coef < 0: "love", "music", "lol", "game"
- Tweet "earthquake in LA, emergency services responding" → alto P(disaster)

#### 8) Práctica guiada
- Entrena LogReg y obtén las 20 palabras más predictivas por clase.
- Verifica que las palabras tienen sentido semántico.

#### 9) Práctica independiente
- Experimenta con diferentes valores de C (0.1, 1, 10) y observa el efecto en F1.
- Ajusta el threshold (0.3, 0.5, 0.7) y analiza el trade-off precision/recall.

#### 10) Autoevaluación
- ¿Por qué class_weight='balanced' mejora el recall?
- ¿Qué pasa si C es muy grande (C=1000)?
- ¿Por qué no usamos Accuracy como métrica principal?

#### 11) Errores comunes
- **Reportar solo Accuracy:** Engañoso con desbalance (57/43).
- **Ignorar convergencia:** Warning "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT" significa que no convergió.
- **No verificar coeficientes:** Si las palabras importantes no tienen sentido, hay bug en preprocesamiento.

#### 12) Retención
- Mantra: "F1 for classification, not Accuracy. Always check class_weight."
- Regla: Si LogReg falla, el problema está en los datos o preprocesamiento.

#### 13) Diferenciación
- Avanzado: Implementar búsqueda de threshold óptimo con curva PR.
- Avanzado: Comparar L1 vs L2 regularización (L1 produce sparsity).

#### 14) Recursos
- Sklearn LogisticRegression docs
- Paper: "Regularization Paths for Generalized Linear Models via Coordinate Descent"

#### 15) Nota docente
- Pide que el alumno explique por qué las palabras más predictivas tienen sentido.
- Ejercicio: Encontrar un tweet mal clasificado y explicar por qué el modelo falló.
</details>

---

## 💻 Parte 3: Deep Learning con LSTM (Semana 23)

### 3.1 Word Embeddings y GloVe

```python
"""SEMANA 23: Word Embeddings con GloVe Pre-entrenados

Word Embeddings representan palabras como vectores densos donde palabras
similares tienen vectores similares. A diferencia de TF-IDF (sparse, ~5000 dims),
los embeddings son densos (~100-300 dims) y capturan semántica.

Evolución de representaciones:
    One-Hot:  "cat" = [1,0,0,...,0]  dim=vocabulario (~50,000)
              "dog" = [0,1,0,...,0]
              Similaridad(cat, dog) = 0  ← PROBLEMA

    Embedding: "cat" = [0.2, -0.4, 0.7, ...]  dim=100-300
               "dog" = [0.3, -0.3, 0.6, ...]
               Similaridad(cat, dog) ≈ 0.85  ← CORRECTO

GloVe (Global Vectors for Word Representation):
- Pre-entrenado en Wikipedia + Gigaword (6B tokens)
- Captura relaciones semánticas: king - man + woman ≈ queen
- Descarga: https://nlp.stanford.edu/projects/glove/
"""  # Cierra docstring del módulo

import numpy as np  # NumPy para operaciones vectoriales
from typing import Dict, Tuple  # Tipos para anotaciones


def load_glove_embeddings(  # Función para cargar GloVe
    glove_path: str,  # Ruta al archivo glove.6B.100d.txt
    embedding_dim: int = 100  # Dimensión de los embeddings (50, 100, 200, 300)
) -> Dict[str, np.ndarray]:
    """Carga embeddings GloVe pre-entrenados desde archivo.

    El archivo tiene formato: palabra dim1 dim2 ... dimN (una línea por palabra).
    Ejemplo: "the 0.418 0.24968 -0.41242 ..."

    Args:
        glove_path: Ruta al archivo GloVe descargado
        embedding_dim: Dimensión esperada (debe coincidir con archivo)

    Returns:
        Dict mapping palabra → vector numpy de shape (embedding_dim,)
    """  # Cierra docstring
    print(f"📥 Cargando GloVe embeddings desde {glove_path}...")  # Status
    embeddings_index = {}  # Dict para almacenar palabra → vector

    with open(glove_path, encoding='utf-8') as f:  # Abre archivo con encoding UTF-8
        for line_num, line in enumerate(f):  # Itera líneas con número de línea
            values = line.split()  # Divide línea por espacios
            word = values[0]  # Primera palabra es el token
            try:
                coefs = np.asarray(values[1:], dtype='float32')  # Resto son coeficientes
                if len(coefs) == embedding_dim:  # Verifica dimensión correcta
                    embeddings_index[word] = coefs  # Guarda en dict
            except ValueError:  # Si hay error de conversión (línea malformada)
                continue  # Salta esa línea

    print(f"   ✅ Cargados {len(embeddings_index):,} word vectors")  # Reporta total
    return embeddings_index  # Retorna diccionario


def create_embedding_matrix(  # Función para crear matriz de embeddings
    word_index: Dict[str, int],  # Mapeo palabra → índice del Tokenizer
    embeddings_index: Dict[str, np.ndarray],  # Embeddings GloVe cargados
    max_words: int = 10000,  # Número máximo de palabras en vocabulario
    embedding_dim: int = 100  # Dimensión de embeddings
) -> Tuple[np.ndarray, int, int]:
    """Crea matriz de embeddings para usar en capa Embedding de Keras.

    La matriz tiene shape (max_words, embedding_dim) donde la fila i
    contiene el embedding de la palabra con índice i en word_index.

    Palabras sin embedding en GloVe (OOV) se inicializan con ceros.

    Args:
        word_index: Dict de Tokenizer.word_index (palabra → índice)
        embeddings_index: Dict de embeddings GloVe cargados
        max_words: Tamaño del vocabulario (debe coincidir con Tokenizer)
        embedding_dim: Dimensión de embeddings GloVe

    Returns:
        Tuple de (embedding_matrix, num_found, num_missing)
    """  # Cierra docstring
    print(f"\n🔨 Creando embedding matrix...")  # Status

    # Inicializar matriz con ceros
    embedding_matrix = np.zeros((max_words, embedding_dim), dtype='float32')  # Shape (vocab, dim)

    found = 0  # Contador de palabras encontradas
    missing = 0  # Contador de palabras no encontradas (OOV)

    for word, i in word_index.items():  # Itera palabras del vocabulario
        if i >= max_words:  # Si índice excede vocabulario máximo
            continue  # Salta (palabras menos frecuentes)

        embedding_vector = embeddings_index.get(word)  # Busca embedding en GloVe
        if embedding_vector is not None:  # Si existe
            embedding_matrix[i] = embedding_vector  # Asigna a la fila i
            found += 1  # Incrementa contador
        else:  # Si no existe (OOV)
            missing += 1  # Incrementa contador de missing
            # La fila queda con ceros (se aprenderá durante training si trainable=True)

    coverage = found / (found + missing) * 100  # Porcentaje de cobertura
    print(f"   ✅ Palabras con embedding: {found:,} ({coverage:.1f}%)")  # Reporta encontradas
    print(f"   ⚠️  Palabras sin embedding (OOV): {missing:,}")  # Reporta faltantes

    return embedding_matrix, found, missing  # Retorna matriz y estadísticas
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 3.1: Word Embeddings</strong></summary>

#### 1) Metadatos
- **Título:** Word Embeddings: de representaciones sparse a semántica densa
- **ID:** `M08-NLP-03_1`
- **Duración estimada:** 60–90 min
- **Nivel:** Intermedio-Avanzado
- **Dependencias:** TF-IDF (2.1), conceptos de similitud coseno

#### 2) Objetivos
- Entender por qué embeddings densos capturan semántica mejor que one-hot.
- Cargar y usar embeddings pre-entrenados (GloVe).
- Crear matriz de embeddings para Keras.
- Analizar cobertura del vocabulario (OOV problem).

#### 3) Relevancia
- Embeddings son la base de todo NLP moderno (LSTM, BERT, GPT).
- Transfer learning: usar conocimiento de Wikipedia para tu tarea.
- Reducen dimensionalidad de ~50,000 (one-hot) a ~100-300 (dense).

#### 4) Mapa conceptual mínimo
```
Palabras → One-Hot (sparse) → No semántica
Palabras → Embedding (dense) → Semántica capturada
king - man + woman ≈ queen
```

#### 5) Definiciones esenciales
- **Embedding:** Vector denso que representa una palabra en espacio continuo.
- **GloVe:** "Global Vectors" - embeddings basados en co-ocurrencia global.
- **OOV (Out of Vocabulary):** Palabras no vistas en el corpus de pre-entrenamiento.
- **Cobertura:** Porcentaje de tu vocabulario con embedding disponible.

#### 6) Explicación didáctica
- Palabras que aparecen en contextos similares → vectores similares.
- "cat" y "dog" aparecen cerca de "pet", "animal" → vectores cercanos.
- Analogías: las direcciones en el espacio codifican relaciones (género, tiempo verbal).

#### 7) Ejemplo modelado
- `glove['disaster']` y `glove['emergency']` tienen alta similitud coseno.
- `glove['fire']` está entre `glove['flames']` (literal) y `glove['passion']` (figurativo).

#### 8) Práctica guiada
- Calcula similitud coseno entre pares de palabras relacionadas con desastres.
- Verifica la analogía: earthquake - ground + water ≈ flood.

#### 9) Práctica independiente
- Analiza palabras OOV del dataset. ¿Son jerga de Twitter, typos, o términos técnicos?
- Implementa inicialización aleatoria para OOV en lugar de ceros.

#### 10) Autoevaluación
- ¿Por qué la cobertura de GloVe puede ser <100% para tweets?
- ¿Qué pasa si inicializamos OOV con el promedio de todos los embeddings?

#### 11) Errores comunes
- **Dimensión incorrecta:** Usar glove.100d con embedding_dim=300.
- **No verificar cobertura:** Asumir que todas las palabras tienen embedding.
- **Encoding incorrecto:** No usar UTF-8 al leer el archivo GloVe.

#### 12) Retención
- Regla: "Embeddings pre-entrenados > entrenar desde cero (para datasets pequeños)."

#### 13) Diferenciación
- Avanzado: Comparar GloVe vs Word2Vec vs FastText (maneja OOV con subwords).

#### 14) Recursos
- Paper GloVe: "GloVe: Global Vectors for Word Representation" (Pennington et al.)
- Descarga: https://nlp.stanford.edu/projects/glove/

#### 15) Nota docente
- Pide que el alumno visualice embeddings con t-SNE y agrupe palabras por tema.
</details>

---

### 3.2 Arquitectura Bidirectional LSTM

```python
"""SEMANA 23: Bidirectional LSTM para Clasificación de Texto

LSTM (Long Short-Term Memory) es una arquitectura de red neuronal recurrente
que puede capturar dependencias de largo alcance en secuencias.

¿Por qué Bidirectional?
    Unidirectional: → → → → →
    Solo ve contexto pasado al procesar cada palabra.

    Bidirectional:  → → → → →
                    ← ← ← ← ←
    Ve contexto pasado Y futuro. Crucial para entender negaciones:
    "The fire was NOT a real emergency" - "NOT" afecta a "emergency" que viene después.

Arquitectura para este proyecto:
    Input (secuencia de índices) → Embedding (GloVe) → Bi-LSTM → Dropout → Dense → Output
"""  # Cierra docstring del módulo

import numpy as np  # NumPy para operaciones numéricas
from tensorflow.keras.models import Model  # API Funcional de Keras
from tensorflow.keras.layers import (  # Capas de Keras
    Input,  # Capa de entrada
    Embedding,  # Capa de embeddings
    LSTM,  # Capa LSTM
    Bidirectional,  # Wrapper para bidireccionalidad
    Dense,  # Capa fully connected
    Dropout,  # Regularización por dropout
    GlobalMaxPooling1D,  # Pooling global
    Concatenate  # Para combinar salidas
)
from tensorflow.keras.callbacks import (  # Callbacks para entrenamiento
    EarlyStopping,  # Detener si no mejora
    ModelCheckpoint,  # Guardar mejor modelo
    ReduceLROnPlateau  # Reducir learning rate si se estanca
)
from tensorflow.keras.preprocessing.text import Tokenizer  # Tokenizador de Keras
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Padding de secuencias
from typing import Tuple  # Tipos


def prepare_sequences(  # Prepara datos para LSTM
    texts: list,  # Lista de textos preprocesados
    max_words: int = 10000,  # Tamaño del vocabulario
    max_len: int = 100  # Longitud máxima de secuencia
) -> Tuple[np.ndarray, Tokenizer]:
    """Convierte textos a secuencias de índices con padding.

    Pipeline:
    1. Tokenizer aprende vocabulario (palabra → índice)
    2. texts_to_sequences convierte textos a listas de índices
    3. pad_sequences asegura longitud uniforme (trunca o añade ceros)

    Args:
        texts: Lista de textos preprocesados
        max_words: Número máximo de palabras en vocabulario
        max_len: Longitud máxima de secuencia (truncar/pad)

    Returns:
        Tuple de (X_padded, tokenizer)
    """  # Cierra docstring
    print(f"\n🔨 Preparando secuencias para LSTM...")  # Status

    # 1. Crear y ajustar Tokenizer
    tokenizer = Tokenizer(  # Crea tokenizer
        num_words=max_words,  # Limita vocabulario a las max_words más frecuentes
        oov_token='<OOV>'  # Token para palabras fuera de vocabulario
    )
    tokenizer.fit_on_texts(texts)  # Aprende vocabulario del corpus

    # 2. Convertir textos a secuencias de índices
    sequences = tokenizer.texts_to_sequences(texts)  # Lista de listas de enteros

    # 3. Aplicar padding para longitud uniforme
    X_padded = pad_sequences(  # Padding
        sequences,
        maxlen=max_len,  # Longitud objetivo
        padding='post',  # Añadir ceros al final
        truncating='post'  # Truncar al final si excede max_len
    )

    # Reportar estadísticas
    print(f"   Vocabulario: {min(len(tokenizer.word_index), max_words):,} palabras")
    print(f"   Shape de secuencias: {X_padded.shape}")
    print(f"   Longitud máxima: {max_len}")

    return X_padded, tokenizer  # Retorna datos y tokenizer


def build_bilstm_model(  # Construye modelo Bi-LSTM
    max_words: int = 10000,  # Tamaño del vocabulario
    max_len: int = 100,  # Longitud de secuencia
    embedding_dim: int = 100,  # Dimensión de embeddings
    embedding_matrix: np.ndarray = None,  # Matriz de embeddings pre-entrenados
    lstm_units: int = 64,  # Unidades en LSTM
    dropout_rate: float = 0.3,  # Tasa de dropout
    trainable_embeddings: bool = False  # Si entrenar embeddings
) -> Model:
    """Construye modelo Bidirectional LSTM para clasificación binaria.

    Arquitectura:
        Input (max_len,) → Embedding (max_words, embedding_dim)
        → Bidirectional(LSTM(lstm_units, return_sequences=True))
        → Dropout → Bidirectional(LSTM(lstm_units//2))
        → Dropout → Dense(64, relu) → Dropout → Dense(1, sigmoid)

    Args:
        max_words: Tamaño del vocabulario
        max_len: Longitud de secuencia de entrada
        embedding_dim: Dimensión de embeddings
        embedding_matrix: Pesos pre-entrenados (GloVe) o None para random
        lstm_units: Número de unidades en primera capa LSTM
        dropout_rate: Tasa de dropout para regularización
        trainable_embeddings: Si True, los embeddings se actualizan en training

    Returns:
        Modelo Keras compilado
    """  # Cierra docstring
    print(f"\n🔨 Construyendo modelo Bi-LSTM...")  # Status

    # Capa de entrada
    inputs = Input(shape=(max_len,), name='input')  # Shape: (batch, max_len)

    # Capa de Embedding
    if embedding_matrix is not None:  # Si tenemos embeddings pre-entrenados
        x = Embedding(  # Capa Embedding con pesos inicializados
            input_dim=max_words,  # Tamaño del vocabulario
            output_dim=embedding_dim,  # Dimensión de salida
            weights=[embedding_matrix],  # Inicializar con GloVe
            input_length=max_len,  # Longitud de secuencia
            trainable=trainable_embeddings,  # Congelar o no los embeddings
            name='embedding_glove'
        )(inputs)
        print(f"   Usando embeddings pre-entrenados (trainable={trainable_embeddings})")
    else:  # Si no tenemos pre-entrenados
        x = Embedding(  # Embedding aleatorio (se aprende)
            input_dim=max_words,
            output_dim=embedding_dim,
            input_length=max_len,
            trainable=True,  # Siempre entrenable si es aleatorio
            name='embedding_random'
        )(inputs)
        print(f"   Usando embeddings aleatorios (se aprenderán)")

    # Primera capa Bidirectional LSTM
    x = Bidirectional(  # Wrapper bidireccional
        LSTM(  # LSTM base
            units=lstm_units,  # Número de unidades
            return_sequences=True,  # Retornar secuencia completa para siguiente LSTM
            dropout=0.2,  # Dropout en input
            recurrent_dropout=0.2  # Dropout en conexiones recurrentes
        ),
        name='bilstm_1'
    )(x)
    x = Dropout(dropout_rate, name='dropout_1')(x)  # Dropout adicional

    # Segunda capa Bidirectional LSTM
    x = Bidirectional(  # Wrapper bidireccional
        LSTM(  # LSTM base
            units=lstm_units // 2,  # Menos unidades (pirámide)
            return_sequences=False  # Solo retornar último output
        ),
        name='bilstm_2'
    )(x)
    x = Dropout(dropout_rate, name='dropout_2')(x)  # Dropout

    # Capas Dense para clasificación
    x = Dense(64, activation='relu', name='dense_1')(x)  # Capa oculta
    x = Dropout(dropout_rate, name='dropout_3')(x)  # Dropout

    # Capa de salida (clasificación binaria)
    outputs = Dense(1, activation='sigmoid', name='output')(x)  # Sigmoid para probabilidad

    # Crear modelo
    model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_Classifier')

    # Compilar
    model.compile(
        optimizer='adam',  # Adam optimizer (adaptativo)
        loss='binary_crossentropy',  # Loss para clasificación binaria
        metrics=['accuracy']  # Métrica de monitoreo (F1 se calcula aparte)
    )

    # Resumen
    print(f"\n📊 Arquitectura del modelo:")
    model.summary()

    return model  # Retorna modelo compilado


def get_callbacks(  # Obtiene callbacks para entrenamiento
    model_path: str = 'models/lstm_best.h5',  # Ruta para guardar mejor modelo
    patience_early: int = 5,  # Paciencia para early stopping
    patience_lr: int = 3  # Paciencia para reducir LR
) -> list:
    """Crea lista de callbacks para entrenamiento.

    Callbacks:
    - EarlyStopping: Detiene si val_loss no mejora en N épocas
    - ModelCheckpoint: Guarda el mejor modelo según val_loss
    - ReduceLROnPlateau: Reduce learning rate si se estanca

    Returns:
        Lista de callbacks configurados
    """  # Cierra docstring
    callbacks = [
        EarlyStopping(  # Detener si no mejora
            monitor='val_loss',  # Métrica a monitorear
            patience=patience_early,  # Épocas sin mejora antes de detener
            restore_best_weights=True,  # Restaurar pesos del mejor epoch
            verbose=1  # Imprimir cuando se detiene
        ),
        ModelCheckpoint(  # Guardar mejor modelo
            filepath=model_path,  # Ruta del archivo
            monitor='val_loss',  # Métrica a monitorear
            save_best_only=True,  # Solo guardar si mejora
            verbose=1  # Imprimir cuando guarda
        ),
        ReduceLROnPlateau(  # Reducir learning rate
            monitor='val_loss',  # Métrica a monitorear
            factor=0.5,  # Factor de reducción (LR *= 0.5)
            patience=patience_lr,  # Épocas sin mejora antes de reducir
            min_lr=1e-6,  # LR mínimo
            verbose=1  # Imprimir cuando reduce
        )
    ]
    return callbacks  # Retorna lista
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 3.2: Bidirectional LSTM</strong></summary>

#### 1) Metadatos
- **Título:** Bi-LSTM: capturando contexto bidireccional para NLP
- **ID:** `M08-NLP-03_2`
- **Duración estimada:** 120–180 min
- **Nivel:** Avanzado
- **Dependencias:** Word embeddings (3.1), conceptos de RNN, backpropagation

#### 2) Objetivos
- Construir arquitectura Bi-LSTM con Keras Functional API.
- Entender por qué bidireccional es mejor que unidireccional para NLP.
- Aplicar regularización (Dropout, EarlyStopping) para evitar overfitting.
- Usar embeddings pre-entrenados vs entrenables.

#### 3) Relevancia
- LSTM es el paso intermedio entre ML clásico y Transformers.
- Bi-LSTM fue state-of-the-art antes de BERT (2018).
- Más rápido de entrenar que BERT, útil para datasets pequeños/recursos limitados.

#### 4) Mapa conceptual mínimo
```
Secuencia → Embedding → LSTM forward  → Concatenate → Dense → Sigmoid
                     → LSTM backward →
```

#### 5) Definiciones esenciales
- **LSTM:** Red recurrente con celdas de memoria que evitan vanishing gradients.
- **Bidirectional:** Procesa secuencia en ambas direcciones y concatena outputs.
- **return_sequences:** Si True, retorna output de cada timestep; si False, solo el último.
- **Dropout:** Desactiva neuronas aleatoriamente durante training para regularizar.

#### 6) Explicación didáctica
- LSTM tiene "compuertas" (forget, input, output) que controlan flujo de información.
- Bidirectional permite que "not" (al inicio) afecte la interpretación de "emergency" (al final).
- Dropout "obliga" a la red a no depender de features específicas.

#### 7) Ejemplo modelado
- Tweet: "This is NOT a real emergency, just a drill"
- Forward LSTM: procesa "NOT" antes de "emergency" → puede modular
- Backward LSTM: procesa "drill" primero, luego "emergency" → contexto adicional

#### 8) Práctica guiada
- Entrena el modelo y observa las curvas de loss/accuracy.
- Compara F1 con trainable_embeddings=True vs False.

#### 9) Práctica independiente
- Experimenta con diferentes lstm_units (32, 64, 128).
- Añade una tercera capa LSTM y observa el efecto en overfitting.

#### 10) Autoevaluación
- ¿Por qué usamos return_sequences=True en la primera LSTM pero no en la segunda?
- ¿Qué indica si train_loss baja pero val_loss sube?
- ¿Por qué congelamos embeddings inicialmente?

#### 11) Errores comunes
- **Overfitting:** Modelo memoriza train, no generaliza. Solución: más dropout, early stopping.
- **OOM (Out of Memory):** batch_size muy grande. Solución: reducir a 16 o 32.
- **No converge:** Learning rate muy alto o bajo. Solución: usar ReduceLROnPlateau.

#### 12) Retención
- Regla: "Siempre usar EarlyStopping. El número de épocas es un upper bound, no un target."

#### 13) Diferenciación
- Avanzado: Añadir capa de Attention sobre los outputs de LSTM.
- Avanzado: Implementar LSTM desde cero para entender las compuertas.

#### 14) Recursos
- Paper LSTM: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- Keras LSTM documentation

#### 15) Nota docente
- Pide que el alumno dibuje el flujo de datos a través del modelo.
- Ejercicio: Identificar tweets donde Bi-LSTM mejora sobre LogReg y explicar por qué.
</details>

---

## � Parte 4: Transfer Learning con BERT (Semana 24)

### 4.1 Introducción a BERT y Transformers

```python
"""SEMANA 24: Transfer Learning con BERT para Clasificación de Texto

BERT (Bidirectional Encoder Representations from Transformers) revolucionó NLP en 2018.
A diferencia de LSTM que procesa secuencialmente, BERT usa atención para ver
TODAS las palabras simultáneamente.

¿Por qué BERT es mejor que LSTM?
1. Atención bidireccional real (no concatenación de forward+backward)
2. Pre-entrenado en corpus masivo (Wikipedia + BookCorpus, 3.3B palabras)
3. Transfer learning: conocimiento de lenguaje general se transfiere a tu tarea
4. State-of-the-art en la mayoría de benchmarks de NLP

Arquitectura BERT:
    Input: [CLS] token1 token2 ... tokenN [SEP]
    ↓
    12 capas de Transformer Encoder (BERT-base) o 24 (BERT-large)
    ↓
    Output: Embedding contextualizado para cada token

Para clasificación: Usamos el embedding de [CLS] como representación del documento.
"""  # Cierra docstring del módulo

import numpy as np  # NumPy para operaciones numéricas
import tensorflow as tf  # TensorFlow para deep learning
from transformers import (  # HuggingFace Transformers
    BertTokenizer,  # Tokenizador de BERT
    TFBertForSequenceClassification,  # Modelo BERT para clasificación
    BertConfig  # Configuración del modelo
)
from typing import Dict, Tuple, List  # Tipos para anotaciones


def load_bert_model(  # Carga modelo BERT pre-entrenado
    model_name: str = 'bert-base-uncased',  # Nombre del modelo en HuggingFace
    num_labels: int = 2  # Número de clases (2 para binario)
) -> Tuple:
    """Carga tokenizer y modelo BERT pre-entrenado.

    Modelos disponibles:
    - bert-base-uncased: 110M params, lowercase (recomendado para empezar)
    - bert-base-cased: 110M params, mantiene mayúsculas
    - bert-large-uncased: 340M params, más potente pero más lento
    - distilbert-base-uncased: 66M params, más rápido, ~97% performance

    Args:
        model_name: Nombre del modelo en HuggingFace Hub
        num_labels: Número de clases de salida

    Returns:
        Tuple de (tokenizer, model)
    """  # Cierra docstring
    print(f"\n📥 Cargando modelo BERT: {model_name}...")  # Status

    # Cargar tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)  # Descarga/carga tokenizer
    print(f"   ✅ Tokenizer cargado (vocab size: {tokenizer.vocab_size:,})")

    # Cargar modelo para clasificación
    model = TFBertForSequenceClassification.from_pretrained(
        model_name,  # Modelo base
        num_labels=num_labels  # Configura capa de clasificación
    )
    print(f"   ✅ Modelo cargado ({model.num_parameters():,} parámetros)")

    return tokenizer, model  # Retorna ambos


def encode_texts_for_bert(  # Prepara datos para BERT
    texts: List[str],  # Lista de textos
    tokenizer: BertTokenizer,  # Tokenizer de BERT
    max_length: int = 128  # Longitud máxima (BERT máximo: 512)
) -> Dict:
    """Tokeniza textos para BERT usando el tokenizer de HuggingFace.

    BERT requiere:
    - input_ids: Índices de tokens en el vocabulario
    - attention_mask: 1 para tokens reales, 0 para padding
    - token_type_ids: 0 para primera oración (no usado en clasificación simple)

    Args:
        texts: Lista de textos a tokenizar
        tokenizer: BertTokenizer cargado
        max_length: Longitud máxima de secuencia (truncar si excede)

    Returns:
        Dict con input_ids, attention_mask, token_type_ids como tensores TF
    """  # Cierra docstring
    print(f"\n🔨 Tokenizando {len(texts):,} textos para BERT...")  # Status

    # Tokenizar batch de textos
    encodings = tokenizer(
        texts,  # Lista de textos (puede ser lista de strings)
        padding='max_length',  # Añadir padding hasta max_length
        truncation=True,  # Truncar si excede max_length
        max_length=max_length,  # Longitud objetivo
        return_tensors='tf'  # Retornar tensores de TensorFlow
    )

    print(f"   ✅ Shape de input_ids: {encodings['input_ids'].shape}")
    print(f"   ✅ Max length: {max_length}")

    return encodings  # Retorna dict de tensores


def create_tf_dataset(  # Crea dataset de TensorFlow
    encodings: Dict,  # Encodings de BERT
    labels: np.ndarray,  # Labels (0/1)
    batch_size: int = 16,  # Tamaño de batch (pequeño por memoria)
    shuffle: bool = True  # Si mezclar datos
) -> tf.data.Dataset:
    """Crea tf.data.Dataset para entrenamiento eficiente.

    tf.data.Dataset permite:
    - Prefetching: carga siguiente batch mientras procesa actual
    - Shuffling: mezcla datos para mejor generalización
    - Batching: agrupa ejemplos para procesamiento paralelo

    Args:
        encodings: Dict de tensores de BERT tokenizer
        labels: Array de labels
        batch_size: Tamaño de batch (16 es típico para BERT por memoria)
        shuffle: Si True, mezcla los datos

    Returns:
        tf.data.Dataset listo para training/evaluation
    """  # Cierra docstring
    # Crear dataset desde tensores
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings),  # Convierte encodings a dict para el modelo
        labels  # Labels
    ))

    if shuffle:  # Si se pide mezclar
        dataset = dataset.shuffle(buffer_size=1000)  # Buffer de 1000 ejemplos

    dataset = dataset.batch(batch_size)  # Agrupa en batches
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch automático

    return dataset  # Retorna dataset


def fine_tune_bert(  # Fine-tuning de BERT
    model: TFBertForSequenceClassification,  # Modelo BERT cargado
    train_dataset: tf.data.Dataset,  # Dataset de entrenamiento
    val_dataset: tf.data.Dataset,  # Dataset de validación
    epochs: int = 3,  # Épocas (BERT converge rápido, 2-4 típico)
    learning_rate: float = 2e-5  # LR bajo para fine-tuning (2e-5 a 5e-5)
) -> Dict:
    """Fine-tune BERT para clasificación de tweets.

    Estrategia de fine-tuning:
    1. Learning rate muy bajo (2e-5) para no destruir pesos pre-entrenados
    2. Pocas épocas (2-4) porque BERT ya sabe mucho de lenguaje
    3. Batch size pequeño (8-32) por limitaciones de memoria GPU

    Args:
        model: Modelo BERT pre-cargado
        train_dataset: tf.data.Dataset de entrenamiento
        val_dataset: tf.data.Dataset de validación
        epochs: Número de épocas (2-4 típico)
        learning_rate: Learning rate (2e-5 a 5e-5 típico)

    Returns:
        Dict con history del entrenamiento
    """  # Cierra docstring
    print(f"\n🚀 Fine-tuning BERT...")  # Status
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")

    # Configurar optimizer con LR bajo
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Loss para clasificación
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compilar modelo
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    # Entrenar
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    print(f"   ✅ Fine-tuning completado")
    return history.history  # Retorna history como dict
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 4.1: BERT Transfer Learning</strong></summary>

#### 1) Metadatos
- **Título:** BERT: Transfer Learning para NLP state-of-the-art
- **ID:** `M08-NLP-04_1`
- **Duración estimada:** 150–240 min
- **Nivel:** Avanzado
- **Dependencias:** Bi-LSTM (3.2), conceptos de attention, GPU recomendada

#### 2) Objetivos
- Entender la arquitectura Transformer y el mecanismo de atención.
- Cargar y usar modelos pre-entrenados de HuggingFace.
- Fine-tune BERT para clasificación de tweets.
- Comparar performance con modelos anteriores (LogReg, LSTM).

#### 3) Relevancia
- BERT y sus variantes son el estándar actual en NLP industrial.
- Transfer learning reduce drásticamente datos necesarios.
- Entender BERT es prerequisito para GPT, T5, y modelos más recientes.

#### 4) Mapa conceptual mínimo
```
Input → BERT Encoder (12 capas) → [CLS] embedding → Dense → Clasificación
         ↑
    Atención: cada token "atiende" a todos los demás
```

#### 5) Definiciones esenciales
- **Transformer:** Arquitectura basada en atención, sin recurrencia.
- **Self-Attention:** Cada posición puede atender a todas las demás.
- **[CLS] token:** Token especial cuyo embedding representa todo el documento.
- **Fine-tuning:** Ajustar pesos pre-entrenados para tarea específica.

#### 6) Explicación didáctica
- BERT "leyó" Wikipedia y aprendió estructura del lenguaje.
- Fine-tuning transfiere ese conocimiento a clasificar desastres.
- Learning rate bajo evita "olvidar" lo aprendido (catastrophic forgetting).

#### 7) Ejemplo modelado
- Sin BERT: Necesitas ~100k ejemplos etiquetados para buen modelo.
- Con BERT: ~7k ejemplos (nuestro dataset) son suficientes para F1 > 0.80.

#### 8) Práctica guiada
- Fine-tune BERT con 2 y 4 épocas, compara val_loss.
- Analiza la curva de entrenamiento: ¿hay overfitting?

#### 9) Práctica independiente
- Prueba distilbert-base-uncased (más rápido) y compara F1.
- Experimenta con diferentes learning rates (1e-5, 2e-5, 5e-5).

#### 10) Autoevaluación
- ¿Por qué usamos learning rate tan bajo (2e-5 vs 1e-3 típico)?
- ¿Qué es el [CLS] token y por qué lo usamos para clasificación?
- ¿Por qué BERT necesita menos épocas que LSTM?

#### 11) Errores comunes
- **OOM (Out of Memory):** Reducir batch_size a 8 o 16.
- **LR muy alto:** Destruye pesos pre-entrenados. Usar 2e-5.
- **Muchas épocas:** BERT overfittea rápido. Máximo 4-5 épocas.
- **No usar GPU:** BERT es muy lento en CPU.

#### 12) Retención
- Regla: "Para BERT: LR bajo (2e-5), pocas épocas (2-4), batch pequeño (16)."

#### 13) Diferenciación
- Avanzado: Implementar gradual unfreezing (descongelar capas progresivamente).
- Avanzado: Probar RoBERTa o ALBERT como alternativas.

#### 14) Recursos
- Paper: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al.)
- HuggingFace Course: https://huggingface.co/course
- The Illustrated Transformer: http://jalammar.github.io/illustrated-transformer/

#### 15) Nota docente
- Pide que el alumno explique por qué BERT puede entender "fire" diferente en contextos diferentes.
- Ejercicio: Comparar F1 de LogReg vs LSTM vs BERT y discutir trade-offs.
</details>

---

### 4.2 Comparación Final de Modelos

```python
"""SEMANA 24: Comparación de Todos los Modelos

Este script compara todos los modelos entrenados:
1. Logistic Regression + TF-IDF (baseline)
2. Naive Bayes + TF-IDF (baseline probabilístico)
3. Bidirectional LSTM + GloVe
4. BERT fine-tuned

Métricas de comparación:
- F1-Score (métrica principal)
- Precision y Recall
- Tiempo de entrenamiento
- Requisitos de recursos (CPU vs GPU)
"""  # Cierra docstring

import pandas as pd  # Pandas para tablas de resultados
import numpy as np  # NumPy para cálculos
import matplotlib.pyplot as plt  # Matplotlib para visualización
from typing import Dict, List  # Tipos


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """Crea tabla comparativa de todos los modelos.

    Args:
        results: Dict con estructura {model_name: {metric: value}}

    Returns:
        DataFrame con comparación
    """  # Cierra docstring
    df = pd.DataFrame(results).T  # Transponer: modelos como filas
    df = df.sort_values('f1_score', ascending=False)  # Ordenar por F1
    return df


def plot_model_comparison(results: Dict[str, Dict], save_path: str = None):
    """Visualiza comparación de modelos con gráfico de barras.

    Args:
        results: Dict de resultados por modelo
        save_path: Ruta para guardar figura (opcional)
    """  # Cierra docstring
    models = list(results.keys())  # Nombres de modelos
    metrics = ['precision', 'recall', 'f1_score']  # Métricas a comparar

    x = np.arange(len(models))  # Posiciones en X
    width = 0.25  # Ancho de barras

    fig, ax = plt.subplots(figsize=(12, 6))  # Figura grande

    # Barras para cada métrica
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in models]
        ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())

    # Configuración
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparación de Modelos - NLP Disaster Tweets', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Añadir valores encima de las barras
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in models]
        for j, v in enumerate(values):
            ax.annotate(f'{v:.2f}', xy=(x[j] + i*width, v), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# === RESULTADOS ESPERADOS (ejemplo) ===
EXPECTED_RESULTS = {
    'LogReg + TF-IDF': {
        'precision': 0.78,
        'recall': 0.75,
        'f1_score': 0.76,
        'training_time': '2s',
        'gpu_required': 'No'
    },
    'Naive Bayes': {
        'precision': 0.72,
        'recall': 0.80,
        'f1_score': 0.76,
        'training_time': '1s',
        'gpu_required': 'No'
    },
    'Bi-LSTM + GloVe': {
        'precision': 0.80,
        'recall': 0.78,
        'f1_score': 0.79,
        'training_time': '5-10min',
        'gpu_required': 'Recomendada'
    },
    'BERT Fine-tuned': {
        'precision': 0.84,
        'recall': 0.82,
        'f1_score': 0.83,
        'training_time': '30-60min',
        'gpu_required': 'Necesaria'
    }
}

# Imprimir tabla de resultados esperados
print("\n" + "="*70)
print("RESULTADOS ESPERADOS (benchmark)")
print("="*70)
df_results = compare_models(EXPECTED_RESULTS)
print(df_results.to_string())
print("\n⭐ Modelo recomendado para producción: LogReg + TF-IDF")
print("   Razón: Balance óptimo entre F1 (0.76) y simplicidad/velocidad")
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 4.2: Comparación de Modelos</strong></summary>

#### 1) Metadatos
- **Título:** Análisis comparativo: elegir el modelo correcto para producción
- **ID:** `M08-NLP-04_2`
- **Duración estimada:** 60–90 min
- **Nivel:** Avanzado
- **Dependencias:** Todos los modelos anteriores entrenados y evaluados

#### 2) Objetivos
- Comparar objetivamente todos los modelos con las mismas métricas.
- Analizar trade-offs: performance vs complejidad vs recursos.
- Tomar decisión informada sobre modelo para producción.
- Documentar resultados en formato académico (REPORT.md).

#### 3) Relevancia
- En la industria, el "mejor" modelo no siempre es el más preciso.
- Costos de deployment (GPU, latencia) importan tanto como F1.
- Saber comunicar trade-offs es skill crítico para data scientists.

#### 4) Mapa conceptual mínimo
```
Modelos → Métricas (F1, P, R) → Trade-offs → Decisión → Producción
              ↓
         Recursos (tiempo, GPU)
```

#### 5) Definiciones esenciales
- **Trade-off:** Sacrificar una cosa por otra (ej: accuracy vs velocidad).
- **Producción:** Ambiente donde el modelo se usa con datos reales.
- **Latencia:** Tiempo que toma una predicción.
- **Deployment:** Proceso de poner modelo en producción.

#### 6) Explicación didáctica
- BERT gana en F1 pero necesita GPU y es lento.
- LogReg es casi tan bueno y funciona en cualquier servidor.
- La decisión depende del contexto: ¿velocidad o precisión importa más?

#### 7) Ejemplo modelado
- Sistema de alertas en tiempo real: LogReg (baja latencia).
- Análisis batch diario: BERT (máxima precisión, tiempo no crítico).

#### 8) Práctica guiada
- Llena la tabla de resultados con tus propios modelos.
- Calcula mejora relativa: (BERT_F1 - LogReg_F1) / LogReg_F1.

#### 9) Práctica independiente
- Añade análisis de errores: ¿qué tweets falla cada modelo?
- Implementa ensemble (combinar predicciones de varios modelos).

#### 10) Autoevaluación
- Si BERT tiene F1=0.83 y LogReg F1=0.76, ¿cuál usarías y por qué?
- ¿En qué casos pagarías el costo extra de BERT?

#### 11) Errores comunes
- **Solo mirar F1:** Ignorar costos de recursos.
- **No reproducibilidad:** No fijar random seeds.
- **Comparación injusta:** Usar diferentes splits de datos.

#### 12) Retención
- Regla: "El mejor modelo es el que resuelve el problema de negocio, no el que tiene mayor F1."

#### 13) Diferenciación
- Avanzado: Calcular costo/beneficio monetario de cada punto de F1.

#### 14) Recursos
- Paper: "Model Selection for NLP" (varios autores)
- Blog: "Deploying ML Models in Production" (Google AI)

#### 15) Nota docente
- Pide que el alumno presente recomendación como si fuera para un cliente.
- Ejercicio: Escribir párrafo de conclusiones para REPORT.md.
</details>

---

## �📊 Evaluación del Proyecto

### Criterios de Evaluación (Total: 100 puntos)

| Componente | Puntos | Criterios |
|------------|--------|-----------|
| **EDA + Preprocessing** | 20 | Pipeline robusto, decisiones justificadas |
| **Baselines ML** | 20 | LogReg + NB funcionando, F1-Score reportado |
| **Deep Learning** | 25 | LSTM entrenando, regularización aplicada |
| **Transfer Learning** | 20 | BERT fine-tuned, comparación con baselines |
| **Reporte REPORT.md** | 15 | Estructura académica, análisis de errores |

### Condición de Aprobación
- **F1-Score mínimo:** 0.75 en test (con al menos un modelo)
- **Reporte completo:** Todas las secciones cubiertas

---

## 📚 Recursos y Referencias

### Papers Fundamentales
1. **"Attention Is All You Need"** (Vaswani et al., 2017) - Transformers
2. **"BERT"** (Devlin et al., 2018) - Pre-training bidireccional
3. **"GloVe"** (Pennington et al., 2014) - Word embeddings

### Documentación
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [NLTK Book](https://www.nltk.org/book/)
- [Keras Text Processing](https://keras.io/api/preprocessing/text/)

---

*Material desarrollado para el MS-AI Pathway - University of Colorado Boulder*
*Semanas 21-24 - Proyecto Capstone NLP*
