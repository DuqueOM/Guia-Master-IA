# Módulo 01 - Python Científico + Pandas

> **🎯 Objetivo:** Dominar Pandas para datos + NumPy para matemáticas
> **Fase:** 1 - Fundamentos | **Semanas 1-2**
> **Prerrequisitos:** Python básico (variables, funciones, listas, loops)

---

<a id="m01-0"></a>

## 🧭 Cómo usar este módulo (modo 0→100)

**Propósito:** que pases de “sé Python básico” a **poder trabajar con datos reales y producir arrays listos para modelos** (lo que usarás en TODO el Pathway).

### Objetivos de aprendizaje (medibles)

Al terminar este módulo podrás:

- **Aplicar** Pandas para cargar, explorar y limpiar datasets reales.
- **Convertir** datasets a `np.ndarray` con shapes correctos para ML (`X` y `y`).
- **Explicar** qué es vectorización y por qué NumPy elimina loops.
- **Diagnosticar** los errores de shapes más comunes (`(n,)` vs `(n,1)`, broadcasting silencioso, vistas vs copias).

### Prerrequisitos

- Python básico (loops, funciones, listas, diccionarios).

Enlaces rápidos:

- [GLOSARIO: NumPy](GLOSARIO.md#numpy)
- [GLOSARIO: Broadcasting](GLOSARIO.md#broadcasting)
- [GLOSARIO: Vectorization](GLOSARIO.md#vectorization)
- [RECURSOS.md](RECURSOS.md)

### Integración con Plan v4/v5

- Drill diario de shapes: `study_tools/DRILL_DIMENSIONES_NUMPY.md`
- Registro de errores: `study_tools/DIARIO_ERRORES.md`
- Evaluación (rúbrica): [study_tools/RUBRICA_v1.md](../study_tools/RUBRICA_v1.md) (scope `M01` en `rubrica.csv`)
- Protocolo completo:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Recursos (cuándo usarlos)

| Prioridad | Recurso | Cuándo usarlo en este módulo | Para qué |
|----------|---------|------------------------------|----------|
| **Obligatorio** | [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/) | Semana 1, antes de empezar con `DataFrame/Series` y limpieza | Referencia oficial para flujo típico de carga/EDA/limpieza |
| **Obligatorio** | [NumPy Documentation (absolute beginners)](https://numpy.org/doc/stable/user/absolute_beginners.html) | Semana 2, cuando aparezcan `ndarray`, `dtype`, `reshape`, `axis`, broadcasting | Fuente oficial para resolver dudas de shapes/axis |
| **Obligatorio** | `study_tools/DRILL_DIMENSIONES_NUMPY.md` | Cada vez que te equivoques en un shape / antes del checklist de salida | Automatizar intuición de shapes |
| **Complementario** | [Real Python - NumPy](https://realpython.com/numpy-tutorial/) | Después de completar broadcasting + vectorización (Semana 2) | Consolidar patrones idiomáticos con ejemplos prácticos |
| **Opcional** | [RECURSOS.md](RECURSOS.md) | Al terminar el módulo (para planificar refuerzo) | Elegir rutas de profundización sin dispersarte |

### Criterio de salida (cuándo puedes avanzar)

- Puedes preparar un `X` y `y` desde un CSV sin errores de dtype/shape.
- Puedes explicar `axis=0` vs `axis=1` y predecir shapes sin ejecutar.
- Puedes demostrar speedup vectorizado (benchmark) y justificarlo.

## 🧠 ¿Por Qué Este Módulo?

### El Problema con Python Puro para ML

```python
# ❌ Así NO se hace en Machine Learning
def dot_product_slow(a: list, b: list) -> float:
    """Producto punto con loop - LENTO."""
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Para vectores de 1 millón de elementos:
# Tiempo: ~200ms
```

```python
# ✅ Así SÍ se hace en Machine Learning
import numpy as np

def dot_product_fast(a: np.ndarray, b: np.ndarray) -> float:
    """Producto punto vectorizado - RÁPIDO."""
    return np.dot(a, b)

# Para vectores de 1 millón de elementos:
# Tiempo: ~2ms (100x más rápido)
```

### Conexión con el Pathway

En los cursos de CU Boulder:
- **Supervised Learning:** Multiplicaciones de matrices para regresión
- **Unsupervised Learning:** PCA requiere descomposición de matrices
- **Deep Learning:** Forward/backward pass son operaciones matriciales

**Sin NumPy, no puedes hacer ML eficiente.**

---

## 📚 Contenido del Módulo

### Semana 1: Pandas + NumPy Básico

```
┌─────────────────────────────────────────────────────────────────┐
│  DÍA 1: Pandas - DataFrame y Series                             │
│  DÍA 2: Pandas - Carga de CSVs (read_csv, head, info)           │
│  DÍA 3: Pandas - Limpieza (dropna, fillna, dtypes)              │
│  DÍA 4: NumPy - Arrays y dtypes                                 │
│  DÍA 5: NumPy - Indexing y Slicing                              │
│  DÍA 6: Pandas → NumPy (df.values, df.to_numpy())               │
└─────────────────────────────────────────────────────────────────┘
```

### Semana 2: NumPy Vectorizado

```
┌─────────────────────────────────────────────────────────────────┐
│  DÍA 1: Broadcasting                                            │
│  DÍA 2: Producto matricial (@, np.dot, np.matmul) + reshape/flatten │
│  DÍA 3: OOP para ML (v5.1): class Tensor (__init__, __add__, @) │
│  DÍA 4: Agregaciones y operaciones con ejes                     │
│  DÍA 5: Random y generación de datos sintéticos                 │
│  DÍA 6: Entregable: Pipeline Pandas → NumPy                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Conceptos Clave

### 0. Pandas Esencial (Días 1-3)

#### ¿Por Qué Pandas?

En el mundo real de ML, los datos vienen en CSVs sucios, no en arrays NumPy perfectos. Antes de aplicar cualquier algoritmo necesitas:

1. **Cargar datos** desde archivos
2. **Explorar** estructura y tipos
3. **Limpiar** valores faltantes y errores
4. **Convertir** a NumPy para el modelo

```python
import pandas as pd
import numpy as np

# ========== CARGA DE DATOS ==========
# Cargar CSV
df = pd.read_csv('data/iris.csv')

# Primeras filas
print(df.head())

# Información del DataFrame
print(df.info())
#  Column         Non-Null Count  Dtype
# ---  ------         --------------  -----
#  0   sepal_length   150 non-null    float64
#  1   sepal_width    150 non-null    float64
#  2   petal_length   150 non-null    float64
#  3   petal_width    150 non-null    float64
#  4   species        150 non-null    object

# Estadísticas básicas
print(df.describe())
```

#### Limpieza de Datos

```python
import pandas as pd

# Crear DataFrame con datos sucios
df = pd.DataFrame({
    'edad': [25, 30, None, 45, 50],
    'salario': [50000, 60000, 70000, None, 90000],
    'ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Sevilla', None]
})

# ========== DETECTAR NULOS ==========
print(df.isnull().sum())
# edad       1
# salario    1
# ciudad     1

# ========== ELIMINAR FILAS CON NULOS ==========
df_clean = df.dropna()  # Elimina filas con cualquier nulo
print(f"Filas después de dropna: {len(df_clean)}")  # 2

# ========== RELLENAR NULOS ==========
df_filled = df.copy()
df_filled['edad'] = df_filled['edad'].fillna(df_filled['edad'].mean())
df_filled['salario'] = df_filled['salario'].fillna(df_filled['salario'].median())
df_filled['ciudad'] = df_filled['ciudad'].fillna('Desconocido')

print(df_filled)
```

#### Selección y Filtrado

```python
import pandas as pd

df = pd.read_csv('data/iris.csv')

# ========== SELECCIONAR COLUMNAS ==========
# Una columna (Serie)
sepal_length = df['sepal_length']

# Múltiples columnas (DataFrame)
features = df[['sepal_length', 'sepal_width']]

# ========== FILTRAR FILAS ==========
# Condición simple
setosa = df[df['species'] == 'setosa']

# Múltiples condiciones
large_setosa = df[(df['species'] == 'setosa') & (df['sepal_length'] > 5)]

# ========== LOC e ILOC ==========
# loc: por etiquetas
df.loc[0:5, ['sepal_length', 'species']]

# iloc: por posición (como NumPy)
df.iloc[0:5, 0:2]
```

#### De Pandas a NumPy (Día 6)

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/iris.csv')

# ========== SEPARAR FEATURES Y TARGET ==========
# Features (X) - todas las columnas numéricas
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()
print(f"X shape: {X.shape}")  # (150, 4)
print(f"X dtype: {X.dtype}")  # float64

# Target (y) - convertir categorías a números
y = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).to_numpy()
print(f"y shape: {y.shape}")  # (150,)

# ========== VERIFICAR ==========
print(f"Tipo X: {type(X)}")  # <class 'numpy.ndarray'>
print(f"Tipo y: {type(y)}")  # <class 'numpy.ndarray'>

# Ahora X e y están listos para algoritmos de ML
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 0: Pandas Esencial (Días 1–3)</strong></summary>

#### 1) Metadatos (organización)
- **Título:** Pandas esencial para preparar datasets reales
- **ID (opcional):** `M01-T00`
- **Duración estimada:** 2–4 horas
- **Nivel:** Intro
- **Dependencias:** Python básico, lectura de CSV, listas/diccionarios

#### 2) Objetivos de aprendizaje (medibles)
- Al terminar, el estudiante podrá **cargar** un CSV en un `DataFrame`, **inspeccionar** su esquema (`info/describe`) y **detectar** valores faltantes **en ≤ 10 min**.
- Al terminar, el estudiante podrá **limpiar** valores faltantes con `dropna/fillna` y **justificar** la elección (media/mediana/categoría) **en un caso dado**.
- Al terminar, el estudiante podrá **construir** `X` y `y` como `np.ndarray` con shapes correctos (`X.shape=(n,d)`, `y.shape=(n,)`) y **verificar** `dtype`.

#### 3) Relevancia y contexto
- **¿Por qué importa?** En ML real, el 80% del trabajo inicial es preparar datos: si `dtype/NaN/shape` están mal, el modelo “falla” o (peor) aprende basura.
- **Dónde encaja:** Esto es el “puente” a todos los módulos (regresión/logística/PCA/redes). Antes: Python básico. Después: NumPy vectorizado y álgebra lineal.

#### 4) Mapa conceptual / conceptos clave
- **Archivo (CSV/Parquet)**
  - **→ `DataFrame`** (tabla con columnas tipadas)
    - **→ EDA rápido** (`head/info/describe`)
      - **→ Limpieza** (NaN, dtypes, outliers básicos)
        - **→ `X` (features) y `y` (target)**
          - **→ `to_numpy()` + validación de shapes**

#### 5) Definiciones, notación y “fórmulas” esenciales
- **`DataFrame`:** estructura tabular (filas/columnas) con columnas tipadas y operaciones vectorizadas.
- **Notación (la usaremos toda la guía):**
  - `n` = número de muestras (filas)
  - `d` = número de features (columnas)
  - `X` = matriz de features, `X.shape == (n, d)`
  - `y` = target, típicamente `y.shape == (n,)`
- **Imputación (idea):**
  - Numéricas: media/mediana (según outliers)
  - Categóricas: categoría “Desconocido” o moda

#### 6) Explicación didáctica (2 niveles)
- **Intuición (2–3 frases):** Pandas es una “mesa de trabajo” para convertir datos sucios en una matriz `X` limpia. El objetivo no es memorizar métodos, sino ejecutar un pipeline estable: cargar → revisar → limpiar → separar → convertir.
- **Paso a paso (pipeline mínimo):**
  - 1. `read_csv` (cargar)
  - 2. `head/info/describe` (entender esquema)
  - 3. `isnull().sum()` (medir faltantes)
  - 4. imputar/eliminar (decisión explícita)
  - 5. separar `y` primero y luego construir `X`
  - 6. `to_numpy()` + checks de `dtype/shape`
- **Visual recomendado:**
  - tabla de `df.info()` + heatmap de faltantes (seaborn) + histograma por columna numérica

#### 7) Ejemplos modelados (worked examples)
- **Ejemplo 1 (sencillo):** Iris → `X` con 4 columnas numéricas, `y` codificada a {0,1,2}.
- **Ejemplo 2 (realista):** dataset con columna de fecha + categorías:
  - parsear fechas con `pd.to_datetime`
  - `get_dummies` para categorías (si aplica)
  - verificar que `X.dtype` sea numérico antes de convertir.
- **Contraejemplo (qué NO hacer):** convertir todo el `DataFrame` a NumPy sin filtrar columnas y terminar con `dtype=object`.

#### 8) Práctica guiada (scaffolded)
- **Ejercicio A (con pista):** carga un CSV y lista columnas por tipo (`df.select_dtypes`).
  - Pista: `df.select_dtypes(include=["number"])`
- **Ejercicio B (con pista):** calcula porcentaje de NaN por columna y decide estrategia.
  - Pista: `df.isnull().mean().sort_values(ascending=False)`
- **Ejercicio C (con pista):** crea `X,y` y verifica shapes.
  - Pista: `assert X.ndim == 2` y `assert y.ndim == 1`

#### 9) Práctica independiente / transferencia
- **Mini-proyecto:** toma un dataset tabular (Titanic / Wine / Housing), define target, limpia faltantes, y entrega `X,y` listos.
- **Entrega:** notebook o script con:
  - impresión de `df.info()`
  - estrategia de limpieza (explicada en 5–10 líneas)
  - `assert` de shapes/dtypes
- **Rúbrica (3 criterios):** precisión (shapes/dtypes), aplicación (pipeline completo), comunicación (justificación breve)

#### 10) Evaluación formativa (rápida)
- **Mini-quiz (5):**
  - 1. ¿Qué diferencia práctica hay entre `df.values` y `df.to_numpy()`?
  - 2. ¿Qué te dice `df.info()` que `df.head()` no te dice?
  - 3. ¿Cuándo usarías mediana en lugar de media para imputar?
  - 4. ¿Qué shapes esperas para `X` y `y` en un problema supervisado?
  - 5. ¿Por qué `dtype=object` es una señal de alerta para ML?

#### 11) Errores comunes y corrección
- **Error:** “`X` contiene strings” → **Corrección:** filtra/convierte columnas y valida `select_dtypes`.
- **Error:** leakage por incluir el target en `X` → **Corrección:** separar `y` primero y luego construir `X`.
- **Error:** imputar sin medir NaN → **Corrección:** reportar `isnull().sum()` / proporción.
- **Error:** `SettingWithCopyWarning` → **Corrección:** usar `.loc[...]` y/o `.copy()`.
- **Error:** no validar shapes → **Corrección:** `assert X.shape[0] == y.shape[0]`.

#### 12) Retención (spaced retrieval)
- **Preguntas (día 2):** ¿Qué hace `axis=0` en medias por columna? ¿Qué shape devuelve?
- **Preguntas (día 7):** ¿Cómo detectas `dtype=object` y qué harías?
- **Checklist rápido:** `info → nulls → split → to_numpy → asserts`.

#### 13) Diferenciación
- **Principiante:** repetir el pipeline en 2 datasets distintos (uno “limpio”, uno “sucio”).
- **Avanzado:** diseñar una función `prepare_xy(df, feature_cols, target_col)` con asserts y tests mínimos.

#### 14) Recursos y materiales
- Pandas: Getting Started (oficial)
- Cheatsheet corta: `df.head/info/describe/isnull/dropna/fillna/loc/iloc/to_numpy`

#### 15) Notas para el docente
- **Tiempo sugerido:** 15 min (carga + schema) + 30 min (limpieza) + 20 min (X/y) + 15 min (debug de errores).
- **Preguntas socráticas:** “¿Qué evidencia tienes de que `X` es numérico?” “¿Qué asumiste al imputar con media?”
</details>

---

### 1. Arrays vs Listas

#### Intuición: “memoria contigua” (NumPy) vs “cajas dispersas” (listas)

Piensa en una **lista de Python** como una fila de cajitas que guardan **referencias** a objetos; esos objetos pueden estar **dispersos** por la memoria. NumPy, en cambio, busca representar un `ndarray` como un **bloque contiguo** de números del mismo tipo (homogéneos). Esa decisión habilita:

- **Vectorización real:** bucles internos en C (muy optimizados).
- **Mejor uso de caché CPU:** leer datos contiguos es más rápido.
- **Menos overhead:** no hay “un objeto por número”.

Mini-diagrama mental:

```
Lista (referencias):  [ * ] -> obj1   [ * ] -> obj2   [ * ] -> obj3   ...
                       |              |              |
                      mem@A          mem@Z          mem@K

NumPy (contiguo):     [ 1.0 ][ 2.0 ][ 3.0 ][ 4.0 ] ...  (mismo dtype)
```

```python
import numpy as np  # Importa NumPy para demostrar cómo axis afecta agregaciones

# Lista de Python
lista = [1, 2, 3, 4, 5]

# Array de NumPy
array = np.array([1, 2, 3, 4, 5])

# Diferencias clave:
# 1. Tipo homogéneo (todos los elementos del mismo tipo)
# 2. Tamaño fijo después de creación
# 3. Operaciones vectorizadas
# 4. Almacenamiento contiguo en memoria
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 1: Arrays vs Listas</strong></summary>

#### 1) Metadatos (organización)
- **Título:** Por qué `ndarray` existe (memoria, dtype y vectorización)
- **ID (opcional):** `M01-T01`
- **Duración estimada:** 30–60 min
- **Nivel:** Intro
- **Dependencias:** listas de Python, loops, tipos numéricos

#### 2) Objetivos de aprendizaje (medibles)
- Al terminar, el estudiante podrá **explicar** (sin jerga) por qué NumPy es más rápido que listas para operaciones numéricas **en ≤ 2 min**.
- Al terminar, el estudiante podrá **predecir** el `dtype` resultante al crear un `np.array(...)` a partir de distintos inputs (ints/floats/mezcla) **en 5 casos**.
- Al terminar, el estudiante podrá **elegir** entre lista vs `ndarray` dado un problema (cómputo numérico vs estructuras heterogéneas) y **justificar** la elección.

#### 3) Relevancia y contexto
- **¿Por qué importa?** Tus modelos (regresión/logística/redes) son multiplicaciones y sumas masivas: hacerlo con listas te hace lento y propenso a errores.
- **Dónde encaja:** fundamento del “por qué” de vectorización, broadcasting, y producto matricial.

#### 4) Mapa conceptual / conceptos clave
- **Lista (Python):** referencias a objetos → heterogénea → overhead por elemento
- **`ndarray` (NumPy):** buffer contiguo + `dtype` homogéneo → operaciones en C/BLAS
- **Consecuencia:** speedup + menos bugs al operar por bloques

#### 5) Definiciones y notación
- **`dtype`:** tipo numérico almacenado en el buffer (p.ej. `float64`, `int64`).
- **`shape`:** tupla con dimensiones (p.ej. `(n,)`, `(n, d)`).
- **“Vectorización”:** operar sobre arrays completos evitando loops en Python.

#### 6) Explicación didáctica (2 niveles)
- **Intuición:** una lista guarda “punteros”; un `ndarray` guarda números contiguos. Leer memoria contigua y aplicar rutinas en C/BLAS suele ser 10x–100x más rápido.
- **Paso a paso:**
  - 1. lista: `for` + operación por elemento
  - 2. NumPy: operación sobre el array completo
  - 3. validación: comparar tiempos y verificar que el resultado coincide
- **Visual recomendado:** gráfico de tiempo (ms) vs tamaño del vector (log-scale).

#### 7) Ejemplos modelados
- **Ejemplo 1:** sumar un escalar a 1e6 números (lista vs array).
- **Ejemplo 2:** producto punto (lista vs `np.dot`).
- **Contraejemplo:** crear un array con mezcla de números y strings → `dtype=object` → pierdes vectorización.

#### 8) Práctica guiada
- **Ejercicio A:** crea `np.array([1,2,3])` y `np.array([1,2,3.0])`. ¿Qué `dtype` obtienes?
  - Pista: imprime `arr.dtype`.
- **Ejercicio B:** mide tiempo de `sum([..])` vs `np.sum(np.array(..))` para `n=10^5`.
  - Pista: usa `time.time()` o `time.perf_counter()`.
- **Ejercicio C:** explica por escrito (5 líneas) cuándo una lista sigue siendo mejor opción.

#### 9) Transferencia
- **Actividad:** dado un “dataset” en lista de diccionarios, conviértelo a `DataFrame` y luego a `ndarray` numérico.
- **Criterio:** no debe quedar `dtype=object` en `X`.

#### 10) Evaluación formativa
- 1. ¿Qué significa que NumPy sea “homogéneo”?
- 2. ¿Qué desventaja tiene `dtype=object`?
- 3. ¿Por qué el loop en Python es costoso?

#### 11) Errores comunes
- **Error:** asumir que `np.array(lista)` siempre crea floats → **Corrección:** inspeccionar `dtype`.
- **Error:** mezclar tipos → **Corrección:** normalizar/convertir antes.

#### 12) Retención
- **Pregunta (día 2):** define `dtype` y por qué importa.
- **Pregunta (día 7):** da un ejemplo donde lista > NumPy.

#### 13) Diferenciación
- **Principiante:** repetir el benchmark con 3 tamaños (1e3, 1e5, 1e6).
- **Avanzado:** explicar (alto nivel) el rol de BLAS y por qué `np.dot` suele estar optimizado.

#### 14) Recursos
- NumPy “absolute beginners” (oficial)

#### 15) Notas para el docente
- **Check for understanding:** pedir que predigan `dtype` antes de imprimirlo.
</details>

### 2. Creación de Arrays

```python
import numpy as np

# Desde lista
a = np.array([1, 2, 3])

# Arrays especiales
zeros = np.zeros((3, 4))        # Matriz 3x4 de ceros
ones = np.ones((2, 3))          # Matriz 2x3 de unos
identity = np.eye(4)            # Matriz identidad 4x4
random = np.random.randn(3, 3)  # Matriz 3x3 valores normales

# Secuencias
rango = np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5) # [0, 0.25, 0.5, 0.75, 1]

print(f"Shape de zeros: {zeros.shape}")  # (3, 4)
print(f"Dtype de zeros: {zeros.dtype}")  # float64
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 2: Creación de Arrays</strong></summary>

#### 1) Metadatos
- **Título:** Crear `ndarray` correctamente (shape, dtype, inicialización)
- **ID (opcional):** `M01-T02`
- **Duración estimada:** 30–45 min
- **Nivel:** Intro
- **Dependencias:** Tema 1 (Arrays vs Listas), noción de `shape` y `dtype`

#### 2) Objetivos de aprendizaje (medibles)
- Al terminar, el estudiante podrá **crear** arrays 1D/2D con `np.array/zeros/ones/eye` y **verificar** `shape` y `dtype` **en ≤ 5 min**.
- Al terminar, el estudiante podrá **elegir** entre `arange` y `linspace` dado un objetivo (paso fijo vs #puntos) y **justificar** la elección.

#### 3) Relevancia y contexto
- **¿Por qué importa?** En ML, crear tensores de parámetros (`W`, `b`) y datos sintéticos es rutina; si creas mal shapes/dtypes, los bugs aparecen tarde.
- **Dónde encaja:** base para broadcasting, producto matricial y generación de datos.

#### 4) Mapa conceptual / conceptos clave
- `shape` → `dtype` → constructor (`zeros/ones/eye/array`) → verificación (`assert`)

#### 5) Definiciones
- **Inicializadores:** `zeros/ones/eye` crean arrays con estructuras útiles para álgebra lineal y debugging.
- **Generadores:** `arange/linspace` construyen secuencias controladas (ojo con floats).

#### 6) Explicación didáctica (2 niveles)
- **Resumen:** decide primero la forma y el tipo; luego el constructor.
- **Paso a paso:**
  - 1. define el `shape`
  - 2. define `dtype` (en ML, casi siempre float)
  - 3. crea
  - 4. valida `shape/dtype`

#### 7) Ejemplos modelados
- **Ejemplo 1:** `np.zeros((n, d), dtype=float)` como “placeholder” de datos.
- **Ejemplo 2:** `np.eye(k)` para confirmar identidad.
- **Contraejemplo:** usar `np.arange(0, 1, 0.1)` esperando incluir exactamente 1.0.

#### 8) Práctica guiada
- **Ejercicio A:** crea una matriz `(3,4)` de unos con dtype float.
- **Ejercicio B:** crea 11 puntos entre 0 y 1 (incluyendo extremos).
- **Ejercicio C:** crea `w` de shape `(d,)` y `b` escalar y verifica tipos.

#### 9) Transferencia
- **Actividad:** genera un dataset sintético `X` con shape `(200, 2)` y verifica `X.ndim == 2`.

#### 10) Evaluación formativa
- ¿Cuándo prefieres `linspace` sobre `arange`?
- ¿Por qué conviene fijar `dtype=float`?

#### 11) Errores comunes
- confundir `(n,)` vs `(n,1)` al crear vectores
- terminar con ints por no fijar dtype

#### 12) Retención
- (día 2) ¿qué shape devuelve `np.eye(3)`?
- (día 7) diferencia práctica `arange` vs `linspace`.

#### 13) Diferenciación
- **Principiante:** escribir `shape/dtype` esperado antes de ejecutar 5 creaciones.
- **Avanzado:** explicar el problema de floats con `arange`.

#### 14) Recursos
- NumPy “absolute beginners” (oficial)

#### 15) Nota para el docente
- Pedir predicción de `shape/dtype` antes de imprimir.
</details>

### 3. Indexing y Slicing

```python
import numpy as np

# Crear matriz 2D
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Acceso a elementos
print(matrix[0, 0])      # 1 (fila 0, columna 0)
print(matrix[1, 2])      # 6 (fila 1, columna 2)

# Slicing
print(matrix[0, :])      # [1, 2, 3] (toda la fila 0)
print(matrix[:, 1])      # [2, 5, 8] (toda la columna 1)
print(matrix[0:2, 1:3])  # [[2, 3], [5, 6]] (submatriz)

# Indexing booleano
print(matrix[matrix > 5])  # [6, 7, 8, 9]
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 3: Indexing y Slicing</strong></summary>

#### 1) Metadatos
- **Título:** Extraer submatrices sin loops (y sin errores de vistas/copias)
- **ID (opcional):** `M01-T03`
- **Duración estimada:** 45–60 min
- **Nivel:** Intro
- **Dependencias:** Tema 2 (creación de arrays) + noción de `shape`

#### 2) Objetivos de aprendizaje (medibles)
- Al terminar, el estudiante podrá **extraer** filas, columnas y submatrices y **predecir** el `shape` resultante **en ≥ 8 de 10 casos**.
- Al terminar, el estudiante podrá **usar** indexing booleano para filtrar valores y **explicar** por qué el resultado suele ser 1D.

#### 3) Relevancia y contexto
- **¿Por qué importa?** Selección de features, minibatches y splits train/val depende de indexar bien; fallar aquí produce bugs silenciosos.

#### 4) Mapa conceptual / conceptos clave
- indexado por enteros → slicing → masks booleanas → vistas/copias

#### 5) Definiciones
- **Slice:** rango semiabierto `[inicio:fin)`.
- **Mask booleana:** array de `True/False` que filtra elementos.

#### 6) Explicación didáctica (2 niveles)
- **Resumen:** primero predice el shape, luego indexa y valida.
- **Paso a paso:**
  - 1. escribe el shape esperado
  - 2. aplica el slicing/indexing
  - 3. valida con `assert`

#### 7) Ejemplos modelados
- **Ejemplo 1:** `X[0]` vs `X[0:1]` (diferencia de shapes).
- **Ejemplo 2:** `X[:, j]` (columna) y por qué queda `(n,)`.
- **Contraejemplo:** modificar un slice pensando que es copia (conecta con la sección de debugging del módulo).

#### 8) Práctica guiada
- **Ejercicio A:** toma un minibatch `X[:32]`.
- **Ejercicio B:** filtra valores en un vector `v` que estén en `[a,b]`.
- **Ejercicio C:** extrae una submatriz y predice su shape antes.

#### 9) Transferencia
- **Actividad:** implementa split 80/20 por slicing y valida shapes de `X_train/X_val`.

#### 10) Evaluación formativa
- ¿Por qué `X[X>0]` suele ser 1D?
- Diferencia entre `X[0]` y `X[0:1]`.

#### 11) Errores comunes
- olvidar paréntesis al usar `&` en masks
- asumir que slices son copias

#### 12) Retención
- (día 2) define “slice semiabierto”.
- (día 7) ¿cuándo usar `.copy()`?

#### 13) Diferenciación
- **Principiante:** 10 predicciones de shape.
- **Avanzado:** verificar vistas/copias con `np.shares_memory`.

#### 14) Recursos
- NumPy indexing (oficial)

#### 15) Nota para el docente
- Pedir “shape antes de ejecutar” + asserts.
</details>

### 4. Broadcasting

#### Worked Example: `(3, 1) + (1, 3)` paso a paso

Objetivo: entender **por qué** funciona sin loops.

1) Define dos arrays con una dimensión “de tamaño 1”:

- `A.shape = (3, 1)` (columna)
- `B.shape = (1, 3)` (fila)

2) Regla clave: si en una dimensión uno de los tamaños es `1`, NumPy puede **“estirar”** esa dimensión para igualar al otro.

3) Resultado final: ambos se ven como `(3, 3)` y se suman elemento a elemento.

```python
import numpy as np

A = np.array([[1], [2], [3]])        # shape: (3, 1)
B = np.array([[10, 20, 30]])         # shape: (1, 3)

# Broadcasting:
# A se repite horizontalmente 3 veces
# B se repite verticalmente 3 veces
C = A + B                             # shape: (3, 3)

print("A:\n", A)
print("B:\n", B)
print("C = A + B:\n", C)
```

```python
import numpy as np

# Broadcasting: operar arrays de diferentes shapes

# Escalar + Array
a = np.array([1, 2, 3])
print(a + 10)  # [11, 12, 13]

# Vector + Matriz (broadcasting automático)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
vector = np.array([10, 20, 30])

# El vector se "expande" para coincidir con la matriz
print(matrix + vector)
# [[11, 22, 33],
#  [14, 25, 36]]

# Regla de broadcasting:
# Las dimensiones deben ser iguales O una de ellas debe ser 1
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 4: Broadcasting</strong></summary>

#### 1) Metadatos
- **Título:** Reglas de broadcasting (sumas/restas por columna/fila)
- **ID (opcional):** `M01-T04`
- **Duración estimada:** 60–90 min
- **Nivel:** Intermedio
- **Dependencias:** Tema 2 (shape) y Tema 3 (indexing)

#### 2) Objetivos de aprendizaje (medibles)
- Al terminar, el estudiante podrá **predecir** el shape resultante de operaciones típicas (`(n,d)+(d,)`, `(n,1)+(1,d)`) **en ≥ 8 de 10 casos**.
- Al terminar, el estudiante podrá **normalizar** por columna `X_norm = (X - mean) / (std + eps)` conservando shape `(n,d)`.

#### 3) Relevancia y contexto
- **¿Por qué importa?** Broadcasting es base de normalización, bias en redes (`+b`) y fórmulas vectorizadas; también es fuente #1 de bugs silenciosos.

#### 4) Mapa conceptual / conceptos clave
- compatibilidad de shapes (alineación por el final) → dimensión 1 “estirable” → `keepdims=True` para control

#### 5) Definición
- **Broadcasting:** regla que permite operar arrays de distinto shape expandiendo virtualmente dimensiones de tamaño 1 sin copiar datos.

#### 6) Explicación didáctica (2 niveles)
- **Resumen:** compara dimensiones desde el final; si son iguales o alguna es 1, son compatibles.
- **Paso a paso:**
  - 1. alinea shapes por la derecha
  - 2. valida compatibilidad por eje
  - 3. el resultado toma el máximo por eje

#### 7) Ejemplos modelados
- **Ejemplo 1:** `(3,1)+(1,3) -> (3,3)`.
- **Ejemplo 2:** `X (n,d) - mean (d,)` para centrar.
- **Contraejemplo:** usar `np.mean(X)` (escalar) cuando querías `axis=0` (vector de size d).

#### 8) Práctica guiada
- **Ejercicio A:** suma un vector `(d,)` a cada fila de `X (n,d)`.
- **Ejercicio B:** calcula `mu = X.mean(axis=0, keepdims=True)` y resta `X - mu`.
- **Ejercicio C:** implementa z-score por columna con `eps`.

#### 9) Transferencia
- **Actividad:** implementa `predict_linear(X,w,b)` que soporte batch: `(n,d)@(d,) + b`.
- **Criterio:** salida con shape `(n,)`.

#### 10) Evaluación formativa
- ¿Qué significa “alineación por la derecha”?
- ¿Qué hace `keepdims=True` y por qué ayuda?

#### 11) Errores comunes
- olvidar `axis=` en agregaciones
- broadcasting “funciona” pero está mal (promedio global)

#### 12) Retención
- (día 2) predice shapes de 5 operaciones.
- (día 7) describe un bug de broadcasting y cómo lo detectarías.

#### 13) Diferenciación
- **Principiante:** practicar `(n,d)+(d,)` y `(n,d)-(d,)`.
- **Avanzado:** justificar por qué `X - X.mean(axis=0)` funciona.

#### 14) Recursos
- NumPy broadcasting rules (oficial)

#### 15) Nota para el docente
- Repetir la regla: “shape antes de ejecutar” + asserts.
</details>

### 5. Agregaciones y Ejes

#### Visualización: ¿qué “colapsa” cada eje?

Regla práctica:

- `axis=0` **colapsa filas** → te queda “una salida por columna”
- `axis=1` **colapsa columnas** → te queda “una salida por fila”

Ejemplo con una matriz `2x3`:

```
X = [[1, 2, 3],
     [4, 5, 6]]

sum(axis=0) = [1+4, 2+5, 3+6] = [5, 7, 9]
sum(axis=1) = [1+2+3, 4+5+6] = [6, 15]
```

```python
import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Agregaciones globales
print(np.sum(matrix))   # 21 (suma de todos)
print(np.mean(matrix))  # 3.5 (promedio de todos)
print(np.std(matrix))   # 1.707... (desviación estándar)

# Agregaciones por eje
# axis=0: colapsar filas (operar columnas)
print(np.sum(matrix, axis=0))  # [5, 7, 9]

# axis=1: colapsar columnas (operar filas)
print(np.sum(matrix, axis=1))  # [6, 15]

# Visualización de ejes:
# ┌─────────────┐
# │ axis=0 ↓    │
# │ [1, 2, 3]   │ → axis=1
# │ [4, 5, 6]   │ → axis=1
# └─────────────┘
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 5: Agregaciones y Ejes</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Agregaciones con `axis` (sum/mean/std) y control de shapes
- **ID (opcional):** `M01-T05`
- **Duración estimada:** 60–90 min
- **Nivel:** Intermedio
- **Dependencias:** Tema 4 (Broadcasting), noción de `shape` y batch `(n,d)`

#### 2) Objetivos de aprendizaje (medibles)
- Al terminar, el estudiante podrá **calcular** agregaciones globales y por eje (`axis=0/1`) y **predecir** el shape resultante **en ≥ 8 de 10 casos**.
- Al terminar, el estudiante podrá **usar** `keepdims=True` para conservar dimensiones y **justificar** cuándo conviene (para broadcasting explícito) **en un ejemplo**.

#### 3) Relevancia y contexto
- **¿Por qué importa?** Normalización por columna, cálculo de métricas y centrado de datos dependen de agregaciones correctas; un `axis` mal puesto produce errores silenciosos.
- **Dónde encaja:** Después de broadcasting, antes de producto matricial y entrenamiento (donde normalizas features y evalúas resultados).

#### 4) Mapa conceptual / lista de conceptos clave
- `X.shape=(n,d)`
  - agregación global → escalar
  - `axis=0` → salida por columna (`(d,)` o `(1,d)`)
  - `axis=1` → salida por fila (`(n,)` o `(n,1)`)
  - `keepdims=True` → habilita broadcasting controlado

#### 5) Definiciones, notación y fórmulas esenciales
- **`axis`:** dimensión que se reduce/colapsa.
- **`keepdims=True`:** conserva la dimensión reducida con tamaño 1 (facilita broadcasting).
- **Notación:** `X∈R^{n×d}`, `μ = mean(X, axis=0)`.

#### 6) Explicación didáctica (2 niveles)
- **Intuición (2–3 frases):** `axis` te dice “qué dirección colapsas”. Si colapsas filas (`axis=0`), te queda una salida por columna. Si colapsas columnas (`axis=1`), te queda una salida por fila.
- **Paso a paso:**
  - 1. escribe el shape de entrada
  - 2. decide si quieres resultado “por columna” o “por fila”
  - 3. elige `axis`
  - 4. si vas a restar/dividir a `X`, evalúa `keepdims=True`
- **Visual recomendado:** matriz dibujada con flechas `axis=0` (↓) y `axis=1` (→).

#### 7) Ejemplos modelados (worked examples)
- **Ejemplo 1 — sencillo:** `X.sum(axis=0)` y `X.sum(axis=1)` y verificación de shapes.
- **Ejemplo 2 — realista:** estandarización por columna con `mu = X.mean(axis=0, keepdims=True)`.
- **Contraejemplo:** usar `X.mean()` (global) cuando se quería media por feature.

#### 8) Práctica guiada (scaffolded)
- **Ejercicio 1:** dado `X (4,3)`, calcula `X.mean(axis=0)` y predice shape.
- **Ejercicio 2:** usa `keepdims=True` y comprueba que `X - mu` conserva shape.
- **Ejercicio 3:** calcula `std` por columna, agrega `eps` y normaliza.
- **Ejercicio 4:** verifica con `assert` que la media por columna de `X_norm` es ~0.

#### 9) Práctica independiente / transferencia
- **Mini-proyecto:** implementa `standardize_columns(X, eps=1e-8)` que retorne `X_norm, mu, std`.
- **Entrega:** función + asserts de shape + breve explicación del uso de `axis`.
- **Rúbrica corta:** precisión (axis/shapes), aplicación (normaliza bien), comunicación (explica keepdims).

#### 10) Evaluación formativa (rápida)
- **Mini-quiz (5):**
  - 1. ¿Qué devuelve `X.mean(axis=0)` si `X` es `(n,d)`?
  - 2. ¿Qué hace `keepdims=True`?
  - 3. Diferencia entre `axis=0` y `axis=1`.
  - 4. ¿Por qué `X.mean()` puede ser un bug?
  - 5. ¿Qué shape debe tener `mu` para restarse a `X` sin ambigüedad?

#### 11) Errores comunes y estrategias correctivas
- **Error:** confundir `axis=0`/`axis=1` → **Corrección:** “axis=0 reduce filas; axis=1 reduce columnas”.
- **Error:** usar agregación global → **Corrección:** forzar `axis=` y revisar shape.
- **Error:** broadcasting accidental → **Corrección:** usar `keepdims=True` + asserts.
- **Error:** normalizar sin `eps` → **Corrección:** sumar `eps` a `std`.
- **Error:** no validar resultado → **Corrección:** `assert np.allclose(X_norm.mean(axis=0), 0, atol=...)`.

#### 12) Apoyos cognitivos y retención
- **Recuperación (día 2):** define `axis` con un ejemplo de `(2,3)`.
- **Recuperación (día 7):** ¿cuándo usarías `keepdims=True`?
- **Checklist:** “shape → axis → keepdims → assert”.

#### 13) Diferenciación
- **Principiante:** practicar solo sum/mean con shapes pequeñas (2x3, 3x2).
- **Avanzado:** derivar por qué `keepdims` evita bugs en pipelines vectorizados.

#### 14) Recursos y materiales
- Documentación NumPy sobre reducciones (`sum`, `mean`, `std`) y `axis`.

#### 15) Notas para el docente / facilitador
- Pide predicción de shape antes de ejecutar; usa 5 casos rápidos en pizarra.
</details>

### 6. Operaciones Matriciales

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Operaciones elemento a elemento
print(A + B)   # Suma
print(A * B)   # Multiplicación elemento a elemento (Hadamard)
print(A / B)   # División elemento a elemento

# Producto matricial (lo que usarás en ML)
print(A @ B)           # Operador @ (Python 3.5+)
print(np.matmul(A, B)) # Función matmul
print(np.dot(A, B))    # Función dot

# Resultado:
# [[19, 22],
#  [43, 50]]

# Transpuesta
print(A.T)
# [[1, 3],
#  [2, 4]]
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 6: Operaciones Matriciales</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Operaciones elemento a elemento vs producto matricial (`@`)
- **ID (opcional):** `M01-T06`
- **Duración estimada:** 60–90 min
- **Nivel:** Intermedio
- **Dependencias:** Tema 2 (shape), Tema 4 (broadcasting), Tema 5 (axis)

#### 2) Objetivos de aprendizaje (medibles)
- Al terminar, el estudiante podrá **distinguir** `A * B` (Hadamard) de `A @ B` (matmul) y **predecir** el shape de salida **en ≥ 8 de 10 casos**.
- Al terminar, el estudiante podrá **implementar** una predicción lineal `y_hat = X @ w + b` y **verificar** shapes correctos con `assert`.

#### 3) Relevancia y contexto
- **¿Por qué importa?** El corazón del ML clásico es álgebra lineal: regresión, logística y redes neuronales dependen de productos matriciales.
- **Dónde encaja:** puente directo a Módulo 02 (Álgebra lineal) y Módulo 05 (regresión/logística).

#### 4) Mapa conceptual / lista de conceptos clave
- `A * B` (element-wise) → requiere mismo shape o broadcasting
- `A @ B` (matmul) → requiere compatibilidad de dimensiones internas
- transpuesta `A.T` → cambia ejes
- caso ML: `(n,d) @ (d,) -> (n,)`

#### 5) Definiciones, notación y fórmulas esenciales
- **Hadamard:** `(A * B)_{ij} = A_{ij}·B_{ij}`
- **Matmul:** si `A∈R^{n×d}` y `B∈R^{d×k}`, entonces `A@B ∈ R^{n×k}`.
- **Transpuesta:** si `A∈R^{n×d}`, entonces `A^T ∈ R^{d×n}`.

#### 6) Explicación didáctica (2 niveles)
- **Intuición:** `@` combina features (columnas) para producir nuevas representaciones; `*` solo “escala” elemento por elemento.
- **Paso a paso:**
  - 1. escribe shapes
  - 2. para `@`, verifica que las dimensiones internas coinciden
  - 3. predice shape de salida
  - 4. valida con asserts
- **Visual recomendado:** diagrama de dimensiones: `(n,d)@(d,k)->(n,k)`.

#### 7) Ejemplos modelados (worked examples)
- **Ejemplo 1 — sencillo:** `A (2,2) @ B (2,2)` y contraste con `A * B`.
- **Ejemplo 2 — realista:** `y_hat = X @ w + b` (regresión lineal) con `X (n,d)`.
- **Contraejemplo:** intentar `A @ B` con shapes incompatibles y “arreglar” a ciegas sin revisar dimensiones.

#### 8) Práctica guiada (scaffolded)
- **Ejercicio 1:** predice shapes de 6 productos (incluye 1D vs 2D).
- **Ejercicio 2:** implementa `y_hat = X @ w + b` y valida `y_hat.shape == (n,)`.
- **Ejercicio 3:** crea `W (d,k)` y calcula `Z = X @ W` y valida `Z.shape == (n,k)`.

#### 9) Práctica independiente / transferencia
- **Mini-proyecto:** implementa `linear_layer(X, W, b)` con asserts de shapes.
- **Entrega:** función + tests mínimos (3 asserts) + explicación breve.

#### 10) Evaluación formativa
- ¿Qué condición de shapes exige `A @ B`?
- ¿Qué operación usarías para aplicar pesos por feature a una matriz: `*` o `@`?
- ¿Qué devuelve `A.T`?

#### 11) Errores comunes y estrategias correctivas
- **Error:** confundir `*` con `@` → **Corrección:** “`@` mezcla columnas; `*` no”.
- **Error:** `X @ w` devuelve `(n,)` pero esperabas `(n,1)` → **Corrección:** decide convención y usa `reshape(-1,1)` si hace falta.
- **Error:** `np.dot` con 1D cambia semántica → **Corrección:** preferir `@` y controlar dims.
- **Error:** shapes incompatibles → **Corrección:** escribir `(n,d)@(d,k)` en papel antes.
- **Error:** broadcasting accidental con `b` → **Corrección:** asegurar `b.shape == (k,)` o `(1,k)`.

#### 12) Apoyos cognitivos y retención
- (día 2) predice shape de `(50,3)@(3,)` y `(50,3)@(3,2)`.
- (día 7) explica diferencia `Hadamard` vs `matmul` en 3 líneas.
- checklist: “internas coinciden → salida externa”.

#### 13) Diferenciación
- **Principiante:** solo 2D×2D con ejemplos pequeños.
- **Avanzado:** discutir cómo BLAS acelera `matmul`.

#### 14) Recursos y materiales
- NumPy `matmul` / operador `@` (docs) + introducción a álgebra lineal (M02).

#### 15) Notas para el docente / facilitador
- Hacer que el estudiante diga en voz alta: “(n,d)@(d,k)->(n,k)” antes de ejecutar.
</details>

### 7. Vectorización: Eliminar Loops

```python
import numpy as np

# ❌ CON LOOP (lento)
def normalize_loop(data: list) -> list:
    """Normalizar datos con loop."""
    mean = sum(data) / len(data)
    std = (sum((x - mean)**2 for x in data) / len(data)) ** 0.5
    return [(x - mean) / std for x in data]

# ✅ VECTORIZADO (rápido)
def normalize_vectorized(data: np.ndarray) -> np.ndarray:
    """Normalizar datos vectorizado."""
    return (data - np.mean(data)) / np.std(data)

# Ejemplo
data = np.random.randn(1000000)

# La versión vectorizada es ~100x más rápida
normalized = normalize_vectorized(data)
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 7: Vectorización (Eliminar Loops)</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Vectorización en NumPy: pensar en batches, no en iteraciones
- **ID (opcional):** `M01-T07`
- **Duración estimada:** 60–90 min
- **Nivel:** Intermedio
- **Dependencias:** Tema 1 (ndarray), Tema 5 (axis)

#### 2) Objetivos de aprendizaje (medibles)
- Al terminar, el estudiante podrá **reescribir** una operación con loops en una versión vectorizada y **verificar** igualdad numérica con `np.allclose`.
- Al terminar, el estudiante podrá **explicar** por qué la versión vectorizada suele ser más rápida (loops en C/BLAS) **en ≤ 2 min**.

#### 3) Relevancia y contexto
- **¿Por qué importa?** Todos los algoritmos del Pathway se vuelven impracticables si operas en Python puro elemento a elemento.
- **Dónde encaja:** base para backprop, gradiente, métricas y entrenamiento batch.

#### 4) Mapa conceptual / conceptos clave
- loop Python → overhead
- ufuncs/BLAS → loops internos en C
- vectorización → operar por bloques
- validación → `allclose` + asserts de shape

#### 5) Definiciones, notación y fórmulas esenciales
- **Vectorización:** expresar el cálculo como operaciones sobre arrays completos.
- **Normalización (z-score):** `x_norm = (x - mean) / std`.

#### 6) Explicación didáctica (2 niveles)
- **Resumen:** “piensa en el batch” (`(n,d)`), no en `for i in range(n)`.
- **Paso a paso:**
  - 1. define entrada como `np.ndarray`
  - 2. escribe el cálculo con operaciones sobre arrays
  - 3. valida con asserts y un test de equivalencia
- **Visual recomendado:** gráfica de tiempo vs tamaño (log) comparando loop vs vectorizado.

#### 7) Ejemplos modelados
- **Ejemplo 1 — sencillo:** normalización 1D con `np.mean/np.std`.
- **Ejemplo 2 — realista:** normalización por columnas de `X (n,d)` con `axis=0`.
- **Contraejemplo:** vectorizar sin controlar `axis`, obteniendo estadísticas globales por accidente.

#### 8) Práctica guiada
- **Ejercicio 1:** implementa suma de dos vectores (loop vs vectorizado) y compara.
- **Ejercicio 2:** implementa distancia L2 (loop vs vectorizado) y compara.
- **Ejercicio 3:** normaliza por columna una matriz y valida `mean≈0`, `std≈1`.
- **Pista:** usa `axis=0` y `keepdims=True` si vas a restar/dividir a `X`.

#### 9) Transferencia
- **Actividad:** implementa `minibatch_mean(X, batch_size)` sin loops sobre features (solo batch slicing permitido).
- **Entrega:** función + asserts de shape + breve explicación.

#### 10) Evaluación formativa
- ¿Qué parte del cómputo se ejecuta en C cuando usas NumPy?
- ¿Qué diferencia hay entre `np.mean(X)` y `np.mean(X, axis=0)`?
- ¿Qué valida `np.allclose`?

#### 11) Errores comunes
- **Error:** “vectoricé pero está mal” → **Corrección:** tests de equivalencia y asserts de shape.
- **Error:** olvido de `axis` → **Corrección:** imprimir shapes de estadísticas.
- **Error:** overflow/underflow en operaciones → **Corrección:** técnicas de estabilidad (p.ej. softmax estable en Tema 8).
- **Error:** mezclar listas con arrays → **Corrección:** normalizar inputs a `np.asarray`.
- **Error:** medir tiempos con ruido → **Corrección:** repetir y promediar.

#### 12) Retención
- (día 2) explica “batch vs iteración” en 3 líneas.
- (día 7) reescribe un loop típico a vectorizado.
- checklist: “shape → axis → vectorizar → validar”.

#### 13) Diferenciación
- **Principiante:** solo 1D (vectores).
- **Avanzado:** vectorizar parte de un algoritmo (p.ej. predicción lineal sobre batch).

#### 14) Recursos
- NumPy ufuncs y broadcasting (docs).

#### 15) Nota docente
- Insistir en: “primero correctness, luego performance”.
</details>

### 8. Funciones Universales (ufuncs)

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])

# Funciones matemáticas (aplicadas elemento a elemento)
print(np.exp(x))      # e^x
print(np.log(x))      # ln(x)
print(np.sqrt(x))     # √x
print(np.sin(x))      # sin(x)

# Importante para ML:
# Sigmoid: σ(x) = 1 / (1 + e^(-x))
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))  # Sigmoid: mapea R -> (0,1) elemento a elemento

# ReLU: max(0, x)
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)  # ReLU: max(0,x) elemento a elemento; anula negativos

print(sigmoid(np.array([-2, -1, 0, 1, 2])))
# [0.119, 0.269, 0.5, 0.731, 0.881]
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 8: Funciones Universales (ufuncs)</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** ufuncs: matemáticas elemento a elemento (base de activaciones y pérdidas)
- **ID (opcional):** `M01-T08`
- **Duración estimada:** 60–90 min
- **Nivel:** Intermedio
- **Dependencias:** Tema 7 (vectorización), noción de `dtype`

#### 2) Objetivos de aprendizaje (medibles)
- Al terminar, el estudiante podrá **identificar** cuándo una función es “element-wise” y **predecir** que preserva el `shape`.
- Al terminar, el estudiante podrá **implementar** `sigmoid` y `relu` vectorizadas y **verificar** propiedades básicas con `assert`.

#### 3) Relevancia y contexto
- **¿Por qué importa?** Activaciones, transformaciones y parte de las pérdidas son ufuncs: sin esto, redes neuronales no se entienden.
- **Dónde encaja:** M07 (Deep Learning) y M05 (logística usa sigmoid).

#### 4) Mapa conceptual / conceptos clave
- ufunc (`exp/log/sqrt/max`) → opera por elemento
- estabilidad numérica → evitar overflow
- activaciones → `sigmoid`, `relu`, `softmax`

#### 5) Definiciones, notación y fórmulas esenciales
- **Sigmoid:** `σ(x)=1/(1+e^{-x})`.
- **ReLU:** `max(0,x)`.
- **Softmax (idea):** `exp(x)/sum(exp(x))` (con shift por estabilidad).

#### 6) Explicación didáctica (2 niveles)
- **Resumen:** una ufunc toma un array y devuelve otro array del mismo shape (salvo agregaciones).
- **Paso a paso:**
  - 1. convierte input con `np.asarray`
  - 2. aplica ufuncs
  - 3. valida rango/propiedades (p.ej. sigmoid en (0,1))
- **Visual recomendado:** curva sigmoid y ReLU (gráfica 2D) y un histograma antes/después.

#### 7) Ejemplos modelados
- **Ejemplo 1:** aplicar `np.exp` a un vector y ver saturación.
- **Ejemplo 2:** softmax estable con `x - x.max()`.
- **Contraejemplo:** softmax sin estabilización con valores grandes.

#### 8) Práctica guiada
- **Ejercicio 1:** implementa `softplus = log(1+exp(x))` con estabilidad.
- **Ejercicio 2:** aplica ReLU a un batch `(n,d)` y valida que no cambia shape.
- **Ejercicio 3:** verifica que `sigmoid(0)=0.5`.

#### 9) Transferencia
- **Actividad:** implementa “capa” `y = relu(X @ W + b)` con asserts de shapes.

#### 10) Evaluación formativa
- ¿Qué parte del cómputo se ejecuta en C cuando usas NumPy?
- ¿Qué propiedad del sigmoid ayuda en clasificación?
- ¿Qué pasa con `dtype=int` si haces operaciones in-place?

#### 11) Errores comunes
- overflow en `exp`
- asumir que ufuncs cambian shape
- usar `sum` sin `axis`

#### 12) Retención
- (día 2) define sigmoid y su rango.
- (día 7) explica softmax estable.

#### 13) Diferenciación
- **Principiante:** graficar ReLU y sigmoid.
- **Avanzado:** derivar qué pasa con gradientes en saturación (intuición).

#### 14) Recursos
- NumPy ufuncs + notas de estabilidad numérica.

#### 15) Nota docente
- Enfatiza “estabilidad primero”: shift en softmax, eps en divisiones.
</details>

### 9. Reshape y Manipulación de Forma

```python
import numpy as np  # Importa NumPy para crear arrays y cambiar su forma (reshape/flatten)
# Crear array 1D
a = np.arange(12)  # Crea un vector 1D con 12 enteros consecutivos (0..11)

# Reshape a 2D
matrix = a.reshape(3, 4)  # Reinterpreta el vector como matriz 2D de shape (3,4); 3*4 debe igualar 12
print(matrix.shape)  # Imprime el shape para verificar que ahora es (3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

# Reshape a 3D
tensor = a.reshape(2, 2, 3)  # Cambia la forma a 3D (2,2,3); 2*2*3=12 conserva el total de elementos
print(tensor.shape)  # Verifica por pantalla la forma del tensor (2, 2, 3)

# Flatten: volver a 1D
flat = matrix.flatten()  # Aplana la matriz 2D y devuelve una copia 1D con todos los elementos
print(flat.shape)  # Comprueba que vuelve a tener 12 elementos en 1D: shape (12,)

# -1 para inferir dimensión automáticamente
auto = a.reshape(4, -1)  # Usa -1 para que NumPy infiera la dimensión faltante: (4, 3)
auto = a.reshape(-1, 6)  # Infiera la primera dimensión para que el total sea 12: (2, 6)

<details>
<summary><strong>📌 Complemento pedagógico — Tema 9: Reshape y Manipulación de Forma</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Control de `shape`: `reshape`, `flatten`, `transpose` y contratos
- **ID (opcional):** `M01-T09`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** Temas 2, 4, 6

#### 2) Objetivo(s) de aprendizaje (medibles)
- Al terminar, el estudiante podrá **transformar** shapes entre 1D/2D/3D con `reshape` sin cambiar `size` y **verificarlo** con `assert`.
- Al terminar, el estudiante podrá **corregir** `(n,)` vs `(n,1)` usando `reshape` o `np.newaxis` en un caso dado.

#### 3) Relevancia y contexto
- **¿Por qué importa?** La mayoría de bugs en ML desde cero son bugs de `shape`. Si controlas shapes, controlas el pipeline.
- **Dónde encaja:** preparación para capas lineales, batches y tensores multi-eje.

#### 4) Mapa conceptual / lista de conceptos clave
- `size` (invariante)
- `reshape` (misma cantidad de elementos)
- `-1` (inferencia)
- `flatten` (copia)
- `transpose` (reordena ejes)

#### 5) Definiciones, notación y fórmulas esenciales
- **Invariante:** `a.size` se conserva al hacer `reshape`.
- **Notación:** batches típicos en ML: `X.shape=(n,d)`.

#### 6) Explicación didáctica (2 niveles)
- **Resumen corto:** `reshape` cambia la interpretación del buffer, no el contenido.
- **Paso a paso:** `size → shape objetivo → reshape → assert`.
- **Visual recomendado:** diagrama “vector → matriz → tensor”.

#### 7) Ejemplos modelados
- **Ejemplo 1:** `(12,) → (3,4)`.
- **Ejemplo 2:** `X_flat (n,784) ↔ X_img (n,28,28)`.
- **Contraejemplo:** “aplanar” para evitar un error sin entender el contrato.

#### 8) Práctica guiada
- 3 ejercicios de predicción de shape + 3 ejercicios de corrección con `assert`.

#### 9) Transferencia
- Implementa `ensure_2d(X)` con casos `(n,)` y `(n,d)`.

#### 10) Evaluación
- ¿Qué garantiza el invariante `size`?
- ¿Qué hace `-1`?

#### 11) Errores comunes
- producto de dimensiones incorrecto
- confundir vector columna vs fila

#### 12) Retención
- (día 2) predice shapes de 5 reshapes.
- (día 7) explica el invariante `size`.

#### 13) Diferenciación
- Principiante: 1D↔2D.
- Avanzado: tensores tipo imagen.

#### 14) Recursos
- NumPy docs: `reshape/transpose/flatten/ravel`.

#### 15) Nota docente
- “Entrada shape → salida shape” antes de ejecutar.
</details>

### 9.1 OOP para ML (v5.1): mini-framework `Tensor`

**Objetivo práctico:** antes de llegar a redes neuronales (donde vas a tener que manejar `self`, estado y operaciones), crea una mini-abstracción que se comporte como un “tensor” simple.

#### Qué debes dominar (sin teoría vacía)

- **Clase vs instancia:** la clase define el “molde”; la instancia es el objeto real en memoria.
- **`self`:** referencia a la instancia actual; ahí vive el estado.
- **Estado:** variables guardadas en el objeto (`self.data`, `self.shape`).
- **Operadores:** `+` llama a `__add__`, `@` llama a `__matmul__`.

#### Entregable (taller)

- Implementar una clase `Tensor` que:
  - acepte lista o `np.ndarray` en `__init__`
  - mantenga un estado interno `self.shape`
  - implemente `__add__` y `__matmul__` usando NumPy por dentro

#### Implementación (referencia)

```python
import numpy as np  # NumPy para convertir entrada a ndarray y reutilizar operaciones vectorizadas
from typing import Union  # Union para aceptar múltiples tipos de entrada en el constructor

ArrayLike = Union[list, np.ndarray]  # Tipo de entrada soportado: lista de Python o ndarray de NumPy

class Tensor:  # Contenedor mínimo para entender OOP aplicado a ML (estado + operadores)
    def __init__(self, data: ArrayLike):  # Constructor: recibe datos y construye el estado interno
        self.data = np.array(data, dtype=float)  # Normaliza a ndarray float para operar consistentemente
        self.shape = self.data.shape  # Guarda shape como parte del estado para inspección y debugging

    def __add__(self, other: "Tensor") -> "Tensor":  # Define el operador + (suma elemento a elemento)
        if not isinstance(other, Tensor):  # Si no es Tensor, delega a Python (permite otros tipos)
            return NotImplemented  # Señal estándar: operación no implementada para ese tipo
        return Tensor(self.data + other.data)  # Suma NumPy y devuelve un nuevo Tensor (no muta self)

    def __matmul__(self, other: "Tensor") -> "Tensor":  # Define el operador @ (producto matricial)
        if not isinstance(other, Tensor):  # Valida tipo para evitar errores silenciosos
            return NotImplemented  # Permite que Python intente la operación reflejada si existe
        return Tensor(self.data @ other.data)  # Usa @ de NumPy (matmul) y envuelve el resultado

    def __repr__(self) -> str:  # Representación útil para ver shape y datos rápido al imprimir
        return f"Tensor(shape={self.shape}, data={self.data})"  # String con información mínima de debugging

```

#### Ejercicios (con `assert`) — tu mínimo aceptable

```python
import numpy as np  # NumPy para comparar arrays con allclose y construir datos de prueba

# 1) Estado: shape debe reflejar el ndarray interno
t = Tensor([1, 2, 3])  # Crea Tensor desde lista (se convierte internamente a ndarray)
assert t.shape == (3,)  # Verifica que el shape se guardó correctamente

# 2) Suma: + llama a __add__
a = Tensor([1, 2, 3])  # Tensor A
b = Tensor([10, 20, 30])  # Tensor B
c = a + b  # Ejecuta __add__ y debe devolver un Tensor nuevo
assert isinstance(c, Tensor)  # Debe devolver Tensor
assert np.allclose(c.data, np.array([11.0, 22.0, 33.0]))  # Verifica el resultado numérico
assert c.shape == (3,)  # El shape debe permanecer (3,)

# 3) Producto matricial: @ llama a __matmul__
A = Tensor([[1, 2], [3, 4]])  # Matriz 2x2
x = Tensor([1, 1])  # Vector de entrada con shape (2,)
y = A @ x  # Producto matriz-vector -> shape (2,)
assert np.allclose(y.data, np.array([3.0, 7.0]))  # [1,2]·[1,1]=3 y [3,4]·[1,1]=7
assert y.shape == (2,)  # Verifica el shape de salida

# 4) Error de shape: debe fallar si dimensiones no son compatibles
try:  # Captura excepción esperada de NumPy cuando shapes no son multiplicables
    _ = Tensor([[1, 2, 3], [4, 5, 6]]) @ Tensor([1, 2])  # (2,3) @ (2,) no es válido
    assert False  # Si no falló, el test debe fallar
except ValueError:  # NumPy lanza ValueError ante incompatibilidad de shapes
    pass  # Éxito: esperábamos el error

```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 9.1: OOP para ML (mini-framework `Tensor`)</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** OOP aplicada a tensores: estado, operadores y contratos
- **ID (opcional):** `M01-T09_1`
- **Duración estimada:** 3–6 horas
- **Nivel:** Intermedio
- **Dependencias:** Temas 2, 6 y 9

#### 2) Objetivo(s) de aprendizaje (medibles)
- Al terminar, el estudiante podrá **implementar** una clase `Tensor` con estado mínimo (`data`, `shape`) y **verificar** invariantes con `assert`.
- Al terminar, el estudiante podrá **explicar** cómo Python resuelve `a + b` y `A @ x` mediante `__add__` y `__matmul__`.

#### 3) Relevancia y contexto
- **¿Por qué importa?** Es el puente entre NumPy “suelo” y los patrones reales de frameworks (módulos con estado + operadores + debugging).
- **Dónde encaja:** prepara para capas (`Linear`), composición y validación temprana de shapes.

#### 4) Mapa conceptual / lista de conceptos clave
- clase vs instancia
- `self` y estado (`self.data`, `self.shape`)
- dunder methods: `__init__`, `__add__`, `__matmul__`, `__repr__`
- `NotImplemented` (protocolo de operadores)
- contrato de shapes (fallar temprano)

#### 5) Definiciones, notación y fórmulas esenciales
- **Estado:** datos guardados en el objeto (persisten entre llamadas).
- **Operador `@`:** producto matricial (matmul). Para `A.shape=(m,n)` y `x.shape=(n,)`, salida `y.shape=(m,)`.

#### 6) Explicación didáctica (2 niveles)
- **Resumen corto:** un “tensor” mínimo es un contenedor con *datos + shape* y métodos que implementan operaciones seguras.
- **Paso a paso:** normaliza entrada → guarda `shape` → implementa operadores → valida con `assert` → mejora `__repr__`.

#### 7) Ejemplos modelados
- `Tensor([1,2,3]) + Tensor([10,20,30])` (element-wise)
- `Tensor([[1,2],[3,4]]) @ Tensor([1,1])` (matmul)
- **Contraejemplo:** permitir `@` sin validar dimensiones y “debuggear” el error tarde.

#### 8) Práctica guiada
- Agrega `assert self.data.ndim in {1,2}` y prueba casos válidos/invalidos.
- Extiende `__repr__` para mostrar `dtype` y `ndim`.

#### 9) Práctica independiente / transferencia
- Implementa `__mul__` (element-wise) y prueba broadcasting controlado.
- Implementa `sum(axis=None)` para conectar con agregaciones del Tema 5.

#### 10) Evaluación (formativa y sumativa)
- **Formativa:** dado un snippet, predice qué método dunder se ejecuta.
- **Sumativa:** implementa `Tensor` + tests con 8–10 `assert` incluyendo 2 casos de error.

#### 11) Errores comunes
- confundir “matmul” con multiplicación elemento a elemento
- no devolver `NotImplemented` y romper compatibilidad con otros tipos
- mutar `self.data` sin intención (efectos colaterales)

#### 12) Apoyos cognitivos / retención
- regla mental: **estado = lo que necesito para debuggear** (shape, dtype).
- checklist: *entrada → shape → operación → salida*.

#### 13) Diferenciación
- **Principiante:** solo `__init__` + `__add__`.
- **Avanzado:** mensajes de error con shapes; soporte para `other` array-like.

#### 14) Recursos
- Data model de Python (dunder methods) y referencia de `operator`.
- NumPy: guía de `matmul` y broadcasting.

#### 15) Nota para el facilitador
- Pide que el alumno escriba primero el “contrato” (shapes esperados) antes de tocar el código.
</details>

### 9.2 OOP para Científicos (ML-first): `__call__`, `__repr__`, `dataclasses`, y debugging

La meta aquí NO es “aprender clases por aprender clases”, sino entrenar los mismos reflejos que usarás en PyTorch/Keras:

- **Objetos invocables:** un objeto que se comporta como función (`layer(x)`), vía `__call__`.
- **Debugging rápido:** `__repr__` te dice estado clave (shapes, dtypes) sin abrir el objeto.
- **Configs reproducibles:** `dataclasses` te obliga a declarar explícitamente hiperparámetros.
- **Shape discipline:** si un shape está mal, que reviente temprano con `assert`.

#### 9.2.1 `dataclass` para configs (hiperparámetros explícitos)

```python
from dataclasses import dataclass  # dataclass: clase “contenedor” para configs, con menos boilerplate
import numpy as np

@dataclass(frozen=True)  # frozen=True: hace la config inmutable (evita cambiar hiperparámetros por accidente)
class LinearConfig:  # Config de una capa lineal
    in_features: int  # D: número de features de entrada
    out_features: int  # K: número de unidades de salida
    seed: int = 0  # Semilla para inicialización reproducible


class Linear:  # “Capa” lineal: implementa y = xW + b
    def __init__(self, cfg: LinearConfig):  # Constructor: recibe config explícita
        self.cfg = cfg  # Guardamos la config como estado: permite debugging y reproducibilidad

        rng = np.random.default_rng(cfg.seed)  # RNG moderno de NumPy; controlado por semilla

        # Pesos W con shape (D, K): D entradas -> K salidas
        self.W = rng.standard_normal((cfg.in_features, cfg.out_features))  # Inicialización normal estándar
        self.b = np.zeros((cfg.out_features,), dtype=float)  # Bias con shape (K,), inicializado en 0

    def __call__(self, x: np.ndarray) -> np.ndarray:  # Permite invocar: y = layer(x)
        # Assert 1: x debe ser 2D: (N, D) porque en ML trabajamos con batches (N muestras)
        assert x.ndim == 2, f"x debe ser 2D (N,D). Recibido ndim={x.ndim}, shape={x.shape}"  # Falla temprano

        # Assert 2: la dimensión D de x debe coincidir con in_features
        assert x.shape[1] == self.cfg.in_features, (
            f"D mismatch: x.shape[1]={x.shape[1]} pero cfg.in_features={self.cfg.in_features}"
        )  # Evita bugs silenciosos en matmul

        # Forward: (N, D) @ (D, K) = (N, K)
        y = x @ self.W  # Producto matricial: combina features con pesos

        # Broadcasting controlado: (N, K) + (K,) => (N, K) (b se suma a cada fila)
        y = y + self.b  # Bias por neurona

        # Assert 3: y debe tener shape (N, K)
        assert y.shape == (x.shape[0], self.cfg.out_features), (
            f"Salida mal shapeada: y.shape={y.shape} pero esperado {(x.shape[0], self.cfg.out_features)}"
        )  # Garantiza contrato de la capa

        return y  # Retorna activaciones lineales (logits si fuera clasificación)

    def __repr__(self) -> str:  # Debugging: qué eres y qué shapes manejas
        return (
            "Linear("  # Prefijo con nombre de clase
            f"in={self.cfg.in_features}, out={self.cfg.out_features}, "  # Hiperparámetros clave
            f"W.shape={self.W.shape}, b.shape={self.b.shape}, "  # Estado interno crítico
            f"W.dtype={self.W.dtype}, b.dtype={self.b.dtype}"  # Dtypes para detectar casts raros
            ")"
        )  # String de diagnóstico


# Demo mínimo (con disciplina de shapes)
cfg = LinearConfig(in_features=4, out_features=3, seed=42)  # Definimos una capa 4->3
layer = Linear(cfg)  # Creamos el objeto capa
print(layer)  # __repr__: imprime “contrato” + estado

X = np.zeros((5, 4), dtype=float)  # Batch: N=5 muestras, D=4 features
assert X.shape == (5, 4)  # Regla de oro: afirmar shapes cuando construyes tensores
Y = layer(X)  # Llamamos como función (gracias a __call__)
assert Y.shape == (5, 3)  # Verificamos el contrato de salida (N,K)
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 9.2: OOP ML-first</strong></summary>

#### 1) Metadatos
- **Título:** `__call__`, `__repr__`, configs y debugging por invariantes
- **ID (opcional):** `M01-T09_2`
- **Duración estimada:** 3–6 horas
- **Nivel:** Intermedio
- **Dependencias:** 9.1, 6

#### 2) Objetivos
- construir módulos invocables y validar shapes/dtypes.
- usar `dataclass(frozen=True)` para reproducibilidad.

#### 3) Relevancia
- replica el patrón de frameworks (PyTorch/Keras).

#### 4) Conceptos clave
- config → estado → contrato (`__call__`) → cómputo (`forward`).

#### 5) Definiciones
- objeto invocable.

#### 6) Didáctica
- separar responsabilidades y fallar temprano.

#### 7) Ejemplos
- `Linear` + `Sequential`.

#### 8) Práctica guiada
- forzar error de shape y mejorar mensajes.

#### 9) Transferencia
- MLP mínimo forward-only.

#### 10) Evaluación
- ¿Qué valida `__call__`?

#### 11) Errores comunes
- no validar invariantes.

#### 12) Retención
- (día 2) lista 3 invariantes.

#### 13) Diferenciación
- avanzado: hooks/logs.

#### 14) Recursos
- dataclasses + lectura conceptual `nn.Module`.

#### 15) Nota docente
- contrato de shapes antes de correr.
</details>

### 10. Generación de Datos Aleatorios

```python
import numpy as np  # Importa NumPy para generar números aleatorios y manipular arrays

# Fijar semilla para reproducibilidad
np.random.seed(42)  # Fija la semilla: hace reproducibles los resultados aleatorios

# Distribución uniforme [0, 1)
uniform = np.random.rand(3, 3)  # Genera una matriz 3x3 con valores uniformes en [0,1)

# Distribución normal (media=0, std=1)
normal = np.random.randn(3, 3)  # Genera una matriz 3x3 con valores ~ N(0,1)

# Distribución normal personalizada
custom_normal = np.random.normal(loc=5, scale=2, size=(100,))  # 100 muestras de N(5,2^2): media 5, std 2

# Enteros aleatorios
integers = np.random.randint(0, 10, size=(3, 3))  # Enteros aleatorios en [0,10) con shape (3,3)

# Shuffle (mezclar)
data = np.arange(10)  # Crea un array 1D [0,1,2,...,9]
np.random.shuffle(data)  # Mezcla el array *in-place* (modifica data directamente)

# Muestreo sin reemplazo
sample = np.random.choice(data, size=5, replace=False)  # Elige 5 elementos distintos de data (sin repetir)
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 10: Generación de Datos Aleatorios</strong></summary>

#### 1) Metadatos
- **Título:** RNG reproducible: semillas, distribuciones, muestreo
- **ID (opcional):** `M01-T10`
- **Duración estimada:** 60–90 min
- **Nivel:** Intermedio
- **Dependencias:** Tema 7 (vectorización), Tema 5 (agregaciones)

#### 2) Objetivos (medibles)
- Al terminar, el estudiante podrá **generar** datos sintéticos reproducibles (seed) y **explicar** por qué importa.
- Al terminar, el estudiante podrá **seleccionar** una distribución adecuada (`rand`, `randn`, `normal`, `randint`) para un caso.

#### 3) Relevancia
- **¿Por qué importa?** Reproducibilidad y experimentos controlados: inicialización, splits y pruebas dependen del RNG.

#### 4) Conceptos clave
- seed → reproducibilidad
- distribución → forma de datos
- `shuffle` in-place → efectos colaterales
- `choice` → muestreo

#### 5) Definiciones
- **In-place:** modifica el objeto original.

#### 6) Didáctica
- fija seed → genera → valida shape → documenta distribución.

#### 7) Ejemplos
- `randn(n,d)` para features gaussianas; `choice` para índices.
- **Contraejemplo:** usar `shuffle` y luego asumir que el array original no cambió.

#### 8) Práctica guiada
- generar `X (200,2)` y `y = X@w + noise`.
- split reproducible usando `choice`.

#### 9) Transferencia
- construir dataset sintético linealmente separable.

#### 10) Evaluación
- diferencia `rand` vs `randn`.
- qué resuelve la seed.

#### 11) Errores comunes
- no fijar seed
- mezclar RNG global sin control

#### 12) Retención
- (día 2) por qué reproducibilidad.
- (día 7) cómo hacer split reproducible.

#### 13) Diferenciación
- **Principiante:** solo shapes.
- **Avanzado:** `default_rng` (conceptual).

#### 14) Recursos
- NumPy Random (docs).

#### 15) Nota docente
- pedir reporte de seed + shape + distribución.
</details>

 ---

## 📊 Type Hints con NumPy

```python
import numpy as np  # Importa NumPy para operaciones numéricas
from numpy.typing import NDArray  # Tipado estático: NDArray permite anotar arrays de NumPy con mypy

# Type hints para arrays
def normalize(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normaliza un array de floats."""
    return (data - np.mean(data)) / np.std(data)  # Estandariza: resta la media y divide por la desviación estándar

# Type hints genéricos
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula el producto punto de dos vectores."""
    return float(np.dot(a, b))  # np.dot devuelve un escalar NumPy; float() lo convierte a float de Python

# Con mypy
# pip install numpy-stubs
```

---

## ⚡ Benchmark: Lista vs NumPy

```python
import numpy as np  # NumPy para operaciones vectorizadas y producto punto rápido (np.dot)
import time  # time.time() para medir tiempos de ejecución (benchmark simple)
from typing import List  # Tipado: lista de floats para la implementación “con Python puro”

def benchmark_dot_product():
    """Compara rendimiento de lista vs NumPy."""
    size = 1_000_000  # Tamaño del vector: suficientemente grande para notar diferencias de rendimiento

    # Crear datos
    list_a: List[float] = [float(i) for i in range(size)]  # Lista de floats: implementación base (no vectorizada)
    list_b: List[float] = [float(i) for i in range(size)]  # Segunda lista de floats
    array_a = np.array(list_a)  # Convierte lista a ndarray: permite operaciones vectorizadas (en C)
    array_b = np.array(list_b)  # Convierte la segunda lista a ndarray

    # Benchmark lista
    start = time.time()  # Marca tiempo inicial
    result_list = sum(a * b for a, b in zip(list_a, list_b))  # Producto punto con generador + zip (Python puro)
    time_list = time.time() - start  # Tiempo total transcurrido para la versión con listas

    # Benchmark NumPy
    start = time.time()  # Marca tiempo inicial para NumPy
    result_numpy = np.dot(array_a, array_b)  # Producto punto vectorizado: usa implementación optimizada (BLAS)
    time_numpy = time.time() - start  # Tiempo total transcurrido para NumPy

    print(f"Lista:  {time_list:.4f}s")  # Reporta tiempo de la implementación con listas
    print(f"NumPy:  {time_numpy:.4f}s")  # Reporta tiempo de la implementación con NumPy
    print(f"Speedup: {time_list/time_numpy:.1f}x")  # Factor de aceleración: cuántas veces NumPy es más rápido

    # Verificar resultados iguales
    assert abs(result_list - result_numpy) < 1e-6  # Confirma que ambos métodos producen el mismo resultado

if __name__ == "__main__":
    benchmark_dot_product()  # Ejecuta el benchmark solo cuando el archivo se corre como script

# Output típico:
# Lista:  0.1523s
# NumPy:  0.0015s
# Speedup: 101.5x
```

---

## 🎯 Ejercicios por tema (progresivos) + Soluciones

Reglas de uso:

- **Primero intenta** sin ver soluciones.
- **Tiempo límite sugerido:** 10–15 min por ejercicio antes de mirar la solución.
- **Éxito mínimo:** que tu solución pase los `assert` de cada ejercicio.

 ---

  ### Ejercicio 1.1: Pandas - DataFrame y Series

  #### Enunciado

  1) **Básico**

  - Crea un `DataFrame` llamado `df` con columnas `edad`, `salario`, `ciudad` (5 filas).
  - Extrae la columna `salario` como `Series` y calcula su media.

  2) **Intermedio**

  - Crea una nueva columna `salario_k` con `salario / 1000`.
  - Ordena el `DataFrame` por `salario` de mayor a menor.

  3) **Avanzado**

  - Calcula, por ciudad, la media de `salario` y el conteo de filas (en una sola tabla).

  #### Solución

```python
import pandas as pd

df = pd.DataFrame(
    {
        "edad": [25, 30, 30, 45, 50],
        "salario": [50000, 60000, 61000, 80000, 90000],
        "ciudad": ["Madrid", "Barcelona", "Madrid", "Sevilla", "Madrid"],
    }
)

salario = df["salario"]
media_salario = float(salario.mean())

df["salario_k"] = df["salario"] / 1000
df_sorted = df.sort_values("salario", ascending=False)

resumen = (
    df.groupby("ciudad", as_index=False)
    .agg(
        salario_mean=("salario", "mean"),
        n=("salario", "size"),
    )
    .sort_values("salario_mean", ascending=False)
)

assert isinstance(media_salario, float)
assert "salario_k" in df.columns
assert df_sorted.iloc[0]["salario"] == df["salario"].max()
assert set(resumen.columns) == {"ciudad", "salario_mean", "n"}
```

  ---

  ### Ejercicio 1.2: Pandas - Limpieza (missing values, dtypes, duplicados)

  #### Enunciado

  1) **Básico**

 - Crea un `DataFrame` con valores faltantes en `edad` y `salario`.
 - Cuenta cuántos nulos hay por columna.

 2) **Intermedio**

 - Rellena `edad` con la media.
 - Rellena `salario` con la mediana.

 3) **Avanzado**

 - Agrega una fila duplicada a propósito.
 - Elimina duplicados.
 - Convierte `edad` a `int` **después** de imputar.

 #### Solución

```python
import pandas as pd  # Pandas para limpieza: nulos, duplicados, casting de tipos
import numpy as np  # NumPy para utilidades numéricas y verificación robusta de dtype

df = pd.DataFrame(  # Crea un DataFrame con missing values (None) para simular datos reales “sucios”
    {  # Diccionario: columnas -> listas
        "edad": [25, None, 30, 45, None],  # 'None' se interpretará como NaN (faltante) en una columna numérica
        "salario": [50000, 60000, None, 80000, 90000],  # Otro faltante en 'salario'
        "ciudad": ["Madrid", "Barcelona", "Madrid", "Sevilla", "Madrid"],  # Columna categórica sin nulos
    }  # Cierra el diccionario
)  # Cierra el constructor del DataFrame

nulls = df.isnull().sum()  # isnull() marca NaN/None; sum() por columna cuenta True => número de nulos

df2 = df.copy()  # Copia explícita: evita mutar df (importante si df se reutiliza en otros pasos)
df2["edad"] = df2["edad"].fillna(df2["edad"].mean())  # Imputa edad con media (supone distribución “razonable”)
df2["salario"] = df2["salario"].fillna(df2["salario"].median())  # Imputa salario con mediana (robusta a outliers)

df3 = pd.concat([df2, df2.iloc[[0]]], ignore_index=True)  # Añade una fila duplicada (la primera) para probar drop_duplicates
df3 = df3.drop_duplicates()  # Elimina filas duplicadas exactas (misma combinación de valores)
df3["edad"] = df3["edad"].round().astype(int)  # Convierte a int al final: redondea y castea (sin NaN ya)

assert nulls["edad"] == 2  # Debe haber 2 nulos originales en edad
assert nulls["salario"] == 1  # Debe haber 1 nulo original en salario
assert df2.isnull().sum().sum() == 0  # Tras imputación, no deben quedar nulos
assert len(df3) == len(df2)  # Agregar un duplicado y luego quitarlo deja el mismo tamaño
assert df3["edad"].dtype == np.int64 or str(df3["edad"].dtype).startswith("int")  # Verifica tipo entero
```

 ---

### Ejercicio 1.3: Pandas - Selección y filtrado (`loc`, `iloc`, boolean masks)

#### Enunciado

 Usa este `DataFrame`:

```python
import pandas as pd  # Importa Pandas para construir el DataFrame de ejemplo

df = pd.DataFrame(  # DataFrame pequeño (similar a Iris) para practicar selección/filtrado
    {  # Diccionario columna -> valores
        "sepal_length": [5.1, 4.9, 5.8, 6.0, 5.4],  # Feature numérica: longitud del sépalo
        "sepal_width": [3.5, 3.0, 2.7, 2.2, 3.9],  # Feature numérica: ancho del sépalo
        "species": ["setosa", "setosa", "versicolor", "virginica", "setosa"],  # Variable categórica: especie
    }  # Cierra diccionario
)  # Cierra DataFrame
```

1) **Básico**

- Extrae las columnas `sepal_length` y `species`.

2) **Intermedio**

- Filtra solo las filas donde `species == "setosa"` y `sepal_length > 5.0`.

3) **Avanzado**

- Calcula el promedio de `sepal_length` por `species`.
- Devuelve el resultado ordenado de mayor a menor.

#### Solución

```python
import pandas as pd  # Pandas para DataFrames, máscaras booleanas y groupby

df = pd.DataFrame(  # Re-crea el DataFrame del enunciado (datos en memoria)
    {  # Columnas definidas con listas de igual longitud
        "sepal_length": [5.1, 4.9, 5.8, 6.0, 5.4],  # Longitud del sépalo
        "sepal_width": [3.5, 3.0, 2.7, 2.2, 3.9],  # Ancho del sépalo
        "species": ["setosa", "setosa", "versicolor", "virginica", "setosa"],  # Clase (string)
    }  # Cierra diccionario
)  # Cierra DataFrame

subset = df[["sepal_length", "species"]]  # Selección de múltiples columnas: devuelve DataFrame con 2 columnas

filtered = df[(df["species"] == "setosa") & (df["sepal_length"] > 5.0)]  # Máscara booleana: combina condiciones con &

means = (  # Agregación por especie para obtener promedios
    df.groupby("species", as_index=False)  # Agrupa por 'species' y conserva 'species' como columna
    .agg(sepal_length_mean=("sepal_length", "mean"))  # Media por grupo: una fila por especie
    .sort_values("sepal_length_mean", ascending=False)  # Ordena para tener ranking de especies por media
)  # Cierra pipeline

assert list(subset.columns) == ["sepal_length", "species"]  # Confirma columnas seleccionadas
assert (filtered["species"] == "setosa").all()  # Todas las filas filtradas deben ser setosa
assert (filtered["sepal_length"] > 5.0).all()  # Todas las filas filtradas deben cumplir sepal_length > 5
assert means.iloc[0]["sepal_length_mean"] >= means.iloc[-1]["sepal_length_mean"]  # Verifica el orden descendente
```

---

### Ejercicio 1.4: NumPy - Arrays y `dtype`

#### Enunciado

1) **Básico**

- Crea:
  - un vector de 10 ceros
  - una matriz `3x3` de unos
  - una identidad `4x4`

2) **Intermedio**

- Crea un vector `v = np.array([1, 2, 3])`.
- Convierte `v` a `float64`.
- Verifica que `v / 2` produce floats.

3) **Avanzado**

- Reproduce el caso típico de bug por `dtype` usando división in-place:
  - crea `a = np.array([1, 2, 3])`
  - aplica `a /= 2`
  - explica el resultado con un `assert` esperado

#### Solución

```python
import numpy as np  # NumPy: base del cómputo numérico y estructuras tipo array

z = np.zeros(10)  # Crea un vector 1D de longitud 10 con ceros (dtype float por defecto)
ones = np.ones((3, 3))  # Crea una matriz 3x3 llena de unos (shape: (3, 3))
I = np.eye(4)  # Crea una matriz identidad 4x4 (1 en diagonal, 0 fuera)

v = np.array([1, 2, 3])  # Crea un array a partir de enteros (dtype típico: int)
v_f = v.astype(np.float64)  # Convierte a float64: evita problemas de división/overflow y habilita decimales

half = v_f / 2  # División “normal”: al ser float, el resultado preserva decimales

a = np.array([1, 2, 3])  # Array entero: aquí preparamos el caso de bug
a /= 2  # División IN-PLACE: si el dtype es int, NumPy trunca/convierte (pierde decimales) para mantener dtype

assert z.shape == (10,)  # Confirma forma del vector
assert ones.shape == (3, 3)  # Confirma forma de la matriz
assert I.shape == (4, 4)  # Confirma forma de la identidad
assert v_f.dtype == np.float64  # Confirma que la conversión a float64 ocurrió
assert half.dtype == np.float64  # Confirma que la división produce floats
assert np.array_equal(a, np.array([0, 1, 1]))  # 1/2->0, 2/2->1, 3/2->1 (truncado por dtype entero)
```

---

### Ejercicio 1.5: NumPy - Indexing y Slicing

#### Enunciado

Dada la matriz:

```python
import numpy as np
X = np.arange(20).reshape(4, 5)
```

1) **Básico**

- Extrae el elemento en fila 2, columna 3.

2) **Intermedio**

- Extrae:
  - toda la fila 1
  - toda la columna 4
  - la submatriz filas 1–2, columnas 2–4

3) **Avanzado**

- Usa indexing booleano para extraer elementos mayores que 10.
- Verifica que todos los elementos del resultado cumplan `> 10`.

#### Solución

```python
import numpy as np  # Importa NumPy: base para trabajar con arrays y hacer slicing/indexing sin bucles

X = np.arange(20).reshape(4, 5)  # Crea 0..19 y lo reorganiza como matriz de 4 filas y 5 columnas

e = X[2, 3]  # Indexado 2D: elemento en fila=2 y columna=3 (índices empiezan en 0)

row1 = X[1, :]  # Slicing: fila 1 completa; ':' significa “todas las columnas”
col4 = X[:, 4]  # Slicing: columna 4 completa; ':' significa “todas las filas”
sub = X[1:3, 2:5]  # Submatriz: filas 1–2 y columnas 2–4 (el extremo final del slice se excluye)

gt10 = X[X > 10]  # Indexado booleano: filtra elementos > 10; el resultado es un vector 1D

assert e == 13  # Verifica el valor esperado en la posición (2,3)
assert row1.shape == (5,)  # Una fila completa de una matriz (4,5) tiene 5 elementos
assert col4.shape == (4,)  # Una columna completa de una matriz (4,5) tiene 4 elementos
assert sub.shape == (2, 3)  # La submatriz seleccionada tiene 2 filas y 3 columnas
assert (gt10 > 10).all()  # Confirma que todos los elementos filtrados cumplen la condición
```

---

### Ejercicio 1.6: NumPy - Broadcasting

#### Enunciado

1) **Básico**

- Sin loops, suma 100 a cada elemento de una matriz `3x3`.

2) **Intermedio**

- Dada una matriz `A` de shape `(4, 3)` y un vector `v` de shape `(3,)`, suma `v` a cada fila.

3) **Avanzado**

- Dado `X` de shape `(n, d)`, normaliza por columna: `X_norm = (X - mean) / (std + eps)`.
- **Importante:** el resultado debe conservar shape `(n, d)`.

#### Solución

```python
import numpy as np  # NumPy: permite operaciones vectorizadas y broadcasting sin bucles

M = np.arange(9).reshape(3, 3)  # Crea una matriz 3x3 con valores 0..8
M2 = M + 100  # Broadcasting con escalar: suma 100 a cada elemento (la forma no cambia)

A = np.arange(12).reshape(4, 3)  # Matriz (4,3) con valores 0..11
v = np.array([10, 20, 30])  # Vector (3,) alineado con las columnas: se sumará a cada fila
B = A + v  # Broadcasting: v se “expande” a (4,3) virtualmente para sumar por filas

X = np.random.randn(100, 5)  # Datos sintéticos: 100 muestras (filas) y 5 features (columnas)
eps = 1e-8  # Epsilon: evita división por cero o números extremadamente pequeños
mean = X.mean(axis=0)  # Media por columna (por feature) => shape (5,)
std = X.std(axis=0)  # Desviación estándar por columna => shape (5,)
X_norm = (X - mean) / (std + eps)  # Normaliza por feature usando broadcasting; conserva shape (100,5)

assert M2.shape == (3, 3)  # Sumar un escalar no cambia la forma
assert B.shape == (4, 3)  # Sumar un vector alineado a columnas no cambia la forma
assert X_norm.shape == (100, 5)  # La normalización por columnas debe conservar (n,d)
```

---

### Ejercicio 1.7: NumPy - Producto matricial (`@`, `np.dot`, `np.matmul`)

#### Enunciado

1) **Básico**

- Calcula `A @ B` con:
  - `A` de shape `(2, 3)`
  - `B` de shape `(3, 2)`

2) **Intermedio**

- Demuestra la diferencia entre:
  - multiplicación elemento a elemento `A * B`
  - producto matricial `A @ B`
  usando matrices cuadradas `2x2`.

3) **Avanzado**

- Implementa una predicción lineal `y_hat = X @ w + b` con:
  - `X` shape `(n, d)`
  - `w` shape `(d,)`
  - `b` escalar
- Verifica el shape de `y_hat`.

#### Solución

```python
import numpy as np  # NumPy: operaciones vectorizadas y álgebra lineal (producto matricial con @)

A = np.array([[1, 2, 3], [4, 5, 6]])  # Matriz A de shape (2,3)
B = np.array([[1, 0], [0, 1], [1, 1]])  # Matriz B de shape (3,2)
C = A @ B  # Producto matricial: (2,3)@(3,2) -> (2,2)

U = np.array([[1, 2], [3, 4]])  # Matriz 2x2 para contrastar Hadamard vs matmul
V = np.array([[10, 20], [30, 40]])  # Matriz 2x2
hadamard = U * V  # Multiplicación elemento a elemento (Hadamard)
matmul = U @ V  # Producto matricial (fila-columna)

X = np.random.randn(50, 3)  # Datos: 50 muestras (n) y 3 features (d)
w = np.array([0.1, -0.2, 0.3])  # Vector de pesos: shape (d,)
b = 0.5  # Bias escalar: se suma a cada predicción por broadcasting
y_hat = X @ w + b  # Predicción lineal: (n,d)@(d,) -> (n,)

assert C.shape == (2, 2)  # Verifica shape del producto matricial A@B
assert hadamard.shape == (2, 2)  # Hadamard mantiene shape
assert matmul.shape == (2, 2)  # Matmul entre 2x2 produce 2x2
assert y_hat.shape == (50,)  # Una predicción por muestra
```

---

### Ejercicio 1.8: NumPy - `reshape`, `flatten`, `transpose`

#### Enunciado

1) **Básico**

- Crea `a = np.arange(12)` y conviértelo a una matriz `(3, 4)`.

2) **Intermedio**

- Transpone la matriz anterior y verifica el shape.

3) **Avanzado**

- Convierte la matriz `(3, 4)` a un tensor `(2, 2, 3)`.
- Vuelve a 1D y verifica que recuperas 12 elementos.

#### Solución

```python
import numpy as np  # NumPy para manipulación de shape y operaciones de reshape/transpose

a = np.arange(12)  # Vector 1D con 12 elementos (0..11)
M = a.reshape(3, 4)  # Reinterpreta como matriz (3,4); 3*4=12 debe coincidir
MT = M.T  # Transpuesta: intercambia ejes (3,4) -> (4,3)

T = a.reshape(2, 2, 3)  # Reinterpreta como tensor 3D (2,2,3); 2*2*3=12
flat = T.reshape(-1)  # Aplana a 1D; -1 indica “infiera el tamaño”

assert M.shape == (3, 4)  # Verifica forma de la matriz
assert MT.shape == (4, 3)  # Verifica forma de la transpuesta
assert T.shape == (2, 2, 3)  # Verifica forma del tensor
assert flat.shape == (12,)  # Verifica que el aplanado recupera 12 elementos
assert np.array_equal(flat, a)  # Verifica que el contenido (y el orden) se conserva
```

---

### Ejercicio 1.9: NumPy - Agregaciones y `axis`

#### Enunciado

Sea:

```python
import numpy as np
X = np.array([[1, 2, 3], [4, 5, 6]])
```

1) **Básico**

- Calcula `X.sum()` y verifica el resultado.

2) **Intermedio**

- Calcula `X.sum(axis=0)` y `X.sum(axis=1)`.
- Predice los shapes antes de ejecutar.

3) **Avanzado**

- Calcula `mean` por columna con `keepdims=True`.
- Resta esa media a `X` y verifica el shape del resultado.

#### Solución

```python
import numpy as np  # NumPy: agregaciones (sum/mean) y control de ejes con axis

X = np.array([[1, 2, 3], [4, 5, 6]])  # Matriz (2,3): 2 filas, 3 columnas

s_all = X.sum()  # Suma total de TODOS los elementos => escalar (sin axis)
s0 = X.sum(axis=0)  # axis=0: reduce filas -> suma por columna => shape (3,)
s1 = X.sum(axis=1)  # axis=1: reduce columnas -> suma por fila => shape (2,)

mu = X.mean(axis=0, keepdims=True)  # Media por columna; keepdims=True deja shape (1,3) para broadcasting explícito
X_centered = X - mu  # Centrado: resta la media de cada columna a cada fila (broadcasting)

assert s_all == 21  # 1+2+3+4+5+6 = 21
assert s0.shape == (3,)  # Una suma por columna
assert s1.shape == (2,)  # Una suma por fila
assert mu.shape == (1, 3)  # Con keepdims, la media conserva el eje reducido como dimensión 1
assert X_centered.shape == (2, 3)  # Restar mu no debe cambiar la forma
assert np.allclose(X_centered.mean(axis=0), 0.0)  # Tras centrar, la media por columna debe ser ~0
```

---

### Ejercicio 1.10: NumPy - `random` y datos sintéticos

#### Enunciado

1) **Básico**

- Fija una semilla y genera 5 números con `np.random.randn`.

2) **Intermedio**

- Genera un dataset sintético para regresión:
  - `X` de shape `(200, 2)`
  - `w_true` de shape `(2,)`
  - `y = X @ w_true + noise`

3) **Avanzado**

- Estandariza `X` por columna (`mean=0`, `std=1` aproximadamente).
- Verifica con `np.allclose` (tolerancia razonable).

#### Solución

```python
import numpy as np  # NumPy para aleatoriedad reproducible, datos sintéticos y estandarización

np.random.seed(42)  # Semilla fija: garantiza reproducibilidad (mismos números aleatorios)
z = np.random.randn(5)  # Genera 5 valores ~ N(0,1) -> vector (5,)

n = 200  # Número de muestras
X = np.random.randn(n, 2)  # Features: matriz (200,2)
w_true = np.array([1.5, -0.7])  # Pesos verdaderos (ground truth) de la relación lineal
noise = 0.1 * np.random.randn(n)  # Ruido gaussiano pequeño para simular variación
y = X @ w_true + noise  # Targets: combinación lineal (X@w) + ruido -> vector (200,)

eps = 1e-8  # Epsilon: evita división por cero (estabilidad numérica)
X_mean = X.mean(axis=0)  # Media por columna (por feature) -> (2,)
X_std = X.std(axis=0)  # Desviación estándar por columna -> (2,)
Xz = (X - X_mean) / (X_std + eps)  # Estandariza por columnas usando broadcasting -> (200,2)

assert z.shape == (5,)  # Confirma 5 valores
assert X.shape == (200, 2)  # Confirma shape del dataset
assert w_true.shape == (2,)  # Confirma shape de pesos
assert y.shape == (200,)  # Confirma un target por muestra
assert np.allclose(Xz.mean(axis=0), np.zeros(2), atol=1e-7)  # Media ~0 por feature
assert np.allclose(Xz.std(axis=0), np.ones(2), atol=1e-6)  # Std ~1 por feature
```

---

### (Bonus) Ejercicio 1.11: Vectorización + funciones de activación (dominio)

#### Enunciado

1) **Vectorización**

- Implementa distancia euclidiana sin loops.

2) **Activaciones**

- Implementa:
  - `sigmoid`
  - `relu`
  - `softmax` (estable numéricamente)

#### Solución

```python
import numpy as np  # NumPy: operaciones vectorizadas y funciones matemáticas (exp, sqrt, sum)

def euclidean_distance_vectorized(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b  # Resta vectorizada: diferencia componente a componente
    return float(np.sqrt(np.sum(diff * diff)))  # Distancia L2: sqrt(sum((a-b)^2)); float() devuelve un escalar nativo

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))  # Sigmoid: 1/(1+exp(-x)), mapea R -> (0,1) elemento a elemento

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)  # ReLU: max(0,x), anula valores negativos y deja positivos

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)  # Asegura np.ndarray (por si llega lista) para operaciones vectorizadas
    x_shift = x - np.max(x)  # Estabilidad numérica: resta el máximo para evitar overflow en exp
    exps = np.exp(x_shift)  # Exponencial elemento a elemento (estable tras el shift)
    return exps / np.sum(exps)  # Softmax: normaliza a probabilidades (la suma debe ser 1)

a = np.array([1.0, 2.0, 3.0])  # Vector de prueba
b = np.array([1.0, 1.0, 1.0])  # Segundo vector de prueba
d = euclidean_distance_vectorized(a, b)  # Distancia euclidiana sin loops

assert np.isclose(d, np.sqrt(0**2 + 1**2 + 2**2))  # Chequeo: sqrt((0)^2+(1)^2+(2)^2)
assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)  # Propiedad clave: sigmoid(0)=0.5
assert relu(np.array([-5.0, 5.0])).tolist() == [0.0, 5.0]  # ReLU anula negativos y deja positivos
assert np.isclose(softmax(np.array([1.0, 2.0, 3.0])).sum(), 1.0)  # Softmax debe sumar 1
```

---

## 📦 Entregable del Módulo

### Script: `benchmark_vectorization.py`

```python
"""
Benchmark: Operaciones vectoriales Lista vs NumPy

Este script compara el rendimiento de operaciones comunes
usando listas de Python puras vs arrays de NumPy.

Operaciones comparadas:
1. Producto punto
2. Normalización
3. Distancia euclidiana
4. Suma de matrices

Autor: [Tu nombre]
Fecha: [Fecha]
"""

import numpy as np  # NumPy: operaciones vectorizadas (arrays), álgebra lineal y cómputo eficiente
import time  # time.time(): medición simple de tiempo (en segundos) para benchmarks
from typing import List, Tuple, Callable  # Tipos: anotar listas, tuplas de argumentos y funciones “callables”
from dataclasses import dataclass  # dataclass: genera automáticamente __init__ y facilita structs de resultados


@dataclass  # Marca la clase como dataclass: simplifica almacenamiento de resultados
class BenchmarkResult:  # Estructura para guardar un resultado de benchmark de forma consistente
    """Resultado de un benchmark."""
    operation: str  # Nombre de la operación evaluada (p.ej. "Producto Punto")
    time_list: float  # Tiempo promedio por iteración usando listas (segundos)
    time_numpy: float  # Tiempo promedio por iteración usando NumPy (segundos)
    speedup: float  # Aceleración: time_list / time_numpy


def benchmark(
    func_list: Callable,  # Implementación “con listas” (más cercana a Python puro)
    func_numpy: Callable,  # Implementación “con NumPy” (vectorizada/optimizada)
    args_list: Tuple,  # Argumentos posicionales para func_list (se expanden con *)
    args_numpy: Tuple,  # Argumentos posicionales para func_numpy
    operation_name: str,  # Nombre legible para imprimir/reportar
    iterations: int = 100  # Cuántas repeticiones para promediar (reduce ruido)
) -> BenchmarkResult:
    """Ejecuta benchmark comparativo."""

    # Benchmark lista
    start = time.time()  # Tiempo inicial (lista)
    for _ in range(iterations):  # Repite para promediar y obtener una medida más estable
        func_list(*args_list)  # Llama la función de listas expandiendo la tupla de argumentos
    time_list = (time.time() - start) / iterations  # Tiempo promedio por iteración (lista)

    # Benchmark NumPy
    start = time.time()  # Tiempo inicial (NumPy)
    for _ in range(iterations):  # Misma cantidad de iteraciones para comparar “justo”
        func_numpy(*args_numpy)  # Llama la función NumPy expandiendo sus argumentos
    time_numpy = (time.time() - start) / iterations  # Tiempo promedio por iteración (NumPy)

    return BenchmarkResult(  # Empaqueta resultados en un objeto con campos con nombre
        operation=operation_name,  # Nombre de la operación
        time_list=time_list,  # Tiempo promedio con listas
        time_numpy=time_numpy,  # Tiempo promedio con NumPy
        speedup=time_list / time_numpy  # Speedup: cuántas veces NumPy es más rápido que listas
    )


# === IMPLEMENTAR TUS FUNCIONES AQUÍ ===

def dot_product_list(a: List[float], b: List[float]) -> float:
    """Producto punto con listas."""
    # TODO: Implementar el producto punto sum(a_i * b_i) recorriendo ambas listas
    pass


def dot_product_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Producto punto con NumPy."""
    # TODO: Implementar usando np.dot(a, b) (o a @ b si son 1D)
    pass


def normalize_list(data: List[float]) -> List[float]:
    """Normalizar con listas."""
    # TODO: Implementar (x - mean) / std calculando mean y std manualmente (Python puro)
    pass


def normalize_numpy(data: np.ndarray) -> np.ndarray:
    """Normalizar con NumPy."""
    # TODO: Implementar (data - data.mean()) / data.std() de forma vectorizada
    pass


def euclidean_distance_list(a: List[float], b: List[float]) -> float:
    """Distancia euclidiana con listas."""
    # TODO: Implementar sqrt(sum((a_i - b_i)^2)) recorriendo ambas listas
    pass


def euclidean_distance_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia euclidiana con NumPy."""
    # TODO: Implementar usando vectorización: np.sqrt(np.sum((a-b)**2)) o np.linalg.norm(a-b)
    pass


def matrix_sum_list(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Suma de matrices con listas."""
    # TODO: Implementar suma elemento a elemento usando loops (filas/columnas)
    pass


def matrix_sum_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Suma de matrices con NumPy."""
    # TODO: Implementar A + B (broadcasting/operación vectorizada)
    pass


def main():
    """Ejecutar todos los benchmarks."""
    size = 10000  # Tamaño de los vectores para las pruebas (no tan grande para que corra rápido)

    # Crear datos de prueba
    list_a = [float(i) for i in range(size)]  # Vector (lista) de floats: 0..size-1
    list_b = [float(i) for i in range(size)]  # Segundo vector (lista) del mismo tamaño
    array_a = np.array(list_a)  # Versión NumPy del vector (ndarray)
    array_b = np.array(list_b)  # Versión NumPy del segundo vector

    matrix_size = 100  # Tamaño de matrices cuadradas (100x100) para prueba de suma de matrices
    list_matrix_a = [[float(i*j) for j in range(matrix_size)]  # Construye matriz A con listas (filas)
                     for i in range(matrix_size)]  # Cada fila i contiene productos i*j
    list_matrix_b = [[float(i+j) for j in range(matrix_size)]  # Construye matriz B con listas
                     for i in range(matrix_size)]  # Cada fila i contiene sumas i+j
    array_matrix_a = np.array(list_matrix_a)  # Convierte matriz A a ndarray (vectorizado)
    array_matrix_b = np.array(list_matrix_b)  # Convierte matriz B a ndarray

    # Ejecutar benchmarks
    results = []  # Acumulador de BenchmarkResult (uno por operación)

    results.append(benchmark(  # Ejecuta y guarda benchmark del producto punto
        dot_product_list, dot_product_numpy,
        (list_a, list_b), (array_a, array_b),
        "Producto Punto"
    ))

    results.append(benchmark(  # Ejecuta y guarda benchmark de normalización
        normalize_list, normalize_numpy,
        (list_a,), (array_a,),
        "Normalización"
    ))

    results.append(benchmark(  # Ejecuta y guarda benchmark de distancia euclidiana
        euclidean_distance_list, euclidean_distance_numpy,
        (list_a, list_b), (array_a, array_b),
        "Distancia Euclidiana"
    ))

    results.append(benchmark(  # Ejecuta y guarda benchmark de suma de matrices
        matrix_sum_list, matrix_sum_numpy,
        (list_matrix_a, list_matrix_b), (array_matrix_a, array_matrix_b),
        "Suma de Matrices"
    ))

    # Mostrar resultados
    print("\n" + "="*60)  # Separador visual
    print("BENCHMARK: Lista vs NumPy")  # Título del reporte
    print("="*60)  # Separador visual
    print(f"{'Operación':<25} {'Lista (ms)':<12} {'NumPy (ms)':<12} {'Speedup':<10}")  # Encabezado de tabla
    print("-"*60)  # Separador para la tabla

    for r in results:  # Itera sobre resultados de cada operación
        print(f"{r.operation:<25} {r.time_list*1000:<12.4f} {r.time_numpy*1000:<12.4f} {r.speedup:<10.1f}x")  # Convierte s->ms

    print("="*60)  # Cierre de la tabla
    print(f"\nSpeedup promedio: {sum(r.speedup for r in results)/len(results):.1f}x")  # Promedio de speedups


if __name__ == "__main__":
    main()  # Punto de entrada: ejecuta benchmarks al correr el script
```

---

## 🐛 Debugging NumPy: Errores que te Harán Perder el Tiempo (v3.2)

> ⚠️ **CRÍTICO:** Estos 5 errores son los más frecuentes en las Fases 1 y 2. Resolverlos ahora previene horas de frustración.

### Error 1: Shape Mismatch - `(5,)` vs `(5,1)`

```python
import numpy as np  # Importa NumPy para crear arrays y analizar shapes (dimensiones)

# PROBLEMA: Vector 1D vs Vector Columna
v1 = np.array([1, 2, 3, 4, 5])      # Shape: (5,) - Vector 1D (una sola dimensión)
v2 = np.array([[1], [2], [3], [4], [5]])  # Shape: (5, 1) - Vector columna (matriz de 5 filas y 1 columna)

print(f"v1.shape: {v1.shape}")  # Imprime el shape real de v1 para confirmar que es (5,)
print(f"v2.shape: {v2.shape}")  # Imprime el shape real de v2 para confirmar que es (5, 1)

# ESTO FALLA en Regresión Lineal:
# Si X tiene shape (100, 5) y theta tiene shape (5,), el resultado es (100,)
# Si theta tiene shape (5, 1), el resultado es (100, 1)

# SOLUCIÓN: Usar reshape o keepdims
v1_columna = v1.reshape(-1, 1)  # Convierte (5,) → (5,1); -1 infiere automáticamente el número de filas
v1_columna_alt = v1[:, np.newaxis]  # Alternativa: inserta un eje nuevo para obtener un vector columna

# REGLA: Para ML, los vectores de features deben ser (n, 1), no (n,)
```

### Error 2: Broadcasting Silencioso Incorrecto

```python
import numpy as np  # Importa NumPy para generar datos y demostrar cómo axis afecta agregaciones/broadcasting

# PROBLEMA: Broadcasting no falla, pero da resultados incorrectos
X = np.random.randn(100, 5)  # 100 samples, 5 features
mean_wrong = np.mean(X)      # ¡INCORRECTO! Media global: colapsa todos los ejes y devuelve un escalar
mean_correct = np.mean(X, axis=0)  # Correcto: media por feature (columna) => shape (5,)

print(f"mean_wrong shape: {np.array(mean_wrong).shape}")  # () - escalar (sin dimensiones)
print(f"mean_correct shape: {mean_correct.shape}")  # (5,) - un valor por columna

# REGLA: Siempre especifica axis= en agregaciones
# axis=0: opera sobre filas (resultado por columna)
# axis=1: opera sobre columnas (resultado por fila)
```

### Error 3: Modificación In-Place Inesperada

```python
import numpy as np  # Importa NumPy para mostrar la diferencia entre vistas (views) y copias (.copy())

# PROBLEMA: Los slices de NumPy son VISTAS, no copias
original = np.array([1, 2, 3, 4, 5])  # Array original
slice_view = original[1:4]  # Slice: por defecto es una vista al mismo buffer de memoria
slice_view[0] = 999  # Modifica la vista; por ser vista, también modifica el array original

print(original)  # [1, 999, 3, 4, 5] - ¡ORIGINAL MODIFICADO! porque slice_view comparte memoria

# SOLUCIÓN: Usar .copy() explícitamente
original = np.array([1, 2, 3, 4, 5])  # Reinicia el array original
slice_copy = original[1:4].copy()  # copy(): crea un nuevo buffer independiente
slice_copy[0] = 999  # Modifica la copia; NO afecta el original

print(original)  # [1, 2, 3, 4, 5] - Original intacto porque slice_copy no comparte memoria
```

### Error 4: División por Cero en Normalización

```python
import numpy as np  # Importa NumPy para ejemplificar el caso std=0 y cómo estabilizar divisiones con epsilon

# PROBLEMA: División por cero cuando std = 0
data = np.array([5, 5, 5, 5, 5])
std = np.std(data)  # 0.0 porque todos los valores son idénticos (varianza cero)
normalized = (data - np.mean(data)) / std  # RuntimeWarning: divide by zero (división por 0)

# SOLUCIÓN: Añadir epsilon
epsilon = 1e-8
normalized_safe = (data - np.mean(data)) / (std + epsilon)  # Evita división por cero y estabiliza el cálculo

# REGLA: Siempre usar epsilon en divisiones (especialmente en softmax, normalizaciones)
```

### Error 5: Tipos de Datos Incorrectos

```python
import numpy as np  # Importa NumPy para demostrar problemas de dtype (int vs float) en operaciones in-place

# PROBLEMA: Operaciones con int cuando necesitas float
a = np.array([1, 2, 3])  # dtype: int64 (enteros)
b = a / 2  # dtype: float64 (OK): en Python 3 la división / produce float

# PERO en operaciones in-place:
a = np.array([1, 2, 3])
a /= 2  # In-place: intenta guardar floats en int64 => trunca (pierde decimales)
print(a)  # [0, 1, 1] - ¡TRUNCADO! por conversión implícita a entero

# SOLUCIÓN: Especificar dtype al crear
a = np.array([1, 2, 3], dtype=np.float64)
a /= 2  # Ahora sí: al ser float64, conserva decimales en la operación in-place
print(a)  # [0.5, 1.0, 1.5] - Correcto (sin truncamiento)

# REGLA: Para ML, siempre usar dtype=np.float64 o np.float32
```

---

## 🛠️ Estándares de Código Profesional (v3.2)

> 💎 **Filosofía v3.2:** El código no se considera terminado hasta que pase `mypy`, `ruff` y `pytest`.

### Configuración del Entorno Profesional

```bash
# Crear entorno virtual
python -m venv .venv  # Crea un entorno virtual local (aislado) en la carpeta .venv
source .venv/bin/activate  # Activa el entorno virtual en Linux/Mac (usa el Python y pip de .venv)
# .venv\Scripts\activate   # Alternativa en Windows para activar el entorno virtual

# Instalar herramientas de calidad
pip install numpy pandas matplotlib  # Instala dependencias principales de ciencia de datos
pip install mypy ruff pytest  # Instala herramientas de calidad: tipos, lint/format y tests

# Archivo pyproject.toml (crear en la raíz del proyecto)
```

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"  # Versión de Python objetivo para el análisis de tipos
warn_return_any = true  # Advierte cuando una función retorna Any (pérdida de precisión de tipos)
warn_unused_ignores = true  # Advierte si hay "# type: ignore" que no son necesarios
disallow_untyped_defs = true  # Exige anotaciones de tipo en funciones (evita defs sin typing)

[tool.ruff]
line-length = 100  # Longitud máxima de línea para lint/format
select = ["E", "F", "W", "I", "UP"]  # Conjunto de reglas: estilo, errores, imports, modernización

[tool.pytest.ini_options]
testpaths = ["tests"]  # Carpeta donde pytest buscará tests por defecto
python_files = "test_*.py"  # Patrón de archivos que pytest considera como tests
```

### Ejemplo: Código con Type Hints

```python
# src/linear_algebra.py
"""Operaciones de álgebra lineal desde cero."""
import numpy as np  # NumPy para operaciones vectorizadas (sum, sqrt) sobre arrays
from numpy.typing import NDArray  # Tipado: NDArray permite anotar ndarrays con dtype para mypy


def dot_product(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """
    Calcula el producto punto de dos vectores.

    Args:
        a: Primer vector (n,)
        b: Segundo vector (n,)

    Returns:
        El producto punto (escalar)

    Raises:
        ValueError: Si los vectores tienen shapes diferentes
    """
    if a.shape != b.shape:  # Validación: el producto punto requiere vectores del mismo tamaño
        raise ValueError(f"Shapes incompatibles: {a.shape} vs {b.shape}")  # Falla explícitamente con mensaje útil
    return float(np.sum(a * b))  # Multiplica elemento a elemento y suma; float() convierte escalar NumPy a float nativo


def norm_l2(v: NDArray[np.float64]) -> float:
    """Calcula la norma L2 (euclidiana) de un vector."""
    return float(np.sqrt(np.sum(v ** 2)))  # sqrt(sum(v^2)): definición de norma L2
```

### Ejemplo: Tests con pytest

```python
# tests/test_linear_algebra.py
"""Tests unitarios para linear_algebra.py"""
import numpy as np  # NumPy para construir vectores de prueba
import pytest  # pytest para asserts avanzados y verificación de excepciones
from src.linear_algebra import dot_product, norm_l2  # Funciones bajo prueba


class TestDotProduct:
    """Tests para la función dot_product."""

    def test_dot_product_basic(self) -> None:
        """Test básico: [1,2,3] · [4,5,6] = 32"""
        a = np.array([1.0, 2.0, 3.0])  # Primer vector
        b = np.array([4.0, 5.0, 6.0])  # Segundo vector
        assert dot_product(a, b) == 32.0  # Verifica 1*4 + 2*5 + 3*6

    def test_dot_product_orthogonal(self) -> None:
        """Vectores ortogonales tienen producto punto = 0"""
        a = np.array([1.0, 0.0])  # Vector unitario en x
        b = np.array([0.0, 1.0])  # Vector unitario en y
        assert dot_product(a, b) == 0.0  # Ortogonales => producto punto 0

    def test_dot_product_shape_mismatch(self) -> None:
        """Debe lanzar ValueError si shapes no coinciden"""
        a = np.array([1.0, 2.0])  # Shape (2,)
        b = np.array([1.0, 2.0, 3.0])  # Shape (3,)
        with pytest.raises(ValueError):  # Espera una excepción por shapes incompatibles
            dot_product(a, b)  # Debe fallar (validación de shapes)


class TestNormL2:
    """Tests para la función norm_l2."""

    def test_norm_unit_vector(self) -> None:
        """Vector unitario tiene norma 1"""
        v = np.array([1.0, 0.0, 0.0])  # Vector unitario en 3D
        assert norm_l2(v) == 1.0  # Norma de un vector unitario es 1

    def test_norm_345(self) -> None:
        """Triángulo 3-4-5: norma de [3,4] = 5"""
        v = np.array([3.0, 4.0])  # Vector (3,4)
        assert norm_l2(v) == 5.0  # sqrt(3^2 + 4^2) = 5
```

### Comandos de Verificación

```bash
# Ejecutar en la raíz del proyecto:

# 1. Verificar tipos (mypy)
mypy src/  # Revisa anotaciones de tipo y detecta inconsistencias en src/

# 2. Verificar estilo (ruff)
ruff check src/  # Lint: encuentra errores comunes (imports, variables no usadas, estilo)
ruff format src/  # Auto-formatea el código según reglas de estilo

# 3. Ejecutar tests (pytest)
pytest tests/ -v  # Ejecuta los tests en modo verboso

# 4. Todo junto (antes de cada commit)
mypy src/ && ruff check src/ && pytest tests/ -v  # Pipeline mínimo de calidad antes de commitear
```

---

## 🎯 El Reto del Tablero Blanco (Metodología Feynman)

> 📝 **Instrucción:** Después de implementar código, debes poder explicar el algoritmo en **máximo 5 líneas** sin usar jerga técnica. Si no puedes, vuelve a la teoría.

### Ejemplo: Broadcasting

**❌ Explicación técnica (mala):**
"Broadcasting es la capacidad de NumPy de realizar operaciones elemento a elemento entre arrays de diferentes shapes mediante la expansión implícita de dimensiones según reglas de compatibilidad."

**✅ Explicación Feynman (buena):**
"Cuando sumas un número a una lista, NumPy automáticamente suma ese número a CADA elemento. Es como si el número se 'copiara' para que tenga el mismo tamaño que la lista. Lo mismo pasa entre listas de diferentes tamaños, siempre que una de ellas tenga tamaño 1 en alguna dimensión."

### Tu Reto para el Módulo 01:

Explica en 5 líneas o menos:
1. ¿Por qué NumPy es más rápido que listas de Python?
2. ¿Qué significa `axis=0` vs `axis=1`?
3. ¿Por qué `.copy()` es importante?

---

## ✅ Checklist de Finalización (v3.2)

### Conocimiento
- [ ] Puedo crear arrays 1D, 2D y 3D con NumPy
- [ ] Entiendo indexing y slicing de arrays
- [ ] Puedo explicar broadcasting y usarlo
- [ ] Sé calcular agregaciones por eje (axis)
- [ ] Puedo reescribir loops como operaciones vectorizadas
- [ ] Conozco las diferencias entre `@`, `np.dot`, `np.matmul`
- [ ] Conozco los 5 errores comunes de NumPy y sus soluciones

### Entregables de Código
- [ ] `benchmark_vectorization.py` implementado
- [ ] El speedup de NumPy vs lista es >50x en mis pruebas
- [ ] `mypy src/` pasa sin errores
- [ ] `ruff check src/` pasa sin errores
- [ ] Al menos 3 tests con `pytest` pasando

### Metodología Feynman
- [ ] Puedo explicar broadcasting en 5 líneas sin jerga
- [ ] Puedo explicar axis=0 vs axis=1 en 5 líneas sin jerga
- [ ] Puedo explicar por qué .copy() es importante

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| - | [00_INDICE](00_INDICE.md) | [02_ALGEBRA_LINEAL_ML](02_ALGEBRA_LINEAL_ML.md) |
