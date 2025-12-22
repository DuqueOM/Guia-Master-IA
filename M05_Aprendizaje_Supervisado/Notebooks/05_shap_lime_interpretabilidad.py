"""
Notebook 05: SHAP y LIME para Interpretabilidad de Modelos
===========================================================

Módulo 5 - Semana 11: Ética en IA e Interpretabilidad (XAI)
Curso Alineado: CSCA 5622 - Supervised Learning

Objetivos:
1. Implementar SHAP values para explicar modelos de caja negra
2. Usar LIME para generar explicaciones locales
3. Comparar ambos métodos en un modelo Random Forest
4. Analizar potencial sesgo en un dataset real

Dependencias:
    pip install shap lime scikit-learn pandas matplotlib numpy

Ejecutar como script o convertir a notebook con jupytext.
"""

# %% [markdown]
# # Interpretabilidad de Modelos con SHAP y LIME
#
# En este notebook exploraremos cómo explicar las predicciones de modelos
# de "caja negra" usando dos técnicas fundamentales:
# - **SHAP** (SHapley Additive exPlanations)
# - **LIME** (Local Interpretable Model-agnostic Explanations)

# %%
# Imports
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Para SHAP y LIME
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP no instalado. Ejecutar: pip install shap")

try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME no instalado. Ejecutar: pip install lime")

warnings.filterwarnings("ignore")
print("✅ Imports completados")

# %% [markdown]
# # Interpretabilidad de Modelos con SHAP y LIME
#
# En este notebook exploraremos cómo explicar las predicciones de modelos
# de "caja negra" usando dos técnicas fundamentales:
# - **SHAP** (SHapley Additive exPlanations)
# - **LIME** (Local Interpretable Model-agnostic Explanations)
#
# %%
# %% [markdown]
# ## 1. Preparación del Dataset
#
# Usaremos el dataset de **Breast Cancer** de sklearn para clasificación binaria.
# Este dataset tiene 30 features numéricas y el objetivo es clasificar tumores
# como malignos (1) o benignos (0).

# %%
# Cargar dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("📊 Dataset: Breast Cancer Wisconsin")
print(f"   Samples: {X.shape[0]}")
print(f"   Features: {X.shape[1]}")
print(f"   Clases: {data.target_names}")
print(f"   Balance: {np.bincount(y)}")

# %%
# Explorar features
print("\n📋 Primeras 5 features:")
print(X.head())

print("\n📈 Estadísticas:")
print(X.describe().T[["mean", "std", "min", "max"]].head(10))

# %%
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✂️ Split:")
print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# %% [markdown]
# ## 2. Entrenar Modelo de Caja Negra
#
# Entrenamos un Random Forest, que es un modelo de "caja gris"
# (tiene cierta interpretabilidad vía feature importance, pero no explica
# predicciones individuales fácilmente).

# %%
# Entrenar Random Forest
model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("🎯 Random Forest entrenado")
print(f"   Accuracy: {accuracy:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=data.target_names)}")

# %%
# Feature Importance (método tradicional)
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

print("\n📊 Top 10 Features (importancia tradicional):")
print(feature_importance.head(10).to_string(index=False))

# Visualizar
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance (Top 15)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. SHAP: SHapley Additive exPlanations
#
# ### 3.1 Fundamento Teórico
#
# SHAP usa valores de Shapley de teoría de juegos para asignar a cada feature
# su contribución a la predicción. Es el único método que satisface:
#
# 1. **Local Accuracy**: Los SHAP values suman exactamente la predicción
# 2. **Missingness**: Features ausentes tienen SHAP = 0
# 3. **Consistency**: Mayor contribución → mayor SHAP value

# %%
if SHAP_AVAILABLE:
    print("🔬 Calculando SHAP values...")

    # Crear explainer para modelo de árboles (más eficiente)
    explainer = shap.TreeExplainer(model)

    # Calcular SHAP values para conjunto de test
    shap_values = explainer.shap_values(X_test)

    print("✅ SHAP values calculados")
    print(f"   Shape: {shap_values[1].shape}")  # Clase 1 (maligno)
    print(f"   Expected value (base): {explainer.expected_value}")
else:
    print("⚠️ SHAP no disponible. Instalar con: pip install shap")

# %% [markdown]
# ### 3.2 Visualizaciones SHAP
#
# SHAP ofrece múltiples visualizaciones para entender el modelo:

# %%
if SHAP_AVAILABLE:
    # Summary Plot: Importancia global con dirección
    print("\n📊 SHAP Summary Plot (Importancia Global)")
    print("   - Cada punto es una muestra")
    print("   - Color: valor de la feature (rojo=alto, azul=bajo)")
    print("   - Posición X: impacto en la predicción")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.title("SHAP Summary Plot - Clase Maligno")
    plt.tight_layout()
    plt.show()

# %%
if SHAP_AVAILABLE:
    # Bar plot de importancia promedio
    print("\n📊 SHAP Feature Importance (valor absoluto promedio)")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

# %%
if SHAP_AVAILABLE:
    # Explicación individual (Force Plot)
    print("\n📊 SHAP Force Plot - Explicación Individual")
    print("   Muestra cómo cada feature empuja la predicción")

    # Seleccionar una muestra
    idx = 0
    print(f"\n   Muestra {idx}:")
    print(f"   - Predicción real: {data.target_names[y_test.iloc[idx]]}")
    print(f"   - Predicción modelo: {data.target_names[y_pred[idx]]}")

    # Force plot (en notebook interactivo)
    # shap.force_plot(explainer.expected_value[1], shap_values[1][idx], X_test.iloc[idx])

    # Waterfall plot (alternativa estática)
    plt.figure(figsize=(12, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[1][idx],
            base_values=explainer.expected_value[1],
            data=X_test.iloc[idx],
            feature_names=X_test.columns.tolist(),
        ),
        show=False,
    )
    plt.title(f"SHAP Waterfall - Muestra {idx}")
    plt.tight_layout()
    plt.show()

# %%
if SHAP_AVAILABLE:
    # Dependence Plot: Relación entre feature y SHAP value
    print("\n📊 SHAP Dependence Plot")
    print("   Muestra cómo el valor de una feature afecta su contribución")

    # Para la feature más importante
    top_feature = feature_importance.iloc[0]["feature"]

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature, shap_values[1], X_test, show=False)
    plt.title(f"SHAP Dependence Plot: {top_feature}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. LIME: Local Interpretable Model-agnostic Explanations
#
# ### 4.1 Fundamento Teórico
#
# LIME genera explicaciones locales aproximando el modelo complejo
# con un modelo simple (ej: regresión lineal) en la vecindad de cada predicción.
#
# **Algoritmo:**
# 1. Generar perturbaciones alrededor de la instancia
# 2. Obtener predicciones del modelo original para cada perturbación
# 3. Pesar por proximidad a la instancia original
# 4. Entrenar modelo lineal en datos pesados

# %%
if LIME_AVAILABLE:
    print("🔬 Configurando LIME explainer...")

    # Crear explainer LIME
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=data.target_names.tolist(),
        mode="classification",
        random_state=42,
    )

    print("✅ LIME explainer configurado")
else:
    print("⚠️ LIME no disponible. Instalar con: pip install lime")

# %%
if LIME_AVAILABLE:
    # Explicar una predicción individual
    idx = 0
    print(f"\n📊 LIME Explicación - Muestra {idx}")
    print(f"   Predicción real: {data.target_names[y_test.iloc[idx]]}")
    print(f"   Predicción modelo: {data.target_names[y_pred[idx]]}")

    # Generar explicación
    explanation = lime_explainer.explain_instance(
        X_test.iloc[idx].values, model.predict_proba, num_features=10, num_samples=1000
    )

    # Mostrar como lista
    print("\n   Top 10 features que influyen en la predicción:")
    for feature, weight in explanation.as_list():
        direction = "↑" if weight > 0 else "↓"
        print(f"   {direction} {feature}: {weight:.4f}")

# %%
if LIME_AVAILABLE:
    # Visualizar explicación LIME
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(12, 6)
    plt.title(f"LIME Explanation - Sample {idx}")
    plt.tight_layout()
    plt.show()

# %%
if LIME_AVAILABLE:
    # Comparar explicaciones para diferentes muestras
    print("\n📊 Comparando explicaciones para múltiples muestras...")

    # Seleccionar muestras de cada clase
    idx_benign = np.where((y_test == 0) & (y_pred == 0))[0][0]
    idx_malign = np.where((y_test == 1) & (y_pred == 1))[0][0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, idx, label in [
        (axes[0], idx_benign, "Benigno"),
        (axes[1], idx_malign, "Maligno"),
    ]:
        exp = lime_explainer.explain_instance(
            X_test.iloc[idx].values, model.predict_proba, num_features=8
        )

        features = [f[0] for f in exp.as_list()]
        weights = [f[1] for f in exp.as_list()]
        colors = ["green" if w > 0 else "red" for w in weights]

        ax.barh(features, weights, color=colors)
        ax.set_xlabel("Weight")
        ax.set_title(f"LIME: Predicción {label}")
        ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Comparación SHAP vs LIME

# %%
if SHAP_AVAILABLE and LIME_AVAILABLE:
    print("\n📊 Comparación SHAP vs LIME para la misma muestra")

    idx = 5

    # SHAP
    shap_importance = (
        pd.DataFrame(
            {"feature": X_test.columns, "shap_value": np.abs(shap_values[1][idx])}
        )
        .sort_values("shap_value", ascending=False)
        .head(10)
    )

    # LIME
    lime_exp = lime_explainer.explain_instance(
        X_test.iloc[idx].values, model.predict_proba, num_features=10
    )
    lime_importance = pd.DataFrame(
        lime_exp.as_list(), columns=["feature", "lime_value"]
    )
    lime_importance["lime_value"] = lime_importance["lime_value"].abs()
    lime_importance = lime_importance.sort_values("lime_value", ascending=False)

    print("\nTop 10 Features:")
    print(f"{'SHAP':<35} {'LIME':<35}")
    print("-" * 70)
    for i in range(10):
        shap_feat = shap_importance.iloc[i]["feature"]
        lime_feat = (
            lime_importance.iloc[i]["feature"] if i < len(lime_importance) else "N/A"
        )
        print(f"{shap_feat:<35} {lime_feat:<35}")

# %% [markdown]
# ## 6. Análisis de Fairness (Introducción)
#
# Una aplicación importante de la interpretabilidad es detectar sesgos.
# Si ciertas features son proxies de atributos protegidos (género, raza, etc.),
# el modelo podría estar discriminando.

# %%
print("\n🔍 ANÁLISIS DE FAIRNESS")
print("=" * 50)

# En el dataset de cáncer no hay atributos demográficos,
# pero podemos simular el análisis

print(
    """
En un dataset real con atributos protegidos:

1. VERIFICAR FEATURES PROXY:
   - ¿El código postal correlaciona con raza?
   - ¿El nombre correlaciona con género?

2. ANALIZAR SHAP POR SUBGRUPOS:
   - Calcular SHAP values separados por grupo demográfico
   - Verificar si las features importantes difieren

3. MÉTRICAS DE FAIRNESS:
   - Demographic Parity: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
   - Equalized Odds: Igual TPR y FPR entre grupos
   - Calibration: Misma precisión de scores entre grupos

4. MITIGACIÓN:
   - Reentrenar sin features sensibles
   - Usar técnicas de fair ML (ej: fairlearn)
   - Post-procesamiento de predicciones
"""
)

# %% [markdown]
# ## 7. Ejercicios Prácticos

# %%
print(
    """
📝 EJERCICIOS

1. SHAP para Regresión:
   - Cargar dataset California Housing
   - Entrenar RandomForestRegressor
   - Calcular y visualizar SHAP values
   - ¿Qué features predicen precios altos vs bajos?

2. Comparar Modelos:
   - Entrenar LogisticRegression y GradientBoosting en el mismo dataset
   - Calcular SHAP para ambos
   - ¿Las features importantes son las mismas?

3. LIME para Texto:
   - Usar LimeTextExplainer con un clasificador de sentimientos
   - Identificar palabras que causan predicciones positivas/negativas

4. Auditoría de Fairness:
   - Descargar dataset Adult Income (UCI)
   - Entrenar modelo de predicción de ingresos
   - Analizar si 'sex' o 'race' influyen en las predicciones
   - Proponer mitigaciones
"""
)

# %% [markdown]
# ## 8. Resumen

# %%
print(
    """
╔══════════════════════════════════════════════════════════════════╗
║                        RESUMEN                                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  SHAP:                                                          ║
║  ✓ Base teórica sólida (Shapley values)                         ║
║  ✓ Explicaciones globales + locales                             ║
║  ✓ Garantías matemáticas (consistency, local accuracy)          ║
║  ✗ Puede ser lento para modelos grandes                         ║
║                                                                  ║
║  LIME:                                                          ║
║  ✓ Agnóstico al modelo (funciona con cualquier clasificador)    ║
║  ✓ Rápido para explicaciones individuales                       ║
║  ✓ Intuitivo (aproximación lineal local)                        ║
║  ✗ Sin garantías teóricas fuertes                               ║
║  ✗ Sensible a hiperparámetros (num_samples, kernel)             ║
║                                                                  ║
║  CUÁNDO USAR:                                                   ║
║  - Auditoría/Compliance → SHAP (más robusto)                    ║
║  - Prototipado rápido → LIME                                    ║
║  - Modelos de árboles → SHAP TreeExplainer (eficiente)          ║
║  - Deep Learning → SHAP DeepExplainer o GradientExplainer       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
)

# %%
print("\n✅ Notebook completado!")
print("   Siguiente: Aplicar estas técnicas en el proyecto Capstone")
