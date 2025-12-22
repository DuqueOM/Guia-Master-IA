# Módulo 05: Aprendizaje Supervisado

> **Semanas:** 9-11 | **Fase:** ML Core ⭐ | **Curso Alineado:** CSCA 5622

---

## 📁 Estructura

```
M05_Aprendizaje_Supervisado/
├── Teoria/
│   ├── 01_regresion_lineal.md
│   ├── 02_regresion_logistica.md
│   ├── 03_regularizacion_l1_l2.md
│   ├── 04_arboles_ensembles.md
│   └── 05_etica_xai.md                    # NUEVO: Ética e Interpretabilidad
├── Notebooks/
│   ├── 01_regresion_lineal_scratch.ipynb
│   ├── 01b_regresion_lineal_sklearn.ipynb # NUEVO: Paridad Scikit-Learn
│   ├── 02_regresion_logistica_scratch.ipynb
│   ├── 02b_regresion_logistica_sklearn.ipynb
│   ├── 03_regularizacion.ipynb
│   ├── 04_arboles_decision_scratch.ipynb
│   ├── 04b_arboles_ensembles_sklearn.ipynb
│   └── 05_shap_lime_interpretabilidad.ipynb # NUEVO: XAI
├── Laboratorios_Interactivos/
│   ├── overfitting_bias_variance_app.py
│   ├── visualizacion_regresion.py
│   └── shap_explainer_app.py              # NUEVO
└── assets/
```

---

## 🎯 Objetivos de Aprendizaje

### Semana 9-10: Modelos Lineales (From Scratch → Production Ready)

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Implementar regresión lineal desde cero | Normal Equation + Gradient Descent funcionando |
| Implementar regresión logística desde cero | Cross-Entropy loss convergiendo |
| **Replicar resultados con Scikit-Learn** | Coeficientes coinciden ±0.01 con `sklearn.linear_model` |
| Dominar regularización L1/L2 | Explicar trade-off bias-variance con ejemplos |

### Semana 10: Árboles y Ensembles

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Implementar árbol de decisión desde cero | Information Gain / Gini funcionando |
| **Usar `sklearn.tree` y `sklearn.ensemble`** | Random Forest con GridSearchCV |
| Entender bagging vs boosting | Comparar RF vs XGBoost en dataset real |

### Semana 11: Ética en IA e Interpretabilidad (XAI) 🆕

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Comprender sesgo algorítmico (Bias/Fairness) | Identificar bias en dataset COMPAS o similar |
| Implementar SHAP values | Explicar predicciones de modelo de caja negra |
| Implementar LIME | Generar explicaciones locales interpretables |
| Documentar consideraciones éticas | Checklist de fairness para modelos ML |

---

## 📚 Lecturas Obligatorias (Semana 11 - Ética)

1. **"Machine Bias" (ProPublica)** - Caso COMPAS y sesgo racial
2. **Documentación SHAP** - https://shap.readthedocs.io/
3. **"Fairness and Machine Learning" (Barocas & Hardt)** - Capítulos 1-2

---

## ⚡ Inicio Rápido

```bash
# Semana 9: Regresión Lineal
jupyter notebook Notebooks/01_regresion_lineal_scratch.ipynb
jupyter notebook Notebooks/01b_regresion_lineal_sklearn.ipynb  # Validar paridad

# Semana 10: Árboles
jupyter notebook Notebooks/04_arboles_decision_scratch.ipynb
jupyter notebook Notebooks/04b_arboles_ensembles_sklearn.ipynb

# Semana 11: Ética y XAI
jupyter notebook Notebooks/05_shap_lime_interpretabilidad.ipynb
streamlit run Laboratorios_Interactivos/shap_explainer_app.py
```

---

## ✅ Entregables del Módulo

- [ ] `linear_regression.py` con tests (from scratch)
- [ ] `logistic_regression.py` con tests (from scratch)
- [ ] Notebook de paridad: resultados manuales == sklearn
- [ ] `decision_tree.py` con tests (from scratch)
- [ ] Análisis SHAP de un modelo Random Forest
- [ ] Documento de reflexión ética (500 palabras)

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [M04 Probabilidad](../M04_Probabilidad_Estadistica/) | [README](../README.md) | [M06 No Supervisado →](../M06_Aprendizaje_No_Supervisado/) |
