#!/usr/bin/env python3
"""
Script de Validación de Carga Horaria
=====================================

Valida que el plan de estudio cumple con los límites de carga horaria
recomendados para programas de posgrado (5-8 horas por crédito por semana).

Uso:
    python scripts/estimate_hours.py

Referencias:
    - Coursera: típicamente 5-8 horas/semana por curso
    - CU Boulder MS-AI: 3 créditos por curso ≈ 9-15 horas/semana
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Activity:
    """Representa una actividad de estudio."""

    name: str
    hours: float
    category: str  # theory, practice, lab, project, exam


@dataclass
class Week:
    """Representa una semana del plan de estudio."""

    number: int
    module: str
    topic: str
    activities: list[Activity]

    @property
    def total_hours(self) -> float:
        return sum(a.hours for a in self.activities)

    @property
    def hours_by_category(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for a in self.activities:
            result[a.category] = result.get(a.category, 0) + a.hours
        return result


# =============================================================================
# PLAN DE ESTUDIO CON ESTIMACIONES DE HORAS
# =============================================================================

STUDY_PLAN: list[dict[str, Any]] = [
    # Mes 1-2: Fundamentos (Semanas 1-8)
    {
        "week": 1,
        "module": "M01",
        "topic": "Python Científico: NumPy",
        "activities": [
            {"name": "Lectura teoría NumPy", "hours": 2, "category": "theory"},
            {"name": "Notebooks NumPy", "hours": 4, "category": "practice"},
            {"name": "Ejercicios", "hours": 3, "category": "practice"},
            {"name": "Quiz", "hours": 1, "category": "exam"},
        ],
    },
    {
        "week": 2,
        "module": "M01",
        "topic": "Pandas DataFrames",
        "activities": [
            {"name": "Lectura teoría Pandas", "hours": 2, "category": "theory"},
            {"name": "Notebooks Pandas", "hours": 4, "category": "practice"},
            {"name": "Mini-proyecto EDA", "hours": 4, "category": "project"},
        ],
    },
    {
        "week": 3,
        "module": "M02",
        "topic": "Vectores y Matrices",
        "activities": [
            {"name": "Lectura álgebra lineal", "hours": 3, "category": "theory"},
            {"name": "Ejercicios escritos", "hours": 3, "category": "practice"},
            {"name": "Implementación from scratch", "hours": 4, "category": "practice"},
            {"name": "Lab interactivo", "hours": 2, "category": "lab"},
        ],
    },
    {
        "week": 4,
        "module": "M02",
        "topic": "Eigenvalues/Eigenvectors",
        "activities": [
            {"name": "Lectura eigendecomposition", "hours": 3, "category": "theory"},
            {"name": "Derivaciones manuales", "hours": 2, "category": "theory"},
            {"name": "Implementación", "hours": 4, "category": "practice"},
            {"name": "Visualizaciones", "hours": 2, "category": "lab"},
        ],
    },
    {
        "week": 5,
        "module": "M02",
        "topic": "SVD y Aplicaciones",
        "activities": [
            {"name": "Lectura SVD", "hours": 2, "category": "theory"},
            {"name": "Implementación SVD", "hours": 4, "category": "practice"},
            {"name": "Lab interactivo", "hours": 3, "category": "lab"},
            {"name": "Ejercicios PCA preview", "hours": 2, "category": "practice"},
        ],
    },
    {
        "week": 6,
        "module": "M03",
        "topic": "Derivadas y Gradientes",
        "activities": [
            {"name": "Lectura cálculo", "hours": 3, "category": "theory"},
            {"name": "Derivaciones manuales", "hours": 3, "category": "theory"},
            {"name": "Implementación gradientes", "hours": 3, "category": "practice"},
            {"name": "Backprop manual", "hours": 2, "category": "practice"},
        ],
    },
    {
        "week": 7,
        "module": "M03",
        "topic": "Optimización",
        "activities": [
            {"name": "Lectura GD/SGD", "hours": 2, "category": "theory"},
            {"name": "Implementación GD", "hours": 4, "category": "practice"},
            {"name": "Lab optimización", "hours": 3, "category": "lab"},
            {"name": "Ejercicios", "hours": 2, "category": "practice"},
        ],
    },
    {
        "week": 8,
        "module": "M04",
        "topic": "Probabilidad y Bayes",
        "activities": [
            {"name": "Lectura probabilidad", "hours": 3, "category": "theory"},
            {"name": "Lab 1: MLE/MAP", "hours": 3, "category": "lab"},
            {"name": "Lab 2: MCMC", "hours": 3, "category": "lab"},
            {"name": "Lab 3: Markov", "hours": 2, "category": "lab"},
            {"name": "Simulacro teórico", "hours": 1, "category": "exam"},
        ],
    },
    # Mes 3: Supervisado (Semanas 9-11)
    {
        "week": 9,
        "module": "M05",
        "topic": "Regresión Lineal/Logística",
        "activities": [
            {"name": "Lectura regresión", "hours": 2, "category": "theory"},
            {"name": "From scratch", "hours": 4, "category": "practice"},
            {"name": "Paridad Sklearn", "hours": 3, "category": "practice"},
            {"name": "Validación", "hours": 2, "category": "practice"},
        ],
    },
    {
        "week": 10,
        "module": "M05",
        "topic": "Árboles, RF, SVM",
        "activities": [
            {"name": "Lectura árboles", "hours": 2, "category": "theory"},
            {"name": "Implementación Gini", "hours": 3, "category": "practice"},
            {"name": "Random Forest", "hours": 3, "category": "practice"},
            {"name": "SVM", "hours": 2, "category": "practice"},
            {"name": "Comparativa", "hours": 2, "category": "project"},
        ],
    },
    {
        "week": 11,
        "module": "M05",
        "topic": "Ética + XAI",
        "activities": [
            {"name": "Lectura ética IA", "hours": 2, "category": "theory"},
            {"name": "SHAP tutorial", "hours": 3, "category": "practice"},
            {"name": "LIME tutorial", "hours": 2, "category": "practice"},
            {"name": "Reporte interpretabilidad", "hours": 3, "category": "project"},
            {"name": "Simulacro CSCA5622", "hours": 2, "category": "exam"},
        ],
    },
    # Mes 4: No Supervisado (Semanas 12-15)
    {
        "week": 12,
        "module": "M06",
        "topic": "K-Means, Clustering",
        "activities": [
            {"name": "Lectura clustering", "hours": 2, "category": "theory"},
            {"name": "K-Means from scratch", "hours": 4, "category": "practice"},
            {"name": "Jerárquico", "hours": 2, "category": "practice"},
            {"name": "Lab interactivo", "hours": 2, "category": "lab"},
        ],
    },
    {
        "week": 13,
        "module": "M06",
        "topic": "PCA",
        "activities": [
            {"name": "Lectura PCA", "hours": 2, "category": "theory"},
            {"name": "PCA from scratch", "hours": 3, "category": "practice"},
            {"name": "t-SNE visualización", "hours": 3, "category": "practice"},
            {"name": "Ejercicios", "hours": 2, "category": "practice"},
        ],
    },
    {
        "week": 14,
        "module": "M06",
        "topic": "GMM y EM",
        "activities": [
            {"name": "Lectura EM", "hours": 3, "category": "theory"},
            {"name": "Derivación matemática", "hours": 2, "category": "theory"},
            {"name": "GMM implementación", "hours": 4, "category": "practice"},
            {"name": "Visualización", "hours": 2, "category": "lab"},
        ],
    },
    {
        "week": 15,
        "module": "M06",
        "topic": "Sistemas de Recomendación",
        "activities": [
            {"name": "Lectura RecSys", "hours": 2, "category": "theory"},
            {"name": "Matrix Factorization", "hours": 4, "category": "practice"},
            {"name": "Proyecto MovieLens", "hours": 4, "category": "project"},
            {"name": "Simulacro CSCA5632", "hours": 2, "category": "exam"},
        ],
    },
    # Mes 5: Deep Learning (Semanas 16-20)
    {
        "week": 16,
        "module": "M07",
        "topic": "Perceptrón, MLP",
        "activities": [
            {"name": "Lectura MLPs", "hours": 2, "category": "theory"},
            {"name": "Backprop manual", "hours": 4, "category": "practice"},
            {"name": "Implementación", "hours": 4, "category": "practice"},
        ],
    },
    {
        "week": 17,
        "module": "M07",
        "topic": "Keras APIs",
        "activities": [
            {"name": "Lectura Keras", "hours": 2, "category": "theory"},
            {"name": "Sequential API", "hours": 3, "category": "practice"},
            {"name": "Functional API", "hours": 3, "category": "practice"},
            {"name": "Modelo híbrido", "hours": 3, "category": "project"},
        ],
    },
    {
        "week": 18,
        "module": "M07",
        "topic": "CNNs",
        "activities": [
            {"name": "Lectura CNNs", "hours": 2, "category": "theory"},
            {"name": "Conv from scratch", "hours": 3, "category": "practice"},
            {"name": "CIFAR-10", "hours": 4, "category": "project"},
            {"name": "Visualización filtros", "hours": 2, "category": "lab"},
        ],
    },
    {
        "week": 19,
        "module": "M07",
        "topic": "RNNs, LSTMs",
        "activities": [
            {"name": "Lectura RNNs", "hours": 2, "category": "theory"},
            {"name": "RNN from scratch", "hours": 3, "category": "practice"},
            {"name": "LSTM Keras", "hours": 4, "category": "practice"},
            {"name": "Secuencias", "hours": 2, "category": "project"},
        ],
    },
    {
        "week": 20,
        "module": "M07",
        "topic": "Regularización, Transfer",
        "activities": [
            {"name": "Lectura regularización", "hours": 2, "category": "theory"},
            {"name": "Dropout, BatchNorm", "hours": 3, "category": "practice"},
            {"name": "Transfer Learning", "hours": 4, "category": "practice"},
            {"name": "Simulacro CSCA5642", "hours": 2, "category": "exam"},
        ],
    },
    # Mes 6: Capstone (Semanas 21-24)
    {
        "week": 21,
        "module": "M08",
        "topic": "EDA + Preprocessing",
        "activities": [
            {"name": "Exploración datos", "hours": 3, "category": "practice"},
            {"name": "Limpieza texto", "hours": 4, "category": "practice"},
            {"name": "Notebook 01", "hours": 4, "category": "project"},
        ],
    },
    {
        "week": 22,
        "module": "M08",
        "topic": "Baseline Models",
        "activities": [
            {"name": "TF-IDF", "hours": 3, "category": "practice"},
            {"name": "LogReg, NB", "hours": 4, "category": "practice"},
            {"name": "Notebook 02", "hours": 4, "category": "project"},
        ],
    },
    {
        "week": 23,
        "module": "M08",
        "topic": "Deep Learning NLP",
        "activities": [
            {"name": "Embeddings", "hours": 3, "category": "practice"},
            {"name": "BiLSTM", "hours": 4, "category": "practice"},
            {"name": "Notebook 03", "hours": 5, "category": "project"},
        ],
    },
    {
        "week": 24,
        "module": "M08",
        "topic": "BERT + Reporte",
        "activities": [
            {"name": "Fine-tuning BERT", "hours": 4, "category": "practice"},
            {"name": "Notebook 04", "hours": 4, "category": "project"},
            {"name": "REPORT.md", "hours": 4, "category": "project"},
        ],
    },
]


def parse_plan() -> list[Week]:
    """Convierte el plan en objetos Week."""
    weeks = []
    for w in STUDY_PLAN:
        activities = [
            Activity(name=a["name"], hours=a["hours"], category=a["category"])
            for a in w["activities"]
        ]
        weeks.append(
            Week(
                number=w["week"],
                module=w["module"],
                topic=w["topic"],
                activities=activities,
            )
        )
    return weeks


def analyze_workload(weeks: list[Week]) -> dict[str, Any]:
    """Analiza la carga de trabajo del plan."""
    total_hours = sum(w.total_hours for w in weeks)
    hours_per_week = [w.total_hours for w in weeks]

    # Por módulo
    by_module: dict[str, float] = {}
    for w in weeks:
        by_module[w.module] = by_module.get(w.module, 0) + w.total_hours

    # Por categoría
    by_category: dict[str, float] = {}
    for w in weeks:
        for cat, hours in w.hours_by_category.items():
            by_category[cat] = by_category.get(cat, 0) + hours

    return {
        "total_hours": total_hours,
        "total_weeks": len(weeks),
        "avg_hours_per_week": total_hours / len(weeks),
        "min_hours_week": min(hours_per_week),
        "max_hours_week": max(hours_per_week),
        "hours_by_module": by_module,
        "hours_by_category": by_category,
    }


def validate_workload(analysis: dict[str, Any]) -> list[str]:
    """Valida que la carga esté dentro de límites aceptables."""
    warnings = []

    # Límites de Coursera: 5-8 horas por crédito
    # MS-AI: ~3 créditos por curso ≈ 9-15 horas/semana recomendadas
    # Siendo conservadores: 8-12 horas/semana

    MIN_HOURS = 8
    MAX_HOURS = 14
    OPTIMAL_RANGE = (10, 12)

    avg = analysis["avg_hours_per_week"]

    if avg < MIN_HOURS:
        warnings.append(
            f"⚠️ Carga muy baja: {avg:.1f} h/semana (mínimo recomendado: {MIN_HOURS})"
        )
    elif avg > MAX_HOURS:
        warnings.append(
            f"⚠️ Carga excesiva: {avg:.1f} h/semana (máximo recomendado: {MAX_HOURS})"
        )
    elif OPTIMAL_RANGE[0] <= avg <= OPTIMAL_RANGE[1]:
        warnings.append(
            f"✅ Carga óptima: {avg:.1f} h/semana (rango ideal: {OPTIMAL_RANGE[0]}-{OPTIMAL_RANGE[1]})"
        )
    else:
        warnings.append(f"ℹ️ Carga aceptable: {avg:.1f} h/semana")

    # Verificar semanas individuales
    for week in parse_plan():
        if week.total_hours > 15:
            warnings.append(
                f"⚠️ Semana {week.number} ({week.topic}): {week.total_hours}h - considerar redistribuir"
            )

    return warnings


def print_report(weeks: list[Week], analysis: dict[str, Any]) -> None:
    """Imprime un reporte detallado."""
    print("=" * 70)
    print("REPORTE DE CARGA HORARIA - GUÍA MASTER IA")
    print("=" * 70)

    print("\n📊 RESUMEN GENERAL")
    print(f"   Total de horas:        {analysis['total_hours']:.0f} horas")
    print(f"   Total de semanas:      {analysis['total_weeks']} semanas")
    print(f"   Promedio por semana:   {analysis['avg_hours_per_week']:.1f} horas")
    print(
        f"   Rango:                 {analysis['min_hours_week']:.0f} - {analysis['max_hours_week']:.0f} horas"
    )

    print("\n📚 HORAS POR MÓDULO")
    for module, hours in sorted(analysis["hours_by_module"].items()):
        print(f"   {module}: {hours:.0f} horas")

    print("\n📋 HORAS POR CATEGORÍA")
    for cat, hours in sorted(
        analysis["hours_by_category"].items(), key=lambda x: -x[1]
    ):
        pct = 100 * hours / analysis["total_hours"]
        print(f"   {cat:12}: {hours:>5.0f} horas ({pct:>5.1f}%)")

    print("\n🔍 VALIDACIÓN")
    for warning in validate_workload(analysis):
        print(f"   {warning}")

    print("\n📅 DETALLE POR SEMANA")
    print(f"   {'Sem':>3} | {'Módulo':>5} | {'Horas':>5} | Tema")
    print("   " + "-" * 50)
    for w in weeks:
        status = "⚠️" if w.total_hours > 12 else "  "
        print(
            f"   {w.number:>3} | {w.module:>5} | {w.total_hours:>5.0f} | {w.topic} {status}"
        )

    print("\n" + "=" * 70)


def export_json(weeks: list[Week], analysis: dict[str, Any], filename: str) -> None:
    """Exporta el análisis a JSON."""
    data = {
        "analysis": analysis,
        "weeks": [
            {
                "number": w.number,
                "module": w.module,
                "topic": w.topic,
                "total_hours": w.total_hours,
                "activities": [
                    {"name": a.name, "hours": a.hours, "category": a.category}
                    for a in w.activities
                ],
            }
            for w in weeks
        ],
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n📁 Exportado a: {filename}")


if __name__ == "__main__":
    weeks = parse_plan()
    analysis = analyze_workload(weeks)
    print_report(weeks, analysis)
    export_json(weeks, analysis, "plan_de_estudio_horas.json")
