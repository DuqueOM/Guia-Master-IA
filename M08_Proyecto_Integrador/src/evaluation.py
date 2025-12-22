"""
Módulo de Evaluación para NLP Disaster Tweets.

Contiene funciones para calcular métricas y generar visualizaciones
de rendimiento de modelos de clasificación.

Uso:
    from src.evaluation import evaluate_model, plot_confusion_matrix

    metrics = evaluate_model(y_true, y_pred, y_proba)
    plot_confusion_matrix(y_true, y_pred)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    average: str = "binary",
) -> dict[str, float]:
    """
    Calcular métricas completas de evaluación.

    Parámetros:
    -----------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_pred : np.ndarray
        Predicciones del modelo.
    y_proba : np.ndarray, opcional
        Probabilidades de predicción (para ROC-AUC).
    average : str
        Tipo de promedio para métricas multiclase.

    Retorna:
    --------
    Dict[str, float]
        Diccionario con todas las métricas.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if y_proba is not None:
        try:
            if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                metrics["avg_precision"] = average_precision_score(
                    y_true, y_proba[:, 1]
                )
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                metrics["avg_precision"] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None
            metrics["avg_precision"] = None

    return metrics


def print_metrics(metrics: dict[str, float], model_name: str = "Model") -> None:
    """Imprimir métricas de forma formateada."""
    print(f"\n{'='*50}")
    print(f"Métricas de Evaluación: {model_name}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        if value is not None:
            print(f"  {metric:20s}: {value:.4f}")
        else:
            print(f"  {metric:20s}: N/A")
    print(f"{'='*50}\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, str] = ("No Disaster", "Disaster"),
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualizar matriz de confusión.

    Parámetros:
    -----------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_pred : np.ndarray
        Predicciones del modelo.
    labels : List[str]
        Nombres de las clases.
    title : str
        Título del gráfico.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    cmap : str
        Colormap a usar.
    save_path : str, opcional
        Ruta para guardar la figura.

    Retorna:
    --------
    plt.Figure
        Figura de matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 14},
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Añadir porcentajes
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total * 100
            ax.text(
                j + 0.5,
                i + 0.7,
                f"({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: tuple[int, int] = (8, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualizar curva ROC.

    Parámetros:
    -----------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_proba : np.ndarray
        Probabilidades de predicción.
    title : str
        Título del gráfico.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    save_path : str, opcional
        Ruta para guardar la figura.

    Retorna:
    --------
    plt.Figure
        Figura de matplotlib.
    """
    if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
        y_proba = y_proba[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    figsize: tuple[int, int] = (8, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualizar curva Precision-Recall.
    """
    if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
        y_proba = y_proba[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        recall,
        precision,
        "b-",
        linewidth=2,
        label=f"PR Curve (AP = {avg_precision:.3f})",
    )

    # Línea base (proporción de positivos)
    baseline = y_true.mean()
    ax.axhline(
        y=baseline,
        color="r",
        linestyle="--",
        linewidth=1,
        label=f"Baseline (Positive Rate = {baseline:.3f})",
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_learning_curves(
    history,
    metrics: list[str] | None = None,
    figsize: tuple[int, int] = (14, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualizar curvas de aprendizaje de entrenamiento Keras.

    Parámetros:
    -----------
    history : keras.callbacks.History
        Objeto history del entrenamiento.
    metrics : List[str]
        Lista de métricas a graficar.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    save_path : str, opcional
        Ruta para guardar la figura.

    Retorna:
    --------
    plt.Figure
        Figura de matplotlib.
    """
    metric_list = metrics or ["loss", "accuracy"]

    fig, axes = plt.subplots(1, len(metric_list), figsize=figsize)

    if len(metric_list) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_list, strict=True):
        if metric in history.history:
            ax.plot(history.history[metric], label=f"Train {metric}")

        val_metric = f"val_{metric}"
        if val_metric in history.history:
            ax.plot(history.history[val_metric], label=f"Validation {metric}")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f"{metric.capitalize()} over Epochs", fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compare_models(
    results: dict[str, dict[str, float]],
    metric: str = "f1_score",
    figsize: tuple[int, int] = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Comparar múltiples modelos en un gráfico de barras.

    Parámetros:
    -----------
    results : Dict[str, Dict[str, float]]
        Diccionario {nombre_modelo: {métrica: valor}}.
    metric : str
        Métrica a comparar.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    save_path : str, opcional
        Ruta para guardar la figura.

    Retorna:
    --------
    plt.Figure
        Figura de matplotlib.
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(models, values, color=colors, edgecolor="black")

    # Añadir valores encima de las barras
    for bar, val in zip(bars, values, strict=True):
        height = bar.get_height()
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=14)
    ax.set_ylim(0, max(values) * 1.15)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def analyze_errors(
    texts: list[str], y_true: np.ndarray, y_pred: np.ndarray, n_examples: int = 10
) -> tuple[list[tuple], list[tuple]]:
    """
    Analizar errores del modelo (falsos positivos y falsos negativos).

    Parámetros:
    -----------
    texts : List[str]
        Textos originales.
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_pred : np.ndarray
        Predicciones del modelo.
    n_examples : int
        Número de ejemplos a retornar por tipo de error.

    Retorna:
    --------
    Tuple[List[Tuple], List[Tuple]]
        (falsos_positivos, falsos_negativos) cada uno como lista de (texto, true, pred).
    """
    false_positives = []
    false_negatives = []

    for text, true, pred in zip(texts, y_true, y_pred, strict=True):
        if true == 0 and pred == 1:
            false_positives.append((text, true, pred))
        elif true == 1 and pred == 0:
            false_negatives.append((text, true, pred))

    return false_positives[:n_examples], false_negatives[:n_examples]


def print_error_analysis(
    texts: list[str], y_true: np.ndarray, y_pred: np.ndarray, n_examples: int = 5
) -> None:
    """Imprimir análisis de errores formateado."""
    fps, fns = analyze_errors(texts, y_true, y_pred, n_examples)

    print("\n" + "=" * 70)
    print("ANÁLISIS DE ERRORES")
    print("=" * 70)

    print(f"\n📛 FALSOS POSITIVOS (Predijo Desastre, era No-Desastre): {len(fps)}")
    print("-" * 70)
    for idx, (text, true_label, predicted_label) in enumerate(fps, 1):
        print(f"{idx}. Pred={predicted_label}, Real={true_label} → {text[:100]}...")
        print()

    print(f"\n📛 FALSOS NEGATIVOS (Predijo No-Desastre, era Desastre): {len(fns)}")
    print("-" * 70)
    for idx, (text, true_label, predicted_label) in enumerate(fns, 1):
        print(f"{idx}. Pred={predicted_label}, Real={true_label} → {text[:100]}...")
        print()


if __name__ == "__main__":
    # Demo con datos sintéticos
    demo_rng = np.random.default_rng(seed=42)
    y_true = demo_rng.integers(0, 2, 100)
    y_pred = demo_rng.integers(0, 2, 100)
    y_proba = demo_rng.random(100)

    metrics = evaluate_model(y_true, y_pred, y_proba)
    print_metrics(metrics, "Demo Model")
