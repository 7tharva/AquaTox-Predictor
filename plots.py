import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, RocCurveDisplay, roc_auc_score
)

from config import FIG_DIR, ENDPOINT_LABEL, CHALLENGE_NAME

# ── Shared style ──────────────────────────────────────────────────────────────
PALETTE   = ["#2C6E94", "#E07B39", "#3A9E6F", "#A85C9E"]
MODEL_COLOURS = {
    "Random Forest":      "#2C6E94",
    "Gaussian NB":        "#E07B39",
    "KNN":                "#3A9E6F",
    "Logistic Regression":"#A85C9E",
}

def _style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#CCCCCC",
        "axes.grid":         True,
        "grid.color":        "#EEEEEE",
        "grid.linewidth":    0.8,
        "font.family":       "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.frameon":    True,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#DDDDDD",
    })

def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → outputs/{CHALLENGE_NAME}/figures/{name}")


# ── 1. Model comparison bar chart ─────────────────────────────────────────────
def plot_model_comparison(comparison_df: pd.DataFrame):
    _style()
    metrics  = ["Accuracy", "Macro F1", "ROC AUC"]
    n_models = len(comparison_df)
    x        = np.arange(n_models)
    width    = 0.22
    offsets  = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (metric, offset) in enumerate(zip(metrics, offsets)):
        values = comparison_df[metric].values
        bars   = ax.barh(
            x + offset, values, width,
            label=metric,
            color=PALETTE[i],
            alpha=0.88,
        )
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=8, color="#333333",
            )

    ax.set_yticks(x)
    ax.set_yticklabels(comparison_df["Model"], fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score")
    ax.set_title(f"Model Comparison — {CHALLENGE_NAME}")
    ax.legend(loc="lower right")
    ax.axvline(0.5, color="#AAAAAA", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    _save(fig, "model_comparison.png")


# ── 2. ROC curves ─────────────────────────────────────────────────────────────
def plot_roc_curves(best_models: dict, comparison_df: pd.DataFrame, X_test, y_test):
    _style()
    fig, ax = plt.subplots(figsize=(7, 6.5))

    for name in comparison_df["Model"]:
        colour = MODEL_COLOURS.get(name, "#555555")
        RocCurveDisplay.from_estimator(
            best_models[name], X_test, y_test,
            ax=ax, name=name, color=colour, lw=2, alpha=0.9,
        )

    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1.2, label="Random")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="#AAAAAA")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {CHALLENGE_NAME}")
    ax.legend(loc="lower right")

    fig.tight_layout()
    _save(fig, f"roc_curves_{CHALLENGE_NAME.replace('-','_')}.png")


# ── 3. Confusion matrix for every model ───────────────────────────────────────
def plot_all_confusion_matrices(best_models: dict, X_test, y_test):
    _style()
    names    = list(best_models.keys())
    n        = len(names)
    cols     = 2
    rows     = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.8 * rows))
    axes      = axes.flatten()

    labels = [f"{ENDPOINT_LABEL} > 1", f"{ENDPOINT_LABEL} ≤ 1"]

    for ax, name in zip(axes, names):
        cm     = confusion_matrix(y_test, best_models[name].predict(X_test))
        colour = MODEL_COLOURS.get(name, "#2C6E94")

        # Custom colourmap per model tinted to its brand colour
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(name, ["#FFFFFF", colour], N=256)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor="#DDDDDD",
            cbar=False, ax=ax,
        )

        # Overlay percentages
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.72,
                    f"({100 * cm[i,j] / total:.1f}%)",
                    ha="center", va="center",
                    fontsize=8, color="#555555",
                )

        auc = roc_auc_score(y_test, best_models[name].predict_proba(X_test)[:, 1]
                            if hasattr(best_models[name], "predict_proba")
                            else best_models[name].decision_function(X_test))
        ax.set_title(f"{name}  |  AUC {auc:.3f}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    # Hide any spare axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Confusion Matrices — {CHALLENGE_NAME}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, f"confusion_matrices_all_{CHALLENGE_NAME.replace('-','_')}.png")


# ── 4. Coefficient comparison ─────────────────────────────────────────────────
def plot_coefficient_comparison(coef_compare: pd.DataFrame):
    _style()
    top    = coef_compare.head(20).copy()
    models = [c for c in top.columns if c != "feature"]
    x      = np.arange(len(top))
    n      = len(models)
    width  = 0.72 / n

    fig, ax = plt.subplots(figsize=(13, 6.5))

    for i, (model, colour) in enumerate(zip(models, PALETTE)):
        offset = (i - n / 2 + 0.5) * width
        ax.bar(
            x + offset,
            top[model].fillna(0),
            width,
            label=model,
            color=colour,
            alpha=0.85,
        )

    ax.axhline(0, color="#999999", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(top["feature"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Standardised Coefficient")
    ax.set_title(f"Top 20 Feature Coefficients — {CHALLENGE_NAME}")
    ax.legend()

    fig.tight_layout()
    _save(fig, "logistic_coefficient_comparison.png")