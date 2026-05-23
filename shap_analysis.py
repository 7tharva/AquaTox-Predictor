import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap

from config import FIG_DIR, CHALLENGE_NAME, ENDPOINT_LABEL

# ── Model colour palette ──────────────────────────────────────────────────────
MODEL_COLOURS = {
    "Random Forest":       "#2C6E94",
    "Logistic Regression": "#A85C9E",
    "XGBoost":             "#E07B39",
    "LightGBM":            "#3A9E6F",
    "MLP":                 "#C0392B",
    "Gaussian NB":         "#7F8C8D",
    "KNN":                 "#1ABC9C",
}

TREE_MODELS   = {"Random Forest", "XGBoost", "LightGBM"}
KERNEL_MODELS = {"Logistic Regression", "Gaussian NB", "KNN", "MLP"}

MAX_SAMPLES   = 300   # rows used for SHAP (keeps runtime reasonable)
MAX_BG        = 100   # background rows for KernelExplainer


# ── Shared style ──────────────────────────────────────────────────────────────
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
    })


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → outputs/{CHALLENGE_NAME}/figures/{name}")


# ── Pipeline utilities ────────────────────────────────────────────────────────
def _transform_X(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Run every pipeline step except the final clf, return a named DataFrame."""
    from sklearn.pipeline import Pipeline as SKPipeline
    Xt = X.copy()
    for step_name, step in pipeline.steps[:-1]:
        Xt = step.transform(Xt)
    feat_names = _get_feature_names(pipeline, X)
    if not isinstance(Xt, pd.DataFrame):
        Xt = pd.DataFrame(Xt, columns=feat_names)
    else:
        Xt.columns = feat_names[:Xt.shape[1]]
    return Xt


def _get_feature_names(pipeline, X_original: pd.DataFrame) -> list:
    """Return column names that survive variance + selector steps."""
    names = np.asarray(X_original.columns)
    if "variance" in pipeline.named_steps:
        names = names[pipeline.named_steps["variance"].get_support()]
    if "selector" in pipeline.named_steps:
        names = names[pipeline.named_steps["selector"].get_support()]
    return list(names)


def _get_clf(pipeline):
    """Extract the final estimator from a pipeline."""
    from sklearn.pipeline import Pipeline as SKPipeline
    if isinstance(pipeline, SKPipeline):
        return pipeline.named_steps["clf"]
    return pipeline


# ── SHAP value computation ────────────────────────────────────────────────────
def _compute_shap(model_name: str, pipeline, X_test: pd.DataFrame):
    """
    Returns (shap_values_2d, X_transformed_df, feature_names).
    shap_values_2d has shape (n_samples, n_features).
    """
    from sklearn.pipeline import Pipeline as SKPipeline

    # Sample rows for speed
    X_sample = X_test.iloc[:MAX_SAMPLES] if len(X_test) > MAX_SAMPLES else X_test
    X_trans  = _transform_X(pipeline, X_sample)
    clf      = _get_clf(pipeline)
    feat_names = list(X_trans.columns)

    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_trans)
        # Binary tree models return list [class0, class1]
        if isinstance(sv, list):
            sv = sv[1]

    else:
        # KernelExplainer — use a small background set
        bg = shap.sample(X_trans, min(MAX_BG, len(X_trans)))
        explainer = shap.KernelExplainer(
            lambda x: clf.predict_proba(
                pd.DataFrame(x, columns=feat_names)
            )[:, 1],
            bg,
        )
        sv = explainer.shap_values(X_trans, silent=True)
        if isinstance(sv, list):
            sv = sv[1]

    return np.array(sv), X_trans, feat_names


# ── 1. Summary bar plots (mean |SHAP|) ───────────────────────────────────────
def plot_shap_summary_bar(best_models: dict, X_test: pd.DataFrame):
    _style()
    print("\n── SHAP Summary Bar plots ──")

    for name, pipeline in best_models.items():
        try:
            sv, X_trans, feat_names = _compute_shap(name, pipeline, X_test)
            mean_abs = np.abs(sv).mean(axis=0)
            top_idx  = np.argsort(mean_abs)[::-1][:20]

            fig, ax = plt.subplots(figsize=(9, 6))
            colour  = MODEL_COLOURS.get(name, "#2C6E94")
            y_pos   = np.arange(len(top_idx))

            bars = ax.barh(y_pos, mean_abs[top_idx], color=colour, alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP value|  (impact on model output)")
            ax.set_title(f"SHAP Feature Importance — {name}\n{CHALLENGE_NAME}", fontsize=12)

            # Value labels on bars
            for bar, val in zip(bars, mean_abs[top_idx]):
                ax.text(val + mean_abs.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=7, color="#444444")

            fig.tight_layout()
            _save(fig, f"shap_summary_bar_{name.replace(' ', '_')}.png")

        except Exception as e:
            print(f"  Skipped {name} summary bar: {e}")


# ── 2. Beeswarm plots ─────────────────────────────────────────────────────────
def plot_shap_beeswarm(best_models: dict, X_test: pd.DataFrame):
    _style()
    print("\n── SHAP Beeswarm plots ──")

    for name, pipeline in best_models.items():
        try:
            sv, X_trans, feat_names = _compute_shap(name, pipeline, X_test)

            # Keep top 15 features by mean |SHAP|
            mean_abs  = np.abs(sv).mean(axis=0)
            top_idx   = np.argsort(mean_abs)[::-1][:15]
            sv_top    = sv[:, top_idx]
            X_top     = X_trans.iloc[:, top_idx]
            names_top = [feat_names[i] for i in top_idx]

            explanation = shap.Explanation(
                values        = sv_top,
                data          = X_top.values,
                feature_names = names_top,
            )

            fig, ax = plt.subplots(figsize=(10, 7))
            # Use matplotlib beeswarm manually so we control the axes
            _draw_beeswarm(ax, sv_top, X_top.values, names_top, name)
            ax.set_title(f"SHAP Beeswarm — {name}\n{CHALLENGE_NAME}", fontsize=12)
            fig.tight_layout()
            _save(fig, f"shap_beeswarm_{name.replace(' ', '_')}.png")

        except Exception as e:
            print(f"  Skipped {name} beeswarm: {e}")


def _draw_beeswarm(ax, shap_vals, data_vals, feat_names, model_name, n_bins=30):
    """
    Manual beeswarm: dots jittered vertically within each feature row,
    coloured by feature value (blue = low, pink/red = high).
    """
    n_features = shap_vals.shape[1]
    cmap = plt.cm.coolwarm

    for fi in range(n_features - 1, -1, -1):   # plot top feature at top
        row_y    = fi
        sv_row   = shap_vals[:, fi]
        dv_row   = data_vals[:, fi]

        # Normalise feature values for colour
        dv_min, dv_max = dv_row.min(), dv_row.max()
        if dv_max > dv_min:
            norm_dv = (dv_row - dv_min) / (dv_max - dv_min)
        else:
            norm_dv = np.full_like(dv_row, 0.5)

        colours = cmap(norm_dv)

        # Bin SHAP values and jitter y within each bin
        bins   = np.linspace(sv_row.min(), sv_row.max() + 1e-9, n_bins + 1)
        jitter = np.zeros(len(sv_row))
        for b in range(n_bins):
            mask = (sv_row >= bins[b]) & (sv_row < bins[b + 1])
            cnt  = mask.sum()
            if cnt > 1:
                spread = 0.35 * min(cnt, 10) / 10
                jitter[mask] = np.linspace(-spread, spread, cnt)

        ax.scatter(sv_row, row_y + jitter, c=colours, s=8, alpha=0.7, linewidths=0)

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feat_names[::-1], fontsize=8)
    ax.axvline(0, color="#999999", linewidth=0.9, linestyle="--")
    ax.set_xlabel("SHAP value  (impact on model output)")

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("Feature value", fontsize=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"], fontsize=7)


# ── 3. Cross-model SHAP comparison ───────────────────────────────────────────
def plot_shap_model_comparison(best_models: dict, X_test: pd.DataFrame, top_n: int = 15):
    _style()
    print("\n── SHAP Cross-Model Comparison ──")

    importance_dict = {}
    for name, pipeline in best_models.items():
        try:
            sv, X_trans, feat_names = _compute_shap(name, pipeline, X_test)
            mean_abs = np.abs(sv).mean(axis=0)
            importance_dict[name] = pd.Series(
                mean_abs, index=feat_names[:len(mean_abs)]
            )
            print(f"  Computed SHAP for {name} ✓")
        except Exception as e:
            print(f"  Skipped {name} in comparison: {e}")

    if not importance_dict:
        print("  No SHAP values computed — skipping comparison plot.")
        return

    # Align all models on the union of feature names
    all_features = sorted(set().union(*[set(s.index) for s in importance_dict.values()]))
    df = pd.DataFrame({
        name: s.reindex(all_features, fill_value=0)
        for name, s in importance_dict.items()
    })
    df["_max"] = df.max(axis=1)
    df = df.sort_values("_max", ascending=False).drop(columns="_max").head(top_n)

    models   = list(df.columns)
    features = list(df.index)
    x        = np.arange(len(features))
    n        = len(models)
    width    = 0.72 / n

    fig, ax = plt.subplots(figsize=(14, 6.5))
    for i, mname in enumerate(models):
        offset = (i - n / 2 + 0.5) * width
        colour = MODEL_COLOURS.get(mname, "#555555")
        ax.bar(x + offset, df[mname].values, width,
               label=mname, color=colour, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance — All Models Compared\n{CHALLENGE_NAME}", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.axhline(0, color="#AAAAAA", linewidth=0.8)
    fig.tight_layout()
    _save(fig, "shap_model_comparison.png")


# ── Entry point ───────────────────────────────────────────────────────────────
def run_shap_analysis(best_models: dict, X_test: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  Running SHAP Analysis")
    print("  (KernelExplainer models may take a few minutes)")
    print("=" * 55)
    plot_shap_summary_bar(best_models, X_test)
    plot_shap_beeswarm(best_models, X_test)
    plot_shap_model_comparison(best_models, X_test)
    print("\n  SHAP analysis complete!")