import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

from config import SEED, FIG_DIR, OUTPUT_DIR, ENDPOINT_LABEL, CHALLENGE_NAME

MODEL_COLOURS = {
    "Ridge":              "#2C6E94",
    "Lasso":              "#A85C9E",
    "Random Forest":      "#E07B39",
    "XGBoost":            "#3A9E6F",
    "LightGBM":           "#C0392B",
    "MLP":                "#1ABC9C",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def default_k(n):
    return min(150, n) if n > 150 else "all"

def k_options(n):
    return [k for k in [50, 150] if k < n] or ["all"]

def make_reg_pipeline(model, *, scale: bool, k) -> Pipeline:
    steps = [
        ("imputer",  SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("selector", SelectKBest(score_func=f_regression, k=k)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("reg", model))
    return Pipeline(steps)

def evaluate_regressor(name, model, X_eval, y_eval):
    y_pred = model.predict(X_eval)
    rmse   = np.sqrt(mean_squared_error(y_eval, y_pred))
    mae    = mean_absolute_error(y_eval, y_pred)
    r2     = r2_score(y_eval, y_pred)
    return {"Model": name, "R²": r2, "RMSE": rmse, "MAE": mae}, y_pred

def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → outputs/{CHALLENGE_NAME}/figures/{name}")

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#CCCCCC",
        "axes.grid":        True,
        "grid.color":       "#EEEEEE",
        "grid.linewidth":   0.8,
        "font.family":      "sans-serif",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
    })


# ── Main regression function ──────────────────────────────────────────────────
def run_regression_models(X_train, X_test, y_train_reg, y_test_reg, groups_train, X_full):
    n  = X_full.shape[1]
    fk = k_options(n)
    cv = GroupKFold(n_splits=3)

    search_spaces = {
        "Ridge": (
            make_reg_pipeline(Ridge(), scale=True, k=fk[0]),
            {"selector__k": fk, "reg__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "Lasso": (
            make_reg_pipeline(Lasso(max_iter=5000), scale=True, k=fk[0]),
            {"selector__k": fk, "reg__alpha": [0.001, 0.01, 0.1, 1.0]},
        ),
        "Random Forest": (
            make_reg_pipeline(
                RandomForestRegressor(n_estimators=120, random_state=SEED, n_jobs=1),
                scale=False, k=default_k(n),
            ),
            {"selector__k": [default_k(n)], "reg__max_depth": [12, None], "reg__min_samples_leaf": [1, 10]},
        ),
        "XGBoost": (
            make_reg_pipeline(
                xgb.XGBRegressor(eval_metric="rmse", random_state=SEED, n_jobs=1),
                scale=False, k=default_k(n),
            ),
            {"selector__k": [default_k(n)], "reg__n_estimators": [100, 200],
             "reg__max_depth": [4, 6], "reg__learning_rate": [0.05, 0.1]},
        ),
        "LightGBM": (
            make_reg_pipeline(
                lgb.LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
                scale=False, k=default_k(n),
            ),
            {"selector__k": [default_k(n)], "reg__n_estimators": [100, 200],
             "reg__max_depth": [4, 6], "reg__learning_rate": [0.05, 0.1]},
        ),
        "MLP": (
            make_reg_pipeline(
                MLPRegressor(max_iter=300, early_stopping=True, random_state=SEED),
                scale=True, k=default_k(n),
            ),
            {"selector__k": [default_k(n)],
             "reg__hidden_layer_sizes": [(128, 64), (256, 128)],
             "reg__alpha": [0.0001, 0.001]},
        ),
    }

    print("\n" + "=" * 55)
    print("  Regression Models (predicting log LC50)")
    print("=" * 55)

    best_models  = {}
    results_rows = []
    predictions  = {}

    for name, (pipeline, param_grid) in search_spaces.items():
        print(f"\n  Tuning {name}...")
        search = GridSearchCV(
            pipeline, param_grid,
            scoring="r2", cv=cv,
            n_jobs=1, refit=True, error_score="raise",
        )
        search.fit(X_train, y_train_reg, groups=groups_train)
        best = clone(search.best_estimator_).fit(X_train, y_train_reg)
        best_models[name] = best
        metrics, y_pred = evaluate_regressor(name, best, X_test, y_test_reg)
        results_rows.append(metrics)
        predictions[name] = y_pred
        print(f"    Best CV R²: {search.best_score_:.4f}  |  Params: {search.best_params_}")
        print(f"    Test  R²: {metrics['R²']:.4f}  RMSE: {metrics['RMSE']:.4f}  MAE: {metrics['MAE']:.4f}")

    results_df = pd.DataFrame(results_rows).sort_values("R²", ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "regression_comparison.csv", index=False)

    print("\n  Final regression comparison:")
    print(results_df.round(4).to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_regression_comparison(results_df)
    _plot_predicted_vs_actual(predictions, y_test_reg)
    _plot_residuals(predictions, y_test_reg)
    _plot_shap_regression(best_models, X_test)

    return best_models, results_df


# ── Plot 1: Model comparison bar ──────────────────────────────────────────────
def _plot_regression_comparison(results_df):
    _style()
    metrics = ["R²", "RMSE", "MAE"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric in zip(axes, metrics):
        colours = [MODEL_COLOURS.get(m, "#555555") for m in results_df["Model"]]
        bars = ax.barh(results_df["Model"], results_df[metric], color=colours, alpha=0.85)
        ax.set_title(metric, fontsize=12)
        ax.invert_yaxis()
        for bar, val in zip(bars, results_df[metric]):
            ax.text(val + results_df[metric].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

    fig.suptitle(f"Regression Model Comparison — {CHALLENGE_NAME}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "regression_model_comparison.png")


# ── Plot 2: Predicted vs actual (one panel per model) ─────────────────────────
def _plot_predicted_vs_actual(predictions, y_test_reg):
    _style()
    n     = len(predictions)
    cols  = 3
    rows  = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes  = axes.flatten()

    y_true = np.array(y_test_reg)

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        colour = MODEL_COLOURS.get(name, "#2C6E94")
        r2     = r2_score(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=colour)

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#AAAAAA", lw=1.2)

        ax.set_xlabel("Actual log(LC50)")
        ax.set_ylabel("Predicted log(LC50)")
        ax.set_title(f"{name}  |  R²={r2:.3f}", fontsize=11)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Predicted vs Actual — {CHALLENGE_NAME}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "regression_predicted_vs_actual.png")


# ── Plot 3: Residual plots ────────────────────────────────────────────────────
def _plot_residuals(predictions, y_test_reg):
    _style()
    n     = len(predictions)
    cols  = 3
    rows  = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes  = axes.flatten()

    y_true = np.array(y_test_reg)

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        colour    = MODEL_COLOURS.get(name, "#2C6E94")
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.3, s=10, color=colour)
        ax.axhline(0, color="#AAAAAA", linestyle="--", lw=1.2)
        ax.set_xlabel("Predicted log(LC50)")
        ax.set_ylabel("Residual")
        ax.set_title(f"{name}", fontsize=11)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Residual Plots — {CHALLENGE_NAME}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "regression_residuals.png")


# ── Plot 4: SHAP for regression ───────────────────────────────────────────────
def _get_reg_feature_names(pipeline, X):
    names = np.asarray(X.columns)
    if "variance" in pipeline.named_steps:
        names = names[pipeline.named_steps["variance"].get_support()]
    if "selector" in pipeline.named_steps:
        names = names[pipeline.named_steps["selector"].get_support()]
    return list(names)

def _transform_reg(pipeline, X):
    Xt = X.copy()
    for _, step in pipeline.steps[:-1]:
        Xt = step.transform(Xt)
    feat_names = _get_reg_feature_names(pipeline, X)
    if not isinstance(Xt, pd.DataFrame):
        Xt = pd.DataFrame(Xt, columns=feat_names)
    return Xt

def _plot_shap_regression(best_models, X_test):
    _style()
    print("\n── SHAP for Regression models ──")
    TREE_REG = {"Random Forest", "XGBoost", "LightGBM"}

    for name, pipeline in best_models.items():
        try:
            X_sample = X_test.iloc[:300] if len(X_test) > 300 else X_test
            X_trans  = _transform_reg(pipeline, X_sample)
            clf      = pipeline.named_steps["reg"]
            feat_names = list(X_trans.columns)

            if name in TREE_REG:
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_trans)
            else:
                bg = shap.sample(X_trans, min(100, len(X_trans)))
                explainer = shap.KernelExplainer(
                    lambda x: clf.predict(pd.DataFrame(x, columns=feat_names)), bg
                )
                sv = explainer.shap_values(X_trans, silent=True)

            if isinstance(sv, list):
                sv = sv[0]

            mean_abs = np.abs(sv).mean(axis=0)
            top_idx  = np.argsort(mean_abs)[::-1][:20]
            colour   = MODEL_COLOURS.get(name, "#2C6E94")

            fig, ax = plt.subplots(figsize=(9, 6))
            y_pos   = np.arange(len(top_idx))
            bars    = ax.barh(y_pos, mean_abs[top_idx], color=colour, alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP value|  (impact on predicted log LC50)")
            ax.set_title(f"SHAP Feature Importance (Regression) — {name}\n{CHALLENGE_NAME}")
            for bar, val in zip(bars, mean_abs[top_idx]):
                ax.text(val + mean_abs.max() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=7)
            fig.tight_layout()
            _save(fig, f"shap_regression_{name.replace(' ', '_')}.png")

        except Exception as e:
            print(f"  Skipped {name} regression SHAP: {e}")