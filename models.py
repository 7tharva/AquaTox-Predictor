import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

from config import SEED, FIG_DIR, OUTPUT_DIR, ENDPOINT_LABEL, ENDPOINT_FILE_STEM, CHALLENGE_NAME


# ── Pipeline builders ─────────────────────────────────────────────────────────

def default_k(n_features: int):
    return min(150, n_features) if n_features > 150 else "all"

def k_options(n_features: int):
    options = [k for k in [50, 150] if k < n_features]
    return options if options else ["all"]


def make_pipeline(model, *, scale: bool, k=None) -> Pipeline:
    selected_k = default_k(0) if k is None else k
    steps = [
        ("imputer",  SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("selector", SelectKBest(score_func=f_classif, k=selected_k)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps)


def make_logistic_no_selector(*, penalty: str, C: float) -> Pipeline:
    return Pipeline([
        ("imputer",  SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler",   StandardScaler()),
        ("clf", LogisticRegression(
            penalty=penalty, C=C,
            solver="liblinear", max_iter=5000, random_state=SEED,
        )),
    ])


# ── Helpers ───────────────────────────────────────────────────────────────────

def selected_feature_names(pipeline: Pipeline, original_columns) -> list:
    names = np.asarray(original_columns)
    if "variance" in pipeline.named_steps:
        names = names[pipeline.named_steps["variance"].get_support()]
    if "selector" in pipeline.named_steps:
        names = names[pipeline.named_steps["selector"].get_support()]
    return list(names)


def positive_probabilities(model, X_eval) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_eval)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_eval)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"{type(model).__name__} has no probability output.")


def evaluate_classifier(name: str, model, X_eval, y_eval) -> dict:
    y_pred = model.predict(X_eval)
    y_prob = positive_probabilities(model, X_eval)
    return {
        "Model":    name,
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Macro F1": f1_score(y_eval, y_pred, average="macro"),
        "Log Loss": log_loss(y_eval, y_prob, labels=[0, 1]),
        "ROC AUC":  roc_auc_score(y_eval, y_prob),
    }


def save_figure(fig, path) -> None:
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ── Main training & evaluation function ──────────────────────────────────────

def run_all_models(X_train, X_test, y_train, y_test, groups_train, X_full):
    n = X_full.shape[1]
    feature_k_grid = k_options(n)
    knn_k_grid = [k for k in [50, 150] if k < n] or ["all"]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    # ── Baseline logistic regression ──────────────────────────────────────────
    print("\n── Baseline Logistic Regression ──")
    baseline = make_pipeline(
        LogisticRegression(penalty="l2", C=1e6, solver="liblinear", max_iter=5000, random_state=SEED),
        scale=True, k=default_k(n),
    )
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_classifier("Weakly regularised LR", baseline, X_test, y_test)
    for k, v in baseline_metrics.items():
        if k != "Model":
            print(f"  {k}: {v:.4f}")
    print("\n" + classification_report(
        y_test, baseline.predict(X_test),
        target_names=[f"{ENDPOINT_LABEL} > 1 mg/L", f"{ENDPOINT_LABEL} <= 1 mg/L"]
    ))

    baseline_features = selected_feature_names(baseline, X_full.columns)

    # ── Coefficient comparison ────────────────────────────────────────────────
    print("── Coefficient Comparison (Baseline / L2 / L1) ──")
    X_train_coef = X_train[baseline_features]
    coef_models = {
        "Baseline (C=1e6)": make_logistic_no_selector(penalty="l2", C=1e6),
        "L2 (C=1.0)":       make_logistic_no_selector(penalty="l2", C=1.0),
        "L1 (C=1.0)":       make_logistic_no_selector(penalty="l1", C=1.0),
    }
    coef_tables = []
    for name, model in coef_models.items():
        model.fit(X_train_coef, y_train)
        feats = np.asarray(baseline_features)[model.named_steps["variance"].get_support()]
        coef_tables.append(pd.DataFrame({"feature": feats, name: model.named_steps["clf"].coef_[0]}))

    coef_compare = coef_tables[0]
    for t in coef_tables[1:]:
        coef_compare = coef_compare.merge(t, on="feature", how="outer")
    coef_compare["max_abs"] = coef_compare.drop(columns="feature").abs().max(axis=1)
    coef_compare = coef_compare.sort_values("max_abs", ascending=False).drop(columns="max_abs")
    coef_compare.to_csv(OUTPUT_DIR / "logistic_coefficient_comparison.csv", index=False)
    print(coef_compare.head(15).round(4).to_string(index=False))

    # ── L2 regularisation sweep ───────────────────────────────────────────────
    print("\n── L2 Regularisation Sweep ──")
    sweep_rows = []
    for C in [100, 10, 1, 0.1, 0.01, 0.001]:
        m = make_pipeline(
            LogisticRegression(penalty="l2", C=C, solver="liblinear", max_iter=5000, random_state=SEED),
            scale=True, k=default_k(n),
        )
        m.fit(X_train, y_train)
        row = evaluate_classifier(f"L2 C={C:g}", m, X_test, y_test)
        row["C"] = C
        sweep_rows.append(row)
    sweep_df = pd.DataFrame(sweep_rows)[["C", "Accuracy", "Macro F1", "Log Loss", "ROC AUC"]]
    sweep_df.to_csv(OUTPUT_DIR / "l2_regularisation_sweep.csv", index=False)
    print(sweep_df.round(4).to_string(index=False))

    # ── Multi-model grid search ───────────────────────────────────────────────
    print("\n── Multi-Model Grid Search ──")

    # Class balance for XGBoost
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos = neg / pos if pos > 0 else 1.0

    search_spaces = {
        "Logistic Regression": (
            make_pipeline(
                LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED),
                scale=True, k=feature_k_grid[0],
            ),
            {
                "selector__k":        feature_k_grid,
                "clf__C":             [0.001, 0.01, 0.1, 1],
                "clf__class_weight":  [None, "balanced"],
            },
        ),
        "Gaussian NB": (
            make_pipeline(GaussianNB(), scale=True, k=feature_k_grid[0]),
            {
                "selector__k":        feature_k_grid,
                "clf__var_smoothing": [1e-9, 1e-8],
            },
        ),
        "Random Forest": (
            make_pipeline(
                RandomForestClassifier(n_estimators=120, random_state=SEED, n_jobs=1),
                scale=False, k=default_k(n),
            ),
            {
                "selector__k":            [default_k(n)],
                "clf__max_depth":         [12, None],
                "clf__min_samples_leaf":  [1, 10],
                "clf__max_features":      ["sqrt"],
            },
        ),
        "KNN": (
            make_pipeline(KNeighborsClassifier(n_jobs=1), scale=True, k=knn_k_grid[0]),
            {
                "selector__k":    [knn_k_grid[0]],
                "clf__n_neighbors": [11, 31],
                "clf__weights":   ["distance"],
                "clf__p":         [2],
            },
        ),
        "XGBoost": (
            make_pipeline(
                xgb.XGBClassifier(
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=SEED,
                    n_jobs=1,
                    scale_pos_weight=scale_pos,
                ),
                scale=False, k=default_k(n),
            ),
            {
                "selector__k":        [default_k(n)],
                "clf__n_estimators":  [100, 200],
                "clf__max_depth":     [4, 6],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample":     [0.8],
            },
        ),
        "LightGBM": (
            make_pipeline(
                lgb.LGBMClassifier(
                    random_state=SEED,
                    n_jobs=1,
                    verbose=-1,
                    is_unbalance=True,
                ),
                scale=False, k=default_k(n),
            ),
            {
                "selector__k":        [default_k(n)],
                "clf__n_estimators":  [100, 200],
                "clf__max_depth":     [4, 6],
                "clf__learning_rate": [0.05, 0.1],
                "clf__num_leaves":    [31, 63],
            },
        ),
        "MLP": (
            make_pipeline(
                MLPClassifier(
                    max_iter=300,
                    early_stopping=True,
                    random_state=SEED,
                ),
                scale=True, k=default_k(n),
            ),
            {
                "selector__k":          [default_k(n)],
                "clf__hidden_layer_sizes": [(128, 64), (256, 128)],
                "clf__alpha":           [0.0001, 0.001],
                "clf__learning_rate_init": [0.001],
            },
        ),
    }

    best_models = {}
    comparison_rows = []

    for name, (pipeline, param_grid) in search_spaces.items():
        print(f"  Tuning {name}...")
        search = GridSearchCV(
            pipeline, param_grid,
            scoring="roc_auc", cv=cv,
            n_jobs=1, refit=True, error_score="raise",
        )
        search.fit(X_train, y_train, groups=groups_train)
        best_models[name] = clone(search.best_estimator_).fit(X_train, y_train)
        comparison_rows.append(evaluate_classifier(name, best_models[name], X_test, y_test))
        print(f"    Best CV ROC AUC: {search.best_score_:.4f}  |  Params: {search.best_params_}")

    comparison_df = pd.DataFrame(comparison_rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nFinal comparison:")
    print(comparison_df.round(4).to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    from plots import (
        plot_model_comparison,
        plot_roc_curves,
        plot_all_confusion_matrices,
        plot_coefficient_comparison,
    )
    plot_model_comparison(comparison_df)
    plot_roc_curves(best_models, comparison_df, X_test, y_test)
    plot_all_confusion_matrices(best_models, X_test, y_test)
    plot_coefficient_comparison(coef_compare)

    return best_models, comparison_df