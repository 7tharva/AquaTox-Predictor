import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*")

import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    CHALLENGE_NAME,
    DATA_PATH,
    ACTIVE_TAX_GROUP_FILTER,
    ENDPOINT_FILTER,
    EFFECT_FILTER,
    SEED,
    OUTPUT_DIR,
)
from data_loader import load_dataset
from models import run_all_models


def main():
    sns.set_theme(style="whitegrid", context="notebook")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"  Challenge : {CHALLENGE_NAME}")
    print(f"  CSV       : {DATA_PATH}")
    print(f"  Tax group : {ACTIVE_TAX_GROUP_FILTER}")
    print(f"  Endpoint  : {ENDPOINT_FILTER}")
    print(f"  Effect    : {EFFECT_FILTER}")
    print("=" * 55)

    print("\nLoading dataset...")
    X, y, y_reg, data_raw, groups, source_counts = load_dataset()

    print(f"  Rows loaded : {len(data_raw):,}")
    print(f"  Features    : {X.shape[1]:,}")
    print("\n  Feature sources:")
    for k, v in source_counts.items():
        print(f"    {k}: {v:,}")

    print("\n  Target class counts:")
    print(y.value_counts().rename(index={0: "0: LC50 > 1 mg/L", 1: "1: LC50 <= 1 mg/L"}).to_string())

    # ── 2. Train / test split ─────────────────────────────────────────────────
    print("\nSplitting into train / test...")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test     = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test     = y.iloc[train_idx], y.iloc[test_idx]
    groups_train        = groups.iloc[train_idx]

    print(f"  Training rows : {len(X_train):,}")
    print(f"  Testing rows  : {len(X_test):,}")

    # ── 3. Run all models ─────────────────────────────────────────────────────
    print("\nTraining models (this may take a few minutes)...")
    best_models, comparison_df = run_all_models(
        X_train, X_test, y_train, y_test, groups_train, X
    )

    # ── 4. Save summary ───────────────────────────────────────────────────────
    best_name = comparison_df.loc[0, "Model"]
    summary = pd.Series({
        "challenge":       CHALLENGE_NAME,
        "rows":            len(data_raw),
        "features":        X.shape[1],
        "best_model":      best_name,
        "best_roc_auc":    round(comparison_df.loc[0, "ROC AUC"], 4),
        "best_accuracy":   round(comparison_df.loc[0, "Accuracy"], 4),
        "best_macro_f1":   round(comparison_df.loc[0, "Macro F1"], 4),
    })
    summary.to_json(OUTPUT_DIR / "summary.json", indent=2)

    print("\n" + "=" * 55)
    print("  DONE!")
    print(f"  Best model  : {best_name}")
    print(f"  ROC AUC     : {comparison_df.loc[0, 'ROC AUC']:.4f}")
    print(f"  Outputs saved in: outputs/{CHALLENGE_NAME}/")
    print("=" * 55)

    # ── SHAP Analysis ─────────────────────────────────────────────────────────
    from shap_analysis import run_shap_analysis
    run_shap_analysis(best_models, X_test)

    # ── Regression Models ─────────────────────────────────────────────────────
    from regression_models import run_regression_models
    reg_models, reg_results_df = run_regression_models(
        X_train, X_test,
        y_reg.iloc[train_idx],
        y_reg.iloc[test_idx],
        groups_train, X
    )

    # ── Save models for Flask ─────────────────────────────────────────────────
    print("\nSaving models to disk for Flask app...")
    from model_store import save_models
    save_models(
        best_models,
        reg_models,
        comparison_df,
        reg_results_df,
        list(X.columns)
    )
    print("  All models saved! You can now run: python app.py")


if __name__ == "__main__":
    main()