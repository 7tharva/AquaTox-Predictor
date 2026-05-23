import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


def save_models(best_clf_models, best_reg_models, comparison_df, reg_results_df, feature_columns):
    """Call this from main.py after training to persist models to disk."""
    joblib.dump(best_clf_models,  MODEL_DIR / "clf_models.joblib")
    joblib.dump(best_reg_models,  MODEL_DIR / "reg_models.joblib")
    joblib.dump(feature_columns,  MODEL_DIR / "feature_columns.joblib")
    comparison_df.to_csv(MODEL_DIR  / "clf_comparison.csv",  index=False)
    reg_results_df.to_csv(MODEL_DIR / "reg_comparison.csv",  index=False)
    print(f"  Models saved to {MODEL_DIR}")


def load_models():
    """Load everything back from disk for Flask."""
    clf_models    = joblib.load(MODEL_DIR / "clf_models.joblib")
    reg_models    = joblib.load(MODEL_DIR / "reg_models.joblib")
    feat_cols     = joblib.load(MODEL_DIR / "feature_columns.joblib")
    clf_compare   = pd.read_csv(MODEL_DIR / "clf_comparison.csv")
    reg_compare   = pd.read_csv(MODEL_DIR / "reg_comparison.csv")
    return clf_models, reg_models, feat_cols, clf_compare, reg_compare


def models_exist():
    return (MODEL_DIR / "clf_models.joblib").exists()