import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from model_store import load_models, models_exist

app = Flask(__name__)

# ── Load models once at startup ───────────────────────────────────────────────
if not models_exist():
    raise RuntimeError("No saved models found! Run main.py first.")

clf_models, reg_models, feat_cols, clf_compare, reg_compare = load_models()
print(f"  Loaded {len(clf_models)} classifiers and {len(reg_models)} regressors")


def build_input_row(form) -> pd.DataFrame:
    """Convert form inputs into a one-row DataFrame matching training features."""
    row = {col: 0.0 for col in feat_cols}

    # Numeric inputs
    numeric_map = {
        "chem_mw":                    "mol_weight",
        "chem_rdkit_clogp":           "clogp",
        "chem_pcp_heavy_atom_count":  "heavy_atoms",
        "chem_rings_count":           "ring_count",
        "chem_pcp_doublebonds_count": "double_bonds",
        "chem_OH_count":              "oh_groups",
        "chem_ws":                    "water_solubility",
        "result_obs_duration_mean":   "duration",
        "media_ph_mean":              "ph",
        "media_temperature_mean":     "temperature",
    }
    for feat_col, form_key in numeric_map.items():
        if feat_col in row and form.get(form_key):
            try:
                row[feat_col] = float(form[form_key])
            except ValueError:
                pass

    # Exposure type one-hot
    exposure = form.get("exposure_type", "static")
    exposure_col = f"test_exposure_type_{exposure}"
    if exposure_col in row:
        row[exposure_col] = 1

    X_input = pd.DataFrame([row], columns=feat_cols)
    return X_input


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    form = request.form
    X_input = build_input_row(form)

    # ── Best classifier (XGBoost) ─────────────────────────────────────────────
    best_clf_name = clf_compare.iloc[0]["Model"]
    best_clf      = clf_models[best_clf_name]
    clf_proba     = best_clf.predict_proba(X_input)[0]
    clf_pred      = int(best_clf.predict(X_input)[0])
    acute1_prob   = round(float(clf_proba[1]) * 100, 1)

    # ── All classifiers for comparison ───────────────────────────────────────
    clf_results = []
    for _, row in clf_compare.iterrows():
        clf_results.append({
            "model":   row["Model"],
            "roc_auc": round(float(row["ROC AUC"]), 3),
            "accuracy": round(float(row["Accuracy"]), 3),
            "f1":      round(float(row["Macro F1"]), 3),
        })

    # ── Best regressor (highest R²) ───────────────────────────────────────────
    best_reg_name = reg_compare.iloc[0]["Model"]
    best_reg      = reg_models[best_reg_name]
    log_lc50_pred = float(best_reg.predict(X_input)[0])
    lc50_pred     = round(np.expm1(log_lc50_pred), 4)

    # ── All regressors for comparison ─────────────────────────────────────────
    reg_results = []
    for _, row in reg_compare.iterrows():
        reg_results.append({
            "model": row["Model"],
            "r2":    round(float(row["R²"]), 3),
            "rmse":  round(float(row["RMSE"]), 3),
            "mae":   round(float(row["MAE"]), 3),
        })

    return jsonify({
        "classification": {
            "prediction":  clf_pred,
            "label":       "LC50 ≤ 1 mg/L (Acute 1 — Highly Toxic)" if clf_pred == 1 else "LC50 > 1 mg/L (Not Acute 1)",
            "acute1_prob": acute1_prob,
            "best_model":  best_clf_name,
        },
        "regression": {
            "lc50_pred":   lc50_pred,
            "log_lc50":    round(log_lc50_pred, 4),
            "best_model":  best_reg_name,
            "best_r2":     round(float(reg_compare.iloc[0]["R²"]), 3),
            "best_rmse":   round(float(reg_compare.iloc[0]["RMSE"]), 3),
        },
        "clf_comparison": clf_results,
        "reg_comparison": reg_results,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)