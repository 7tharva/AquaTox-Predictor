import numpy as np
import pandas as pd

from config import (
    ACTIVE_TAX_GROUP_FILTER,
    DATA_PATH,
    EFFECT_FILTER,
    ENDPOINT_FILTER,
)
from features import make_feature_matrix, unique_preserve_order, BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES, FINGERPRINT_COLUMNS

TARGET_COLUMN   = "result_conc1_mean_binary"
LC50_RAW_COLUMN = "result_conc1_mean"
GROUP_COLUMN    = "chem_dtxsid"
FILTER_COLUMNS  = ["tax_group", "result_effect", "result_endpoint"]


def load_dataset():
    """Load the CSV, apply filters, and return feature matrix + labels."""

    # Read just the header first to know what columns exist
    header = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()

    from features import (
        BASE_NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        FINGERPRINT_COLUMNS,
        unique_preserve_order,
    )

    mol2vec_cols = [c for c in header if c.startswith("chem_mol2vec")]
    mordred_cols = [c for c in header if c.startswith("chem_mordred_")]

    all_feature_inputs = unique_preserve_order(
        BASE_NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + FINGERPRINT_COLUMNS
        + mol2vec_cols
        + mordred_cols
    )

    usecols = unique_preserve_order(
        FILTER_COLUMNS
        + [TARGET_COLUMN, LC50_RAW_COLUMN, GROUP_COLUMN]
        + [c for c in all_feature_inputs if c in header]
    )

    dtype = {
        col: "string"
        for col in FINGERPRINT_COLUMNS + CATEGORICAL_FEATURES + [GROUP_COLUMN]
        if col in header
    }

    df = pd.read_csv(DATA_PATH, usecols=usecols, dtype=dtype, low_memory=False)

    # ── Apply filters ─────────────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)
    if ACTIVE_TAX_GROUP_FILTER is not None:
        mask &= df["tax_group"] == ACTIVE_TAX_GROUP_FILTER
    if EFFECT_FILTER is not None:
        mask &= df["result_effect"] == EFFECT_FILTER
    if ENDPOINT_FILTER is not None:
        mask &= df["result_endpoint"] == ENDPOINT_FILTER

    data = df[mask].copy()
    data = data.dropna(subset=[TARGET_COLUMN, LC50_RAW_COLUMN])

    if data.empty:
        raise ValueError(
            "No rows left after filtering! "
            "Check TAX_GROUP_FILTER, ENDPOINT_FILTER, and EFFECT_FILTER in config.py"
        )

    # ── Classification target ─────────────────────────────────────────────────
    y = data[TARGET_COLUMN].astype(int)

    # ── Regression target (log-transformed raw LC50) ──────────────────────────
    y_reg = np.log1p(pd.to_numeric(data[LC50_RAW_COLUMN], errors="coerce"))
    y_reg = y_reg.fillna(y_reg.median())

    # ── Feature matrix ────────────────────────────────────────────────────────
    X, source_counts = make_feature_matrix(data, header)

    # ── Groups (chemicals) for grouped cross-validation ───────────────────────
    groups = data[GROUP_COLUMN].astype("string")
    fallback = pd.Series(
        [f"missing_group_{i}" for i in range(len(data))],
        index=data.index,
        dtype="string",
    )
    groups = groups.where(groups.notna(), fallback)

    return X, y, y_reg, data, groups, source_counts