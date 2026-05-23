import numpy as np
import pandas as pd

MAX_BITS_PER_FINGERPRINT = 128

FINGERPRINT_COLUMNS = [
    "chem_pcp_fp",
    "chem_MACCS_fp",
    "chem_Morgan_fp",
    "chem_ToxPrint_fp",
]

BASE_NUMERIC_FEATURES = [
    # Experimental conditions
    "result_obs_duration_mean",
    "media_ph_mean",
    "media_temperature_mean",
    # Species / ecology numeric descriptors
    "tax_pdm_available",
    "tax_ps_ampv",
    "tax_ps_ampkap",
    "tax_ps_amppm",
    "tax_lh_amd",
    "tax_lh_lbcm",
    "tax_lh_lpcm",
    "tax_lh_licm",
    "tax_lh_ri#/d",
    # Curated chemical properties
    "chem_mw",
    "chem_mp",
    "chem_ws",
    "chem_ws_binary",
    "chem_rdkit_clogp",
    "chem_pcp_heavy_atom_count",
    "chem_pcp_bonds_count",
    "chem_pcp_doublebonds_count",
    "chem_pcp_triplebonds_count",
    "chem_rings_count",
    "chem_OH_count",
    "chem_mol2vec_allowed",
    "chem_pka_median",
]

CATEGORICAL_FEATURES = [
    "test_location",
    "test_exposure_type",
    "test_control_type",
    "test_media_type",
    "test_application_freq_unit",
    "test_organism_lifestage",
    "tax_eco_climate",
    "tax_eco_ecozone",
    "tax_eco_food",
    "tax_eco_migrate5",
    "tax_eco_migrate2",
]

LEAKAGE_SUBSTRINGS = [
    "conc1", "lc50", "endpoint", "effect",
    "result_id", "test_id", "cas", "dtx",
    "inchi", "smiles", "name",
]


def unique_preserve_order(values: list) -> list:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def expand_fingerprint_bits(series: pd.Series, prefix: str, max_bits: int = MAX_BITS_PER_FINGERPRINT) -> pd.DataFrame:
    """Expand a 0/1 bit-string fingerprint column into individual bit columns."""
    text  = series.astype("string").fillna("")
    valid = text.str.fullmatch(r"[01]+").fillna(False)
    lengths = text.loc[valid].str.len()

    if lengths.empty:
        return pd.DataFrame(index=series.index)

    width = int(lengths.mode().iloc[0])
    valid = valid & (text.str.len() == width)
    default = "0" * width
    values = text.where(valid, default).astype(str).tolist()

    joined = "".join(values).encode("ascii")
    arr = np.frombuffer(joined, dtype=np.uint8).reshape(len(values), width) - ord("0")

    variances = arr.var(axis=0)
    keep = np.flatnonzero(variances > 0)

    if len(keep) > max_bits:
        keep = keep[np.argsort(variances[keep])[::-1][:max_bits]]
        keep = np.sort(keep)

    columns = [f"{prefix}_bit_{i:04d}" for i in keep]
    return pd.DataFrame(arr[:, keep].astype(np.uint8), columns=columns, index=series.index)


def make_feature_matrix(data: pd.DataFrame, header: list) -> tuple:
    """Build the full feature matrix from numeric, categorical, and fingerprint columns."""
    mol2vec_cols = [c for c in header if c.startswith("chem_mol2vec")]
    mordred_cols = [c for c in header if c.startswith("chem_mordred_")]

    numeric_cols = unique_preserve_order(
        [c for c in BASE_NUMERIC_FEATURES + mol2vec_cols + mordred_cols if c in data.columns]
    )
    categorical_cols  = [c for c in CATEGORICAL_FEATURES  if c in data.columns]
    fingerprint_cols  = [c for c in FINGERPRINT_COLUMNS   if c in data.columns]

    # ── Numeric ───────────────────────────────────────────────────────────────
    numeric = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    # ── Categorical (one-hot) ─────────────────────────────────────────────────
    categorical = pd.get_dummies(
        data[categorical_cols].astype("string").fillna("missing"),
        prefix=categorical_cols,
        dtype=np.uint8,
    )

    # ── Fingerprints ──────────────────────────────────────────────────────────
    fingerprint_frames = [
        expand_fingerprint_bits(data[col], col)
        for col in fingerprint_cols
    ]

    # ── Combine ───────────────────────────────────────────────────────────────
    pieces = [numeric, categorical] + fingerprint_frames
    X = pd.concat([p for p in pieces if p.shape[1] > 0], axis=1)

    # Drop columns that are mostly missing or have no variance
    X = X.loc[:, X.notna().mean() >= 0.45]
    X = X.loc[:, X.nunique(dropna=True) > 1]

    # ── Leakage check ─────────────────────────────────────────────────────────
    leaky = [col for col in X.columns if any(t in col.lower() for t in LEAKAGE_SUBSTRINGS)]
    if leaky:
        raise ValueError(f"Leakage features detected: {leaky[:20]}")

    source_counts = {
        "numeric_after_filter":      int(sum(c in X.columns for c in numeric.columns)),
        "categorical_after_filter":  int(sum(c in X.columns for c in categorical.columns)),
        "fingerprint_after_filter":  int(sum(any(c.startswith(fp) for fp in FINGERPRINT_COLUMNS) for c in X.columns)),
        "total_features":            int(X.shape[1]),
    }

    return X, source_counts