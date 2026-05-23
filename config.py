from pathlib import Path

# ── Where your CSVs live ──────────────────────────────────────────────────────
DATASET_DIR = Path(r"C:\Users\Atharva\Desktop\lc50_project\data\processed")

# ── Which challenge to run ────────────────────────────────────────────────────
CHALLENGE_NAME = "t-F2F"

# ── All available challenges and their CSV filenames ─────────────────────────
CHALLENGE_FILES = {
    "a-CA2F-same": "a-CA2F-same_mortality.csv",
    "a-CA2F-diff": "a-CA2F-diff_mortality.csv",
    "a-FCA2FCA":   "a-FCA2FCA_mortality.csv",
    "t-F2F":       "t-F2F_mortality.csv",
    "t-C2C":       "t-C2C_mortality.csv",
    "t-A2A":       "t-A2A_mortality.csv",
    "s-F2F-1":     "s-F2F-1_mortality.csv",
    "s-F2F-2":     "s-F2F-2_mortality.csv",
    "s-F2F-3":     "s-F2F-3_mortality.csv",
    "s-C2C":       "s-C2C_mortality.csv",
    "s-A2C":       "s-A2C_mortality.csv",
}

CHALLENGE_TAX_GROUPS = {
    "a-CA2F-same": None,
    "a-CA2F-diff": None,
    "a-FCA2FCA":   None,
    "t-F2F":       "fish",
    "t-C2C":       "crusta",
    "t-A2A":       "algae",
    "s-F2F-1":     "fish",
    "s-F2F-2":     "fish",
    "s-F2F-3":     "fish",
    "s-C2C":       "crusta",
    "s-A2C":       "algae",
}

# ── Filters ───────────────────────────────────────────────────────────────────
# For fish/crustaceans: LC50 + MOR
# For algae challenges: change to EC50 + POP
ENDPOINT_FILTER = "LC50"
EFFECT_FILTER   = "MOR"
TAX_GROUP_FILTER = "auto"

# ── Output folder ─────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs" / CHALLENGE_NAME.replace("-", "_")
FIG_DIR    = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Derived values (don't touch these) ───────────────────────────────────────
SEED = 42
DATA_PATH = DATASET_DIR / CHALLENGE_FILES[CHALLENGE_NAME]
ACTIVE_TAX_GROUP_FILTER = (
    CHALLENGE_TAX_GROUPS[CHALLENGE_NAME]
    if TAX_GROUP_FILTER == "auto"
    else TAX_GROUP_FILTER
)
ENDPOINT_LABEL     = ENDPOINT_FILTER if ENDPOINT_FILTER else "toxicity endpoint"
ENDPOINT_FILE_STEM = ENDPOINT_LABEL.lower().replace(" ", "_")