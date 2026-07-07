from pathlib import Path


# Root della repository.
PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# ARTIFACT GENERATI LOCALMENTE
# =========================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATA_DIR = ARTIFACTS_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DATASETS_DIR = DATA_DIR / "datasets"

MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"


# =========================================================
# DATASET
# =========================================================

CLEAN_DIR = DATASETS_DIR / "clean"
CLEAN_DATA_PATH = CLEAN_DIR / "all_stores_cashflow.csv"

LEVEL_SHIFT_DATA_DIR = DATASETS_DIR / "level_shift"
LEVEL_SHIFT_SENSITIVITY_DIR = LEVEL_SHIFT_DATA_DIR / "sensitivity"
LEVEL_SHIFT_CONTAMINATION_DIR = (
    LEVEL_SHIFT_DATA_DIR / "contamination"
)

POS_DELAY_DATA_DIR = DATASETS_DIR / "pos_delay"
POS_DELAY_SENSITIVITY_DIR = POS_DELAY_DATA_DIR / "sensitivity"
POS_DELAY_CONTAMINATION_DIR = (
    POS_DELAY_DATA_DIR / "contamination"
)


# =========================================================
# CACHE DATI ESTERNI
# =========================================================

EXTERNAL_INDICES_CACHE_PATH = (
    CACHE_DIR / "external_indices_2018_2024.csv"
)


# =========================================================
# MODELLI E RISULTATI
# =========================================================

SALES_MODEL_DIR = MODELS_DIR / "sales"
POS_MODEL_DIR = MODELS_DIR / "pos"

LEVEL_SHIFT_RESULTS_DIR = RESULTS_DIR / "level_shift"
POS_DELAY_RESULTS_DIR = RESULTS_DIR / "pos_delay"


def ensure_artifact_directories():
    """Crea le cartelle principali degli artifact, se mancanti."""

    directories = [
        CACHE_DIR,
        CLEAN_DIR,
        LEVEL_SHIFT_SENSITIVITY_DIR,
        LEVEL_SHIFT_CONTAMINATION_DIR,
        POS_DELAY_SENSITIVITY_DIR,
        POS_DELAY_CONTAMINATION_DIR,
        SALES_MODEL_DIR,
        POS_MODEL_DIR,
        LEVEL_SHIFT_RESULTS_DIR,
        POS_DELAY_RESULTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
