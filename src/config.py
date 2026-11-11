"""Central project settings: paths and constants live here and nowhere else.

Any "magic number" (like -6 or 2.0) is written once in this file and imported
everywhere, so changing our mind later means editing a single place.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dry run: if the environment variable COLLISION_DATA_DIR is set, the project reads
# that (synthetic) data and writes models and reports to separate folders, so dry-run
# files never get mixed up with the real results.
_DATA_OVERRIDE = os.environ.get("COLLISION_DATA_DIR")
DRY_RUN = _DATA_OVERRIDE is not None

DATA_DIR = Path(_DATA_OVERRIDE) if DRY_RUN else PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / ("models_dryrun" if DRY_RUN else "models")
REPORTS_DIR = PROJECT_ROOT / ("reports_dryrun" if DRY_RUN else "reports")

# Small bundled demo sample (committed to git) and the archived research results.
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
RESEARCH_RESULTS_DIR = PROJECT_ROOT / "research" / "results"

# Core column names
ID_COL = "event_id"
TIME_COL = "time_to_tca"
RISK_COL = "risk"

# Competition rules
HIGH_RISK_THRESHOLD = -6.0        # risk is already log10, so 1e-6 equals -6
OFFICIAL_CLIP_LOW = -6.001         # the official ESA scoring clips predictions below this value
MIN_TIME_TO_TCA_FOR_INPUT = 2.0   # model inputs: CDMs at least 2 days before TCA
MAX_FINAL_TIME_TO_TCA = 1.0       # test events: last CDM within 1 day of TCA

# Experiment settings
RANDOM_SEED = 42
N_SPLITS = 5
