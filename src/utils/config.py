"""
src/utils/config.py
============================================================
Central configuration — single source of truth
============================================================
v2: intent taxonomy reduced from 6 -> 5 classes after Milestone 2
    revealed forward_guidance had only 20 rows (insufficient to
    train or evaluate). Merged into market_commentary.

WHY A CONFIG FILE:
    Every path, label name, and training hyperparameter lives
    here. Every other file imports from this module. Change a
    setting once, it changes everywhere — no drift, no surprises.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Data directories ───────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
LABELED_DIR = DATA_DIR / "labeled"

# ── Model directories ──────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
ENCODER_DIR = MODELS_DIR / "encoder"
LORA_ADAPTER_DIR = MODELS_DIR / "lora_adapter"
BASELINE_DIR = MODELS_DIR / "baseline"

# ── Reports ────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVAL_DIR = REPORTS_DIR / "eval"

# ── Key data files ─────────────────────────────────────────
MASTER_UNLABELED = INTERIM_DIR / "master_unlabeled.csv"
MASTER_LABELED = LABELED_DIR / "master_labeled.csv"

# ============================================================
# THE LABEL TAXONOMY — the project's core spec
# ============================================================

# ---- Head 1: Sentiment (3 classes, unchanged) ----
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# ---- Head 2: Intent (5 classes, after merge) ----
# DESIGN NOTE for your case study:
#   We originally specified 6 intents. After labeling, the
#   `forward_guidance` class held only 20 rows — too few to train
#   on or evaluate meaningfully. It was merged into
#   `market_commentary` (the semantically nearest class) to
#   preserve a defensible evaluation. This kind of mid-project
#   taxonomy revision is normal industry practice.
INTENT_LABELS = [
    "market_commentary",   # general observation; absorbed forward_guidance
    "performance_report",  # past results (earnings, revenue, etc.)
    "policy_signal",       # central-bank/regulatory direction
    "risk_warning",        # downside risks, warnings
    "analyst_question",    # analyst / investor questions
]

# Integer mappings — models predict ints, humans read strings.
SENTIMENT_TO_ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID_TO_SENTIMENT = {i: label for label, i in SENTIMENT_TO_ID.items()}
INTENT_TO_ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID_TO_INTENT = {i: label for label, i in INTENT_TO_ID.items()}

# The merge map — applied when loading the labeled master so the
# rest of the pipeline never sees the old class.
INTENT_MERGES = {
    "forward_guidance": "market_commentary",
}

# Special sentinel for "no label" — multi-task loss masks these.
LABEL_NA_ID = -100  # HuggingFace convention; CrossEntropyLoss ignores this index

# ============================================================
# MODEL CHOICES
# ============================================================
# Production model: a finance-domain encoder pre-trained on
# financial text. FinBERT-tone is small (~110M params) and
# CPU-fine-tunable in 10-30 min.
BASE_ENCODER_MODEL = "yiyanghkust/finbert-tone"

# Phi-3 extension (Colab notebook only — kept for reference).
LORA_BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
MAX_SEQ_LENGTH = 128        # finance headlines/sentences are short
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# CPU-friendly training settings. Small batch + few epochs + a
# learning rate proven to work for BERT fine-tuning. If you later
# get a GPU, bump batch_size to 32 and epochs to 4.
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Loss weighting between the two heads. Equal weight is a fine
# default; you can re-tune in Milestone 4 if one head dominates.
SENTIMENT_LOSS_WEIGHT = 1.0
INTENT_LOSS_WEIGHT = 1.0

# ── Weights & Biases ───────────────────────────────────────
WANDB_PROJECT = "canadian-finance-sentiment"

# ── Ensure directories exist on import ─────────────────────
for _dir in [
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR, LABELED_DIR,
    ENCODER_DIR, LORA_ADAPTER_DIR, BASELINE_DIR,
    FIGURES_DIR, EVAL_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
