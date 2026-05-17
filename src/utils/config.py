"""
src/utils/config.py
============================================================
Central configuration — single source of truth
============================================================

WHY A CONFIG FILE:
    In the fraud project, paths and constants drifted across files
    and caused bugs. Here, EVERY path, label name, and key setting
    lives in one place. Every other file imports from here. Change
    something once, it changes everywhere.

    This also documents the project's design decisions in code:
    the label taxonomy below IS the spec for what this classifier
    predicts.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────
# This file is src/utils/config.py, so root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Data directories ───────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
LABELED_DIR = DATA_DIR / "labeled"

# ── Model directories ──────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
ENCODER_DIR = MODELS_DIR / "encoder"          # production model
LORA_ADAPTER_DIR = MODELS_DIR / "lora_adapter"  # Phi-3 extension

# ── Reports ────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVAL_DIR = REPORTS_DIR / "eval"

# ── Key data files ─────────────────────────────────────────
MASTER_UNLABELED = INTERIM_DIR / "master_unlabeled.csv"
MASTER_LABELED = LABELED_DIR / "master_labeled.csv"

# ============================================================
# THE LABEL TAXONOMY — this is the project's core spec
# ============================================================
# This classifier has TWO heads. Both taxonomies are fixed here
# so labelling, training, and inference can never disagree.

# ---- Head 1: Sentiment (3 classes) ----
# Standard financial sentiment. PhraseBank gives us real labels
# for this; weak rules extend it to the Canadian sources.
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# ---- Head 2: Intent (6 classes) ----
# THIS taxonomy is your own design — no dataset hands it to you,
# and designing it well is what makes the project portfolio-worthy.
# Each label answers "what is this text trying to DO?"
INTENT_LABELS = [
    "forward_guidance",    # statements about future expectations/outlook
    "risk_warning",        # flagging downside risks, threats, concerns
    "performance_report",  # reporting past results — earnings, growth figures
    "policy_signal",       # central-bank / regulatory policy direction
    "analyst_question",    # analyst/investor questions or information requests
    "market_commentary",   # general observation/description of market conditions
]

# Integer mappings — models predict integers, humans read strings.
SENTIMENT_TO_ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID_TO_SENTIMENT = {i: label for label, i in SENTIMENT_TO_ID.items()}
INTENT_TO_ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID_TO_INTENT = {i: label for label, i in INTENT_TO_ID.items()}

# ============================================================
# MODEL CHOICES
# ============================================================
# Production model: a finance-domain encoder. FinBERT is BERT
# pre-trained on financial text — a strong, small, deployable
# starting point. Swap to "microsoft/deberta-v3-small" if you
# want a more modern general encoder.
BASE_ENCODER_MODEL = "yiyanghkust/finbert-tone"

# Extension model: small LLM for the LoRA fine-tune (Colab).
# Phi-3-mini is ~3.8B params — fits in Colab's free T4 with 4-bit
# quantization. This is the "research extension" piece.
LORA_BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# ── Tokenization / training settings ───────────────────────
MAX_SEQ_LENGTH = 256          # finance sentences/headlines are short
TRAIN_TEST_SPLIT = 0.15       # 15% test
VAL_SPLIT = 0.15              # 15% of remaining for validation
RANDOM_SEED = 42              # reproducibility — same seed everywhere

# ── Weights & Biases ───────────────────────────────────────
WANDB_PROJECT = "canadian-finance-sentiment"

# ── Ensure directories exist on import ─────────────────────
# Importing config guarantees the folder structure is present —
# no more "FileNotFoundError: data/processed" surprises.
for _dir in [
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR, LABELED_DIR,
    ENCODER_DIR, LORA_ADAPTER_DIR, FIGURES_DIR, EVAL_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
