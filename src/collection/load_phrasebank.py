"""
src/collection/load_phrasebank.py
============================================================
Source 1 of 3 — Financial PhraseBank (the labeled foundation)
============================================================

WHY THIS SOURCE FIRST:
    Every classification project needs a trustworthy labeled core
    before you start weak-labeling messier data. Financial PhraseBank
    is ~4,800 finance sentences hand-annotated by 16 people with
    finance backgrounds. It is the closest thing to "ground truth"
    for financial *sentiment* that exists publicly.

    It does NOT have intent labels — that's fine. This source
    teaches the model financial sentiment language. Sources 2 and 3
    add the Canadian context and the intent dimension.

WHAT "sentences_allagree" MEANS:
    The dataset ships in 4 configs based on annotator agreement:
        - sentences_50agree   (>50% agreed)
        - sentences_66agree   (>66% agreed)
        - sentences_75agree   (>75% agreed)
        - sentences_allagree  (100% agreed)
    We take the 75%-agree config: a good balance between volume
    and label quality. allagree is cleaner but small; 50agree is
    noisy. This is a real modelling decision worth mentioning in
    your case study.

OUTPUT:
    data/raw/phrasebank.csv  with columns: [text, sentiment, source]
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from loguru import logger

# ── Paths ──────────────────────────────────────────────────
# Resolve project root relative to this file so the script works
# no matter what directory you run it from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "phrasebank.csv"

# The HuggingFace dataset id and the config we want.
DATASET_ID = "financial_phrasebank"
CONFIG = "sentences_75agree"

# PhraseBank ships labels as integers — map them to readable strings
# so every downstream file speaks the same vocabulary.
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_phrasebank() -> pd.DataFrame:
    """Download Financial PhraseBank and return a clean DataFrame."""
    logger.info(f"Downloading {DATASET_ID} ({CONFIG}) from HuggingFace Hub...")

    # trust_remote_code is required for this particular dataset's
    # loading script. It's a well-known, safe community dataset.
    dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        split="train",
        trust_remote_code=True,
    )

    df = dataset.to_pandas()
    logger.info(f"Raw rows downloaded: {len(df)}")

    # The dataset's columns are 'sentence' and 'label'.
    df = df.rename(columns={"sentence": "text", "label": "sentiment"})

    # Convert integer labels -> strings.
    df["sentiment"] = df["sentiment"].map(LABEL_MAP)

    # Tag the provenance. Every row in every source carries a
    # 'source' column so that in Milestone 2 we can analyse
    # label quality per-source and weight the training set.
    df["source"] = "phrasebank"

    # Basic hygiene: drop exact-duplicate sentences and any nulls.
    before = len(df)
    df = df.dropna(subset=["text", "sentiment"])
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} duplicate/null rows")

    return df[["text", "sentiment", "source"]]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = load_phrasebank()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    logger.success(f"Saved {len(df)} rows -> {OUTPUT_PATH}")

    # Print the class balance — you WANT to see this now, because
    # PhraseBank is heavily skewed toward 'neutral'. Knowing the
    # imbalance early shapes decisions in Milestone 3 (class weights).
    logger.info("Sentiment distribution:")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
