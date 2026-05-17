"""
src/labeling/build_labeled_master.py
============================================================
Build the final labeled dataset — merge all 3 label tiers
============================================================

WHY THIS STEP (the Milestone 2 equivalent of build_master.py):
    Milestone 2 produced labels from three places:
      tier1  PhraseBank          -> sentiment only   (expert)
      tier2  Twitter finance     -> sentiment + intent (annotated)
      tier3  Canadian weak labels -> sentiment + intent (rules)
    This script unifies them into ONE labeled file with a
    consistent schema, ready for tokenization in Milestone 3.

THE DESIGN DECISION — what to do about partial labels:
    PhraseBank has sentiment but no intent. The Twitter SENTIMENT
    set has sentiment but no intent; the Twitter TOPIC set has
    intent but no sentiment. Rather than throw away half-labeled
    rows, we keep them and let the training stage handle missing
    labels per-head (a row with only a sentiment label contributes
    to the sentiment loss but not the intent loss). This is called
    "partial supervision" — it's efficient and worth explaining in
    your case study.

    Each row keeps its `label_tier` so Milestone 3 can:
      - build the TEST set mostly from tier1 + tier2 (trustworthy)
      - use tier3 mainly for TRAINING (more data, lower precision)

OUTPUT:
    data/labeled/master_labeled.csv
        [text, sentiment, intent, source, label_tier]
"""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    RAW_DIR, LABELED_DIR, SENTIMENT_LABELS, INTENT_LABELS,
)

FINAL_COLUMNS = ["text", "sentiment", "intent", "source", "label_tier"]
OUTPUT_PATH = LABELED_DIR / "master_labeled.csv"


def _load(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"{name} not found at {path} — skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {name}")
    return df


def _conform(df: pd.DataFrame) -> pd.DataFrame:
    """Force a frame into the FINAL_COLUMNS schema, filling gaps with NA."""
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[FINAL_COLUMNS]


def build() -> pd.DataFrame:
    frames = []

    # ---- Tier 1: PhraseBank (sentiment only) ----
    pb = _load(RAW_DIR / "phrasebank.csv", "PhraseBank (tier1)")
    if not pb.empty:
        pb["intent"] = pd.NA
        pb["label_tier"] = "tier1"
        frames.append(_conform(pb))

    # ---- Tier 2: Twitter sentiment (sentiment only) ----
    ts = _load(LABELED_DIR / "twitter_sentiment.csv", "Twitter sentiment (tier2)")
    if not ts.empty:
        ts["intent"] = pd.NA
        frames.append(_conform(ts))

    # ---- Tier 2: Twitter intent (intent only) ----
    ti = _load(LABELED_DIR / "twitter_intent.csv", "Twitter intent (tier2)")
    if not ti.empty:
        ti["sentiment"] = pd.NA
        frames.append(_conform(ti))

    # ---- Tier 3: Canadian weak labels (sentiment + intent) ----
    cw = _load(LABELED_DIR / "canadian_weak_labeled.csv", "Canadian weak (tier3)")
    if not cw.empty:
        frames.append(_conform(cw))

    if not frames:
        raise RuntimeError(
            "No labeled sources found. Run, in order:\n"
            "  python -m src.collection.load_phrasebank\n"
            "  python -m src.labeling.load_twitter_finance\n"
            "  python -m src.labeling.weak_labeler\n"
            "  python -m src.labeling.build_labeled_master"
        )

    df = pd.concat(frames, ignore_index=True)

    # ---- Validation: every non-null label must be in the taxonomy ----
    bad_sent = set(df["sentiment"].dropna()) - set(SENTIMENT_LABELS)
    bad_int = set(df["intent"].dropna()) - set(INTENT_LABELS)
    if bad_sent:
        raise ValueError(f"Unknown sentiment labels found: {bad_sent}")
    if bad_int:
        raise ValueError(f"Unknown intent labels found: {bad_int}")

    # ---- Hygiene ----
    before = len(df)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 15]
    # Drop rows that have NEITHER label — useless for training.
    df = df[df["sentiment"].notna() | df["intent"].notna()]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} short/empty/unlabeled/dup rows")

    return df


def main() -> None:
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    df = build()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logger.success(f"Final labeled dataset: {len(df)} rows -> {OUTPUT_PATH}")

    # The situational report you carry into Milestone 3.
    logger.info("Rows per label tier:")
    print(df["label_tier"].value_counts())

    sent_n = df["sentiment"].notna().sum()
    int_n = df["intent"].notna().sum()
    both_n = (df["sentiment"].notna() & df["intent"].notna()).sum()
    logger.info(
        f"Sentiment-labeled: {sent_n} | Intent-labeled: {int_n} | "
        f"Both: {both_n}"
    )
    logger.info("Sentiment distribution:")
    print(df["sentiment"].value_counts(dropna=False))
    logger.info("Intent distribution:")
    print(df["intent"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
