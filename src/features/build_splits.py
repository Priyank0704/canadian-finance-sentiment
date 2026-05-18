"""
src/features/build_splits.py
============================================================
Build the train/val/test splits — tier-aware stratification
============================================================

WHY THIS FILE:
    Naive train_test_split would mix Tier-1 (expert) and Tier-3
    (weak rule-based) labels into all three splits. That makes
    your test scores partly an evaluation of your weak-labeling
    rules, not just your model. We want the opposite: TEST on
    high-quality labels, TRAIN on everything.

THE SPLITTING STRATEGY:
    1. Apply the 6->5 intent merge from config.INTENT_MERGES.
    2. Carve out a clean test set ONLY from tier1 + tier2.
       This is the trustworthy evaluation set.
    3. Carve a clean validation set the same way.
    4. Everything else (including all tier3) -> training set.
    5. Stratify each split by sentiment AND intent where
       possible so rare classes (risk_warning, negative) aren't
       absent from val/test by chance.

CLASS WEIGHTS:
    We compute and save class weights for both heads so the
    trainer can use them in the loss function. Class weights are
    inversely proportional to class frequency — they tell the
    loss to "pay more attention" to rare classes (negative
    sentiment, risk_warning intent).

OUTPUT:
    data/processed/train.parquet
    data/processed/val.parquet
    data/processed/test.parquet
    data/processed/class_weights.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    MASTER_LABELED, PROCESSED_DIR, INTENT_MERGES,
    SENTIMENT_LABELS, INTENT_LABELS,
    SENTIMENT_TO_ID, INTENT_TO_ID,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
)


def apply_merges(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the intent-class merges from config (6 -> 5 classes)."""
    if INTENT_MERGES:
        before_classes = df["intent"].dropna().unique()
        df["intent"] = df["intent"].replace(INTENT_MERGES)
        after_classes = df["intent"].dropna().unique()
        logger.info(
            f"Applied intent merges: {INTENT_MERGES}. "
            f"Classes: {len(before_classes)} -> {len(after_classes)}"
        )
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string labels to int ids. NaN stays NaN (the trainer
    converts to LABEL_NA_ID=-100 at batch time, which CrossEntropy
    ignores). Keeping NaN here is cleaner than encoding -100 in
    the parquet — easier to debug.
    """
    df = df.copy()
    df["sentiment_id"] = df["sentiment"].map(SENTIMENT_TO_ID)
    df["intent_id"] = df["intent"].map(INTENT_TO_ID)

    # Sanity check — every non-null label must encode to an int.
    bad_s = df.loc[df["sentiment"].notna() & df["sentiment_id"].isna(), "sentiment"]
    bad_i = df.loc[df["intent"].notna() & df["intent_id"].isna(), "intent"]
    if not bad_s.empty:
        raise ValueError(f"Unknown sentiment labels: {bad_s.unique()}")
    if not bad_i.empty:
        raise ValueError(f"Unknown intent labels: {bad_i.unique()}")

    return df


def build_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tier-aware train/val/test split.

    Test and val are sampled ONLY from tier1+tier2 (trustworthy
    labels). Tier3 (weak rule-based) goes entirely into train.
    """
    trustworthy = df[df["label_tier"].isin(["tier1", "tier2"])].copy()
    weak = df[df["label_tier"] == "tier3"].copy()

    logger.info(
        f"Trustworthy rows (tier1+2): {len(trustworthy)}, "
        f"Weak rows (tier3): {len(weak)}"
    )

    # First split trustworthy -> (train_t + val_t + test_t).
    # We split sequentially: first carve test, then carve val from
    # the remainder. Stratifying on sentiment_id is the cleanest
    # signal we have for most rows; rows missing sentiment use a
    # placeholder so stratification can still run.
    trustworthy["_strat"] = trustworthy["sentiment_id"].fillna(-1).astype(int)

    train_val_t, test_t = train_test_split(
        trustworthy,
        test_size=TEST_SPLIT,
        stratify=trustworthy["_strat"],
        random_state=RANDOM_SEED,
    )

    # val proportion among the remaining 85%
    val_relative = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_t, val_t = train_test_split(
        train_val_t,
        test_size=val_relative,
        stratify=train_val_t["_strat"],
        random_state=RANDOM_SEED,
    )

    train_t = train_t.drop(columns=["_strat"])
    val_t = val_t.drop(columns=["_strat"])
    test_t = test_t.drop(columns=["_strat"])

    # Tier3 (weak) goes entirely into train — never used to score.
    train = pd.concat([train_t, weak], ignore_index=True)

    # Shuffle the training set so weak rows aren't all clumped at
    # the end of an epoch. Stable seed for reproducibility.
    train = train.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    val = val_t.reset_index(drop=True)
    test = test_t.reset_index(drop=True)

    logger.info(
        f"Split sizes — train: {len(train)} (incl. {len(weak)} weak), "
        f"val: {len(val)}, test: {len(test)}"
    )
    return train, val, test


def compute_class_weights(train: pd.DataFrame) -> dict:
    """
    Inverse-frequency class weights for both heads.

    Formula: weight_i = n_total / (n_classes * n_i)
    This is sklearn's 'balanced' formula. Result: rare classes
    get higher weights, frequent ones get weights near 1.

    Weights are computed ON THE TRAINING SET ONLY — using val or
    test would leak information into training.
    """
    weights = {}

    # Sentiment
    sent_counts = train["sentiment_id"].dropna().astype(int).value_counts()
    n_classes = len(SENTIMENT_LABELS)
    n_total = sent_counts.sum()
    sent_weights = [
        float(n_total / (n_classes * sent_counts.get(i, 1)))
        for i in range(n_classes)
    ]
    weights["sentiment"] = sent_weights

    # Intent
    int_counts = train["intent_id"].dropna().astype(int).value_counts()
    n_classes = len(INTENT_LABELS)
    n_total = int_counts.sum()
    int_weights = [
        float(n_total / (n_classes * int_counts.get(i, 1)))
        for i in range(n_classes)
    ]
    weights["intent"] = int_weights

    return weights


def main() -> None:
    if not MASTER_LABELED.exists():
        raise FileNotFoundError(
            f"{MASTER_LABELED} not found. Run "
            "`python -m src.labeling.build_labeled_master` first."
        )

    df = pd.read_csv(MASTER_LABELED)
    logger.info(f"Loaded {len(df)} rows from master_labeled.csv")

    df = apply_merges(df)
    df = encode_labels(df)

    train, val, test = build_splits(df)

    # Save splits as parquet — faster reads than CSV during training.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    logger.success(f"Splits saved -> {PROCESSED_DIR}")

    # Compute and save class weights.
    weights = compute_class_weights(train)
    weights_path = PROCESSED_DIR / "class_weights.json"
    weights_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    logger.success(f"Class weights saved -> {weights_path}")

    logger.info("Sentiment class weights (negative, neutral, positive):")
    print([f"{w:.3f}" for w in weights["sentiment"]])
    logger.info("Intent class weights " + str(INTENT_LABELS) + ":")
    print([f"{w:.3f}" for w in weights["intent"]])

    # Per-split label distribution — sanity check before training.
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        logger.info(f"\n--- {name} ---")
        print(f"  sentiment: {split['sentiment'].value_counts(dropna=False).to_dict()}")
        print(f"  intent:    {split['intent'].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
