"""
src/labeling/load_twitter_finance.py
============================================================
Tier 2 labels — Twitter Financial News (sentiment + topic)
============================================================

WHY THIS FILE EXISTS:
    You chose "mix of both" for labels. This is the "both" part:
    instead of relying only on rule-based weak labels for the
    Canadian sources, we bring in TWO real annotated datasets.

    1. zeroshot/twitter-financial-news-sentiment
         ~11.9k finance tweets, 3 sentiment classes.
         Maps directly onto our sentiment head.

    2. zeroshot/twitter-financial-news-topic
         ~21k finance tweets, 20 topic classes.
         We MAP those 20 topics down onto our 6-class intent
         taxonomy. This gives the intent head real labels.

WHY MAP 20 TOPICS -> 6 INTENTS INSTEAD OF USING 20 CLASSES:
    20 classes on a ~21k dataset means very few examples per class
    and a model that's hard to evaluate. Our 6-intent taxonomy is
    a deliberate, defensible design choice. The mapping below IS a
    piece of design work — document it in your case study.

    Note: the topic dataset has labels about *what the text is
    about* (Earnings, Fed, M&A). Our intent taxonomy is about
    *what the text is doing*. The mapping is approximate — that's
    honest and fine. We mark these rows label_tier="tier2" so
    later analysis can weight them appropriately.

OUTPUT:
    data/labeled/twitter_sentiment.csv   [text, sentiment, source, label_tier]
    data/labeled/twitter_intent.csv      [text, intent, source, label_tier]
"""

import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from loguru import logger

# Make `from src.utils.config import ...` work when run as a module.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import LABELED_DIR, INTENT_LABELS  # noqa: E402

# ── Sentiment dataset ──────────────────────────────────────
SENTIMENT_DATASET = "zeroshot/twitter-financial-news-sentiment"
# This dataset's integer labels: 0=Bearish, 1=Bullish, 2=Neutral.
# We translate to OUR vocabulary: bearish->negative, bullish->positive.
SENTIMENT_MAP = {0: "negative", 1: "positive", 2: "neutral"}

# ── Topic dataset ──────────────────────────────────────────
TOPIC_DATASET = "zeroshot/twitter-financial-news-topic"

# THE MAPPING: 20 source topics -> our 6 intent classes.
# This is the core design decision of this file. Each line is a
# judgement call about what that topic's text is usually *doing*.
TOPIC_TO_INTENT = {
    0:  "analyst_question",     # Analyst Update
    1:  "policy_signal",        # Fed | Central Banks
    2:  "performance_report",   # Company | Product News
    3:  "market_commentary",    # Treasuries | Corporate Debt
    4:  "performance_report",   # Dividend
    5:  "performance_report",   # Earnings
    6:  "market_commentary",    # Energy | Oil
    7:  "market_commentary",    # Financials
    8:  "market_commentary",    # Currencies
    9:  "market_commentary",    # General News | Opinion
    10: "market_commentary",    # Gold | Metals | Materials
    11: "performance_report",   # IPO
    12: "risk_warning",         # Legal | Regulation
    13: "performance_report",   # M&A | Investments
    14: "policy_signal",        # Macro
    15: "market_commentary",    # Markets
    16: "policy_signal",        # Politics
    17: "performance_report",   # Personnel Change
    18: "market_commentary",    # Stock Commentary
    19: "market_commentary",    # Stock Movement
}
# Note: "forward_guidance" isn't well represented in the topic
# dataset — that's OK. The weak-labeling rules in Milestone 2
# and the BoC data fill that gap. Knowing WHERE each label comes
# from is part of the project's story.


def load_twitter_sentiment() -> pd.DataFrame:
    """Load and remap the Twitter financial sentiment dataset."""
    logger.info(f"Downloading {SENTIMENT_DATASET}...")
    ds = load_dataset(SENTIMENT_DATASET)

    # Combine train + validation — we'll make our own splits later.
    df = pd.concat(
        [ds["train"].to_pandas(), ds["validation"].to_pandas()],
        ignore_index=True,
    )
    df = df.rename(columns={"text": "text", "label": "sentiment"})
    df["sentiment"] = df["sentiment"].map(SENTIMENT_MAP)

    df["source"] = "twitter_finance"
    df["label_tier"] = "tier2"

    df = df.dropna(subset=["text", "sentiment"])
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    logger.info(f"Twitter sentiment rows: {len(df)}")
    return df[["text", "sentiment", "source", "label_tier"]]


def load_twitter_intent() -> pd.DataFrame:
    """Load the Twitter topic dataset and map topics -> intents."""
    logger.info(f"Downloading {TOPIC_DATASET}...")
    ds = load_dataset(TOPIC_DATASET)

    df = pd.concat(
        [ds["train"].to_pandas(), ds["validation"].to_pandas()],
        ignore_index=True,
    )
    df = df.rename(columns={"text": "text", "label": "topic_id"})

    # Apply the 20 -> 6 mapping.
    df["intent"] = df["topic_id"].map(TOPIC_TO_INTENT)

    df["source"] = "twitter_finance"
    df["label_tier"] = "tier2"

    df = df.dropna(subset=["text", "intent"])
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Sanity check: every mapped intent must be in our taxonomy.
    bad = set(df["intent"]) - set(INTENT_LABELS)
    if bad:
        raise ValueError(f"Mapping produced unknown intents: {bad}")

    logger.info(f"Twitter intent rows: {len(df)}")
    return df[["text", "intent", "source", "label_tier"]]


def main() -> None:
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    sent_df = load_twitter_sentiment()
    sent_df.to_csv(LABELED_DIR / "twitter_sentiment.csv", index=False, encoding="utf-8")
    logger.success(f"Saved {len(sent_df)} rows -> twitter_sentiment.csv")
    logger.info("Sentiment distribution:")
    print(sent_df["sentiment"].value_counts())

    intent_df = load_twitter_intent()
    intent_df.to_csv(LABELED_DIR / "twitter_intent.csv", index=False, encoding="utf-8")
    logger.success(f"Saved {len(intent_df)} rows -> twitter_intent.csv")
    logger.info("Intent distribution (after 20->6 mapping):")
    print(intent_df["intent"].value_counts())


if __name__ == "__main__":
    main()
