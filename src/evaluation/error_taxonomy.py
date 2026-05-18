"""
src/evaluation/error_taxonomy.py
============================================================
Sample FinBERT's actual mistakes for manual categorization
============================================================

WHY THIS FILE:
    Confusion matrices tell you THAT classes get confused.
    They don't tell you WHY. The only way to know why is to read
    the actual texts the model got wrong.

    This script samples 50 sentiment errors and 50 intent errors,
    stratified across error types (which true class -> which
    predicted class), and writes them to a CSV you fill in.

    You categorize each error into one of:
       genuinely_ambiguous   — even a human would struggle
       label_noise           — true label looks wrong to you
       model_bias            — model favors a frequent class
       missing_context       — text needs surrounding text/world knowledge
       other                 — annotate in notes

    Then run this script in score mode to get the breakdown.

CASE-STUDY PAYOFF:
    "Of 100 random errors I categorized, 42% were genuinely
    ambiguous, 31% were label noise from the crowd-annotated
    Twitter dataset, and 19% were model bias toward the majority
    class. The remaining 8% were missing-context cases I'd
    address with longer windows in production."
    That single paragraph is worth more than a 0.93 F1 score.

TWO MODES:
    python -m src.evaluation.error_taxonomy sample   # make the audit sheet
    python -m src.evaluation.error_taxonomy score    # tally categories
"""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    EVAL_DIR, SENTIMENT_LABELS, INTENT_LABELS, RANDOM_SEED,
)

ERRORS_SHEET = EVAL_DIR / "error_taxonomy_sheet.csv"
ERRORS_SUMMARY = EVAL_DIR / "error_taxonomy_summary.txt"

SAMPLE_PER_HEAD = 50  # 50 per head, 100 total — feasible to audit in <1 hour

VALID_CATEGORIES = {
    "genuinely_ambiguous",
    "label_noise",
    "model_bias",
    "missing_context",
    "other",
}


def make_sample() -> None:
    pred_path = EVAL_DIR / "test_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found. Run evaluate_finbert.py first."
        )

    df = pd.read_parquet(pred_path)

    sentiment_id_to_str = dict(enumerate(SENTIMENT_LABELS))
    intent_id_to_str = dict(enumerate(INTENT_LABELS))

    samples = []

    # ---- Sentiment errors ----
    sent_mask = (
        df["sentiment_id"].notna()
        & (df["sentiment_id"].astype("Int64") != df["pred_sentiment_id"].astype("Int64"))
    )
    sent_errors = df[sent_mask].copy()
    sent_errors["head"] = "sentiment"
    sent_errors["true_label"] = sent_errors["sentiment_id"].astype(int).map(sentiment_id_to_str)
    sent_errors["pred_label"] = sent_errors["pred_sentiment_id"].astype(int).map(sentiment_id_to_str)

    if len(sent_errors) > SAMPLE_PER_HEAD:
        # Stratify by (true_label, pred_label) so we cover all
        # error TYPES, not just the most frequent mistake.
        sent_errors = (
            sent_errors.groupby(["true_label", "pred_label"], group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, SAMPLE_PER_HEAD // 6)),
                random_state=RANDOM_SEED,
            ))
        )
        # Top up if stratification underfilled.
        if len(sent_errors) < SAMPLE_PER_HEAD:
            extra = df[sent_mask].drop(sent_errors.index, errors="ignore")
            extra = extra.sample(
                min(SAMPLE_PER_HEAD - len(sent_errors), len(extra)),
                random_state=RANDOM_SEED,
            )
            extra["head"] = "sentiment"
            extra["true_label"] = extra["sentiment_id"].astype(int).map(sentiment_id_to_str)
            extra["pred_label"] = extra["pred_sentiment_id"].astype(int).map(sentiment_id_to_str)
            sent_errors = pd.concat([sent_errors, extra], ignore_index=True)

    samples.append(sent_errors.head(SAMPLE_PER_HEAD))
    logger.info(f"Sampled {len(samples[-1])} sentiment errors "
                f"(out of {sent_mask.sum()} total)")

    # ---- Intent errors ----
    intent_mask = (
        df["intent_id"].notna()
        & (df["intent_id"].astype("Int64") != df["pred_intent_id"].astype("Int64"))
    )
    intent_errors = df[intent_mask].copy()
    intent_errors["head"] = "intent"
    intent_errors["true_label"] = intent_errors["intent_id"].astype(int).map(intent_id_to_str)
    intent_errors["pred_label"] = intent_errors["pred_intent_id"].astype(int).map(intent_id_to_str)

    if len(intent_errors) > SAMPLE_PER_HEAD:
        intent_errors = (
            intent_errors.groupby(["true_label", "pred_label"], group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, SAMPLE_PER_HEAD // 10)),
                random_state=RANDOM_SEED,
            ))
        )
        if len(intent_errors) < SAMPLE_PER_HEAD:
            extra = df[intent_mask].drop(intent_errors.index, errors="ignore")
            extra = extra.sample(
                min(SAMPLE_PER_HEAD - len(intent_errors), len(extra)),
                random_state=RANDOM_SEED,
            )
            extra["head"] = "intent"
            extra["true_label"] = extra["intent_id"].astype(int).map(intent_id_to_str)
            extra["pred_label"] = extra["pred_intent_id"].astype(int).map(intent_id_to_str)
            intent_errors = pd.concat([intent_errors, extra], ignore_index=True)

    samples.append(intent_errors.head(SAMPLE_PER_HEAD))
    logger.info(f"Sampled {len(samples[-1])} intent errors "
                f"(out of {intent_mask.sum()} total)")

    sheet = pd.concat(samples, ignore_index=True)
    sheet = sheet[["head", "text", "true_label", "pred_label",
                   "source", "label_tier"]].copy()
    sheet["category"] = ""   # YOU fill this in
    sheet["notes"] = ""      # optional

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(ERRORS_SHEET, index=False, encoding="utf-8")
    logger.success(f"Audit sheet created: {ERRORS_SHEET}")
    logger.info(
        f"Open it. For each of the {len(sheet)} errors, read `text` "
        f"and put one of these in `category`:\n"
        f"  {sorted(VALID_CATEGORIES)}\n"
        f"Then: python -m src.evaluation.error_taxonomy score"
    )


def score_sample() -> None:
    if not ERRORS_SHEET.exists():
        raise FileNotFoundError(f"{ERRORS_SHEET} not found. Run `sample` mode first.")
    df = pd.read_csv(ERRORS_SHEET)
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    reviewed = df[df["category"].isin(VALID_CATEGORIES)]

    if reviewed.empty:
        logger.error("No rows categorized yet. Fill in the sheet first.")
        return

    lines = ["ERROR TAXONOMY SUMMARY", "=" * 50]

    for head in ["sentiment", "intent"]:
        sub = reviewed[reviewed["head"] == head]
        if sub.empty:
            continue
        counts = sub["category"].value_counts()
        pct = (counts / counts.sum() * 100).round(1)
        lines.append(f"\n{head.upper()}  (n={len(sub)})")
        for cat in sorted(VALID_CATEGORIES):
            c = int(counts.get(cat, 0))
            p = float(pct.get(cat, 0))
            lines.append(f"  {cat:22s}  {c:3d}  ({p:5.1f}%)")

    lines.append("\n" + "=" * 50)
    lines.append(
        "Use these percentages in your case study. The bigger the "
        "ambiguous/noise share, the closer your model is to the "
        "ceiling imposed by the data quality itself."
    )

    out = "\n".join(lines)
    ERRORS_SUMMARY.write_text(out, encoding="utf-8")
    print(out)
    logger.success(f"Summary saved -> {ERRORS_SUMMARY}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "sample":
        make_sample()
    elif mode == "score":
        score_sample()
    else:
        print("Usage:")
        print("  python -m src.evaluation.error_taxonomy sample")
        print("  python -m src.evaluation.error_taxonomy score")


if __name__ == "__main__":
    main()
