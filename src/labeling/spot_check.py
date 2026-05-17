"""
src/labeling/spot_check.py
============================================================
Spot-check weak labels — measure rule accuracy honestly
============================================================

WHY THIS FILE EXISTS:
    Weak labels are only credible if you VERIFY them. This tool
    samples a small random set of weak-labeled rows and writes
    them to a CSV with blank "correct?" columns. You open that CSV,
    eyeball each row, and mark whether the rule got it right.

    Run it again in "score" mode and it computes your weak-label
    accuracy. THAT NUMBER goes in your case study:
        "Rule-based weak labeling achieved 78% sentiment accuracy
         and 71% intent accuracy on a 50-row manual audit."
    That sentence demonstrates rigor most candidates skip.

TWO MODES:
    python -m src.labeling.spot_check sample   # make the audit sheet
    python -m src.labeling.spot_check score    # score it after you fill it in

WORKFLOW:
    1. Run `sample`  -> creates reports/eval/spot_check_sheet.csv
    2. Open it in Excel. For each row, look at `text`, then put
       "y" or "n" in `sentiment_ok` and `intent_ok`.
    3. Save it.
    4. Run `score`   -> prints accuracy and saves a summary.
"""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import LABELED_DIR, EVAL_DIR, RANDOM_SEED  # noqa: E402

WEAK_LABELED_PATH = LABELED_DIR / "canadian_weak_labeled.csv"
SHEET_PATH = EVAL_DIR / "spot_check_sheet.csv"
SUMMARY_PATH = EVAL_DIR / "spot_check_summary.txt"

SAMPLE_SIZE = 50  # 50 rows is enough to estimate accuracy, small
                  # enough to audit in ~15 minutes by hand.


def make_sample() -> None:
    """Create the blank audit sheet for manual review."""
    if not WEAK_LABELED_PATH.exists():
        logger.error(f"{WEAK_LABELED_PATH} not found — run weak_labeler.py first.")
        return

    df = pd.read_csv(WEAK_LABELED_PATH)

    # Stratify lightly: sample across sentiment classes so you
    # don't audit 48 neutrals and 2 of everything else.
    n_per_class = max(1, SAMPLE_SIZE // df["sentiment"].nunique())
    sample = (
        df.groupby("sentiment", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), n_per_class), random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )

    # Top up to SAMPLE_SIZE if stratification under-filled.
    if len(sample) < SAMPLE_SIZE:
        extra = df.drop(sample.index, errors="ignore").sample(
            min(SAMPLE_SIZE - len(sample), len(df) - len(sample)),
            random_state=RANDOM_SEED,
        )
        sample = pd.concat([sample, extra], ignore_index=True)

    # Add the blank columns YOU fill in.
    sample["sentiment_ok"] = ""   # put "y" or "n"
    sample["intent_ok"] = ""      # put "y" or "n"
    sample["notes"] = ""          # optional: why it was wrong

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SHEET_PATH, index=False, encoding="utf-8")

    logger.success(f"Audit sheet created: {SHEET_PATH}")
    logger.info(
        f"Open it in Excel. For each of the {len(sample)} rows, read "
        f"the `text` and mark `sentiment_ok` and `intent_ok` with y/n. "
        f"Then run: python -m src.labeling.spot_check score"
    )


def score_sample() -> None:
    """Read the filled-in audit sheet and compute accuracy."""
    if not SHEET_PATH.exists():
        logger.error(f"{SHEET_PATH} not found — run `sample` mode first.")
        return

    df = pd.read_csv(SHEET_PATH)

    # Normalise the y/n entries.
    for col in ["sentiment_ok", "intent_ok"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    reviewed = df[df["sentiment_ok"].isin(["y", "n"])]
    if reviewed.empty:
        logger.error(
            "No rows marked yet. Open the sheet and fill in y/n first."
        )
        return

    sent_acc = (reviewed["sentiment_ok"] == "y").mean()
    intent_reviewed = df[df["intent_ok"].isin(["y", "n"])]
    intent_acc = (
        (intent_reviewed["intent_ok"] == "y").mean()
        if not intent_reviewed.empty else float("nan")
    )

    summary = (
        f"WEAK-LABEL SPOT-CHECK SUMMARY\n"
        f"{'=' * 40}\n"
        f"Rows audited (sentiment): {len(reviewed)}\n"
        f"Sentiment rule accuracy : {sent_acc:.1%}\n"
        f"Rows audited (intent)   : {len(intent_reviewed)}\n"
        f"Intent rule accuracy    : {intent_acc:.1%}\n"
        f"{'=' * 40}\n"
        f"Use these numbers in your case study. If accuracy is\n"
        f"below ~65%, tune the rules in weak_labeler.py and re-audit.\n"
    )

    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)
    logger.success(f"Summary saved -> {SUMMARY_PATH}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "sample":
        make_sample()
    elif mode == "score":
        score_sample()
    else:
        print("Usage:")
        print("  python -m src.labeling.spot_check sample   # create audit sheet")
        print("  python -m src.labeling.spot_check score    # score it after filling in")


if __name__ == "__main__":
    main()
