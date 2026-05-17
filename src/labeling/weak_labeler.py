"""
src/labeling/weak_labeler.py
============================================================
Tier 3 labels — rule-based weak supervision for Canadian data
============================================================

WHY WEAK LABELING:
    Your Bank of Canada releases and Canadian news headlines came
    out of Milestone 1 with NO labels. Hand-labeling thousands of
    rows is infeasible for a portfolio project. Weak supervision —
    writing rules that assign labels programmatically — is the
    industry-standard answer (it's the core idea behind tools like
    Snorkel).

    The rules are imperfect. That's expected and FINE, as long as:
      1. You mark these rows as tier3 (lower confidence).
      2. You spot-check a sample to estimate rule accuracy.
      3. You don't let tier3 dominate your test set (Milestone 3
         builds the test set mostly from tier1/tier2).

HOW THE RULES WORK:
    Sentiment: financial text has fairly reliable signal words.
    "rose, gains, beat, strong, growth" lean positive; "fell,
    losses, warns, weak, decline" lean negative. We count matches
    and take the stronger side; ties or no-matches -> neutral.

    Intent: each intent class has characteristic phrasing.
    "expects, forecast, outlook, will" -> forward_guidance.
    "warns, risk, concern, threat" -> risk_warning. Etc.
    First matching rule wins, with a sensible default.

    This is keyword/phrase matching, same philosophy as the query
    router in your RAG project — simple, fast, transparent, and
    good enough to bootstrap. A model trained on these labels
    generalizes BEYOND the keywords.

OUTPUT:
    data/labeled/canadian_weak_labeled.csv
        [text, sentiment, intent, source, label_tier]
"""

import re
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import RAW_DIR, LABELED_DIR  # noqa: E402

# ============================================================
# SENTIMENT RULES
# ============================================================
# Word stems that signal each polarity. Using stems (e.g. "gain"
# matches gain/gains/gained/gaining) keeps the lists short.
POSITIVE_TERMS = [
    r"\brose\b", r"\brise[sn]?\b", r"\bgain", r"\bbeat\b", r"\bbeats\b",
    r"\bstrong", r"\bgrowth\b", r"\bgrew\b", r"\bsurge", r"\brally",
    r"\bprofit", r"\boutperform", r"\bupgrade", r"\brecord high",
    r"\bboost", r"\bimprove", r"\bexceed", r"\boptimis", r"\bbullish\b",
]
NEGATIVE_TERMS = [
    r"\bfell\b", r"\bfall[sn]?\b", r"\bdrop", r"\bloss", r"\blosses\b",
    r"\bweak", r"\bdecline", r"\bslump", r"\bplunge", r"\bwarns?\b",
    r"\bwarning\b", r"\bcut[s]?\b", r"\bmiss(?:es|ed)?\b", r"\bdowngrade",
    r"\brecession\b", r"\bcrisis\b", r"\bconcern", r"\bpressure\b",
    r"\bbearish\b", r"\bslowdown\b", r"\bdeficit\b", r"\bdefault\b",
]

POSITIVE_RX = [re.compile(p, re.IGNORECASE) for p in POSITIVE_TERMS]
NEGATIVE_RX = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_TERMS]


def label_sentiment(text: str) -> str:
    """Count positive vs negative signal words; stronger side wins."""
    pos = sum(1 for rx in POSITIVE_RX if rx.search(text))
    neg = sum(1 for rx in NEGATIVE_RX if rx.search(text))

    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    # Tie or no signal -> neutral. Most BoC text is genuinely
    # neutral in tone, so this default is reasonable for our data.
    return "neutral"


# ============================================================
# INTENT RULES
# ============================================================
# Ordered list of (intent, compiled-patterns). First intent with
# ANY pattern match wins. Order matters — more specific intents
# (risk_warning, forward_guidance) are checked before the
# catch-all market_commentary.
INTENT_RULES = [
    (
        "forward_guidance",
        [r"\bexpect", r"\bforecast", r"\boutlook\b", r"\bprojec",
         r"\bguidance\b", r"\banticipat", r"\bwill likely\b",
         r"\bgoing forward\b", r"\bin the coming\b", r"\bahead\b"],
    ),
    (
        "risk_warning",
        [r"\bwarn", r"\brisk", r"\bconcern", r"\bthreat", r"\bcaution",
         r"\bvulnerab", r"\bexposure\b", r"\buncertain", r"\bdownside\b",
         r"\bheadwind"],
    ),
    (
        "policy_signal",
        [r"\bbank of canada\b", r"\binterest rate", r"\bovernight rate\b",
         r"\bmonetary policy\b", r"\bbasis points?\b", r"\bgoverning council\b",
         r"\brate (?:hike|cut|hold|decision)", r"\binflation target",
         r"\bcentral bank\b", r"\bregulat"],
    ),
    (
        "analyst_question",
        [r"\banalyst", r"\bestimate[sd]?\b", r"\bconsensus\b",
         r"\bprice target\b", r"\brating\b", r"\bupgrade[sd]?\b",
         r"\bdowngrade[sd]?\b", r"\bcoverage\b"],
    ),
    (
        "performance_report",
        [r"\bearnings\b", r"\bquarter", r"\brevenue\b", r"\bprofit",
         r"\breported\b", r"\bresults\b", r"\bQ[1-4]\b", r"\bfiscal\b",
         r"\bdividend\b", r"\bsales\b", r"\bnet income\b"],
    ),
    # market_commentary has no rules — it's the default fallback
    # for text that doesn't clearly match any specific intent.
]

# Pre-compile every pattern once.
COMPILED_INTENT_RULES = [
    (intent, [re.compile(p, re.IGNORECASE) for p in patterns])
    for intent, patterns in INTENT_RULES
]


def label_intent(text: str) -> str:
    """Return the first intent whose patterns match; else default."""
    for intent, patterns in COMPILED_INTENT_RULES:
        if any(rx.search(text) for rx in patterns):
            return intent
    return "market_commentary"  # catch-all default


def weak_label_file(path: Path, source_name: str) -> pd.DataFrame:
    """Apply both rule sets to one raw CSV."""
    if not path.exists():
        logger.warning(f"{source_name} not found at {path} — skipping.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Weak-labeling {len(df)} rows from {source_name}...")

    df["sentiment"] = df["text"].astype(str).apply(label_sentiment)
    df["intent"] = df["text"].astype(str).apply(label_intent)
    df["label_tier"] = "tier3"

    return df[["text", "sentiment", "intent", "source", "label_tier"]]


def main() -> None:
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    frames = [
        weak_label_file(RAW_DIR / "boc_releases.csv", "Bank of Canada"),
        weak_label_file(RAW_DIR / "news_headlines.csv", "Canadian news"),
    ]
    frames = [f for f in frames if not f.empty]

    if not frames:
        logger.error("No Canadian raw files found — run Milestone 1 scrapers first.")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    out_path = LABELED_DIR / "canadian_weak_labeled.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.success(f"Saved {len(df)} weak-labeled rows -> {out_path}")

    # ALWAYS print these distributions. If one class is 95% of the
    # data, your rules are too blunt and need tuning before you
    # train anything on them.
    logger.info("Weak sentiment distribution:")
    print(df["sentiment"].value_counts())
    logger.info("Weak intent distribution:")
    print(df["intent"].value_counts())

    logger.warning(
        "These are TIER 3 (rule-based) labels. Next step: run "
        "spot_check.py to sample and manually verify ~50 rows so "
        "you can report rule accuracy in your case study."
    )


if __name__ == "__main__":
    main()
