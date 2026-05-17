"""
src/collection/build_master.py
============================================================
Master dataset builder — merges all 3 raw sources
============================================================

WHY THIS STEP EXISTS (same role as the real estate project's
"Step 1.6 master builder"):
    The three collection scripts each produce their own raw CSV in
    their own shape. This script unifies them into ONE interim
    dataset with a consistent schema, so every downstream stage
    (labelling, tokenizing, training) reads a single file.

    Keeping this separate from the scrapers means: re-run a scraper
    without re-running everything; change the merge logic without
    re-scraping; and have one obvious place where "the dataset" is
    defined.

THE UNIFIED SCHEMA:
    Every row, regardless of source, ends up with:
        text        — the text to classify (required)
        sentiment   — negative/neutral/positive, or NaN if unlabeled
        intent      — always NaN here; intent is assigned in Milestone 2
        source      — phrasebank / bank_of_canada / canadian_news
        meta        — JSON-ish string with any extra source fields
                      (title, url, published) so nothing is lost

    PhraseBank arrives WITH sentiment. BoC and news arrive WITHOUT
    any labels — that's expected. Milestone 2 handles labelling.

OUTPUT:
    data/interim/master_unlabeled.csv
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_PATH = INTERIM_DIR / "master_unlabeled.csv"

# The unified column order every downstream file expects.
UNIFIED_COLUMNS = ["text", "sentiment", "intent", "source", "meta"]


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load a raw CSV, or return an empty frame with a warning."""
    if not path.exists():
        logger.warning(
            f"{label} file not found at {path} — "
            f"did you run its collection script? Skipping."
        )
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {label}")
    return df


def _pack_meta(row: pd.Series, fields: list[str]) -> str:
    """
    Bundle source-specific extra columns into a single JSON string.

    WHY: different sources have different extra fields (news has
    feed_name, BoC has url, etc.). Rather than a sparse table full
    of NaNs, we pack the extras into one 'meta' column. Nothing is
    lost and the main schema stays clean.
    """
    meta = {f: row[f] for f in fields if f in row and pd.notna(row[f])}
    return json.dumps(meta, ensure_ascii=False)


def build_master() -> pd.DataFrame:
    frames = []

    # ---- Source 1: PhraseBank (already has sentiment) ----
    pb = _load_csv(RAW_DIR / "phrasebank.csv", "PhraseBank")
    if not pb.empty:
        pb_clean = pd.DataFrame(
            {
                "text": pb["text"],
                "sentiment": pb["sentiment"],   # real labels
                "intent": pd.NA,                # assigned in Milestone 2
                "source": pb["source"],
                "meta": "{}",                   # PhraseBank has no extra fields
            }
        )
        frames.append(pb_clean)

    # ---- Source 2: Bank of Canada (no labels yet) ----
    boc = _load_csv(RAW_DIR / "boc_releases.csv", "Bank of Canada")
    if not boc.empty:
        boc_clean = pd.DataFrame(
            {
                "text": boc["text"],
                "sentiment": pd.NA,             # weak-labeled in Milestone 2
                "intent": pd.NA,
                "source": boc["source"],
                "meta": boc.apply(
                    _pack_meta, axis=1, fields=["title", "url", "published"]
                ),
            }
        )
        frames.append(boc_clean)

    # ---- Source 3: Canadian news (no labels yet) ----
    news = _load_csv(RAW_DIR / "news_headlines.csv", "Canadian news")
    if not news.empty:
        news_clean = pd.DataFrame(
            {
                "text": news["text"],
                "sentiment": pd.NA,
                "intent": pd.NA,
                "source": news["source"],
                "meta": news.apply(
                    _pack_meta,
                    axis=1,
                    fields=["title", "url", "published", "feed_name"],
                ),
            }
        )
        frames.append(news_clean)

    if not frames:
        raise RuntimeError(
            "No source files found. Run the three collection scripts first:\n"
            "  python -m src.collection.load_phrasebank\n"
            "  python -m src.collection.scrape_boc\n"
            "  python -m src.collection.scrape_news"
        )

    master = pd.concat(frames, ignore_index=True)[UNIFIED_COLUMNS]

    # Final hygiene across the merged set.
    before = len(master)
    master = master.dropna(subset=["text"])
    master["text"] = master["text"].str.strip()
    master = master[master["text"].str.len() > 15]
    master = master.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info(f"Dropped {before - len(master)} bad/duplicate rows in merge")

    return master


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    master = build_master()
    master.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    logger.success(f"Master dataset built: {len(master)} rows -> {OUTPUT_PATH}")

    # A quick situational report. You want to SEE this — it tells
    # you how much labeled vs unlabeled data you have going into
    # Milestone 2, which shapes the labelling effort.
    logger.info("Rows per source:")
    print(master["source"].value_counts())

    labeled = master["sentiment"].notna().sum()
    logger.info(
        f"Sentiment-labeled rows: {labeled} / {len(master)} "
        f"({labeled / len(master):.0%}). "
        f"The rest get weak-labeled in Milestone 2."
    )


if __name__ == "__main__":
    main()
