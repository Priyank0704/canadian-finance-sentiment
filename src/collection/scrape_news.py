"""
src/collection/scrape_news.py
============================================================
Source 3 of 3 — Canadian financial news (the intent layer)
============================================================
PATCHED v2: adds per-feed HTTP timeout + try/except so a single
            unresponsive RSS server can never hang the script.

WHY THIS PATCH:
    feedparser.parse(url) has NO network timeout by default. If a
    publisher's RSS server stops responding (CBC and BNN do this
    intermittently), feedparser blocks forever. The fix:
      1. Use `requests.get(url, timeout=...)` to fetch the XML.
      2. Hand the resulting bytes to `feedparser.parse(bytes)`.
      3. Wrap each feed in try/except so one bad feed doesn't
         kill the whole run.
    This is the same robustness pattern your fraud project used
    for external API calls.
"""

from pathlib import Path

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "news_headlines.csv"

# Canadian-market financial news RSS feeds.
NEWS_FEEDS = {
    "financial_post_economy": "https://financialpost.com/category/news/economy/feed",
    "financial_post_markets": "https://financialpost.com/category/investing/feed",
    "cbc_business": "https://www.cbc.ca/webfeed/rss/rss-business",
    "bnn_bloomberg": "https://www.bnnbloomberg.ca/rss",
}

# Hard timeout per feed. 15s is generous — a healthy RSS feed
# responds in under 2s. If we hit this, we move on.
FEED_TIMEOUT_SECONDS = 15

# Identify ourselves politely; some servers reject requests with
# no User-Agent, which can look like a hang.
HEADERS = {
    "User-Agent": (
        "CanFinanceSentiment-Research/1.0 "
        "(portfolio project; contact: your-email@example.com)"
    )
}

FINANCE_KEYWORDS = {
    "rate", "rates", "inflation", "earnings", "market", "markets",
    "stock", "stocks", "tsx", "economy", "gdp", "bank", "banks",
    "investor", "investors", "profit", "loss", "losses", "revenue",
    "quarter", "fiscal", "dividend", "bond", "yield", "recession",
    "growth", "forecast", "guidance", "loan", "mortgage", "housing",
}


def is_finance_related(text: str) -> bool:
    words = set(text.lower().split())
    return len(words & FINANCE_KEYWORDS) > 0


def fetch_feed_with_timeout(url: str):
    """
    Fetch the RSS XML via requests (which respects timeout), then
    feed the bytes to feedparser. Returns feedparser's parsed
    object, or None on any failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=FEED_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        logger.warning(f"TIMEOUT after {FEED_TIMEOUT_SECONDS}s: {url}")
        return None
    except requests.RequestException as exc:
        logger.warning(f"HTTP error on {url}: {exc}")
        return None

    return feedparser.parse(resp.content)


def scrape_news() -> pd.DataFrame:
    """Pull all configured news feeds into one DataFrame."""
    rows = []

    for feed_name, feed_url in NEWS_FEEDS.items():
        logger.info(f"Parsing feed '{feed_name}': {feed_url}")

        feed = fetch_feed_with_timeout(feed_url)
        if feed is None:
            logger.warning(f"  '{feed_name}' unreachable - skipping.")
            continue

        if feed.bozo:
            logger.warning(f"  '{feed_name}' had parsing issues (skipping if empty)")

        entries = feed.entries
        logger.info(f"  found {len(entries)} entries")

        kept = 0
        for entry in entries:
            title = entry.get("title", "").strip()
            url = entry.get("link", "")
            published = entry.get("published", "")

            summary_html = entry.get("summary", "")
            summary = BeautifulSoup(summary_html, "lxml").get_text(strip=True)

            text = f"{title}. {summary}".strip()

            if not is_finance_related(text):
                continue

            rows.append(
                {
                    "text": text,
                    "title": title,
                    "published": published,
                    "url": url,
                    "feed_name": feed_name,
                    "source": "canadian_news",
                }
            )
            kept += 1

        logger.info(f"  kept {kept} finance-related entries")

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    before = len(df)
    df = df[df["text"].str.len() > 25]
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} short/duplicate rows")

    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = scrape_news()

    if df.empty:
        logger.error(
            "No news rows collected. All feeds may be stale/unreachable - "
            "you can proceed without this source; tiers 1+2 carry the load."
        )
        return

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logger.success(f"Saved {len(df)} news rows -> {OUTPUT_PATH}")

    logger.info("Rows per feed:")
    print(df["feed_name"].value_counts())


if __name__ == "__main__":
    main()
