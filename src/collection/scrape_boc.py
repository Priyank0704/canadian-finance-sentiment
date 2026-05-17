"""
src/collection/scrape_boc.py
============================================================
Source 2 of 3 — Bank of Canada press releases (Canadian layer)
============================================================

WHY THIS SOURCE:
    Financial PhraseBank teaches the model sentiment language, but
    it's American/European corporate text. The Bank of Canada
    publishes the canonical Canadian financial-policy text: rate
    announcements, Monetary Policy Reports, press releases.

    This is what makes the project *Canadian*. An employer at RBC
    or a Canadian fintech sees "fine-tuned on Bank of Canada
    communications" and immediately understands you built for the
    local market — exactly the signal the real estate project sent
    with CREA data.

WHY RSS, NOT HTML SCRAPING:
    The Bank of Canada publishes a clean RSS feed of its press
    content. RSS is structured XML — far more stable than scraping
    HTML, which breaks every time a site redesigns. feedparser
    handles all the XML parsing for us.

    We pull the feed, then for richer entries optionally fetch the
    full article body. Headlines + summaries alone are already
    useful short text for classification.

WEAK LABELLING NOTE:
    These rows come out UNLABELED. BoC language is formulaic
    ("the Governing Council decided to raise/hold/lower..."), so
    in Milestone 2 we'll weak-label them with simple rules and
    then spot-check. We do NOT label here — collection and
    labelling are deliberately separate stages.

OUTPUT:
    data/raw/boc_releases.csv  with columns:
        [text, title, published, url, source]
"""

import time
from pathlib import Path

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "boc_releases.csv"

# Bank of Canada publishes several feeds. The press-content feed
# carries rate announcements and press releases — the highest-
# signal text for our task.
BOC_FEEDS = [
    "https://www.bankofcanada.ca/content_type/press-releases/feed/",
    "https://www.bankofcanada.ca/content_type/speeches/feed/",
]

# Identify ourselves politely. Scrapers that send a real
# User-Agent and pace their requests get blocked far less often.
HEADERS = {
    "User-Agent": (
        "CanFinanceSentiment-Research/1.0 "
        "(portfolio project; contact: your-email@example.com)"
    )
}

# Be a good citizen: pause between full-article fetches so we
# don't hammer the server. 1 second is generous and safe.
REQUEST_DELAY_SECONDS = 1.0


def fetch_full_text(url: str) -> str:
    """
    Fetch the full article body for one release.

    Returns an empty string on any failure — collection should
    never crash because one URL is down. We log it and move on.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning(f"Could not fetch {url}: {exc}")
        return ""

    soup = BeautifulSoup(resp.text, "lxml")

    # BoC article bodies live in the main content region. We grab
    # all paragraph tags inside it and join them. If the selector
    # ever changes, only this line needs updating.
    content = soup.find("div", class_="content-area") or soup.find("main")
    if content is None:
        return ""

    paragraphs = [p.get_text(strip=True) for p in content.find_all("p")]
    # Drop very short fragments (nav text, captions, disclaimers).
    paragraphs = [p for p in paragraphs if len(p) > 40]
    return " ".join(paragraphs)


def scrape_boc() -> pd.DataFrame:
    """Pull all configured BoC feeds into one DataFrame."""
    rows = []

    for feed_url in BOC_FEEDS:
        logger.info(f"Parsing feed: {feed_url}")
        feed = feedparser.parse(feed_url)

        if feed.bozo:
            # bozo=1 means feedparser hit a malformed feed. Log and
            # continue — one bad feed shouldn't kill the whole run.
            logger.warning(f"Feed had parsing issues: {feed_url}")

        logger.info(f"  found {len(feed.entries)} entries")

        for entry in feed.entries:
            title = entry.get("title", "").strip()
            url = entry.get("link", "")
            published = entry.get("published", "")

            # The RSS summary is a useful short-text sample on its
            # own. Strip any HTML tags the feed includes.
            summary_html = entry.get("summary", "")
            summary = BeautifulSoup(summary_html, "lxml").get_text(strip=True)

            # Try to enrich with the full article body.
            full_text = fetch_full_text(url) if url else ""
            time.sleep(REQUEST_DELAY_SECONDS)

            # Prefer full text; fall back to summary; fall back to title.
            text = full_text or summary or title

            rows.append(
                {
                    "text": text,
                    "title": title,
                    "published": published,
                    "url": url,
                    "source": "bank_of_canada",
                }
            )

    df = pd.DataFrame(rows)

    # Hygiene: drop empties and duplicates.
    before = len(df)
    df = df[df["text"].str.len() > 40]
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} empty/duplicate rows")

    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = scrape_boc()

    if df.empty:
        logger.error(
            "No rows collected. The feed URLs may have changed — "
            "check https://www.bankofcanada.ca/rss-feeds/ for current feeds."
        )
        return

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logger.success(f"Saved {len(df)} BoC rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
