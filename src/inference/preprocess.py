"""
src/inference/preprocess.py
============================================================
Input text normalization for inference
============================================================

WHY THIS FILE EXISTS:
    During the Milestone 4 error audit we found that Twitter-
    source training rows often contain noise tokens:
      - shortened URLs (https://t.co/...)
      - dangling cashtags ($BAC, $TSLA)
      - HTML entities (&amp;)
      - irregular whitespace
    The model learned to mostly ignore them, but in production
    we want to strip them before tokenizing so:
      1. Real-world inputs (which usually DON'T have URL noise)
         get a cleaner signal to the model.
      2. The output looks professional in the demo.

WHY NOT FIX THE TRAINING DATA AND RETRAIN?
    Diminishing returns. The model already handles the noisy
    text well (F1 = 0.858 / 0.912). The standard production
    pattern is: keep the trained model, clean inputs to match
    its expected distribution. That's all this file does.

DELIBERATELY NOT DOING:
    - Lowercasing (BERT tokenizers handle case themselves)
    - Stopword removal (the model needs every token)
    - Stemming (subword tokenization makes it irrelevant)
    These are bag-of-words preprocessing steps that would HURT
    a transformer.
"""

import html
import re

# Compiled regexes — built once, reused per call. Order matters:
# URLs first (they may contain punctuation), then other patterns.
URL_RX = re.compile(r"https?://\S+|www\.\S+")
CASHTAG_RX = re.compile(r"\$[A-Z]{1,5}\b")     # $TSLA, $BAC
MENTION_RX = re.compile(r"@[A-Za-z0-9_]+")      # @username
HASHTAG_RX = re.compile(r"#(\w+)")              # #earnings -> "earnings"
MULTI_SPACE_RX = re.compile(r"\s+")
TRAILING_ELLIPSIS_RX = re.compile(r"\.{2,}\s*$")


def preprocess(text: str) -> str:
    """
    Clean a piece of text for FinBERT inference.

    Steps, in order:
      1. Decode HTML entities (&amp; -> &)
      2. Strip URLs
      3. Drop @mentions (no semantic content for our task)
      4. Keep cashtags as plain tickers ($TSLA -> TSLA)
      5. Unwrap hashtags (#earnings -> earnings)
      6. Collapse whitespace
      7. Strip trailing "..."
    """
    if not isinstance(text, str):
        return ""

    s = html.unescape(text)
    s = URL_RX.sub("", s)
    s = MENTION_RX.sub("", s)
    s = CASHTAG_RX.sub(lambda m: m.group(0)[1:], s)   # drop the $
    s = HASHTAG_RX.sub(r"\1", s)                       # drop the #
    s = TRAILING_ELLIPSIS_RX.sub("", s)
    s = MULTI_SPACE_RX.sub(" ", s).strip()

    return s


# Quick self-test you can run as a script:
#   python -m src.inference.preprocess
if __name__ == "__main__":
    examples = [
        "California boosts pot taxes, shocking unsteady industry https://t.co/0j0H",
        "$TSLA delivered 466,000 vehicles in Q4 &amp; missed estimates @elonmusk",
        "BoC holds rate at 5%   #monetarypolicy ...",
    ]
    for ex in examples:
        print(f"IN : {ex}")
        print(f"OUT: {preprocess(ex)}")
        print()
