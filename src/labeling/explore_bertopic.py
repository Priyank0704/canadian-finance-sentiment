"""
src/labeling/explore_bertopic.py
============================================================
BERTopic exploration — validate the intent taxonomy with data
============================================================
PATCHED v2: 
    - Now builds the corpus from ALL labeled sources (PhraseBank,
      Twitter sentiment, Twitter intent, Canadian weak) so we
      actually have enough text for topic discovery.
    - Vectorizer min_df now scales to corpus size so tiny corpora
      don't trigger sklearn's "max_df < min_df" error.

WHY THE PATCH:
    The first run found only 40 weak-labeled Canadian rows — far
    too few for topic modeling. BERTopic needs hundreds to
    thousands of documents to find meaningful clusters. The
    Twitter datasets you already downloaded give ~33k labeled
    rows — that's the corpus we should be exploring.

    Also: sklearn's CountVectorizer enforces max_df > min_df. On
    a tiny corpus, min_df=3 with the default max_df=1.0 can
    invert. We now set min_df adaptively (1 for small, 3 for
    large) so the code works at any scale.

WHAT BERTOPIC DOES (so you can explain it in interviews):
    1. Embeds every document with a sentence transformer.
    2. Reduces dimensionality with UMAP.
    3. Clusters the reduced embeddings with HDBSCAN.
    4. Names each cluster using class-based TF-IDF (c-TF-IDF) —
       the words most distinctive to that cluster.
    HDBSCAN decides how many topics exist; you don't.

CPU NOTE:
    This runs on CPU. With ~4000 short docs, total runtime is a
    few minutes. We cap MAX_DOCS so it stays manageable.

OUTPUT:
    reports/figures/bertopic_topics.html    interactive topic viz
    reports/eval/bertopic_topic_words.csv   top words per topic
    reports/eval/bertopic_vs_intent.csv     cross-tab vs weak intents
"""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    INTERIM_DIR, LABELED_DIR, RAW_DIR, FIGURES_DIR, EVAL_DIR, RANDOM_SEED,
)

# Cap corpus size so CPU runtime stays reasonable.
MAX_DOCS = 4000


def load_corpus() -> pd.DataFrame:
    """
    Build the corpus from EVERY available labeled source.

    Each row carries:
      text      — the document
      intent    — if known (Twitter intent set + Canadian weak)
      source    — which dataset it came from

    The `intent` column is what powers the validation cross-tab
    later. Sources without intent labels still contribute to the
    topic clustering — they just won't appear in the cross-tab.
    """
    frames = []

    # ---- Tier 1: PhraseBank (sentiment only, no intent) ----
    pb_path = RAW_DIR / "phrasebank.csv"
    if pb_path.exists():
        pb = pd.read_csv(pb_path)
        pb["intent"] = pd.NA
        pb["source"] = "phrasebank"
        frames.append(pb[["text", "intent", "source"]])
        logger.info(f"Loaded {len(pb)} rows from PhraseBank")

    # ---- Tier 2a: Twitter sentiment (no intent) ----
    ts_path = LABELED_DIR / "twitter_sentiment.csv"
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        ts["intent"] = pd.NA
        frames.append(ts[["text", "intent", "source"]])
        logger.info(f"Loaded {len(ts)} rows from Twitter sentiment")

    # ---- Tier 2b: Twitter intent (HAS intent — gold for the cross-tab) ----
    ti_path = LABELED_DIR / "twitter_intent.csv"
    if ti_path.exists():
        ti = pd.read_csv(ti_path)
        frames.append(ti[["text", "intent", "source"]])
        logger.info(f"Loaded {len(ti)} rows from Twitter intent")

    # ---- Tier 3: Canadian weak labels ----
    cw_path = LABELED_DIR / "canadian_weak_labeled.csv"
    if cw_path.exists():
        cw = pd.read_csv(cw_path)
        frames.append(cw[["text", "intent", "source"]])
        logger.info(f"Loaded {len(cw)} rows from Canadian weak-labeled")

    if not frames:
        raise FileNotFoundError(
            "No labeled sources found. Run, in order:\n"
            "  python -m src.collection.load_phrasebank\n"
            "  python -m src.labeling.load_twitter_finance\n"
            "  python -m src.labeling.weak_labeler"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 15]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    logger.info(f"Combined corpus: {len(df)} unique documents")

    if len(df) > MAX_DOCS:
        # Stratified sample by source so every source is represented
        # in the topic map (not just the biggest one).
        df = (
            df.groupby("source", group_keys=False)
            .apply(
                lambda g: g.sample(
                    min(len(g), MAX_DOCS // df["source"].nunique()),
                    random_state=RANDOM_SEED,
                )
            )
            .reset_index(drop=True)
        )
        logger.info(f"Stratified sample taken: {len(df)} docs for BERTopic")

    return df


def run_bertopic(df: pd.DataFrame) -> None:
    # Heavy imports kept inside the function so importing this
    # module doesn't pay the cost unless we actually run it.
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    docs = df["text"].astype(str).tolist()
    n_docs = len(docs)

    # ── Corpus-size-adaptive vectorizer settings ───────────
    # min_df = minimum number of documents a term must appear in.
    # On large corpora, 3 keeps the topic words clean (drops typos
    # and one-offs). On small corpora, 3 can wipe out every term
    # and inversely violate max_df > min_df. Scale it.
    if n_docs < 200:
        min_df, max_df = 1, 1.0
        min_topic_size = 5
    elif n_docs < 1000:
        min_df, max_df = 2, 0.95
        min_topic_size = 15
    else:
        min_df, max_df = 3, 0.95
        min_topic_size = 30

    logger.info(
        f"Vectorizer config for {n_docs} docs: "
        f"min_df={min_df}, max_df={max_df}, min_topic_size={min_topic_size}"
    )

    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),  # allow bigrams like "interest rate"
    )

    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=True,
    )

    logger.info("Fitting BERTopic (embed -> UMAP -> HDBSCAN -> c-TF-IDF)...")
    topics, _ = topic_model.fit_transform(docs)
    df = df.copy()
    df["bertopic"] = topics

    # ---- Output 1: interactive topic visualization ----
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fig = topic_model.visualize_topics()
        fig_path = FIGURES_DIR / "bertopic_topics.html"
        fig.write_html(str(fig_path))
        logger.success(f"Topic viz saved -> {fig_path}")
    except Exception as exc:
        # visualize_topics needs >=2 topics; tiny corpora can fail.
        logger.warning(f"Could not render topic viz ({exc}). Continuing.")

    # ---- Output 2: top words per topic ----
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(
        EVAL_DIR / "bertopic_topic_words.csv", index=False, encoding="utf-8"
    )
    logger.success("Top words per topic saved -> bertopic_topic_words.csv")

    logger.info("Discovered topics (topic -1 = outliers/noise):")
    print(topic_info[["Topic", "Count", "Name"]].head(15).to_string(index=False))

    # ---- Output 3: cross-tab BERTopic vs known intent labels ----
    # ONLY uses rows that have a real intent label (Twitter intent +
    # Canadian weak). If every topic is dominated by one intent
    # column, your taxonomy is data-grounded.
    intent_df = df[df["intent"].notna()]
    if not intent_df.empty:
        crosstab = pd.crosstab(intent_df["bertopic"], intent_df["intent"])
        crosstab.to_csv(EVAL_DIR / "bertopic_vs_intent.csv", encoding="utf-8")
        logger.success(
            "Cross-tab (BERTopic x intent) saved -> bertopic_vs_intent.csv "
            f"(based on {len(intent_df)} intent-labeled rows)"
        )
        logger.info(
            "Open that file: if each row (topic) is dominated by ONE "
            "intent column, your taxonomy matches the data's natural "
            "structure. If topics smear across intents, consider "
            "revising the taxonomy before Milestone 3."
        )
    else:
        logger.warning(
            "No intent-labeled rows in corpus — skipping cross-tab. "
            "Make sure load_twitter_finance.py and weak_labeler.py ran."
        )


def main() -> None:
    df = load_corpus()
    run_bertopic(df)
    logger.success(
        "BERTopic exploration complete. Screenshot bertopic_topics.html "
        "for your portfolio — it's a striking visual."
    )


if __name__ == "__main__":
    main()
