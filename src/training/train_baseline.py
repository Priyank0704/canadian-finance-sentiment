"""
src/training/train_baseline.py
============================================================
Baseline: TF-IDF + Logistic Regression
============================================================

WHY A BASELINE FIRST:
    Always train a simple model before a complex one. Three reasons:
      1. It tells you what "easy" performance looks like — if
         FinBERT only beats it by 2 F1 points, you're paying a
         huge complexity cost for very little.
      2. It catches data leaks, label-encoding bugs, and class
         imbalance issues early, while iteration is fast.
      3. In your case study you say:
            "TF-IDF + logistic regression scored X F1.
             Fine-tuned FinBERT scored Y F1, a +Z improvement."
         That comparison IS the result. Without the baseline,
         FinBERT's number is just a number.

WHY TF-IDF + LOGISTIC REGRESSION:
    TF-IDF turns text into sparse word/n-gram count vectors,
    weighted by how rare each term is across the corpus. Logistic
    regression on those vectors is the gold-standard text-
    classification baseline. It's interpretable (you can see which
    words drive each class) and trains in seconds.

WHAT THIS FILE DOES:
    Trains TWO independent models — one for sentiment, one for
    intent. Uses class_weight='balanced' for imbalance. Saves
    both models, the shared TF-IDF vectorizer, and a metrics JSON.

OUTPUT:
    models/baseline/sentiment_clf.joblib
    models/baseline/intent_clf.joblib
    models/baseline/tfidf.joblib
    reports/eval/baseline_metrics.json
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    PROCESSED_DIR, BASELINE_DIR, EVAL_DIR,
    SENTIMENT_LABELS, INTENT_LABELS, RANDOM_SEED,
)


def load_splits():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    return train, val, test


def fit_tfidf(train_text: pd.Series) -> TfidfVectorizer:
    """
    Fit one shared TF-IDF vectorizer on ALL training text.

    Why shared: the vectorizer only learns the vocabulary from
    raw text — it doesn't see labels. Both heads benefit from
    the largest possible fitting corpus.
    """
    vec = TfidfVectorizer(
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=3,                # ignore terms in <3 docs
        max_df=0.95,             # ignore terms in >95% of docs
        sublinear_tf=True,       # 1 + log(tf) — dampens common words
        strip_accents="unicode",
        lowercase=True,
        max_features=50_000,     # cap vocab for memory
    )
    vec.fit(train_text.astype(str))
    logger.info(f"TF-IDF vocabulary size: {len(vec.vocabulary_):,}")
    return vec


def train_head(
    head_name: str,
    label_col: str,
    class_names: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    tfidf: TfidfVectorizer,
) -> dict:
    """Train one logistic regression head; return metrics dict."""
    # Each head trains only on rows where ITS label is present.
    # This is the same partial-supervision idea we use in FinBERT.
    train_lbl = train.dropna(subset=[label_col])
    val_lbl = val.dropna(subset=[label_col])
    test_lbl = test.dropna(subset=[label_col])

    logger.info(
        f"[{head_name}] train={len(train_lbl)}, "
        f"val={len(val_lbl)}, test={len(test_lbl)}"
    )

    X_train = tfidf.transform(train_lbl["text"].astype(str))
    y_train = train_lbl[label_col].astype(int)
    X_val = tfidf.transform(val_lbl["text"].astype(str))
    y_val = val_lbl[label_col].astype(int)
    X_test = tfidf.transform(test_lbl["text"].astype(str))
    y_test = test_lbl[label_col].astype(int)

    # class_weight='balanced' applies inverse-frequency weights —
    # the same idea as our manual weights for FinBERT.
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    logger.info(f"[{head_name}] fitting logistic regression...")
    clf.fit(X_train, y_train)

    metrics = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = clf.predict(X)
        # 'macro' F1 weights every class equally — the right metric
        # when classes are imbalanced and you care about minorities.
        macro_f1 = f1_score(y, y_pred, average="macro")
        weighted_f1 = f1_score(y, y_pred, average="weighted")
        report = classification_report(
            y, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        metrics[split_name] = {
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": report,
        }
        logger.info(
            f"[{head_name}] {split_name} macro F1={macro_f1:.3f} "
            f"weighted F1={weighted_f1:.3f}"
        )
        print(classification_report(
            y, y_pred, target_names=class_names, zero_division=0
        ))

    # Save the trained model.
    out = BASELINE_DIR / f"{head_name}_clf.joblib"
    joblib.dump(clf, out)
    logger.success(f"[{head_name}] model saved -> {out}")

    return metrics


def main() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    train, val, test = load_splits()
    logger.info(
        f"Loaded splits: train={len(train)}, val={len(val)}, test={len(test)}"
    )

    tfidf = fit_tfidf(train["text"])
    joblib.dump(tfidf, BASELINE_DIR / "tfidf.joblib")
    logger.success(f"TF-IDF vectorizer saved -> {BASELINE_DIR / 'tfidf.joblib'}")

    all_metrics = {}
    all_metrics["sentiment"] = train_head(
        "sentiment", "sentiment_id", SENTIMENT_LABELS,
        train, val, test, tfidf,
    )
    all_metrics["intent"] = train_head(
        "intent", "intent_id", INTENT_LABELS,
        train, val, test, tfidf,
    )

    out_path = EVAL_DIR / "baseline_metrics.json"
    out_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    logger.success(f"Baseline metrics saved -> {out_path}")

    # The one-line summary you'll quote in your case study.
    logger.info("=" * 60)
    logger.info("BASELINE NUMBERS TO BEAT (test set, macro F1):")
    logger.info(
        f"  Sentiment: {all_metrics['sentiment']['test']['macro_f1']:.3f}"
    )
    logger.info(
        f"  Intent:    {all_metrics['intent']['test']['macro_f1']:.3f}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
