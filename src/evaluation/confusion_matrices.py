"""
src/evaluation/confusion_matrices.py
============================================================
Plot side-by-side confusion matrices: Baseline vs FinBERT
============================================================

WHY CONFUSION MATRICES MATTER:
    A single macro F1 number tells you the model is good. A
    confusion matrix tells you WHICH classes are still getting
    confused with WHICH. For your case study, this is the figure
    that makes the story concrete.

    Example reading: if FinBERT's intent matrix shows
    market_commentary <-> performance_report confusions, that's a
    real semantic ambiguity ("Q3 revenue was strong" reports
    performance AND comments on the market). That's a finding,
    not a failure.

DESIGN CHOICES:
    - Normalized by true class (each row sums to 1). This is the
      "recall view" — diagonal = recall per class. Better than
      raw counts because it doesn't get distorted by class size.
    - Side-by-side baseline vs FinBERT so you can see the
      diagonal getting darker (recall improving) at a glance.
    - Numbers shown on each cell. Tiny corpus by ML standards,
      so cells are readable.

OUTPUTS:
    reports/figures/confusion_sentiment.png
    reports/figures/confusion_intent.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    EVAL_DIR, FIGURES_DIR, BASELINE_DIR,
    SENTIMENT_LABELS, INTENT_LABELS,
)


def plot_pair(
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_fb: np.ndarray,
    class_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    """One figure, two confusion-matrix subplots side by side."""
    # Row-normalize: each row is "of all true class X, what
    # fraction got predicted as Y". Diagonal == recall.
    labels = list(range(len(class_names)))
    cm_b = confusion_matrix(y_true, y_pred_base, labels=labels, normalize="true")
    cm_f = confusion_matrix(y_true, y_pred_fb, labels=labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cm, name in [
        (axes[0], cm_b, "Baseline (TF-IDF + LR)"),
        (axes[1], cm_f, "FinBERT (fine-tuned)"),
    ]:
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0, vmax=1,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=ax,
            annot_kws={"size": 10},
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved -> {out_path}")


def main() -> None:
    # The evaluator saved FinBERT predictions; we need to re-run
    # the baseline on test to get its predictions too. Fast.
    pred_path = EVAL_DIR / "test_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found. Run evaluate_finbert.py first."
        )
    test_df = pd.read_parquet(pred_path)

    # Baseline predictions — load and predict on the spot.
    tfidf = joblib.load(BASELINE_DIR / "tfidf.joblib")
    sent_clf = joblib.load(BASELINE_DIR / "sentiment_clf.joblib")
    intent_clf = joblib.load(BASELINE_DIR / "intent_clf.joblib")
    X = tfidf.transform(test_df["text"].astype(str))
    test_df["pred_sentiment_baseline"] = sent_clf.predict(X)
    test_df["pred_intent_baseline"] = intent_clf.predict(X)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Sentiment ----
    mask = test_df["sentiment_id"].notna()
    if mask.any():
        sub = test_df[mask]
        plot_pair(
            y_true=sub["sentiment_id"].astype(int).to_numpy(),
            y_pred_base=sub["pred_sentiment_baseline"].astype(int).to_numpy(),
            y_pred_fb=sub["pred_sentiment_id"].astype(int).to_numpy(),
            class_names=SENTIMENT_LABELS,
            title="Sentiment — Confusion Matrices (row-normalized)",
            out_path=FIGURES_DIR / "confusion_sentiment.png",
        )

    # ---- Intent ----
    mask = test_df["intent_id"].notna()
    if mask.any():
        sub = test_df[mask]
        plot_pair(
            y_true=sub["intent_id"].astype(int).to_numpy(),
            y_pred_base=sub["pred_intent_baseline"].astype(int).to_numpy(),
            y_pred_fb=sub["pred_intent_id"].astype(int).to_numpy(),
            class_names=INTENT_LABELS,
            title="Intent — Confusion Matrices (row-normalized)",
            out_path=FIGURES_DIR / "confusion_intent.png",
        )


if __name__ == "__main__":
    main()
