"""
src/evaluation/slice_and_calibration.py
============================================================
Slice-by-tier scoring + calibration check
============================================================

WHY THESE TWO ANALYSES TOGETHER:
    Both ask "should we trust the headline number?" — from
    different angles.

    TIER BREAKDOWN: scores the model separately on tier1
    (PhraseBank, expert-labeled) and tier2 (Twitter, crowd-
    labeled) test rows. If tier2 is much lower than tier1, the
    Twitter labels are noisier and the model is being penalized
    for matching the underlying ambiguity. That's a label-quality
    finding, not a model failure.

    CALIBRATION: does the model's confidence score match its
    accuracy? A model that says 90% confidence should be right
    90% of the time. We compute the reliability diagram for
    sentiment (the head where calibration matters most for
    downstream use).

OUTPUTS:
    reports/eval/tier_breakdown.csv
    reports/figures/calibration_sentiment.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    EVAL_DIR, FIGURES_DIR, ENCODER_DIR, PROCESSED_DIR,
    SENTIMENT_LABELS, INTENT_LABELS, MAX_SEQ_LENGTH,
)
from src.training.model import MultiTaskFinBERT, get_tokenizer  # noqa: E402


# ============================================================
# Tier breakdown
# ============================================================
def tier_breakdown() -> None:
    pred_path = EVAL_DIR / "test_predictions.parquet"
    df = pd.read_parquet(pred_path)

    rows = []
    for tier in df["label_tier"].dropna().unique():
        sub = df[df["label_tier"] == tier]

        for head, true_col, pred_col, names in [
            ("sentiment", "sentiment_id", "pred_sentiment_id", SENTIMENT_LABELS),
            ("intent", "intent_id", "pred_intent_id", INTENT_LABELS),
        ]:
            mask = sub[true_col].notna()
            if not mask.any():
                continue
            y_true = sub.loc[mask, true_col].astype(int).to_numpy()
            y_pred = sub.loc[mask, pred_col].astype(int).to_numpy()

            rows.append({
                "tier": tier,
                "head": head,
                "n": int(mask.sum()),
                "macro_f1": float(f1_score(y_true, y_pred,
                                           average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred,
                                              average="weighted", zero_division=0)),
            })

    out_df = pd.DataFrame(rows).sort_values(["head", "tier"])
    out_csv = EVAL_DIR / "tier_breakdown.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    logger.success(f"Tier breakdown saved -> {out_csv}")
    print("\n" + out_df.to_string(index=False))

    logger.info(
        "\nReading the breakdown: if tier2 scores much lower than tier1, "
        "your model is hitting the ceiling imposed by Twitter label noise — "
        "a finding to put in your case study."
    )


# ============================================================
# Calibration (reliability diagram for sentiment)
# ============================================================
def calibration_curve(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Bin predictions by confidence; compute mean accuracy per bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs, bin_edges[1:-1])

    accs, confs, sizes = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            accs.append(np.nan)
            confs.append((bin_edges[b] + bin_edges[b + 1]) / 2)
            sizes.append(0)
        else:
            accs.append(correct[mask].mean())
            confs.append(probs[mask].mean())
            sizes.append(int(mask.sum()))
    return np.array(confs), np.array(accs), np.array(sizes)


def calibration_plot() -> None:
    """
    Re-run FinBERT to get softmax probabilities (the saved
    predictions are argmax-only). We only need ~1500 rows for a
    stable reliability diagram, so we sample to keep runtime low.
    """
    final_dir = ENCODER_DIR / "final"
    if not (final_dir / "pytorch_model.bin").exists():
        logger.warning("No trained model — skipping calibration plot.")
        return

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    test_df = test_df.dropna(subset=["sentiment_id"]).reset_index(drop=True)

    SAMPLE_N = min(1500, len(test_df))
    test_df = test_df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)
    logger.info(f"Calibration: scoring {len(test_df)} test rows with softmax probs")

    tokenizer = get_tokenizer()
    model = MultiTaskFinBERT()
    state = torch.load(
        final_dir / "pytorch_model.bin",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state, strict=False)
    model.eval()

    BATCH = 32
    all_probs, all_true = [], []
    texts = test_df["text"].astype(str).tolist()
    truths = test_df["sentiment_id"].astype(int).to_numpy()

    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            enc = tokenizer(
                texts[start:start + BATCH],
                padding=True, truncation=True,
                max_length=MAX_SEQ_LENGTH, return_tensors="pt",
            )
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"])
            probs = torch.softmax(out["sentiment_logits"], dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_true.append(truths[start:start + BATCH])

    probs = np.concatenate(all_probs, axis=0)
    truths = np.concatenate(all_true, axis=0)

    # Confidence = probability assigned to the predicted class.
    pred = probs.argmax(axis=1)
    confidence = probs[np.arange(len(pred)), pred]
    correct = (pred == truths).astype(int)

    confs, accs, sizes = calibration_curve(confidence, correct, n_bins=10)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # Plot points sized by bin size — empty bins not drawn.
    nonempty = sizes > 0
    ax.scatter(
        confs[nonempty], accs[nonempty],
        s=np.clip(sizes[nonempty] * 1.5, 30, 400),
        alpha=0.7, color="#2563eb", edgecolor="white",
        label="FinBERT sentiment",
    )
    ax.plot(confs[nonempty], accs[nonempty], color="#2563eb", alpha=0.5)

    # Expected Calibration Error — a single summary number for
    # the case study. ECE = sum_bin (bin_weight * |conf - acc|)
    total = sizes.sum()
    ece = float(np.nansum(sizes[nonempty] / total
                          * np.abs(confs[nonempty] - accs[nonempty])))
    ax.text(
        0.05, 0.95,
        f"ECE = {ece:.3f}\nPoint size = bin count",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=10, va="top",
    )

    ax.set_xlabel("Confidence (predicted class probability)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram — sentiment head", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = FIGURES_DIR / "calibration_sentiment.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Calibration plot saved -> {out_path} (ECE={ece:.3f})")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "calibration.json").write_text(
        json.dumps({"sentiment_ece": ece, "n_samples": int(total)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    logger.info("=== Tier breakdown ===")
    tier_breakdown()
    logger.info("\n=== Calibration ===")
    calibration_plot()


if __name__ == "__main__":
    main()
