"""
src/evaluation/per_class_delta.py
============================================================
Per-class F1 deltas: where exactly did FinBERT beat baseline?
============================================================

WHY THIS FIGURE:
    Your headline number says "FinBERT +0.110 on sentiment, +0.088
    on intent." The bar chart that breaks that down by class is
    THE figure that goes in your case study. It answers the
    interview question "what did fine-tuning actually learn?"

    Expected pattern from your numbers:
      sentiment[negative]    : +0.14  (largest gain — rare class)
      sentiment[positive]    : +0.15  (rare class)
      sentiment[neutral]     : +0.05  (majority class, baseline already strong)
      intent[analyst_question]: +0.17 (rare class)
      intent[risk_warning]   : +0.07
      intent[market/perf/pol]: +0.07-0.09

    The narrative writes itself: "Fine-tuning's biggest gains
    came on rare classes where bag-of-words has no semantic
    knowledge to fall back on."

OUTPUTS:
    reports/figures/per_class_delta.png
    reports/eval/per_class_delta.csv
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    EVAL_DIR, FIGURES_DIR, SENTIMENT_LABELS, INTENT_LABELS,
)


def main() -> None:
    comp_path = EVAL_DIR / "comparison.json"
    if not comp_path.exists():
        raise FileNotFoundError(
            f"{comp_path} not found. Run evaluate_finbert.py first."
        )
    comparison = json.loads(comp_path.read_text(encoding="utf-8"))

    rows = []

    # For each head, walk the per-class report and pull baseline +
    # FinBERT F1 for every class.
    for head, class_names in [
        ("sentiment", SENTIMENT_LABELS),
        ("intent", INTENT_LABELS),
    ]:
        if head not in comparison:
            continue
        base = comparison[head]["baseline"]["per_class"]
        finbert = comparison[head]["finbert"]["per_class"]

        for cls in class_names:
            if cls in base and cls in finbert:
                base_f1 = base[cls]["f1-score"]
                fb_f1 = finbert[cls]["f1-score"]
                support = base[cls].get("support", 0)
                rows.append({
                    "head": head,
                    "class": cls,
                    "baseline_f1": base_f1,
                    "finbert_f1": fb_f1,
                    "delta": fb_f1 - base_f1,
                    "support": int(support),
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["head", "delta"], ascending=[True, False])

    # ---- Save the numeric table ----
    out_csv = EVAL_DIR / "per_class_delta.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.success(f"Per-class delta table saved -> {out_csv}")
    print("\n" + df.to_string(index=False))

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, head in zip(axes, ["sentiment", "intent"]):
        sub = df[df["head"] == head].copy()
        # Sort by FinBERT F1 descending so the strongest class is on top.
        sub = sub.sort_values("finbert_f1", ascending=True)

        y_pos = np.arange(len(sub))
        ax.barh(y_pos - 0.2, sub["baseline_f1"], height=0.4,
                label="Baseline", color="#9aa5b1")
        ax.barh(y_pos + 0.2, sub["finbert_f1"], height=0.4,
                label="FinBERT", color="#2563eb")

        # Annotate the delta next to each FinBERT bar.
        for i, (b, f) in enumerate(zip(sub["baseline_f1"], sub["finbert_f1"])):
            delta = f - b
            ax.text(
                max(f, b) + 0.01, i + 0.2,
                f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}",
                va="center", fontsize=9,
                color="#16a34a" if delta >= 0 else "#dc2626",
                fontweight="bold",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["class"])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("F1-score")
        ax.set_title(f"{head.capitalize()} — per-class F1", fontsize=11)
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Per-class F1: Baseline vs FinBERT (test set)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out_fig = FIGURES_DIR / "per_class_delta.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Per-class delta figure saved -> {out_fig}")


if __name__ == "__main__":
    main()
