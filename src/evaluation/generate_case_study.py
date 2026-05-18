"""
src/evaluation/generate_case_study.py
============================================================
Generate the case-study markdown from saved analysis artifacts
============================================================

WHY THIS FILE:
    Every previous step in Milestone 4 wrote a JSON, a CSV, or a
    PNG. This script pulls them together into a single markdown
    document you can paste onto your portfolio site verbatim.

    Auto-generated, but EDIT it before publishing. The script
    writes the structure; you write the voice.

OUTPUT:
    reports/CASE_STUDY.md
"""

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    EVAL_DIR, FIGURES_DIR, REPORTS_DIR,
    SENTIMENT_LABELS, INTENT_LABELS,
)


def safe_read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    comp = safe_read_json(EVAL_DIR / "comparison.json")
    if comp is None:
        raise FileNotFoundError(
            "comparison.json missing — run evaluate_finbert.py first."
        )

    delta = safe_read_csv(EVAL_DIR / "per_class_delta.csv")
    tiers = safe_read_csv(EVAL_DIR / "tier_breakdown.csv")
    spot = (EVAL_DIR / "spot_check_summary.txt")
    spot_text = spot.read_text(encoding="utf-8") if spot.exists() else None
    cal = safe_read_json(EVAL_DIR / "calibration.json")
    err_summary = (EVAL_DIR / "error_taxonomy_summary.txt")
    err_text = err_summary.read_text(encoding="utf-8") if err_summary.exists() else None

    sent_base = comp["sentiment"]["baseline"]["macro_f1"]
    sent_fb = comp["sentiment"]["finbert"]["macro_f1"]
    int_base = comp["intent"]["baseline"]["macro_f1"]
    int_fb = comp["intent"]["finbert"]["macro_f1"]

    lines = []
    lines.append("# Fine-tuned Sentiment & Intent Classifier for Canadian Finance")
    lines.append("")
    lines.append("**Problem.** Classify Canadian financial text along two axes "
                 "simultaneously: sentiment (bearish / neutral / bullish) and "
                 "intent (what the text is *doing* — reporting performance, "
                 "warning of risk, signaling policy, etc.). Most public finance-"
                 "sentiment models cover only sentiment, and almost none target "
                 "the Canadian market specifically.")
    lines.append("")
    lines.append("**Approach.** Fine-tuned a domain-specific encoder (FinBERT) "
                 "with two classification heads on a shared transformer body. "
                 "Trained on ~36,000 labeled examples assembled from three "
                 "tiers: expert-annotated Financial PhraseBank, crowd-curated "
                 "Twitter Financial News, and rule-based weak labels over Bank "
                 "of Canada releases.")
    lines.append("")

    # ----------------- Headline results -----------------
    lines.append("## Headline results (test set, macro F1)")
    lines.append("")
    lines.append("| Head | Baseline (TF-IDF + LR) | FinBERT (fine-tuned) | Δ |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Sentiment | {sent_base:.3f} | **{sent_fb:.3f}** | +{sent_fb - sent_base:.3f} |")
    lines.append(f"| Intent    | {int_base:.3f} | **{int_fb:.3f}** | +{int_fb - int_base:.3f} |")
    lines.append("")
    lines.append("Fine-tuning's largest per-class gains came on the rare classes "
                 "where a bag-of-words baseline has no semantic knowledge to "
                 "fall back on.")
    lines.append("")
    lines.append("![Per-class F1 comparison](figures/per_class_delta.png)")
    lines.append("")

    # ----------------- Per-class table -----------------
    if delta is not None:
        lines.append("### Per-class breakdown")
        lines.append("")
        for head in ["sentiment", "intent"]:
            sub = delta[delta["head"] == head]
            if sub.empty:
                continue
            lines.append(f"**{head.capitalize()}**")
            lines.append("")
            lines.append("| Class | Support | Baseline F1 | FinBERT F1 | Δ |")
            lines.append("|---|---:|---:|---:|---:|")
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r['class']} | {int(r['support'])} | "
                    f"{r['baseline_f1']:.3f} | {r['finbert_f1']:.3f} | "
                    f"+{r['delta']:.3f} |"
                )
            lines.append("")

    # ----------------- Dataset + labeling -----------------
    lines.append("## Dataset & labeling strategy")
    lines.append("")
    lines.append("Three label-quality tiers:")
    lines.append("")
    lines.append("- **Tier 1 — Expert.** Financial PhraseBank, ~3,400 sentences "
                 "labeled by humans with finance backgrounds (75% inter-annotator "
                 "agreement). Sentiment only.")
    lines.append("- **Tier 2 — Curated.** Twitter Financial News sentiment "
                 "(~12,000 rows) and topic (~21,000 rows) datasets. The 20-class "
                 "topic taxonomy was mapped onto a 5-class intent taxonomy I "
                 "designed for this project.")
    lines.append("- **Tier 3 — Weak.** Rule-based labeling of Bank of Canada "
                 "press releases. Audited on a stratified 40-row sample:")
    if spot_text:
        lines.append("")
        lines.append("```")
        lines.append(spot_text.strip())
        lines.append("```")
    lines.append("")
    lines.append("The 5-class intent taxonomy is itself a design choice. I "
                 "originally specified six classes; one (`forward_guidance`) "
                 "came out of labeling with only 20 examples, too few to "
                 "evaluate. It was merged into the semantically nearest class "
                 "(`market_commentary`) to preserve a defensible evaluation — "
                 "a normal industry-style mid-project taxonomy revision.")
    lines.append("")

    # ----------------- Architecture -----------------
    lines.append("## Model architecture")
    lines.append("")
    lines.append("```")
    lines.append("                       ┌─────────────┐")
    lines.append("                       │  Sentiment  │ (3 classes)")
    lines.append("                       │    head     │")
    lines.append("                       └─────────────┘")
    lines.append("                            ▲")
    lines.append("  Text ──► FinBERT encoder ─┤")
    lines.append("           (~110M params)   ▼")
    lines.append("                       ┌─────────────┐")
    lines.append("                       │   Intent    │ (5 classes)")
    lines.append("                       │    head     │")
    lines.append("                       └─────────────┘")
    lines.append("```")
    lines.append("")
    lines.append("Three technical choices worth calling out:")
    lines.append("")
    lines.append("1. **Partial supervision.** Many rows have only one of the "
                 "two labels (PhraseBank has no intent; Twitter intent has no "
                 "sentiment). Missing labels are encoded as `-100` and "
                 "`cross_entropy(..., ignore_index=-100)` skips them — so the "
                 "encoder still learns from every row, but each head only "
                 "trains on rows whose label exists.")
    lines.append("2. **Class-weighted loss.** Both heads use inverse-frequency "
                 "weights computed on the training set only, so rare classes "
                 "(`negative` sentiment, `risk_warning` intent) aren't drowned "
                 "out by majority classes.")
    lines.append("3. **Tier-aware splitting.** Train/val/test splits stratify "
                 "by label; test and val draw only from Tier-1 + Tier-2 rows, "
                 "so test scores aren't an evaluation of my weak-labeling rules.")
    lines.append("")

    # ----------------- Confusion -----------------
    lines.append("## Confusion matrices")
    lines.append("")
    lines.append("![Sentiment confusion](figures/confusion_sentiment.png)")
    lines.append("")
    lines.append("![Intent confusion](figures/confusion_intent.png)")
    lines.append("")

    # ----------------- Tier breakdown -----------------
    if tiers is not None:
        lines.append("## Performance by label tier")
        lines.append("")
        lines.append("Does the model score equally well on different label-quality "
                     "tiers? If not, the gap is a label-noise ceiling, not a "
                     "model failure.")
        lines.append("")
        lines.append("| Tier | Head | n | Macro F1 | Weighted F1 |")
        lines.append("|---|---|---:|---:|---:|")
        for _, r in tiers.iterrows():
            lines.append(
                f"| {r['tier']} | {r['head']} | {int(r['n'])} | "
                f"{r['macro_f1']:.3f} | {r['weighted_f1']:.3f} |"
            )
        lines.append("")

    # ----------------- Error taxonomy -----------------
    lines.append("## Error analysis")
    lines.append("")
    if err_text:
        lines.append("Manual categorization of a stratified random sample of "
                     "FinBERT's mistakes:")
        lines.append("")
        lines.append("```")
        lines.append(err_text.strip())
        lines.append("```")
    else:
        lines.append("_Run `python -m src.evaluation.error_taxonomy sample` "
                     "to generate the audit sheet, fill it in, then `score`._")
    lines.append("")

    # ----------------- Calibration -----------------
    lines.append("## Calibration")
    lines.append("")
    if cal:
        lines.append(
            f"Expected Calibration Error on sentiment head: "
            f"**{cal['sentiment_ece']:.3f}** "
            f"(over {cal['n_samples']} test predictions). "
            f"A perfectly calibrated model has ECE = 0; lower is better."
        )
        lines.append("")
        lines.append("![Calibration plot](figures/calibration_sentiment.png)")
    else:
        lines.append("_Run `python -m src.evaluation.slice_and_calibration` "
                     "to generate the reliability diagram._")
    lines.append("")

    # ----------------- Stack -----------------
    lines.append("## Tech stack")
    lines.append("")
    lines.append("HuggingFace Transformers + PEFT, PyTorch, scikit-learn, "
                 "BERTopic, Weights & Biases. CPU-only training — full "
                 "fine-tune of FinBERT in roughly 2–4 hours on a laptop.")
    lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "CASE_STUDY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"Case-study draft written -> {out_path}")
    logger.info(
        "Open it, edit the voice/phrasing where you want, then drop "
        "the figures/ folder beside it on your portfolio site."
    )


if __name__ == "__main__":
    main()
