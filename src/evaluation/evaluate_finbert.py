"""
src/evaluation/evaluate_finbert.py
============================================================
Final FinBERT evaluation on the held-out test set
============================================================
PATCHED v2: load state dict with strict=False so the trainer's
            saved loss-function class-weight buffers don't break
            inference-time loading (when we don't pass weights).

WHY THIS PATCH:
    During training we instantiated MultiTaskFinBERT WITH class
    weights, so CrossEntropyLoss registered tensor buffers like
    `sentiment_loss_fn.weight`. Those got serialized into the
    state dict. At inference we don't need loss functions at all
    — we just take argmax. So we build the model WITHOUT weights,
    which means those keys are "unexpected" at load time.

    Fix: pass strict=False to ignore the unused keys. We also log
    which keys were skipped so any *real* missing weights would
    still surface as a warning.

WHY A SEPARATE EVAL FILE:
    During training we only watched val. The test set was
    untouched. This script scores the trained model on test
    exactly once, alongside the baseline. Touching test
    repeatedly until you "like" the numbers is data leakage.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report, f1_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    PROCESSED_DIR, ENCODER_DIR, BASELINE_DIR, EVAL_DIR,
    SENTIMENT_LABELS, INTENT_LABELS, LABEL_NA_ID, MAX_SEQ_LENGTH,
)
from src.training.model import MultiTaskFinBERT, get_tokenizer  # noqa: E402


# ============================================================
# FinBERT inference on the test set
# ============================================================
def predict_finbert(test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run the trained model over the test set and return preds."""
    final_dir = ENCODER_DIR / "final"
    if not (final_dir / "pytorch_model.bin").exists():
        raise FileNotFoundError(
            f"Trained model not found at {final_dir}. "
            "Run train_finbert.py first."
        )

    tokenizer = get_tokenizer()

    # Build model WITHOUT class weights — we don't need loss at
    # inference, just argmax of the logits.
    model = MultiTaskFinBERT()

    # weights_only=False because PyTorch 2.6+ flips the default;
    # our state dict is one we wrote ourselves so this is safe.
    state = torch.load(
        final_dir / "pytorch_model.bin",
        map_location="cpu",
        weights_only=False,
    )

    # strict=False: the saved state has loss-fn buffers
    # (sentiment_loss_fn.weight, intent_loss_fn.weight) which the
    # weight-less model has no slot for. load_state_dict reports
    # them as "unexpected" — fine to ignore at inference.
    result = model.load_state_dict(state, strict=False)

    # Sanity check: log what was skipped. If MISSING keys show up
    # here, something is actually wrong (real weights aren't being
    # loaded). UNEXPECTED keys = only the loss-fn buffers = fine.
    if result.missing_keys:
        logger.warning(f"Missing keys when loading: {result.missing_keys}")
    if result.unexpected_keys:
        expected_unexpected = {"sentiment_loss_fn.weight", "intent_loss_fn.weight"}
        unrecognized = set(result.unexpected_keys) - expected_unexpected
        if unrecognized:
            logger.warning(f"Unrecognized extra keys in state dict: {unrecognized}")
        else:
            logger.info(
                f"Skipped {len(result.unexpected_keys)} loss-fn buffers "
                "(expected — they're only used during training)."
            )

    model.eval()

    texts = test_df["text"].astype(str).tolist()
    sent_preds_all, intent_preds_all = [], []

    BATCH = 32
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            batch_texts = texts[start:start + BATCH]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            sent_preds_all.append(
                out["sentiment_logits"].argmax(dim=-1).cpu().numpy()
            )
            intent_preds_all.append(
                out["intent_logits"].argmax(dim=-1).cpu().numpy()
            )
            if (start // BATCH) % 20 == 0:
                logger.info(f"  predicted {start + len(batch_texts)}/{len(texts)}")

    return (
        np.concatenate(sent_preds_all),
        np.concatenate(intent_preds_all),
    )


# ============================================================
# Baseline inference on the test set
# ============================================================
def predict_baseline(test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run the trained baseline models over the test set."""
    tfidf = joblib.load(BASELINE_DIR / "tfidf.joblib")
    sent_clf = joblib.load(BASELINE_DIR / "sentiment_clf.joblib")
    intent_clf = joblib.load(BASELINE_DIR / "intent_clf.joblib")

    X = tfidf.transform(test_df["text"].astype(str))
    return sent_clf.predict(X), intent_clf.predict(X)


# ============================================================
# Scoring helpers
# ============================================================
def score_head(name: str, y_true: np.ndarray, y_pred: np.ndarray,
               class_names: list[str]) -> dict:
    """Compute and pretty-print metrics for one head."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    logger.info(f"{name}: macro F1={macro_f1:.3f} | weighted F1={weighted_f1:.3f}")
    print(classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0,
    ))
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": report,
    }


def main() -> None:
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    logger.info(f"Test set size: {len(test_df)}")

    logger.info("Running baseline predictions...")
    base_sent, base_int = predict_baseline(test_df)

    logger.info("Running FinBERT predictions...")
    fb_sent, fb_int = predict_finbert(test_df)

    results = {}
    for head, y_pred_base, y_pred_fb, label_col, class_names in [
        ("sentiment", base_sent, fb_sent, "sentiment_id", SENTIMENT_LABELS),
        ("intent", base_int, fb_int, "intent_id", INTENT_LABELS),
    ]:
        mask = test_df[label_col].notna().to_numpy()
        if not mask.any():
            logger.warning(f"{head}: no labeled rows in test - skipping")
            continue
        y_true = test_df.loc[mask, label_col].astype(int).to_numpy()

        logger.info(f"\n=== {head.upper()} - Baseline (TF-IDF + LR) ===")
        base_score = score_head("Baseline", y_true, y_pred_base[mask], class_names)

        logger.info(f"\n=== {head.upper()} - FinBERT ===")
        fb_score = score_head("FinBERT", y_true, y_pred_fb[mask], class_names)

        results[head] = {
            "baseline": base_score,
            "finbert": fb_score,
            "delta_macro_f1": fb_score["macro_f1"] - base_score["macro_f1"],
        }

    # Save predictions for Milestone 4 error analysis.
    test_df = test_df.copy()
    test_df["pred_sentiment_id"] = fb_sent
    test_df["pred_intent_id"] = fb_int
    test_df["pred_sentiment"] = pd.Series(fb_sent).map(
        {i: l for i, l in enumerate(SENTIMENT_LABELS)}
    )
    test_df["pred_intent"] = pd.Series(fb_int).map(
        {i: l for i, l in enumerate(INTENT_LABELS)}
    )
    pred_path = EVAL_DIR / "test_predictions.parquet"
    test_df.to_parquet(pred_path, index=False)
    logger.success(f"Test predictions saved -> {pred_path}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    logger.success(f"Comparison saved -> {EVAL_DIR / 'comparison.json'}")

    print("\n" + "=" * 60)
    print("CASE-STUDY HEADLINE NUMBERS (test set, macro F1)")
    print("=" * 60)
    for head, r in results.items():
        print(
            f"  {head.upper():10s}  "
            f"Baseline: {r['baseline']['macro_f1']:.3f}  ->  "
            f"FinBERT: {r['finbert']['macro_f1']:.3f}  "
            f"({r['delta_macro_f1']:+.3f})"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
