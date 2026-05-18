"""
src/inference/predictor.py
============================================================
The inference engine - one model loaded once, many predictions
============================================================

WHY A SINGLETON:
    Loading FinBERT takes ~5-10 seconds. If we loaded it per-
    request, every prediction would feel like a cold start. A
    module-level singleton means the model loads once at process
    start and stays in memory.

    FastAPI calls get_predictor() inside its startup hook, so the
    load happens BEFORE the first request arrives - when the user
    types a headline they get a ~50ms response, not 5 seconds.

WHAT THIS RETURNS:
    For each input text, a dict like:
      {
        "text": "<original>",
        "cleaned_text": "<after preprocess>",
        "sentiment": {
            "label": "negative",
            "confidence": 0.87,
            "all_scores": {"negative": 0.87, "neutral": 0.10, "positive": 0.03}
        },
        "intent": { ... same shape ... }
      }
    Confidence + per-class scores are critical for the demo -
    they let users see when the model is unsure (which connects
    back to your calibration analysis).
"""

import json
import sys
from pathlib import Path
from threading import Lock
from typing import Iterable

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    ENCODER_DIR, MAX_SEQ_LENGTH, SENTIMENT_LABELS, INTENT_LABELS,
)
from src.training.model import MultiTaskFinBERT, get_tokenizer  # noqa: E402
from src.inference.preprocess import preprocess  # noqa: E402


class Predictor:
    """One model, many predictions. Thread-safe loading."""

    _instance: "Predictor | None" = None
    _lock = Lock()

    def __init__(self):
        final_dir = ENCODER_DIR / "final"
        if not (final_dir / "pytorch_model.bin").exists():
            raise FileNotFoundError(
                f"No trained model at {final_dir}. "
                "Run `python -m src.training.train_finbert` first."
            )

        self.tokenizer = get_tokenizer()
        self.model = MultiTaskFinBERT()

        state = torch.load(
            final_dir / "pytorch_model.bin",
            map_location="cpu",
            weights_only=False,
        )
        # strict=False because the saved state may still contain
        # the loss-fn class-weight buffers from training.
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # Load label mappings - prefer the JSON saved with the
        # model, fall back to config so old models still work.
        labels_path = final_dir / "labels.json"
        if labels_path.exists():
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            self.sentiment_labels = labels["sentiment"]
            self.intent_labels = labels["intent"]
        else:
            self.sentiment_labels = SENTIMENT_LABELS
            self.intent_labels = INTENT_LABELS

    @classmethod
    def get(cls) -> "Predictor":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def predict(self, texts) -> list[dict]:
        """
        Predict on one string or a list of strings. Always returns
        a list - even for a single input - so callers don't have to
        special-case batch size 1.
        """
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)

        cleaned = [preprocess(t) for t in batch]

        enc = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )

        # Softmax -> probabilities, then pull out per-row results.
        sent_probs = F.softmax(out["sentiment_logits"], dim=-1).cpu().numpy()
        intent_probs = F.softmax(out["intent_logits"], dim=-1).cpu().numpy()

        results = []
        for i, original in enumerate(batch):
            sent_idx = int(sent_probs[i].argmax())
            intent_idx = int(intent_probs[i].argmax())

            results.append({
                "text": original,
                "cleaned_text": cleaned[i],
                "sentiment": {
                    "label": self.sentiment_labels[sent_idx],
                    "confidence": float(sent_probs[i][sent_idx]),
                    "all_scores": {
                        self.sentiment_labels[j]: float(sent_probs[i][j])
                        for j in range(len(self.sentiment_labels))
                    },
                },
                "intent": {
                    "label": self.intent_labels[intent_idx],
                    "confidence": float(intent_probs[i][intent_idx]),
                    "all_scores": {
                        self.intent_labels[j]: float(intent_probs[i][j])
                        for j in range(len(self.intent_labels))
                    },
                },
            })

        return results


def get_predictor() -> Predictor:
    """Module-level convenience accessor."""
    return Predictor.get()


# Self-test:  python -m src.inference.predictor
if __name__ == "__main__":
    p = get_predictor()
    samples = [
        "Bank of Canada holds overnight rate at 5%, cites sticky inflation.",
        "Shopify Q3 revenue rose 25% to $1.7B, beating estimates.",
        "Manulife flags growing exposure to U.S. commercial real estate.",
        "TSX closed flat as energy gains offset bank weakness.",
        "RBC Capital upgrades Loblaw to Outperform, raises target to $185.",
    ]
    for r in p.predict(samples):
        print(f"\nText: {r['text']}")
        print(
            f"  Sentiment: {r['sentiment']['label']} "
            f"({r['sentiment']['confidence']:.2f})"
        )
        print(
            f"  Intent:    {r['intent']['label']} "
            f"({r['intent']['confidence']:.2f})"
        )
