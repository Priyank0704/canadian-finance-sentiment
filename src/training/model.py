"""
src/training/model.py
============================================================
The multi-task FinBERT model — shared encoder + 2 heads
============================================================
PATCHED v2: class weights moved out of nn.CrossEntropyLoss and
            into a non-persistent buffer + functional cross_entropy
            call. This keeps the saved state dict CLEAN (no
            sentiment_loss_fn.weight / intent_loss_fn.weight keys),
            so inference-time loading works with strict=True.

WHY THE PATCH:
    The original used `nn.CrossEntropyLoss(weight=tensor)`. That
    stores the weight tensor as a buffer inside the loss module,
    which then ends up in `model.state_dict()`. At inference we
    don't pass weights (we don't compute loss at all), so loading
    the saved dict fails with "unexpected keys".

    Fix: register weights as non-persistent buffers and use
    `F.cross_entropy(...)` directly. Non-persistent buffers behave
    like regular tensors (move to GPU with .to(), no grad) but
    are NOT included in state_dict(). Clean save, clean load.

WHAT YOU'RE BUILDING:
    A single neural network with three parts:
      1. A FinBERT encoder (pre-trained on financial text).
      2. A sentiment classification head (3 classes).
      3. An intent classification head (5 classes).
    Both heads share the encoder. The encoder learns
    representations useful for BOTH tasks — that's multi-task
    learning.

CORE TECHNICAL POINTS (interview gold):
    1. PARTIAL SUPERVISION — missing labels are -100, ignored
       by cross_entropy via ignore_index=-100.
    2. CLASS-WEIGHTED LOSS — per-class inverse-frequency weights
       handle the imbalance (neutral is 64% of sentiment data).
    3. WEIGHTED TWO-HEAD LOSS — total = w_s * s_loss + w_i * i_loss.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.utils.config import (
    BASE_ENCODER_MODEL, SENTIMENT_LABELS, INTENT_LABELS,
    LABEL_NA_ID, PROCESSED_DIR,
    SENTIMENT_LOSS_WEIGHT, INTENT_LOSS_WEIGHT,
)


class MultiTaskFinBERT(nn.Module):
    """
    FinBERT encoder + two classification heads.

    Inputs (from the data collator):
      input_ids        (B, L) — token ids
      attention_mask   (B, L) — 1 for real tokens, 0 for padding
      sentiment_labels (B,)   — int label or -100 for "missing"
      intent_labels    (B,)   — int label or -100 for "missing"

    Outputs:
      dict with keys: loss, sentiment_logits, intent_logits
      (loss only present when labels are supplied)
    """

    def __init__(
        self,
        encoder_name: str = BASE_ENCODER_MODEL,
        n_sentiment: int = len(SENTIMENT_LABELS),
        n_intent: int = len(INTENT_LABELS),
        sentiment_weights: list[float] | None = None,
        intent_weights: list[float] | None = None,
        dropout: float = 0.1,
        sentiment_loss_w: float = SENTIMENT_LOSS_WEIGHT,
        intent_loss_w: float = INTENT_LOSS_WEIGHT,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for FinBERT

        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden_size, n_sentiment)
        self.intent_head = nn.Linear(hidden_size, n_intent)

        # Register class weights as NON-PERSISTENT buffers.
        # Non-persistent means: included in .to(device) and .cuda(),
        # but NOT saved into state_dict(). That's what keeps the
        # saved file clean for inference-time loading.
        if sentiment_weights is not None:
            self.register_buffer(
                "_sent_w",
                torch.tensor(sentiment_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self._sent_w = None

        if intent_weights is not None:
            self.register_buffer(
                "_intent_w",
                torch.tensor(intent_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self._intent_w = None

        self.sentiment_loss_w = sentiment_loss_w
        self.intent_loss_w = intent_loss_w

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentiment_labels: torch.Tensor | None = None,
        intent_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # [CLS] token = pooled summary. Using last_hidden_state[:,0]
        # explicitly so it works with or without a pooler layer.
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)

        sentiment_logits = self.sentiment_head(cls)
        intent_logits = self.intent_head(cls)

        result = {
            "sentiment_logits": sentiment_logits,
            "intent_logits": intent_logits,
        }

        if sentiment_labels is not None and intent_labels is not None:
            # Functional cross_entropy: no module to register
            # buffers under. ignore_index=-100 handles partial
            # supervision; weight handles class imbalance.
            s_loss = F.cross_entropy(
                sentiment_logits,
                sentiment_labels,
                weight=self._sent_w,
                ignore_index=LABEL_NA_ID,
            )
            i_loss = F.cross_entropy(
                intent_logits,
                intent_labels,
                weight=self._intent_w,
                ignore_index=LABEL_NA_ID,
            )

            # Edge case: a whole batch with NO valid labels for one
            # head -> cross_entropy returns NaN. Replace with 0.
            if torch.isnan(s_loss):
                s_loss = torch.tensor(0.0, device=cls.device)
            if torch.isnan(i_loss):
                i_loss = torch.tensor(0.0, device=cls.device)

            total = (
                self.sentiment_loss_w * s_loss
                + self.intent_loss_w * i_loss
            )
            result["loss"] = total
            result["sentiment_loss"] = s_loss.detach()
            result["intent_loss"] = i_loss.detach()

        return result


def load_class_weights() -> tuple[list[float], list[float]]:
    """Load class weights JSON written by build_splits.py."""
    path = PROCESSED_DIR / "class_weights.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_splits.py first."
        )
    w = json.loads(path.read_text(encoding="utf-8"))
    return w["sentiment"], w["intent"]


def get_tokenizer(encoder_name: str = BASE_ENCODER_MODEL):
    """Convenience accessor used by training and inference."""
    return AutoTokenizer.from_pretrained(encoder_name)
