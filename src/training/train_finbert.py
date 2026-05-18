"""
src/training/train_finbert.py
============================================================
Fine-tune FinBERT with two heads on CPU
============================================================

WHAT THIS SCRIPT DOES:
    1. Loads the splits + class weights from Milestone 3 step 1.
    2. Tokenizes every row to a fixed length.
    3. Wires up the MultiTaskFinBERT model with class weights.
    4. Trains with HuggingFace Trainer for NUM_EPOCHS.
    5. Saves the best-on-val checkpoint to models/encoder/.
    6. Logs everything to Weights & Biases.

CPU NOTES (read this BEFORE running):
    With 36k rows, batch size 16, 3 epochs, this takes roughly
    2-5 HOURS on a typical laptop CPU. That's the deal — fine-
    tuning a 110M-param transformer is genuinely heavy. To run
    a smoke test first (5 min), use --fast which subsamples to
    2000 train rows and 1 epoch.

WEIGHTS & BIASES:
    First run: you'll be prompted for an API key. Get one free
    at wandb.ai. After that it logs to your account. To skip
    W&B entirely, set the env var:  set WANDB_MODE=disabled
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from transformers import (
    Trainer, TrainingArguments,
    DataCollatorWithPadding,
)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    PROCESSED_DIR, ENCODER_DIR, EVAL_DIR,
    SENTIMENT_LABELS, INTENT_LABELS,
    LABEL_NA_ID, MAX_SEQ_LENGTH,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, WARMUP_RATIO,
    RANDOM_SEED, WANDB_PROJECT,
)
from src.training.model import (  # noqa: E402
    MultiTaskFinBERT, load_class_weights, get_tokenizer,
)


# ============================================================
# Dataset wrapper
# ============================================================
class FinanceDataset(torch.utils.data.Dataset):
    """
    Wraps a pandas DataFrame as a PyTorch dataset.

    Tokenization happens here, ONCE per row, so the Trainer's
    data loader can iterate cheaply. Missing labels become
    LABEL_NA_ID so the loss ignores them.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["text"].astype(str).tolist()
        # NaN -> LABEL_NA_ID (-100). CrossEntropyLoss ignores this.
        self.sentiment_ids = (
            df["sentiment_id"].fillna(LABEL_NA_ID).astype(int).tolist()
        )
        self.intent_ids = (
            df["intent_id"].fillna(LABEL_NA_ID).astype(int).tolist()
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,  # padding happens at batch time
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "sentiment_labels": self.sentiment_ids[idx],
            "intent_labels": self.intent_ids[idx],
        }


# ============================================================
# Custom collator
# ============================================================
class MultiTaskCollator:
    """
    Pads input_ids/attention_mask via HF's DataCollatorWithPadding,
    then stacks our two label tensors. The HF default collator
    doesn't know about our two label columns, so we wrap it.
    """

    def __init__(self, tokenizer):
        self.pad_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        # Split off our labels — pad_collator chokes on extra keys.
        sentiment = [f.pop("sentiment_labels") for f in features]
        intent = [f.pop("intent_labels") for f in features]

        batch = self.pad_collator(features)
        batch["sentiment_labels"] = torch.tensor(sentiment, dtype=torch.long)
        batch["intent_labels"] = torch.tensor(intent, dtype=torch.long)
        return batch


# ============================================================
# Metric function — called by Trainer every eval
# ============================================================
def compute_metrics(eval_pred):
    """
    Trainer hands us model predictions and true labels at each eval.

    We compute macro F1 per head, IGNORING the -100 rows (rows
    where that head's label was missing). 'Macro' weights every
    class equally — the right metric under imbalance.
    """
    predictions, labels = eval_pred
    # Our model returns a dict; Trainer flattens it. The order is
    # the dict's insertion order, so: sentiment_logits, intent_logits
    # (plus losses if present — but Trainer strips those).
    sent_logits, intent_logits = predictions[:2]
    sent_labels, intent_labels = labels

    sent_preds = np.argmax(sent_logits, axis=-1)
    intent_preds = np.argmax(intent_logits, axis=-1)

    # Mask out -100 rows per head.
    s_mask = sent_labels != LABEL_NA_ID
    i_mask = intent_labels != LABEL_NA_ID

    metrics = {}
    if s_mask.any():
        metrics["sentiment_macro_f1"] = float(
            f1_score(sent_labels[s_mask], sent_preds[s_mask],
                     average="macro", zero_division=0)
        )
    if i_mask.any():
        metrics["intent_macro_f1"] = float(
            f1_score(intent_labels[i_mask], intent_preds[i_mask],
                     average="macro", zero_division=0)
        )
    # The combined number used for best-model selection.
    metrics["avg_macro_f1"] = float(
        np.mean([
            metrics.get("sentiment_macro_f1", 0.0),
            metrics.get("intent_macro_f1", 0.0),
        ])
    )
    return metrics


# ============================================================
# Trainer override — our model returns a dict; HF Trainer expects
# a specific shape. The default works, but we override
# `prediction_step` to ensure labels come out as a tuple.
# ============================================================
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # HF's default works but uses outputs["logits"]; we have two.
        # Override to return both heads' logits as a tuple.
        labels_tuple = (
            inputs.pop("sentiment_labels"),
            inputs.pop("intent_labels"),
        )
        with torch.no_grad():
            outputs = model(
                **inputs,
                sentiment_labels=labels_tuple[0],
                intent_labels=labels_tuple[1],
            )
            loss = outputs.get("loss")
        if prediction_loss_only:
            return (loss, None, None)
        logits_tuple = (
            outputs["sentiment_logits"].cpu(),
            outputs["intent_logits"].cpu(),
        )
        return (loss, logits_tuple, labels_tuple)


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast", action="store_true",
        help="Smoke test: 2000 train rows, 1 epoch (~5 min on CPU)",
    )
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ---- W&B setup ----
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    run_name = "finbert-fast" if args.fast else "finbert-full"

    # ---- Load data ----
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")

    if args.fast:
        train_df = train_df.sample(
            min(2000, len(train_df)), random_state=RANDOM_SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            min(500, len(val_df)), random_state=RANDOM_SEED
        ).reset_index(drop=True)
        logger.warning(f"FAST MODE: train={len(train_df)}, val={len(val_df)}")

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ---- Tokenizer ----
    tokenizer = get_tokenizer()

    train_ds = FinanceDataset(train_df, tokenizer, MAX_SEQ_LENGTH)
    val_ds = FinanceDataset(val_df, tokenizer, MAX_SEQ_LENGTH)

    # ---- Model ----
    sent_w, intent_w = load_class_weights()
    logger.info(f"Sentiment weights: {[f'{w:.2f}' for w in sent_w]}")
    logger.info(f"Intent weights:    {[f'{w:.2f}' for w in intent_w]}")

    model = MultiTaskFinBERT(
        sentiment_weights=sent_w,
        intent_weights=intent_w,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {n_params:,} parameters")

    # ---- Training config ----
    epochs = 1 if args.fast else NUM_EPOCHS
    out_dir = ENCODER_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,           # keep last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="avg_macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=["wandb"],
        run_name=run_name,
        seed=RANDOM_SEED,
        # CPU-specific: pin to fp32 (no mixed precision on CPU)
        fp16=False,
        bf16=False,
        # Pre-tokenizing in __getitem__ keeps memory low; workers
        # would just add overhead on Windows.
        dataloader_num_workers=0,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=MultiTaskCollator(tokenizer),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # ---- Train ----
    logger.info("Starting training. This will take a while on CPU.")
    trainer.train()
    logger.success("Training complete.")

    # ---- Final eval on val ----
    val_metrics = trainer.evaluate()
    logger.info(f"Final val metrics: {val_metrics}")

    # ---- Save the model + tokenizer in a clean location ----
    final_dir = ENCODER_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    # We save the encoder weights + head weights together.
    torch.save(model.state_dict(), final_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(final_dir)
    # Save the label maps so inference doesn't depend on config edits.
    (final_dir / "labels.json").write_text(json.dumps({
        "sentiment": SENTIMENT_LABELS,
        "intent": INTENT_LABELS,
    }, indent=2), encoding="utf-8")
    logger.success(f"Final model saved -> {final_dir}")

    # ---- Save val metrics ----
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "finbert_val_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in val_metrics.items()
                    if isinstance(v, (int, float))}, indent=2),
        encoding="utf-8",
    )
    logger.success("Done. Run evaluate_finbert.py next for test-set scores.")


if __name__ == "__main__":
    main()
