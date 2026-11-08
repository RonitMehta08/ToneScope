"""
Training script for the sarcasm detection model.

Fine-tunes RoBERTa-base on iSarcasmEval (SemEval-2022) for binary sarcasm
classification. This model acts as the sarcasm detection head in the
ToneScope pipeline.

Usage:
    python scripts/train_sarcasm.py
    python scripts/train_sarcasm.py --epochs 5 --lr 2e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_CONFIG, TRAINING_CONFIG, MODELS_DIR
from src.data.dataset_loader import load_isarcasmeval
from src.data.preprocessing import clean_chat_text


class SarcasmDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for sarcasm detection."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute sarcasm detection metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary", pos_label=1, zero_division=0),
        "precision": precision_score(labels, predictions, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(labels, predictions, average="binary", pos_label=1, zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ToneScope Sarcasm Detector")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default=str(MODELS_DIR / "sarcasm_model"))
    args = parser.parse_args()

    print("=" * 60)
    print("ToneScope — Sarcasm Detector Training")
    print("=" * 60)
    print(f"  Model: {MODEL_CONFIG.base_model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Effective Batch: {args.batch_size} × {args.accumulation} = {args.batch_size * args.accumulation}")

    # GPU check
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG.base_model_name)

    # Load data
    print("\nLoading iSarcasmEval data...")
    train_df = load_isarcasmeval("train")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Sarcastic: {train_df['is_sarcastic'].sum()} | Not: {(~train_df['is_sarcastic'].astype(bool)).sum()}")

    # Split train into train/val (80/20) since iSarcasmEval doesn't have a validation split
    from sklearn.model_selection import train_test_split
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["is_sarcastic"]
    )

    print(f"  Train split: {len(train_split)} | Val split: {len(val_split)}")

    # Clean texts
    train_texts = [clean_chat_text(t) for t in train_split["text"].tolist()]
    val_texts = [clean_chat_text(t) for t in val_split["text"].tolist()]
    train_labels = train_split["is_sarcastic"].astype(int).tolist()
    val_labels = val_split["is_sarcastic"].astype(int).tolist()

    # Datasets
    train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = SarcasmDataset(val_texts, val_labels, tokenizer, args.max_length)

    # Model — standard sequence classification for sarcasm (simpler, lighter)
    print("\nInitializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CONFIG.base_model_name,
        num_labels=2,
        problem_type="single_label_classification",
    )

    if TRAINING_CONFIG.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    # Compute warmup steps
    total_steps = (len(train_dataset) // (args.batch_size * args.accumulation)) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.accumulation,
        fp16=TRAINING_CONFIG.fp16 and torch.cuda.is_available(),
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting sarcasm training...")
    print("=" * 60)
    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")

    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    print("\nClassification Report:")
    print(classification_report(val_labels, pred_labels, target_names=["Not Sarcastic", "Sarcastic"], zero_division=0))

    print(f"\n✓ Sarcasm model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
