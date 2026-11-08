"""
Training script for the multi-task emotion classifier.

Fine-tunes RoBERTa-base on combined emotion datasets (GoEmotions + MELD + DailyDialog)
with the multi-task architecture (emotion + sarcasm + escalation heads).

Optimized for RTX 4050 (6GB VRAM) using:
    - FP16 mixed precision
    - Gradient accumulation (effective batch = 32)
    - Gradient checkpointing
    - Short sequence lengths (128 tokens)

Usage:
    python scripts/train_emotion.py
    python scripts/train_emotion.py --epochs 6 --lr 3e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    MODELS_DIR,
    EKMAN_EMOTIONS,
)
from src.data.dataset_loader import load_combined_emotion_dataset
from src.data.preprocessing import clean_chat_text
from src.models.emotion_classifier import create_emotion_classifier


class EmotionDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for emotion classification."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "emotion_labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute evaluation metrics for the Trainer."""
    logits_tuple, labels = eval_pred

    # Handle multi-task output (emotion logits are the first element)
    if isinstance(logits_tuple, tuple):
        emotion_logits = logits_tuple[0]
    else:
        emotion_logits = logits_tuple

    predictions = np.argmax(emotion_logits, axis=-1)

    weighted_f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    accuracy = accuracy_score(labels, predictions)

    return {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ToneScope Emotion Classifier")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG.num_train_epochs)
    parser.add_argument("--lr", type=float, default=TRAINING_CONFIG.learning_rate)
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONFIG.per_device_train_batch_size)
    parser.add_argument("--accumulation", type=int, default=TRAINING_CONFIG.gradient_accumulation_steps)
    parser.add_argument("--max_length", type=int, default=MODEL_CONFIG.max_seq_length)
    parser.add_argument("--output_dir", type=str, default=str(MODELS_DIR / "emotion_model"))
    parser.add_argument("--no_goemotions", action="store_true", help="Exclude GoEmotions dataset")
    parser.add_argument("--no_meld", action="store_true", help="Exclude MELD dataset")
    parser.add_argument("--no_dailydialog", action="store_true", help="Exclude DailyDialog dataset")
    parser.add_argument("--no_dair", action="store_true", help="Exclude dair-ai/emotion dataset")
    args = parser.parse_args()

    print("=" * 60)
    print("ToneScope — Emotion Classifier Training")
    print("=" * 60)
    print(f"  Model: {MODEL_CONFIG.base_model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size} × {args.accumulation} = {args.batch_size * args.accumulation}")
    print(f"  Max Seq Length: {args.max_length}")
    print(f"  FP16: {TRAINING_CONFIG.fp16}")
    print(f"  Gradient Checkpointing: {TRAINING_CONFIG.gradient_checkpointing}")
    print(f"  Output: {args.output_dir}")

    # Check GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu} ({vram:.1f} GB)")
    else:
        print("  ⚠ No GPU detected — training will be slow!")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG.base_model_name)

    # Load datasets
    print("\nLoading training data...")
    train_df = load_combined_emotion_dataset(
        include_goemotions=not args.no_goemotions,
        include_meld=not args.no_meld,
        include_dailydialog=not args.no_dailydialog,
        include_dair_emotion=not args.no_dair,
        split="train",
    )
    print(f"  Train samples: {len(train_df)}")
    print(f"  Label distribution:\n{train_df['emotion'].value_counts().to_string()}")

    print("\nLoading validation data...")
    val_df = load_combined_emotion_dataset(
        include_goemotions=not args.no_goemotions,
        include_meld=not args.no_meld,
        include_dailydialog=not args.no_dailydialog,
        include_dair_emotion=not args.no_dair,
        split="validation",
    )
    print(f"  Validation samples: {len(val_df)}")

    # Clean texts
    print("\nCleaning texts...")
    train_texts = [clean_chat_text(t) for t in train_df["text"].tolist()]
    val_texts = [clean_chat_text(t) for t in val_df["text"].tolist()]
    train_labels = train_df["emotion_id"].tolist()
    val_labels = val_df["emotion_id"].tolist()

    # Create datasets
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, args.max_length)

    # Create model
    print("\nInitializing model...")
    model = create_emotion_classifier(MODEL_CONFIG, pretrained=True)

    # Enable gradient checkpointing for memory savings
    if TRAINING_CONFIG.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Compute warmup steps from ratio
    total_steps = (len(train_dataset) // (args.batch_size * args.accumulation)) * args.epochs
    warmup_steps = int(total_steps * TRAINING_CONFIG.warmup_ratio)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=TRAINING_CONFIG.per_device_eval_batch_size,
        gradient_accumulation_steps=args.accumulation,
        fp16=TRAINING_CONFIG.fp16 and torch.cuda.is_available(),
        learning_rate=args.lr,
        weight_decay=TRAINING_CONFIG.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=TRAINING_CONFIG.lr_scheduler_type,
        logging_steps=TRAINING_CONFIG.logging_steps,
        eval_strategy=TRAINING_CONFIG.eval_strategy,
        save_strategy=TRAINING_CONFIG.save_strategy,
        save_total_limit=TRAINING_CONFIG.save_total_limit,
        load_best_model_at_end=TRAINING_CONFIG.load_best_model_at_end,
        metric_for_best_model=TRAINING_CONFIG.metric_for_best_model,
        greater_is_better=TRAINING_CONFIG.greater_is_better,
        seed=TRAINING_CONFIG.seed,
        report_to="none",
        remove_unused_columns=False,
        label_names=["emotion_labels"],
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    train_result = trainer.train()

    # Save
    print("\nSaving best model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # Detailed classification report
    predictions = trainer.predict(val_dataset)
    if isinstance(predictions.predictions, tuple):
        pred_logits = predictions.predictions[0]
    else:
        pred_logits = predictions.predictions
    pred_labels = np.argmax(pred_logits, axis=-1)

    print("\nDetailed Classification Report:")
    print(classification_report(val_labels, pred_labels, target_names=EKMAN_EMOTIONS, zero_division=0))

    print("\n✓ Training complete!")
    print(f"  Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
