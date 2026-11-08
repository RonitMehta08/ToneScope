"""
Evaluation script — runs full evaluation on standard benchmarks.

Usage:
    python scripts/evaluate.py --dataset meld --split test
    python scripts/evaluate.py --dataset dailydialog --split test
    python scripts/evaluate.py --dataset all --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EKMAN_EMOTIONS, MODELS_DIR
from src.data.dataset_loader import load_meld, load_dailydialog, load_isarcasmeval
from src.data.preprocessing import clean_chat_text


def evaluate_emotion_model(dataset_name: str, split: str, model_path: str | None = None) -> dict:
    """Evaluate the emotion model on a benchmark dataset."""
    from transformers import pipeline as hf_pipeline

    # Load model
    custom_path = Path(model_path) if model_path else MODELS_DIR / "emotion_model"

    if custom_path.exists() and (custom_path / "config.json").exists():
        print(f"Using custom model: {custom_path}")
        pipe = hf_pipeline(
            "text-classification",
            model=str(custom_path),
            tokenizer=str(custom_path),
            device=0,
            top_k=1,
            truncation=True,
            max_length=128,
        )
        mode = "custom"
    else:
        print("Using pre-trained model: SamLowe/roberta-base-go_emotions")
        pipe = hf_pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            device=0,
            top_k=None,
            truncation=True,
            max_length=128,
        )
        mode = "pretrained"

    # Load dataset
    print(f"\nLoading {dataset_name} ({split})...")
    if dataset_name == "meld":
        df = load_meld(split)
    elif dataset_name == "dailydialog":
        df = load_dailydialog(split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    texts = [clean_chat_text(t) for t in df["text"].tolist()]
    true_labels = df["emotion"].tolist()
    true_ids = df["emotion_id"].tolist()

    print(f"  Samples: {len(texts)}")

    # Predict
    print("Running predictions...")
    from src.config import GOEMOTIONS_TO_EKMAN, EMOTION_TO_ID

    pred_labels = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [t if t else "empty" for t in batch]

        results = pipe(batch)

        for result in results:
            if mode == "pretrained":
                # Map GoEmotions → Ekman
                ekman_scores = {}
                items = result if isinstance(result, list) else [result]
                for item in items:
                    label = item["label"].lower()
                    score = item["score"]
                    ekman = GOEMOTIONS_TO_EKMAN.get(label, "neutral")
                    ekman_scores[ekman] = max(ekman_scores.get(ekman, 0), score)
                pred_labels.append(max(ekman_scores, key=ekman_scores.get))
            else:
                if isinstance(result, list):
                    result = result[0]
                pred_labels.append(result["label"])

        if (i + batch_size) % 500 == 0:
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    # Metrics
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results — {dataset_name} ({split})")
    print(f"{'=' * 60}")

    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)

    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Accuracy:    {accuracy:.4f}")

    print(f"\nDetailed Classification Report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))

    return {
        "dataset": dataset_name,
        "split": split,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ToneScope models")
    parser.add_argument("--dataset", type=str, default="all", choices=["meld", "dailydialog", "all"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    results = []

    if args.dataset in ("meld", "all"):
        r = evaluate_emotion_model("meld", args.split, args.model_path)
        results.append(r)

    if args.dataset in ("dailydialog", "all"):
        r = evaluate_emotion_model("dailydialog", args.split, args.model_path)
        results.append(r)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['dataset']:15s} WF1={r['weighted_f1']:.4f}  MF1={r['macro_f1']:.4f}  Acc={r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
