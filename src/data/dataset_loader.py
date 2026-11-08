"""
Unified dataset loader for ToneScope.

Loads and normalizes GoEmotions, MELD, DailyDialog, iSarcasmEval, and RECCON
into a common schema for training and evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

from src.config import (
    GOEMOTIONS_TO_EKMAN,
    EKMAN_EMOTIONS,
    EMOTION_TO_ID,
    RAW_DATASETS_DIR,
)


def load_goemotions(split: str = "train") -> pd.DataFrame:
    """Load GoEmotions and map 27-label to Ekman 7-label scheme.

    Args:
        split: 'train', 'validation', or 'test'.

    Returns:
        DataFrame with columns: [text, emotion, emotion_id].
    """
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)

    records: list[dict[str, Any]] = []
    goemotions_labels = ds.features["labels"].feature.names

    for row in ds:
        text = row["text"]
        label_ids = row["labels"]

        # Map each GoEmotions label to Ekman
        for lid in label_ids:
            go_label = goemotions_labels[lid]
            ekman_label = GOEMOTIONS_TO_EKMAN.get(go_label, "neutral")
            records.append({
                "text": text,
                "emotion": ekman_label,
                "emotion_id": EMOTION_TO_ID[ekman_label],
                "source": "goemotions",
            })

    df = pd.DataFrame(records)
    # De-duplicate (same text can map to multiple Ekman labels; keep most common)
    df = df.drop_duplicates(subset=["text", "emotion"])
    return df


def load_meld(split: str = "train") -> pd.DataFrame:
    """Load MELD dataset for conversational emotion recognition.

    Args:
        split: 'train', 'validation', or 'test'.

    Returns:
        DataFrame with columns: [text, emotion, emotion_id, speaker,
        dialogue_id, utterance_id, sentiment].
    """
    split_map = {"train": "train", "validation": "validation", "test": "test"}
    ds = load_dataset("declare-lab/MELD", split=split_map[split])

    records: list[dict[str, Any]] = []
    label_map = {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "joy",
        "neutral": "neutral",
        "sadness": "sadness",
        "surprise": "surprise",
    }

    for row in ds:
        raw_emotion = row["Emotion"].lower().strip() if isinstance(row["Emotion"], str) else row["Emotion"]
        # Handle both string and integer label formats
        if isinstance(raw_emotion, int):
            emotion_names = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
            raw_emotion = emotion_names[raw_emotion] if raw_emotion < len(emotion_names) else "neutral"

        emotion = label_map.get(raw_emotion, "neutral")

        records.append({
            "text": row.get("Utterance", ""),
            "emotion": emotion,
            "emotion_id": EMOTION_TO_ID.get(emotion, EMOTION_TO_ID["neutral"]),
            "speaker": row.get("Speaker", "Unknown"),
            "dialogue_id": row.get("Dialogue_ID", 0),
            "utterance_id": row.get("Utterance_ID", 0),
            "sentiment": row.get("Sentiment", ""),
            "source": "meld",
        })

    return pd.DataFrame(records)


def load_dailydialog(split: str = "train") -> pd.DataFrame:
    """Load DailyDialog dataset.

    Args:
        split: 'train', 'validation', or 'test'.

    Returns:
        DataFrame with columns: [text, emotion, emotion_id, dialogue_id, utterance_id].
    """
    ds = load_dataset("daily_dialog", split=split)

    records: list[dict[str, Any]] = []
    dd_emotions = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]

    for dial_idx, row in enumerate(ds):
        utterances = row["dialog"]
        emotions = row["emotion"]

        for utt_idx, (text, emo_id) in enumerate(zip(utterances, emotions)):
            emotion = dd_emotions[emo_id] if emo_id < len(dd_emotions) else "neutral"
            # DailyDialog alternates speakers (A, B)
            speaker = "Speaker_A" if utt_idx % 2 == 0 else "Speaker_B"

            records.append({
                "text": text.strip(),
                "emotion": emotion,
                "emotion_id": EMOTION_TO_ID.get(emotion, EMOTION_TO_ID["neutral"]),
                "speaker": speaker,
                "dialogue_id": dial_idx,
                "utterance_id": utt_idx,
                "source": "dailydialog",
            })

    return pd.DataFrame(records)


def load_isarcasmeval(split: str = "train") -> pd.DataFrame:
    """Load iSarcasmEval dataset for sarcasm detection.

    Args:
        split: 'train' or 'test'.

    Returns:
        DataFrame with columns: [text, is_sarcastic, sarcasm_type].
    """
    # iSarcasmEval is available on HuggingFace
    try:
        ds = load_dataset("Ammar-alhaj-ali/iSarcasmEval", split=split)

        records: list[dict[str, Any]] = []
        for row in ds:
            records.append({
                "text": row.get("text", row.get("tweet", "")),
                "is_sarcastic": int(row.get("sarcastic", row.get("label", 0))),
                "sarcasm_type": row.get("sarcasm_type", row.get("irony_type", "none")),
                "source": "isarcasmeval",
            })
        return pd.DataFrame(records)
    except Exception:
        # Fallback: tweet_eval irony (binary irony detection, ~3K train)
        print("  iSarcasmEval unavailable, using tweet_eval irony as fallback...")
        tweet_split = split if split != "validation" else "validation"
        ds = load_dataset("cardiffnlp/tweet_eval", "irony", split=tweet_split)
        records = []
        for row in ds:
            records.append({
                "text": row.get("text", ""),
                "is_sarcastic": int(row.get("label", 0)),
                "sarcasm_type": "none",
                "source": "tweet_eval_irony",
            })
        return pd.DataFrame(records)


def load_dair_emotion(split: str = "train") -> pd.DataFrame:
    """Load dair-ai/emotion dataset.

    Clean, well-balanced dataset with ~20K samples across 6 emotions.
    Helps fill training gaps in anger, fear, sadness, and surprise.

    Args:
        split: 'train', 'validation', or 'test'.

    Returns:
        DataFrame with columns: [text, emotion, emotion_id].
    """
    ds = load_dataset("dair-ai/emotion", split=split)

    # Label mapping: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
    dair_to_ekman = {
        0: "sadness",
        1: "joy",
        2: "joy",       # love → joy
        3: "anger",
        4: "fear",
        5: "surprise",
    }

    records: list[dict[str, Any]] = []
    for row in ds:
        emotion = dair_to_ekman.get(row["label"], "neutral")
        records.append({
            "text": row["text"],
            "emotion": emotion,
            "emotion_id": EMOTION_TO_ID[emotion],
            "source": "dair_emotion",
        })

    return pd.DataFrame(records)


def load_combined_emotion_dataset(
    include_goemotions: bool = True,
    include_meld: bool = True,
    include_dailydialog: bool = True,
    include_dair_emotion: bool = True,
    split: str = "train",
) -> pd.DataFrame:
    """Load and combine multiple emotion datasets.

    Args:
        include_goemotions: Include GoEmotions data.
        include_meld: Include MELD data.
        include_dailydialog: Include DailyDialog data.
        include_dair_emotion: Include dair-ai/emotion data.
        split: Dataset split to load.

    Returns:
        Combined DataFrame with standardized schema.
    """
    frames: list[pd.DataFrame] = []

    if include_goemotions:
        try:
            frames.append(load_goemotions(split))
        except Exception as e:
            print(f"Warning: Could not load GoEmotions ({split}): {e}")

    if include_meld:
        try:
            frames.append(load_meld(split))
        except Exception as e:
            print(f"Warning: Could not load MELD ({split}): {e}")

    if include_dailydialog:
        try:
            frames.append(load_dailydialog(split))
        except Exception as e:
            print(f"Warning: Could not load DailyDialog ({split}): {e}")

    if include_dair_emotion:
        try:
            frames.append(load_dair_emotion(split))
        except Exception as e:
            print(f"Warning: Could not load dair-ai/emotion ({split}): {e}")
    if not frames:
        raise RuntimeError(f"No datasets could be loaded for split='{split}'")

    combined = pd.concat(frames, ignore_index=True)
    # Ensure required columns
    for col in ["text", "emotion", "emotion_id"]:
        if col not in combined.columns:
            raise ValueError(f"Missing column '{col}' after combining datasets")

    return combined
