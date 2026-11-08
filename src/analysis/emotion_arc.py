"""
Emotion arc construction and tone shift detection.

Builds the emotional trajectory of a conversation — per-speaker and aggregate —
and identifies statistically significant tone shift points using gradient-based
change point detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import EKMAN_EMOTIONS, EMOTION_TO_ID
from src.inference.pipeline import TurnPrediction


@dataclass
class ToneShift:
    """A detected tone shift in the conversation."""

    turn_index: int
    speaker: str
    from_emotion: str
    to_emotion: str
    shift_magnitude: float  # 0-1, how dramatic the shift is
    text: str


@dataclass
class EmotionArcData:
    """Data structure for emotion arc visualization."""

    turns: list[int]
    speakers: list[str]
    emotions: list[str]
    confidences: list[float]
    texts: list[str]

    # Per-emotion intensity timeseries (for multi-line charts)
    emotion_timeseries: dict[str, list[float]]  # emotion → [intensity per turn]

    # Aggregate sentiment (positive/negative polarity)
    sentiment_scores: list[float]  # -1 (very negative) to +1 (very positive)
    smoothed_sentiment: list[float]  # Smoothed version

    # Tone shifts
    tone_shifts: list[ToneShift]

    # Per-speaker arcs
    speaker_arcs: dict[str, dict[str, list]]  # speaker → {turns, emotions, ...}


# Sentiment mapping: each Ekman emotion gets a polarity score
EMOTION_POLARITY: dict[str, float] = {
    "joy": 0.9,
    "surprise": 0.3,
    "neutral": 0.0,
    "fear": -0.5,
    "sadness": -0.7,
    "disgust": -0.8,
    "anger": -0.9,
}


def build_emotion_arc(predictions: list[TurnPrediction]) -> EmotionArcData:
    """Construct the full emotion arc from turn-level predictions.

    Args:
        predictions: List of TurnPrediction from the inference pipeline.

    Returns:
        EmotionArcData ready for visualization.
    """
    if not predictions:
        return EmotionArcData(
            turns=[], speakers=[], emotions=[], confidences=[], texts=[],
            emotion_timeseries={e: [] for e in EKMAN_EMOTIONS},
            sentiment_scores=[], smoothed_sentiment=[],
            tone_shifts=[], speaker_arcs={},
        )

    turns = [p.turn_index for p in predictions]
    speakers = [p.speaker for p in predictions]
    emotions = [p.corrected_emotion for p in predictions]
    confidences = [p.emotion_confidence for p in predictions]
    texts = [p.text for p in predictions]

    # Build per-emotion intensity timeseries
    emotion_timeseries: dict[str, list[float]] = {e: [] for e in EKMAN_EMOTIONS}
    for pred in predictions:
        for emotion in EKMAN_EMOTIONS:
            prob = pred.emotion_probabilities.get(emotion, 0.0)
            emotion_timeseries[emotion].append(prob)

    # Compute sentiment polarity scores
    sentiment_scores = [
        EMOTION_POLARITY.get(e, 0.0) * c
        for e, c in zip(emotions, confidences)
    ]

    # Smoothed sentiment (exponential moving average)
    smoothed = _exponential_moving_average(sentiment_scores, alpha=0.3)

    # Detect tone shifts
    tone_shifts = detect_tone_shifts(predictions)

    # Build per-speaker arcs
    speaker_set = list(dict.fromkeys(speakers))  # Preserve order
    speaker_arcs: dict[str, dict[str, list]] = {}
    for spk in speaker_set:
        spk_preds = [p for p in predictions if p.speaker == spk]
        speaker_arcs[spk] = {
            "turns": [p.turn_index for p in spk_preds],
            "emotions": [p.corrected_emotion for p in spk_preds],
            "confidences": [p.emotion_confidence for p in spk_preds],
            "sentiment": [
                EMOTION_POLARITY.get(p.corrected_emotion, 0.0) * p.emotion_confidence
                for p in spk_preds
            ],
            "texts": [p.text for p in spk_preds],
        }

    return EmotionArcData(
        turns=turns,
        speakers=speakers,
        emotions=emotions,
        confidences=confidences,
        texts=texts,
        emotion_timeseries=emotion_timeseries,
        sentiment_scores=sentiment_scores,
        smoothed_sentiment=smoothed,
        tone_shifts=tone_shifts,
        speaker_arcs=speaker_arcs,
    )


def detect_tone_shifts(
    predictions: list[TurnPrediction],
    min_magnitude: float = 0.3,
) -> list[ToneShift]:
    """Detect significant tone shifts in the conversation.

    A tone shift is detected when:
        1. The dominant emotion changes between consecutive turns
        2. The polarity change exceeds min_magnitude

    Args:
        predictions: Ordered list of turn predictions.
        min_magnitude: Minimum polarity change to qualify as a tone shift.

    Returns:
        List of ToneShift objects.
    """
    shifts: list[ToneShift] = []

    for i in range(1, len(predictions)):
        prev = predictions[i - 1]
        curr = predictions[i]

        prev_polarity = EMOTION_POLARITY.get(prev.corrected_emotion, 0.0) * prev.emotion_confidence
        curr_polarity = EMOTION_POLARITY.get(curr.corrected_emotion, 0.0) * curr.emotion_confidence
        magnitude = abs(curr_polarity - prev_polarity)

        if (
            curr.corrected_emotion != prev.corrected_emotion
            and magnitude >= min_magnitude
        ):
            shifts.append(
                ToneShift(
                    turn_index=curr.turn_index,
                    speaker=curr.speaker,
                    from_emotion=prev.corrected_emotion,
                    to_emotion=curr.corrected_emotion,
                    shift_magnitude=round(magnitude, 3),
                    text=curr.text,
                )
            )

    return shifts


def _exponential_moving_average(values: list[float], alpha: float = 0.3) -> list[float]:
    """Compute exponential moving average for smoothing."""
    if not values:
        return []

    smoothed = [values[0]]
    for val in values[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return [round(v, 4) for v in smoothed]
