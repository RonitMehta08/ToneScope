"""
Sarcasm-aware emotion correction layer.

When sarcasm or irony is detected, the literal emotion prediction is often
inverted (e.g., "joy" when the speaker actually means "frustration").
This module applies intelligent corrections to emotion predictions based
on sarcasm detection output.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import EMOTION_TO_ID, ID_TO_EMOTION


# Emotion inversion map: what the speaker *likely* means when being sarcastic
# about a given literal emotion
SARCASM_EMOTION_FLIP: dict[str, str] = {
    "joy": "anger",          # "Oh great, just wonderful" → actually angry
    "surprise": "anger",     # "Oh wow, what a shock" → actually irritated
    "neutral": "anger",      # Deadpan sarcasm is often masking frustration
    "sadness": "joy",        # "I'm so heartbroken..." (sarcastically) → actually fine
    "anger": "neutral",      # Exaggerated anger in sarcasm → actually dismissive
    "disgust": "neutral",    # Over-the-top disgust → often theatrical
    "fear": "neutral",       # "Oh I'm terrified" → dismissive sarcasm
}


@dataclass
class CorrectedPrediction:
    """An emotion prediction that may have been adjusted for sarcasm."""

    original_emotion: str
    corrected_emotion: str
    is_sarcastic: bool
    sarcasm_confidence: float
    emotion_confidence: float
    was_corrected: bool
    explanation: str


def correct_emotion_for_sarcasm(
    emotion_label: str,
    emotion_confidence: float,
    is_sarcastic: bool,
    sarcasm_confidence: float,
    correction_threshold: float = 0.6,
) -> CorrectedPrediction:
    """Apply sarcasm-aware correction to an emotion prediction.

    The correction only fires when:
        1. Sarcasm is detected with sufficient confidence
        2. The detected emotion is one that commonly gets inverted by sarcasm

    Args:
        emotion_label: Predicted emotion (e.g., "joy").
        emotion_confidence: Confidence of the emotion prediction.
        is_sarcastic: Whether sarcasm was detected.
        sarcasm_confidence: Confidence of sarcasm detection.
        correction_threshold: Minimum sarcasm confidence to trigger correction.

    Returns:
        CorrectedPrediction with original and potentially corrected emotion.
    """
    if not is_sarcastic or sarcasm_confidence < correction_threshold:
        return CorrectedPrediction(
            original_emotion=emotion_label,
            corrected_emotion=emotion_label,
            is_sarcastic=is_sarcastic,
            sarcasm_confidence=sarcasm_confidence,
            emotion_confidence=emotion_confidence,
            was_corrected=False,
            explanation=f"No sarcasm correction needed (sarcasm={'detected' if is_sarcastic else 'not detected'}, conf={sarcasm_confidence:.2f})",
        )

    # Look up the flip
    flipped = SARCASM_EMOTION_FLIP.get(emotion_label, emotion_label)
    was_corrected = flipped != emotion_label

    if was_corrected:
        explanation = (
            f"Sarcasm detected (conf={sarcasm_confidence:.2f}). "
            f"Literal emotion '{emotion_label}' flipped to '{flipped}' — "
            f"the speaker likely means the opposite of what they expressed."
        )
    else:
        explanation = (
            f"Sarcasm detected (conf={sarcasm_confidence:.2f}) but "
            f"emotion '{emotion_label}' does not have a clear inversion. "
            f"Keeping original prediction."
        )

    return CorrectedPrediction(
        original_emotion=emotion_label,
        corrected_emotion=flipped,
        is_sarcastic=True,
        sarcasm_confidence=sarcasm_confidence,
        emotion_confidence=emotion_confidence,
        was_corrected=was_corrected,
        explanation=explanation,
    )


def batch_correct_emotions(
    emotion_labels: list[str],
    emotion_confidences: list[float],
    sarcasm_flags: list[bool],
    sarcasm_confidences: list[float],
    correction_threshold: float = 0.6,
) -> list[CorrectedPrediction]:
    """Apply sarcasm correction to a batch of predictions.

    Args:
        emotion_labels: List of predicted emotions.
        emotion_confidences: Corresponding confidence scores.
        sarcasm_flags: Whether each message is sarcastic.
        sarcasm_confidences: Confidence of sarcasm detection.
        correction_threshold: Minimum confidence to trigger correction.

    Returns:
        List of CorrectedPrediction objects.
    """
    results: list[CorrectedPrediction] = []

    for emo, emo_conf, sarc, sarc_conf in zip(
        emotion_labels, emotion_confidences, sarcasm_flags, sarcasm_confidences
    ):
        results.append(
            correct_emotion_for_sarcasm(
                emotion_label=emo,
                emotion_confidence=emo_conf,
                is_sarcastic=sarc,
                sarcasm_confidence=sarc_conf,
                correction_threshold=correction_threshold,
            )
        )

    return results
