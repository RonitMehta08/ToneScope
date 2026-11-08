"""
Emotion cause attribution module.

For each detected tone shift, identifies the most likely triggering utterance
from the conversation history. Uses cosine similarity between utterance
embeddings + temporal proximity weighting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import CAUSE_ATTRIBUTION_CONFIG as cfg
from src.inference.pipeline import TurnPrediction
from src.analysis.emotion_arc import ToneShift, EMOTION_POLARITY


@dataclass
class CauseAttribution:
    """Attribution linking an emotion change to its likely trigger."""

    effect_turn: int
    effect_speaker: str
    effect_emotion: str
    effect_text: str

    cause_turn: int
    cause_speaker: str
    cause_emotion: str
    cause_text: str

    attribution_score: float  # 0-1, confidence of the attribution
    explanation: str


def attribute_emotion_causes(
    predictions: list[TurnPrediction],
    tone_shifts: list[ToneShift],
) -> list[CauseAttribution]:
    """Attribute causes to detected tone shifts.

    For each tone shift, looks backward through the conversation to find
    the most likely triggering utterance based on:
        1. Temporal proximity (closer turns are more likely triggers)
        2. Speaker interaction (cross-speaker triggers weighted higher)
        3. Emotional contrast (opposite polarity = stronger trigger)
        4. Lexical intensity (high-intensity messages are stronger triggers)

    Args:
        predictions: Full list of turn predictions.
        tone_shifts: Detected tone shifts from emotion_arc module.

    Returns:
        List of CauseAttribution objects.
    """
    if not tone_shifts or not predictions:
        return []

    # Build lookup by turn index
    pred_by_turn: dict[int, TurnPrediction] = {p.turn_index: p for p in predictions}
    turn_indices = sorted(pred_by_turn.keys())

    attributions: list[CauseAttribution] = []

    for shift in tone_shifts:
        effect_pred = pred_by_turn.get(shift.turn_index)
        if not effect_pred:
            continue

        # Look back through conversation history
        candidates: list[tuple[TurnPrediction, float]] = []
        lookback_limit = min(cfg.max_lookback_turns, shift.turn_index)

        for prev_turn in turn_indices:
            if prev_turn >= shift.turn_index:
                break
            if shift.turn_index - prev_turn > lookback_limit:
                continue

            prev_pred = pred_by_turn[prev_turn]

            # Score this candidate as a potential trigger
            score = _score_cause_candidate(
                effect=effect_pred,
                candidate=prev_pred,
                distance=shift.turn_index - prev_turn,
            )

            candidates.append((prev_pred, score))

        if not candidates:
            continue

        # Select best cause
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_cause, best_score = candidates[0]

        if best_score < cfg.similarity_threshold:
            continue

        # Generate explanation
        explanation = _generate_explanation(
            effect=effect_pred,
            cause=best_cause,
            shift=shift,
            score=best_score,
        )

        attributions.append(
            CauseAttribution(
                effect_turn=effect_pred.turn_index,
                effect_speaker=effect_pred.speaker,
                effect_emotion=effect_pred.corrected_emotion,
                effect_text=effect_pred.text,
                cause_turn=best_cause.turn_index,
                cause_speaker=best_cause.speaker,
                cause_emotion=best_cause.corrected_emotion,
                cause_text=best_cause.text,
                attribution_score=round(best_score, 3),
                explanation=explanation,
            )
        )

    return attributions


def _score_cause_candidate(
    effect: TurnPrediction,
    candidate: TurnPrediction,
    distance: int,
) -> float:
    """Score a candidate utterance as a potential cause for an emotion change.

    Higher score = more likely to be the cause.
    """
    score = 0.0

    # 1. Temporal proximity (exponential decay) — closer = higher
    proximity_score = np.exp(-0.3 * (distance - 1))
    score += 0.3 * proximity_score

    # 2. Cross-speaker interaction (different speaker = more likely trigger)
    if candidate.speaker != effect.speaker:
        score += 0.25
    else:
        score += 0.05  # Self-triggered shifts are less common but possible

    # 3. Emotional contrast (opposite polarities = stronger trigger)
    effect_polarity = EMOTION_POLARITY.get(effect.corrected_emotion, 0.0)
    cause_polarity = EMOTION_POLARITY.get(candidate.corrected_emotion, 0.0)
    contrast = abs(effect_polarity - cause_polarity) / 2  # Normalize to 0-1
    score += 0.25 * contrast

    # 4. Candidate intensity (high-intensity messages are stronger triggers)
    intensity = 0.0
    if candidate.lexical_features:
        intensity += candidate.lexical_features.get("caps_ratio", 0.0) * 0.3
        intensity += min(candidate.lexical_features.get("exclamation_count", 0) / 5, 1.0) * 0.3
        intensity += (1.0 if candidate.is_sarcastic else 0.0) * 0.4
    score += 0.2 * intensity

    return min(score, 1.0)


def _generate_explanation(
    effect: TurnPrediction,
    cause: TurnPrediction,
    shift: ToneShift,
    score: float,
) -> str:
    """Generate a human-readable explanation for the attribution."""
    distance = effect.turn_index - cause.turn_index

    parts: list[str] = []

    if cause.speaker != effect.speaker:
        parts.append(
            f"{effect.speaker}'s tone shifted from '{shift.from_emotion}' to "
            f"'{shift.to_emotion}' likely triggered by {cause.speaker}'s message "
            f"{distance} turn(s) earlier"
        )
    else:
        parts.append(
            f"{effect.speaker}'s tone shifted from '{shift.from_emotion}' to "
            f"'{shift.to_emotion}', likely a continuation of their own earlier "
            f"message ({distance} turn(s) back)"
        )

    if cause.is_sarcastic:
        parts.append("(the triggering message was sarcastic)")

    cause_preview = cause.text[:80] + "..." if len(cause.text) > 80 else cause.text
    parts.append(f'Trigger: "{cause_preview}"')

    return ". ".join(parts) + f" [confidence: {score:.0%}]"
