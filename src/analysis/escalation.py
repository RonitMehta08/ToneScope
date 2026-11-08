"""
Escalation detection and risk assessment.

Computes a rolling escalation risk score across the conversation and
identifies danger zones where conflict is building or has peaked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import ESCALATION_CONFIG
from src.inference.pipeline import TurnPrediction


@dataclass
class EscalationZone:
    """A contiguous zone of elevated escalation risk."""

    start_turn: int
    end_turn: int
    peak_score: float
    peak_turn: int
    avg_score: float
    level: str  # "caution" or "alert"
    contributing_factors: list[str]


@dataclass
class EscalationReport:
    """Full escalation analysis for a conversation."""

    per_turn_scores: list[float]
    per_turn_levels: list[str]
    escalation_zones: list[EscalationZone]
    max_escalation: float
    avg_escalation: float
    escalation_trend: str  # "rising", "falling", "stable", "volatile"
    risk_summary: str


def compute_escalation_profile(
    predictions: list[TurnPrediction],
) -> EscalationReport:
    """Compute the full escalation profile for a conversation.

    Args:
        predictions: List of TurnPrediction from inference pipeline.

    Returns:
        EscalationReport with per-turn scores, zones, and summary.
    """
    if not predictions:
        return EscalationReport(
            per_turn_scores=[],
            per_turn_levels=[],
            escalation_zones=[],
            max_escalation=0.0,
            avg_escalation=0.0,
            escalation_trend="stable",
            risk_summary="No messages to analyze.",
        )

    scores = [p.escalation_score for p in predictions]
    levels = [p.escalation_level for p in predictions]

    # Detect escalation zones (contiguous runs above "safe")
    zones = _find_escalation_zones(predictions)

    # Determine trend
    trend = _compute_trend(scores)

    # Summary
    max_esc = max(scores)
    avg_esc = sum(scores) / len(scores)
    alert_count = levels.count("alert")
    caution_count = levels.count("caution")

    if alert_count == 0 and caution_count == 0:
        summary = "✅ Conversation is calm and healthy. No escalation detected."
    elif alert_count == 0:
        summary = (
            f"⚠️ Mild tension detected in {caution_count} turn(s). "
            f"Average risk: {avg_esc:.0%}. Overall tone is manageable."
        )
    else:
        summary = (
            f"🚨 Significant escalation detected! {alert_count} high-risk turn(s) "
            f"and {caution_count} caution turn(s). Peak risk: {max_esc:.0%}. "
            f"Trend: {trend}."
        )

    return EscalationReport(
        per_turn_scores=scores,
        per_turn_levels=levels,
        escalation_zones=zones,
        max_escalation=round(max_esc, 3),
        avg_escalation=round(avg_esc, 3),
        escalation_trend=trend,
        risk_summary=summary,
    )


def _find_escalation_zones(predictions: list[TurnPrediction]) -> list[EscalationZone]:
    """Identify contiguous zones of elevated escalation."""
    zones: list[EscalationZone] = []
    in_zone = False
    zone_start = 0
    zone_scores: list[float] = []
    zone_turns: list[int] = []

    threshold = ESCALATION_CONFIG.safe_threshold

    for i, pred in enumerate(predictions):
        if pred.escalation_score >= threshold:
            if not in_zone:
                zone_start = pred.turn_index
                zone_scores = []
                zone_turns = []
                in_zone = True
            zone_scores.append(pred.escalation_score)
            zone_turns.append(pred.turn_index)
        elif in_zone:
            # End of zone
            _append_zone(zones, zone_start, zone_turns, zone_scores, predictions)
            in_zone = False

    # Handle zone at end of conversation
    if in_zone:
        _append_zone(zones, zone_start, zone_turns, zone_scores, predictions)

    return zones


def _append_zone(
    zones: list[EscalationZone],
    start: int,
    turns: list[int],
    scores: list[float],
    predictions: list[TurnPrediction],
) -> None:
    """Create and append an EscalationZone."""
    peak_idx = scores.index(max(scores))
    peak_score = max(scores)
    avg_score = sum(scores) / len(scores)

    # Determine contributing factors
    factors: list[str] = []
    zone_preds = [p for p in predictions if p.turn_index in turns]

    negative_count = sum(1 for p in zone_preds if p.corrected_emotion in {"anger", "disgust", "sadness"})
    sarcasm_count = sum(1 for p in zone_preds if p.is_sarcastic)
    high_intensity = sum(1 for p in zone_preds if p.lexical_features.get("caps_ratio", 0) > 0.3)

    if negative_count > 0:
        factors.append(f"negative emotions ({negative_count} turns)")
    if sarcasm_count > 0:
        factors.append(f"sarcasm detected ({sarcasm_count} turns)")
    if high_intensity > 0:
        factors.append(f"high-intensity language ({high_intensity} turns)")

    level = "alert" if peak_score >= ESCALATION_CONFIG.caution_threshold else "caution"

    zones.append(
        EscalationZone(
            start_turn=start,
            end_turn=turns[-1],
            peak_score=round(peak_score, 3),
            peak_turn=turns[peak_idx],
            avg_score=round(avg_score, 3),
            level=level,
            contributing_factors=factors or ["general tension"],
        )
    )


def _compute_trend(scores: list[float]) -> str:
    """Determine the overall escalation trend."""
    if len(scores) < 3:
        return "stable"

    # Simple linear regression slope
    n = len(scores)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(scores) / n

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, scores))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if denominator == 0:
        return "stable"

    slope = numerator / denominator

    # Volatility check
    diffs = [abs(scores[i] - scores[i - 1]) for i in range(1, n)]
    avg_diff = sum(diffs) / len(diffs) if diffs else 0

    if avg_diff > 0.2:
        return "volatile"
    elif slope > 0.02:
        return "rising"
    elif slope < -0.02:
        return "falling"
    return "stable"
