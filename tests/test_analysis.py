"""
Tests for analysis modules — emotion arc, escalation, cause attribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import TurnPrediction
from src.analysis.emotion_arc import build_emotion_arc, detect_tone_shifts
from src.analysis.escalation import compute_escalation_profile
from src.analysis.cause_attribution import attribute_emotion_causes


def _make_prediction(
    turn: int, speaker: str, text: str, emotion: str,
    confidence: float = 0.8, sarcastic: bool = False,
    escalation: float = 0.1,
) -> TurnPrediction:
    """Helper to create a TurnPrediction for testing."""
    from src.config import EKMAN_EMOTIONS
    probs = {e: 0.05 for e in EKMAN_EMOTIONS}
    probs[emotion] = confidence

    return TurnPrediction(
        turn_index=turn,
        speaker=speaker,
        text=text,
        cleaned_text=text.lower(),
        emotion=emotion,
        emotion_confidence=confidence,
        emotion_probabilities=probs,
        is_sarcastic=sarcastic,
        sarcasm_confidence=0.8 if sarcastic else 0.1,
        corrected_emotion=emotion,
        was_corrected=False,
        correction_explanation="",
        escalation_score=escalation,
        escalation_level="safe" if escalation < 0.3 else "caution" if escalation < 0.6 else "alert",
        lexical_features={"caps_ratio": 0.0, "exclamation_count": 0, "repeated_punct": 0},
    )


# ─── Emotion Arc Tests ──────────────────────────────────────────────

class TestEmotionArc:
    def test_builds_arc_from_predictions(self):
        predictions = [
            _make_prediction(0, "Alice", "Hello!", "joy"),
            _make_prediction(1, "Bob", "Hi there", "neutral"),
            _make_prediction(2, "Alice", "This is terrible", "anger"),
        ]

        arc = build_emotion_arc(predictions)

        assert len(arc.turns) == 3
        assert len(arc.emotions) == 3
        assert "joy" in arc.emotion_timeseries

    def test_detects_tone_shifts(self):
        predictions = [
            _make_prediction(0, "Alice", "Happy day!", "joy", confidence=0.9),
            _make_prediction(1, "Bob", "That's awful", "anger", confidence=0.9),
        ]

        shifts = detect_tone_shifts(predictions, min_magnitude=0.3)
        assert len(shifts) >= 1
        assert shifts[0].from_emotion == "joy"
        assert shifts[0].to_emotion == "anger"

    def test_per_speaker_arcs(self):
        predictions = [
            _make_prediction(0, "Alice", "Hi!", "joy"),
            _make_prediction(1, "Bob", "Hey", "neutral"),
            _make_prediction(2, "Alice", "How are you?", "joy"),
        ]

        arc = build_emotion_arc(predictions)
        assert "Alice" in arc.speaker_arcs
        assert len(arc.speaker_arcs["Alice"]["turns"]) == 2

    def test_empty_predictions(self):
        arc = build_emotion_arc([])
        assert arc.turns == []
        assert arc.tone_shifts == []


# ─── Escalation Tests ────────────────────────────────────────────────

class TestEscalation:
    def test_safe_conversation(self):
        predictions = [
            _make_prediction(0, "A", "Hi!", "joy", escalation=0.1),
            _make_prediction(1, "B", "Hey!", "joy", escalation=0.1),
        ]

        report = compute_escalation_profile(predictions)
        assert report.max_escalation < 0.3
        assert "calm" in report.risk_summary.lower() or "no escalation" in report.risk_summary.lower()

    def test_escalation_detection(self):
        predictions = [
            _make_prediction(0, "A", "Hi", "neutral", escalation=0.1),
            _make_prediction(1, "B", "This is bad", "anger", escalation=0.7),
            _make_prediction(2, "A", "How dare you!", "anger", escalation=0.9),
        ]

        report = compute_escalation_profile(predictions)
        assert report.max_escalation >= 0.7
        assert len(report.escalation_zones) >= 1

    def test_empty_predictions(self):
        report = compute_escalation_profile([])
        assert report.per_turn_scores == []
        assert report.escalation_trend == "stable"


# ─── Cause Attribution Tests ────────────────────────────────────────

class TestCauseAttribution:
    def test_attributes_cross_speaker_cause(self):
        predictions = [
            _make_prediction(0, "Alice", "Everything is great!", "joy", confidence=0.9),
            _make_prediction(1, "Bob", "Actually, you messed up.", "anger", confidence=0.8),
            _make_prediction(2, "Alice", "What?! That's not true!", "anger", confidence=0.9),
        ]

        from src.analysis.emotion_arc import ToneShift
        shifts = [
            ToneShift(
                turn_index=2, speaker="Alice",
                from_emotion="joy", to_emotion="anger",
                shift_magnitude=0.8, text="What?! That's not true!",
            )
        ]

        attributions = attribute_emotion_causes(predictions, shifts)
        assert len(attributions) >= 1
        # Bob's message (turn 1) should be the attributed cause
        assert attributions[0].cause_speaker == "Bob"

    def test_no_shifts_no_attributions(self):
        predictions = [_make_prediction(0, "A", "Hi", "neutral")]
        attributions = attribute_emotion_causes(predictions, [])
        assert len(attributions) == 0
