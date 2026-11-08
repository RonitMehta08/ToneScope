"""
Conversation summary generator.

Produces a comprehensive, human-readable analysis report including:
    - Overall conversation health score
    - Per-speaker emotional profiles
    - Key tone shifts and their causes
    - Escalation risk assessment
    - Sarcasm/irony density metrics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

from src.inference.pipeline import TurnPrediction, ConversationAnalysis
from src.analysis.emotion_arc import EmotionArcData, ToneShift
from src.analysis.escalation import EscalationReport
from src.analysis.cause_attribution import CauseAttribution


@dataclass
class SpeakerProfile:
    """Emotional profile of a single speaker."""

    name: str
    num_messages: int
    dominant_emotion: str
    emotion_distribution: dict[str, float]  # emotion → percentage
    sarcasm_rate: float  # percentage of sarcastic messages
    avg_escalation: float
    positivity_score: float  # 0-1


@dataclass
class ConversationSummary:
    """Complete conversation analysis summary."""

    # Overall metrics
    health_score: float  # 0-100 composite score
    health_label: str    # "Healthy", "Tense", "Hostile"
    total_turns: int
    num_speakers: int

    # Dominant characteristics
    dominant_emotion: str
    positivity_ratio: float
    sarcasm_density: float  # percentage of turns with sarcasm
    escalation_risk: str    # from EscalationReport

    # Per-speaker profiles
    speaker_profiles: list[SpeakerProfile]

    # Key events
    num_tone_shifts: int
    most_dramatic_shift: ToneShift | None
    num_escalation_zones: int

    # Narrative summary
    narrative: str


def generate_summary(
    analysis: ConversationAnalysis,
    arc_data: EmotionArcData,
    escalation_report: EscalationReport,
    attributions: list[CauseAttribution],
) -> ConversationSummary:
    """Generate a comprehensive conversation summary.

    Args:
        analysis: ConversationAnalysis from the inference pipeline.
        arc_data: EmotionArcData from emotion_arc module.
        escalation_report: EscalationReport from escalation module.
        attributions: CauseAttribution list from cause_attribution module.

    Returns:
        ConversationSummary with all metrics and narrative.
    """
    predictions = analysis.predictions

    if not predictions:
        return ConversationSummary(
            health_score=100.0,
            health_label="N/A",
            total_turns=0,
            num_speakers=0,
            dominant_emotion="neutral",
            positivity_ratio=0.0,
            sarcasm_density=0.0,
            escalation_risk="none",
            speaker_profiles=[],
            num_tone_shifts=0,
            most_dramatic_shift=None,
            num_escalation_zones=0,
            narrative="No messages to analyze.",
        )

    # Compute per-speaker profiles
    speaker_profiles = _compute_speaker_profiles(predictions, analysis.speakers)

    # Overall metrics
    emotion_counts = Counter(p.corrected_emotion for p in predictions)
    dominant_emotion = emotion_counts.most_common(1)[0][0]

    positive_emotions = {"joy", "surprise"}
    positivity_ratio = sum(
        1 for p in predictions if p.corrected_emotion in positive_emotions
    ) / len(predictions)

    sarcasm_density = sum(1 for p in predictions if p.is_sarcastic) / len(predictions)

    # Health score (0-100)
    health_score = _compute_health_score(
        positivity_ratio=positivity_ratio,
        sarcasm_density=sarcasm_density,
        avg_escalation=escalation_report.avg_escalation,
        num_tone_shifts=len(arc_data.tone_shifts),
        total_turns=len(predictions),
    )

    if health_score >= 70:
        health_label = "Healthy"
    elif health_score >= 40:
        health_label = "Tense"
    else:
        health_label = "Hostile"

    # Most dramatic tone shift
    most_dramatic = None
    if arc_data.tone_shifts:
        most_dramatic = max(arc_data.tone_shifts, key=lambda s: s.shift_magnitude)

    # Generate narrative
    narrative = _generate_narrative(
        predictions=predictions,
        speaker_profiles=speaker_profiles,
        tone_shifts=arc_data.tone_shifts,
        escalation_report=escalation_report,
        attributions=attributions,
        health_label=health_label,
    )

    return ConversationSummary(
        health_score=round(health_score, 1),
        health_label=health_label,
        total_turns=len(predictions),
        num_speakers=len(analysis.speakers),
        dominant_emotion=dominant_emotion,
        positivity_ratio=round(positivity_ratio, 3),
        sarcasm_density=round(sarcasm_density, 3),
        escalation_risk=escalation_report.escalation_trend,
        speaker_profiles=speaker_profiles,
        num_tone_shifts=len(arc_data.tone_shifts),
        most_dramatic_shift=most_dramatic,
        num_escalation_zones=len(escalation_report.escalation_zones),
        narrative=narrative,
    )


def _compute_speaker_profiles(
    predictions: list[TurnPrediction],
    speakers: list[str],
) -> list[SpeakerProfile]:
    """Build emotional profile for each speaker."""
    profiles: list[SpeakerProfile] = []

    for speaker in speakers:
        spk_preds = [p for p in predictions if p.speaker == speaker]
        if not spk_preds:
            continue

        emotion_counts = Counter(p.corrected_emotion for p in spk_preds)
        total = len(spk_preds)

        distribution = {e: count / total for e, count in emotion_counts.items()}
        dominant = emotion_counts.most_common(1)[0][0]

        sarcasm_rate = sum(1 for p in spk_preds if p.is_sarcastic) / total

        positive = {"joy", "surprise"}
        positivity = sum(1 for p in spk_preds if p.corrected_emotion in positive) / total

        avg_esc = sum(p.escalation_score for p in spk_preds) / total

        profiles.append(
            SpeakerProfile(
                name=speaker,
                num_messages=total,
                dominant_emotion=dominant,
                emotion_distribution=distribution,
                sarcasm_rate=round(sarcasm_rate, 3),
                avg_escalation=round(avg_esc, 3),
                positivity_score=round(positivity, 3),
            )
        )

    return profiles


def _compute_health_score(
    positivity_ratio: float,
    sarcasm_density: float,
    avg_escalation: float,
    num_tone_shifts: int,
    total_turns: int,
) -> float:
    """Compute a 0-100 health score for the conversation."""
    # Start at 100 and deduct
    score = 100.0

    # Positivity bonus/penalty
    score += (positivity_ratio - 0.5) * 30  # +15 if mostly positive, -15 if mostly negative

    # Sarcasm penalty
    score -= sarcasm_density * 20

    # Escalation penalty
    score -= avg_escalation * 40

    # Tone shift density penalty (more shifts per turn = more chaotic)
    if total_turns > 0:
        shift_density = num_tone_shifts / total_turns
        score -= shift_density * 15

    return max(0.0, min(100.0, score))


def _generate_narrative(
    predictions: list[TurnPrediction],
    speaker_profiles: list[SpeakerProfile],
    tone_shifts: list[ToneShift],
    escalation_report: EscalationReport,
    attributions: list[CauseAttribution],
    health_label: str,
) -> str:
    """Generate a human-readable narrative summary."""
    parts: list[str] = []

    total = len(predictions)
    num_speakers = len(speaker_profiles)

    parts.append(
        f"This {total}-message conversation between {num_speakers} participant(s) "
        f"is classified as **{health_label}**."
    )

    # Speaker highlights
    for profile in speaker_profiles:
        parts.append(
            f"**{profile.name}** primarily expressed **{profile.dominant_emotion}** "
            f"({profile.positivity_score:.0%} positivity, "
            f"{profile.sarcasm_rate:.0%} sarcasm rate)."
        )

    # Tone shifts
    if tone_shifts:
        parts.append(
            f"\n{len(tone_shifts)} significant tone shift(s) were detected."
        )
        most_dramatic = max(tone_shifts, key=lambda s: s.shift_magnitude)
        parts.append(
            f"The most dramatic shift: {most_dramatic.speaker} went from "
            f"'{most_dramatic.from_emotion}' → '{most_dramatic.to_emotion}' "
            f"at turn {most_dramatic.turn_index} (magnitude: {most_dramatic.shift_magnitude:.2f})."
        )
    else:
        parts.append("\nNo significant tone shifts were detected — the emotional tone was stable.")

    # Escalation
    parts.append(f"\n{escalation_report.risk_summary}")

    # Attributions
    if attributions:
        parts.append(f"\n{len(attributions)} emotion cause(s) were attributed:")
        for attr in attributions[:3]:  # Show top 3
            parts.append(f"  • {attr.explanation}")

    return "\n".join(parts)
