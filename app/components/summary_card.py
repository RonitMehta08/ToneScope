"""
Conversation health summary card component.

Displays the overall health score, key metrics, and per-speaker
radar charts in a visually striking summary card.
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from src.config import EMOTION_COLORS, EKMAN_EMOTIONS
from src.analysis.conversation_summary import ConversationSummary, SpeakerProfile


def render_summary_card(summary: ConversationSummary) -> None:
    """Render the main conversation health summary card."""
    # Health score color
    if summary.health_score >= 70:
        score_color = "#22C55E"
        score_emoji = "💚"
    elif summary.health_score >= 40:
        score_color = "#F59E0B"
        score_emoji = "⚠️"
    else:
        score_color = "#EF4444"
        score_emoji = "🚨"

    # Main health card
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        ">
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
                <div style="
                    width: 80px; height: 80px;
                    border-radius: 50%;
                    border: 4px solid {score_color};
                    display: flex; align-items: center; justify-content: center;
                    font-size: 1.8em; font-weight: 700; color: {score_color};
                ">{summary.health_score:.0f}</div>
                <div>
                    <h2 style="margin: 0; color: #E2E8F0;">
                        {score_emoji} Conversation Health: {summary.health_label}
                    </h2>
                    <p style="margin: 4px 0 0; color: #94A3B8;">
                        {summary.total_turns} messages · {summary.num_speakers} speakers
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dominant Emotion",
            summary.dominant_emotion.capitalize(),
            delta=None,
        )

    with col2:
        st.metric(
            "Positivity",
            f"{summary.positivity_ratio:.0%}",
            delta=f"{'↑' if summary.positivity_ratio > 0.5 else '↓'}",
        )

    with col3:
        st.metric(
            "Sarcasm Rate",
            f"{summary.sarcasm_density:.0%}",
            delta=f"{summary.sarcasm_density:.0%}" if summary.sarcasm_density > 0 else None,
            delta_color="inverse",
        )

    with col4:
        st.metric(
            "Tone Shifts",
            str(summary.num_tone_shifts),
            delta=f"{summary.num_escalation_zones} escalation zone(s)",
            delta_color="inverse" if summary.num_escalation_zones > 0 else "normal",
        )


def create_speaker_radar(profiles: list[SpeakerProfile]) -> go.Figure:
    """Create a radar chart comparing speaker emotion profiles."""
    if not profiles:
        fig = go.Figure()
        fig.add_annotation(text="No speaker data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    emotions = EKMAN_EMOTIONS
    colors = ["#8B5CF6", "#22C55E", "#3B82F6", "#F59E0B", "#EF4444", "#EC4899"]

    fig = go.Figure()

    for i, profile in enumerate(profiles):
        values = [profile.emotion_distribution.get(e, 0.0) for e in emotions]
        values.append(values[0])  # Close the polygon

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=[e.capitalize() for e in emotions] + [emotions[0].capitalize()],
                fill="toself",
                name=profile.name,
                line=dict(color=colors[i % len(colors)], width=2),
                fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)}, "
                          f"{int(colors[i % len(colors)][3:5], 16)}, "
                          f"{int(colors[i % len(colors)][5:7], 16)}, 0.15)",
            )
        )

    fig.update_layout(
        title=dict(
            text="Speaker Emotion Profiles",
            font=dict(size=18, color="#E2E8F0"),
        ),
        polar=dict(
            bgcolor="rgba(15, 23, 42, 0.5)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
            angularaxis=dict(color="#94A3B8"),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        height=400,
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig
