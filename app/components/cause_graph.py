"""
Emotion cause attribution graph component.

Visualizes the cause → effect relationships as a directed graph,
linking triggering utterances to their emotional impact.
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from src.analysis.cause_attribution import CauseAttribution
from src.config import EMOTION_COLORS


def create_cause_graph(attributions: list[CauseAttribution]) -> go.Figure:
    """Create a visual graph connecting emotion causes to effects.

    Displays as a timeline with arrows from cause turns to effect turns.
    """
    if not attributions:
        fig = go.Figure()
        fig.add_annotation(text="No emotion causes attributed", xref="paper", yref="paper", x=0.5, y=0.5)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(15, 23, 42, 0.8)", height=250)
        return fig

    fig = go.Figure()

    for i, attr in enumerate(attributions):
        cause_color = EMOTION_COLORS.get(attr.cause_emotion, "#888888")
        effect_color = EMOTION_COLORS.get(attr.effect_emotion, "#888888")

        # Cause point
        fig.add_trace(
            go.Scatter(
                x=[attr.cause_turn],
                y=[i],
                mode="markers+text",
                marker=dict(size=14, color=cause_color, symbol="circle",
                            line=dict(width=2, color="white")),
                text=[f"T{attr.cause_turn}"],
                textposition="bottom center",
                textfont=dict(size=9),
                hovertemplate=(
                    f"<b>CAUSE — Turn {attr.cause_turn}</b><br>"
                    f"Speaker: {attr.cause_speaker}<br>"
                    f"Emotion: {attr.cause_emotion}<br>"
                    f'"{attr.cause_text[:60]}..."<br>'
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # Effect point
        fig.add_trace(
            go.Scatter(
                x=[attr.effect_turn],
                y=[i],
                mode="markers+text",
                marker=dict(size=14, color=effect_color, symbol="diamond",
                            line=dict(width=2, color="white")),
                text=[f"T{attr.effect_turn}"],
                textposition="bottom center",
                textfont=dict(size=9),
                hovertemplate=(
                    f"<b>EFFECT — Turn {attr.effect_turn}</b><br>"
                    f"Speaker: {attr.effect_speaker}<br>"
                    f"Emotion: {attr.effect_emotion}<br>"
                    f'"{attr.effect_text[:60]}..."<br>'
                    f"Confidence: {attr.attribution_score:.0%}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # Arrow connecting cause → effect
        fig.add_annotation(
            x=attr.effect_turn,
            y=i,
            ax=attr.cause_turn,
            ay=i,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="rgba(139, 92, 246, 0.6)",
        )

    fig.update_layout(
        title=dict(
            text="Emotion Cause Attribution Graph",
            font=dict(size=18, color="#E2E8F0"),
        ),
        xaxis_title="Turn Index",
        yaxis=dict(visible=False),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        height=max(200, 80 * len(attributions)),
        margin=dict(l=30, r=30, t=60, b=50),
    )

    return fig


def render_cause_details(attributions: list[CauseAttribution]) -> None:
    """Render detailed cause attribution cards in Streamlit."""
    if not attributions:
        st.info("No emotion cause attributions were detected (tone was stable).")
        return

    for attr in attributions:
        cause_emoji = "💬"
        effect_emoji = "💥"

        st.markdown(
            f"""
            <div style="
                background: rgba(30, 41, 59, 0.8);
                border: 1px solid rgba(139, 92, 246, 0.3);
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
            ">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.2em;">{cause_emoji}</span>
                    <span style="color: #C4B5FD; font-weight: 600;">
                        Turn {attr.cause_turn} ({attr.cause_speaker}) — {attr.cause_emotion}
                    </span>
                    <span style="color: #64748B;">→</span>
                    <span style="font-size: 1.2em;">{effect_emoji}</span>
                    <span style="color: #FCA5A5; font-weight: 600;">
                        Turn {attr.effect_turn} ({attr.effect_speaker}) — {attr.effect_emotion}
                    </span>
                </div>
                <p style="color: #94A3B8; font-size: 0.9em; margin: 4px 0;">
                    {attr.explanation}
                </p>
                <div style="
                    background: rgba(139, 92, 246, 0.15);
                    height: 4px;
                    border-radius: 2px;
                    margin-top: 8px;
                ">
                    <div style="
                        background: #8B5CF6;
                        height: 100%;
                        width: {attr.attribution_score * 100}%;
                        border-radius: 2px;
                    "></div>
                </div>
                <span style="font-size: 0.75em; color: #64748B;">
                    Attribution confidence: {attr.attribution_score:.0%}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
