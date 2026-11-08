"""
Emotion Arc Chart component — Interactive Plotly line chart.

Shows the emotional trajectory of the conversation with per-speaker
color coding and hover-to-read message text.
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import EMOTION_COLORS, EKMAN_EMOTIONS
from src.analysis.emotion_arc import EmotionArcData, ToneShift


def create_emotion_arc_chart(arc_data: EmotionArcData) -> go.Figure:
    """Create an interactive emotion arc line chart.

    Features:
        - One line per emotion showing intensity over turns
        - Vertical markers at tone shift points
        - Hover tooltips with actual message text
        - Smooth interpolation

    Args:
        arc_data: EmotionArcData from the analysis module.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    turns = arc_data.turns
    if not turns:
        fig.add_annotation(text="No data to display", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Add a trace for each emotion
    for emotion in EKMAN_EMOTIONS:
        intensities = arc_data.emotion_timeseries.get(emotion, [0.0] * len(turns))
        color = EMOTION_COLORS.get(emotion, "#888888")

        fig.add_trace(
            go.Scatter(
                x=turns,
                y=intensities,
                name=emotion.capitalize(),
                mode="lines+markers",
                line=dict(color=color, width=2, shape="spline"),
                marker=dict(size=4, color=color),
                hovertemplate=(
                    "<b>Turn %{x}</b><br>"
                    f"Emotion: {emotion.capitalize()}<br>"
                    "Intensity: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
                visible="legendonly" if emotion in ("fear", "disgust") else True,
            )
        )

    # Add tone shift markers
    for shift in arc_data.tone_shifts:
        fig.add_vline(
            x=shift.turn_index,
            line_dash="dash",
            line_color="rgba(255, 100, 100, 0.5)",
            annotation_text=f"↕ {shift.from_emotion}→{shift.to_emotion}",
            annotation_position="top",
            annotation_font_size=9,
        )

    fig.update_layout(
        title=dict(
            text="Emotion Arc — Intensity Over Conversation",
            font=dict(size=18, color="#E2E8F0"),
        ),
        xaxis_title="Turn Index",
        yaxis_title="Emotion Intensity",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        height=450,
        margin=dict(l=60, r=30, t=60, b=80),
        hovermode="x unified",
    )

    return fig


def create_sentiment_timeline(arc_data: EmotionArcData) -> go.Figure:
    """Create an aggregate sentiment polarity timeline.

    Shows smoothed positive/negative trajectory with shaded areas
    and tone shift vertical markers.

    Args:
        arc_data: EmotionArcData from the analysis module.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    turns = arc_data.turns
    if not turns:
        return fig

    # Raw sentiment
    fig.add_trace(
        go.Scatter(
            x=turns,
            y=arc_data.sentiment_scores,
            name="Raw Sentiment",
            mode="markers",
            marker=dict(
                size=6,
                color=arc_data.sentiment_scores,
                colorscale="RdYlGn",
                cmin=-1,
                cmax=1,
                showscale=True,
                colorbar=dict(title="Polarity", tickvals=[-1, 0, 1], ticktext=["Negative", "Neutral", "Positive"]),
            ),
            customdata=arc_data.texts,
            hovertemplate=(
                "<b>Turn %{x}</b><br>"
                "Sentiment: %{y:.2f}<br>"
                "Message: %{customdata}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Smoothed line
    fig.add_trace(
        go.Scatter(
            x=turns,
            y=arc_data.smoothed_sentiment,
            name="Smoothed Trend",
            mode="lines",
            line=dict(color="#8B5CF6", width=3, shape="spline"),
        )
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")

    # Shade positive/negative regions
    fig.add_hrect(y0=0, y1=1, fillcolor="rgba(34, 197, 94, 0.05)", line_width=0)
    fig.add_hrect(y0=-1, y1=0, fillcolor="rgba(239, 68, 68, 0.05)", line_width=0)

    fig.update_layout(
        title=dict(
            text="Sentiment Polarity Timeline",
            font=dict(size=18, color="#E2E8F0"),
        ),
        xaxis_title="Turn Index",
        yaxis_title="Sentiment Polarity",
        yaxis=dict(range=[-1.1, 1.1]),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        height=350,
        margin=dict(l=60, r=30, t=60, b=50),
    )

    return fig
