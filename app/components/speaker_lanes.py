"""
Per-speaker emotion lanes visualization.

Displays parallel swim lanes showing each speaker's emotional trajectory
independently, enabling cross-speaker comparison.
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import EMOTION_COLORS
from src.analysis.emotion_arc import EmotionArcData


def create_speaker_lanes(arc_data: EmotionArcData) -> go.Figure:
    """Create per-speaker emotion swim lanes.

    Each speaker gets their own subplot row showing their emotion over time.
    """
    speakers = list(arc_data.speaker_arcs.keys())
    if not speakers:
        fig = go.Figure()
        fig.add_annotation(text="No speakers to display", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = make_subplots(
        rows=len(speakers),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"🎤 {s}" for s in speakers],
        vertical_spacing=0.08,
    )

    for idx, speaker in enumerate(speakers, 1):
        arc = arc_data.speaker_arcs[speaker]
        turns = arc["turns"]
        emotions = arc["emotions"]
        sentiments = arc["sentiment"]
        texts = arc["texts"]

        colors = [EMOTION_COLORS.get(e, "#888888") for e in emotions]

        fig.add_trace(
            go.Scatter(
                x=turns,
                y=sentiments,
                mode="lines+markers",
                marker=dict(
                    size=8,
                    color=colors,
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                line=dict(color="rgba(139, 92, 246, 0.4)", width=1, shape="spline"),
                customdata=list(zip(emotions, texts)),
                hovertemplate=(
                    "<b>Turn %{x}</b><br>"
                    "Emotion: %{customdata[0]}<br>"
                    "Sentiment: %{y:.2f}<br>"
                    "Message: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
                name=speaker,
                showlegend=False,
            ),
            row=idx,
            col=1,
        )

        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=idx, col=1)

    fig.update_layout(
        title=dict(
            text="Speaker Emotion Lanes",
            font=dict(size=18, color="#E2E8F0"),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        height=250 * len(speakers),
        margin=dict(l=60, r=30, t=60, b=50),
    )

    return fig
