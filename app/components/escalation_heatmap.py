"""
Escalation heatmap component.

Displays a heatmap of escalation risk scores across conversation turns,
with danger zones highlighted.
"""
from __future__ import annotations

import plotly.graph_objects as go

from src.analysis.escalation import EscalationReport


def create_escalation_heatmap(
    escalation_report: EscalationReport,
    speakers: list[str] | None = None,
) -> go.Figure:
    """Create an escalation risk heatmap.

    Shows a color-coded bar for each turn (green→yellow→red) and
    highlights escalation zones.
    """
    scores = escalation_report.per_turn_scores
    if not scores:
        fig = go.Figure()
        fig.add_annotation(text="No escalation data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    turns = list(range(len(scores)))

    # Create bar chart with gradient coloring
    colors = []
    for score in scores:
        if score < 0.3:
            colors.append(f"rgba(34, 197, 94, {0.3 + score})")
        elif score < 0.6:
            colors.append(f"rgba(245, 158, 11, {0.3 + score})")
        else:
            colors.append(f"rgba(239, 68, 68, {0.3 + score})")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=turns,
            y=scores,
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate=(
                "<b>Turn %{x}</b><br>"
                "Escalation: %{y:.1%}<br>"
                "<extra></extra>"
            ),
            name="Escalation Risk",
        )
    )

    # Add threshold lines
    fig.add_hline(y=0.3, line_dash="dot", line_color="#22C55E",
                  annotation_text="Safe", annotation_position="right")
    fig.add_hline(y=0.6, line_dash="dot", line_color="#F59E0B",
                  annotation_text="Caution", annotation_position="right")

    # Highlight escalation zones
    for zone in escalation_report.escalation_zones:
        color = "rgba(239, 68, 68, 0.15)" if zone.level == "alert" else "rgba(245, 158, 11, 0.1)"
        fig.add_vrect(
            x0=zone.start_turn - 0.5,
            x1=zone.end_turn + 0.5,
            fillcolor=color,
            line_width=0,
            annotation_text=f"⚠ {zone.level}",
            annotation_position="top",
            annotation_font_size=9,
        )

    fig.update_layout(
        title=dict(
            text=f"Escalation Risk Map — Trend: {escalation_report.escalation_trend.capitalize()}",
            font=dict(size=18, color="#E2E8F0"),
        ),
        xaxis_title="Turn Index",
        yaxis_title="Escalation Score",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        height=300,
        margin=dict(l=60, r=60, t=60, b=50),
        showlegend=False,
    )

    return fig
