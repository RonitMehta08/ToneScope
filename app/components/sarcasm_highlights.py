"""
Sarcasm/irony highlight component.

Displays messages with sarcasm/irony/passive-aggression annotations
and the sarcasm correction explanations.
"""
from __future__ import annotations

import streamlit as st

from src.inference.pipeline import TurnPrediction


def render_sarcasm_highlights(predictions: list[TurnPrediction]) -> None:
    """Render sarcasm-annotated message cards in the Streamlit UI.

    Shows each message with badges for:
        🎭 Sarcasm
        🔄 Irony (if detected as sarcasm but with correction)
        😤 Passive-Aggression indicators
    """
    sarcastic_messages = [p for p in predictions if p.is_sarcastic]

    if not sarcastic_messages:
        st.info("✅ No sarcasm or irony detected in this conversation.")
        return

    st.markdown(
        f"**{len(sarcastic_messages)}** message(s) flagged for sarcasm/irony "
        f"out of {len(predictions)} total ({len(sarcastic_messages)/len(predictions):.0%})"
    )

    for pred in sarcastic_messages:
        badge = "🎭 Sarcasm"
        if pred.was_corrected:
            badge += " | 🔄 Emotion Corrected"

        with st.container():
            st.markdown(
                f"""
                <div style="
                    background: rgba(139, 92, 246, 0.1);
                    border-left: 4px solid #8B5CF6;
                    padding: 12px 16px;
                    border-radius: 0 8px 8px 0;
                    margin-bottom: 12px;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: #C4B5FD;">
                            {pred.speaker} — Turn {pred.turn_index}
                        </span>
                        <span style="
                            background: rgba(139, 92, 246, 0.3);
                            padding: 2px 8px;
                            border-radius: 12px;
                            font-size: 0.8em;
                        ">{badge}</span>
                    </div>
                    <p style="margin: 4px 0; color: #E2E8F0;">"{pred.text}"</p>
                    <div style="font-size: 0.85em; color: #94A3B8; margin-top: 6px;">
                        Sarcasm confidence: {pred.sarcasm_confidence:.0%}
                        {f' | {pred.emotion} → {pred.corrected_emotion}' if pred.was_corrected else ''}
                    </div>
                    <div style="font-size: 0.8em; color: #64748B; margin-top: 4px;">
                        {pred.correction_explanation}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
