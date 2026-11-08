"""
ToneScope — Streamlit Dashboard

The main application interface for analyzing multi-turn conversations.
Supports file upload (WhatsApp, Slack, Discord, Telegram) and provides
rich interactive visualizations of emotional arcs, escalation, sarcasm,
and emotion cause attribution.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Page config — must be first Streamlit call
st.set_page_config(
    page_title="ToneScope — Conversation Emotion Arc Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 0 10px;
        background: linear-gradient(90deg, #8B5CF6, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8em;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .sub-header {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1em;
        margin-bottom: 30px;
    }

    /* Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 20px;
        transition: transform 0.2s, border-color 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.5);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        padding: 8px 16px;
        color: #94A3B8;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(139, 92, 246, 0.3) !important;
        color: #E2E8F0 !important;
    }

    /* Dividers */
    hr { border-color: rgba(139, 92, 246, 0.2); }

    /* Message display */
    .msg-card {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-left: 3px solid;
    }

    .stMarkdown { color: #E2E8F0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # ─── Header ──────────────────────────────────────────────────────
    st.markdown('<h1 class="main-header">🧠 ToneScope</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multi-Turn Conversation Emotion Arc Analyzer<br>'
        '<em>Where minds decode emotion at machine speed.</em></p>',
        unsafe_allow_html=True,
    )

    # ─── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📁 Upload Conversation")
        st.markdown("Supported formats: WhatsApp (.txt), Slack (.json), Discord (.json/.csv), Telegram (.json)")

        uploaded_file = st.file_uploader(
            "Drop your chat export here",
            type=["txt", "json", "csv"],
            help="Export a chat from WhatsApp, Slack, Discord, or Telegram and upload it here.",
        )

        format_hint = st.selectbox(
            "Chat Format",
            options=["Auto-Detect", "WhatsApp", "Slack", "Discord", "Telegram"],
            index=0,
        )

        st.markdown("---")
        st.markdown("### ⚙️ Settings")

        sarcasm_threshold = st.slider(
            "Sarcasm Correction Threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Minimum sarcasm confidence to trigger emotion correction.",
        )

        escalation_window = st.slider(
            "Escalation Window Size",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of turns to consider for escalation trend.",
        )

        st.markdown("---")
        st.markdown("### 🎯 About ToneScope")
        st.markdown(
            """
            **ToneScope** analyzes the emotional DNA of multi-turn conversations.

            **Unique Features:**
            - 🎭 Sarcasm-corrected emotion detection
            - 📈 Escalation risk prediction
            - 🔍 Emotion cause attribution
            - 👥 Per-speaker emotion tracking
            - 📊 Interactive visualizations

            **Built for CEREBRO 2026** — PS-01: Tone Intelligence in Conversations
            """
        )

    # ─── Demo Mode ───────────────────────────────────────────────────
    use_demo = False
    if not uploaded_file:
        st.info(
            "👆 Upload a chat export in the sidebar to begin analysis, "
            "or click below to try a demo conversation."
        )
        use_demo = st.button("🎮 Load Demo Conversation", use_container_width=True)

    if not uploaded_file and not use_demo:
        _render_landing_page()
        return

    # ─── Process ─────────────────────────────────────────────────────
    with st.spinner("🔄 Loading ToneScope pipeline..."):
        from src.inference.pipeline import ToneScopePipeline, ConversationAnalysis
        from src.chat_parsers import detect_and_parse
        from src.analysis.emotion_arc import build_emotion_arc
        from src.analysis.escalation import compute_escalation_profile
        from src.analysis.cause_attribution import attribute_emotion_causes
        from src.analysis.conversation_summary import generate_summary

        # Initialize pipeline (cached)
        pipeline = _get_pipeline()

    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_file.name).suffix,
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        hint = None if format_hint == "Auto-Detect" else format_hint.lower()

        with st.spinner("📝 Parsing chat export..."):
            try:
                conversation = detect_and_parse(tmp_path, format_hint=hint)
            except Exception as e:
                st.error(f"❌ Failed to parse chat: {e}")
                return
    else:
        # Demo conversation
        conversation = _get_demo_conversation()

    st.success(
        f"✅ Parsed **{conversation.num_turns}** messages from "
        f"**{conversation.num_speakers}** speakers ({conversation.source_format})"
    )

    # Run analysis
    with st.spinner("🧠 Analyzing emotions, sarcasm, and escalation..."):
        analysis = pipeline.analyze_conversation(conversation)
        arc_data = build_emotion_arc(analysis.predictions)
        escalation_report = compute_escalation_profile(analysis.predictions)
        attributions = attribute_emotion_causes(analysis.predictions, arc_data.tone_shifts)
        summary = generate_summary(analysis, arc_data, escalation_report, attributions)

    # ─── Results ─────────────────────────────────────────────────────
    from app.components.summary_card import render_summary_card, create_speaker_radar
    from app.components.emotion_arc_chart import create_emotion_arc_chart, create_sentiment_timeline
    from app.components.speaker_lanes import create_speaker_lanes
    from app.components.escalation_heatmap import create_escalation_heatmap
    from app.components.sarcasm_highlights import render_sarcasm_highlights
    from app.components.cause_graph import create_cause_graph, render_cause_details

    # Summary card
    render_summary_card(summary)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Emotion Arc",
        "👥 Speaker Lanes",
        "🔥 Escalation Map",
        "🎭 Sarcasm/Irony",
        "🔍 Cause Attribution",
        "💬 Conversation View",
    ])

    with tab1:
        st.plotly_chart(create_emotion_arc_chart(arc_data), use_container_width=True)
        st.plotly_chart(create_sentiment_timeline(arc_data), use_container_width=True)

    with tab2:
        st.plotly_chart(create_speaker_lanes(arc_data), use_container_width=True)
        st.plotly_chart(create_speaker_radar(summary.speaker_profiles), use_container_width=True)

    with tab3:
        st.plotly_chart(create_escalation_heatmap(escalation_report), use_container_width=True)
        st.markdown(f"**Risk Summary:** {escalation_report.risk_summary}")
        if escalation_report.escalation_zones:
            st.markdown("**Escalation Zones:**")
            for zone in escalation_report.escalation_zones:
                st.markdown(
                    f"- Turns {zone.start_turn}–{zone.end_turn}: "
                    f"**{zone.level.upper()}** (peak: {zone.peak_score:.0%}) — "
                    f"{', '.join(zone.contributing_factors)}"
                )

    with tab4:
        render_sarcasm_highlights(analysis.predictions)

    with tab5:
        st.plotly_chart(create_cause_graph(attributions), use_container_width=True)
        render_cause_details(attributions)

    with tab6:
        _render_conversation_view(analysis)

    # Narrative summary
    st.markdown("---")
    st.markdown("### 📝 Narrative Summary")
    st.markdown(summary.narrative)


@st.cache_resource
def _get_pipeline():
    """Load and cache the ToneScope pipeline."""
    from src.inference.pipeline import ToneScopePipeline
    return ToneScopePipeline()


def _get_demo_conversation():
    """Create a demo conversation for showcase."""
    from src.chat_parsers.base import ParsedConversation, ParsedMessage
    from datetime import datetime, timedelta

    base_time = datetime(2026, 9, 20, 14, 0, 0)

    demo_messages = [
        ("Alice", "Hey team! How's the project going? I'm really excited about our progress!"),
        ("Bob", "Yeah it's going... great. We've only missed three deadlines so far."),
        ("Alice", "Wait what? I thought we were on track? The last update said everything was fine!"),
        ("Bob", "Oh sure, everything is ABSOLUTELY fine. If you count being two weeks behind as fine."),
        ("Charlie", "Guys, let's not point fingers. We had unexpected issues with the API integration."),
        ("Alice", "I get that, but why wasn't this communicated earlier? I've been presenting positive updates to the stakeholders!"),
        ("Bob", "Because every time someone raises a concern, they get told to 'stay positive'. Sound familiar?"),
        ("Alice", "That's not fair, Bob. I always listen to feedback!"),
        ("Charlie", "Okay okay, both of you calm down. Let's focus on solutions. What's the current blocker?"),
        ("Bob", "Fine. The authentication module is broken. Has been for days. I flagged it but nobody responded."),
        ("Alice", "I never saw that message... I'm sorry. Let me look into it right now."),
        ("Bob", "Thank you. That's all I was asking for."),
        ("Charlie", "Great. See? We can work this out. Let's set up a daily sync to catch these things earlier."),
        ("Alice", "That's a good idea, Charlie. I'll set it up. And Bob, I genuinely apologize for missing your flag."),
        ("Bob", "Appreciated. Let's just get this done. I'll prepare a status doc with the actual blockers."),
        ("Charlie", "Perfect. Now we're talking! Let's crush this."),
    ]

    messages = []
    for i, (speaker, text) in enumerate(demo_messages):
        messages.append(
            ParsedMessage(
                turn_index=i,
                speaker=speaker,
                text=text,
                timestamp=base_time + timedelta(minutes=i * 3),
            )
        )

    return ParsedConversation(
        source_format="demo",
        source_file="demo_conversation",
        messages=messages,
        speakers=["Alice", "Bob", "Charlie"],
        metadata={"type": "demo"},
    )


def _render_conversation_view(analysis) -> None:
    """Render the full conversation with inline emotion annotations."""
    from src.config import EMOTION_COLORS

    for pred in analysis.predictions:
        color = EMOTION_COLORS.get(pred.corrected_emotion, "#888888")
        sarcasm_badge = " 🎭" if pred.is_sarcastic else ""
        esc_badge = ""
        if pred.escalation_level == "alert":
            esc_badge = " 🔴"
        elif pred.escalation_level == "caution":
            esc_badge = " 🟡"

        st.markdown(
            f"""
            <div class="msg-card" style="border-left-color: {color};">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-weight: 600; color: #E2E8F0;">
                        {pred.speaker}{sarcasm_badge}{esc_badge}
                    </span>
                    <span style="font-size: 0.8em; color: {color};">
                        {pred.corrected_emotion} ({pred.emotion_confidence:.0%})
                    </span>
                </div>
                <p style="margin: 0; color: #CBD5E1;">{pred.text}</p>
                <div style="font-size: 0.75em; color: #64748B; margin-top: 4px;">
                    Turn {pred.turn_index}
                    {'| Escalation: ' + f'{pred.escalation_score:.0%}' if pred.escalation_score > 0.2 else ''}
                    {' | ' + pred.timestamp if pred.timestamp else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_landing_page() -> None:
    """Render the landing page when no file is uploaded."""
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### 🎯 Emotion Detection
            Classifies 7 Ekman emotions per message using
            fine-tuned RoBERTa with conversational context.
            """
        )

    with col2:
        st.markdown(
            """
            ### 🎭 Sarcasm Correction
            Detects sarcasm and automatically flips
            misclassified literal emotions to their true meaning.
            """
        )

    with col3:
        st.markdown(
            """
            ### 📈 Escalation Prediction
            A rolling 0→1 risk score that predicts
            when a conversation is heading toward conflict.
            """
        )

    st.markdown("---")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(
            """
            ### 🔍 Cause Attribution
            Identifies *which prior message* triggered each
            emotional tone shift — answers the "why."
            """
        )

    with col5:
        st.markdown(
            """
            ### 👥 Speaker Profiling
            Independent emotional trajectory tracking per speaker
            with radar chart emotion profiles.
            """
        )

    with col6:
        st.markdown(
            """
            ### 📊 Rich Visualizations
            Interactive Plotly charts: emotion arcs, sentiment
            timelines, escalation heatmaps, and cause graphs.
            """
        )


if __name__ == "__main__":
    main()
