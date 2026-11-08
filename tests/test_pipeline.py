"""
End-to-end pipeline integration tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPreprocessing:
    def test_clean_chat_text(self):
        from src.data.preprocessing import clean_chat_text

        assert clean_chat_text("Hello world!") == "Hello world!"
        assert "[URL]" in clean_chat_text("Check https://example.com out")
        assert clean_chat_text("   spaces   ") == "spaces"
        assert clean_chat_text("") == ""

    def test_extract_lexical_features(self):
        from src.data.preprocessing import extract_lexical_features

        features = extract_lexical_features("HELLO WORLD!!!")
        assert features["caps_ratio"] > 0.5
        assert features["exclamation_count"] == 3

    def test_context_window(self):
        from src.data.preprocessing import prepare_context_window

        messages = ["Hi", "How are you?", "Fine", "Good", "Bye"]
        context = prepare_context_window(messages, 3, window_size=2)
        assert "How are you?" in context
        assert "Fine" in context
        assert "Good" in context


class TestSarcasmCorrector:
    def test_correction_on_sarcastic_joy(self):
        from src.models.sarcasm_corrector import correct_emotion_for_sarcasm

        result = correct_emotion_for_sarcasm(
            emotion_label="joy",
            emotion_confidence=0.9,
            is_sarcastic=True,
            sarcasm_confidence=0.8,
        )
        assert result.was_corrected
        assert result.corrected_emotion == "anger"

    def test_no_correction_without_sarcasm(self):
        from src.models.sarcasm_corrector import correct_emotion_for_sarcasm

        result = correct_emotion_for_sarcasm(
            emotion_label="joy",
            emotion_confidence=0.9,
            is_sarcastic=False,
            sarcasm_confidence=0.1,
        )
        assert not result.was_corrected
        assert result.corrected_emotion == "joy"

    def test_no_correction_below_threshold(self):
        from src.models.sarcasm_corrector import correct_emotion_for_sarcasm

        result = correct_emotion_for_sarcasm(
            emotion_label="joy",
            emotion_confidence=0.9,
            is_sarcastic=True,
            sarcasm_confidence=0.3,
            correction_threshold=0.5,
        )
        assert not result.was_corrected
