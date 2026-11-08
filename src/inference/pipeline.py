"""
End-to-end inference pipeline for ToneScope.

Orchestrates: parse → preprocess → predict → correct → analyze → output.

Supports two modes:
    1. Custom-trained models (from scripts/train_*.py output)
    2. Pre-trained HuggingFace models (fallback, no training needed)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.config import (
    MODEL_CONFIG,
    MODELS_DIR,
    ID_TO_EMOTION,
    EKMAN_EMOTIONS,
    EMOTION_TO_ID,
)
from src.chat_parsers.base import ParsedConversation, ParsedMessage
from src.data.preprocessing import clean_chat_text, extract_lexical_features
from src.models.sarcasm_corrector import (
    correct_emotion_for_sarcasm,
    CorrectedPrediction,
)


@dataclass
class TurnPrediction:
    """Complete prediction for a single conversation turn."""

    turn_index: int
    speaker: str
    text: str
    cleaned_text: str

    # Emotion
    emotion: str
    emotion_confidence: float
    emotion_probabilities: dict[str, float]

    # Sarcasm
    is_sarcastic: bool
    sarcasm_confidence: float

    # Sarcasm-corrected emotion
    corrected_emotion: str
    was_corrected: bool
    correction_explanation: str

    # Escalation
    escalation_score: float
    escalation_level: str  # "safe", "caution", "alert"

    # Lexical features
    lexical_features: dict[str, float]

    # Metadata
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "text": self.text,
            "emotion": self.corrected_emotion,
            "original_emotion": self.emotion,
            "emotion_confidence": round(self.emotion_confidence, 3),
            "is_sarcastic": self.is_sarcastic,
            "sarcasm_confidence": round(self.sarcasm_confidence, 3),
            "was_corrected": self.was_corrected,
            "escalation_score": round(self.escalation_score, 3),
            "escalation_level": self.escalation_level,
            "timestamp": self.timestamp,
        }


@dataclass
class ConversationAnalysis:
    """Complete analysis result for a conversation."""

    source_format: str
    num_turns: int
    speakers: list[str]
    predictions: list[TurnPrediction]
    metadata: dict[str, Any] = field(default_factory=dict)


class ToneScopePipeline:
    """Main inference pipeline for ToneScope.

    Loads models, runs predictions, applies sarcasm correction,
    and produces per-turn analysis.
    """

    def __init__(
        self,
        emotion_model_path: str | None = None,
        sarcasm_model_path: str | None = None,
        device: str | None = None,
    ):
        """Initialize the pipeline.

        Args:
            emotion_model_path: Path to custom-trained emotion model.
                               If None, uses pre-trained HuggingFace model.
            sarcasm_model_path: Path to custom-trained sarcasm model.
                               If None, uses pre-trained HuggingFace model.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"ToneScope Pipeline — Device: {self.device}")

        # Load emotion model
        self._load_emotion_model(emotion_model_path)

        # Load sarcasm model
        self._load_sarcasm_model(sarcasm_model_path)

    def _load_emotion_model(self, model_path: str | None) -> None:
        """Load emotion classification model."""
        custom_path = Path(model_path) if model_path else MODELS_DIR / "emotion_model"

        if custom_path.exists() and (custom_path / "config.json").exists():
            print(f"  Loading custom emotion model from {custom_path}")
            self.emotion_pipe = pipeline(
                "text-classification",
                model=str(custom_path),
                tokenizer=str(custom_path),
                device=0 if self.device == "cuda" else -1,
                top_k=None,
                truncation=True,
                max_length=MODEL_CONFIG.max_seq_length,
            )
            self._emotion_mode = "custom"
        else:
            print("  Loading pre-trained emotion model (SamLowe/roberta-base-go_emotions)")
            self.emotion_pipe = pipeline(
                "text-classification",
                model="SamLowe/roberta-base-go_emotions",
                device=0 if self.device == "cuda" else -1,
                top_k=None,
                truncation=True,
                max_length=MODEL_CONFIG.max_seq_length,
            )
            self._emotion_mode = "pretrained"

    def _load_sarcasm_model(self, model_path: str | None) -> None:
        """Load sarcasm detection model."""
        custom_path = Path(model_path) if model_path else MODELS_DIR / "sarcasm_model"

        if custom_path.exists() and (custom_path / "config.json").exists():
            print(f"  Loading custom sarcasm model from {custom_path}")
            self.sarcasm_pipe = pipeline(
                "text-classification",
                model=str(custom_path),
                tokenizer=str(custom_path),
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=MODEL_CONFIG.max_seq_length,
            )
            self._sarcasm_mode = "custom"
        else:
            print("  Loading pre-trained sarcasm model (mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis → fallback to rule-based)")
            # Use a lightweight sentiment model as fallback sarcasm detector
            # Combined with lexical heuristics for sarcasm detection
            self.sarcasm_pipe = None
            self._sarcasm_mode = "heuristic"

    def _predict_emotion(self, text: str) -> tuple[str, float, dict[str, float]]:
        """Predict emotion for a single text.

        Returns:
            (emotion_label, confidence, all_probabilities)
        """
        from src.config import GOEMOTIONS_TO_EKMAN

        results = self.emotion_pipe(text)

        if self._emotion_mode == "pretrained":
            # Map GoEmotions labels to Ekman
            ekman_scores: dict[str, float] = {e: 0.0 for e in EKMAN_EMOTIONS}
            for item in results[0] if isinstance(results[0], list) else results:
                label = item["label"].lower()
                score = item["score"]
                ekman = GOEMOTIONS_TO_EKMAN.get(label, "neutral")
                ekman_scores[ekman] = max(ekman_scores[ekman], score)

            # Normalize
            total = sum(ekman_scores.values())
            if total > 0:
                ekman_scores = {k: v / total for k, v in ekman_scores.items()}

            best_emotion = max(ekman_scores, key=ekman_scores.get)
            return best_emotion, ekman_scores[best_emotion], ekman_scores
        else:
            # Custom model output
            all_probs: dict[str, float] = {}
            for item in results[0] if isinstance(results[0], list) else results:
                all_probs[item["label"]] = item["score"]

            best = max(all_probs, key=all_probs.get)
            return best, all_probs[best], all_probs

    def _predict_sarcasm(self, text: str) -> tuple[bool, float]:
        """Predict sarcasm for a single text.

        Returns:
            (is_sarcastic, confidence)
        """
        if self._sarcasm_mode == "heuristic":
            return self._heuristic_sarcasm(text)

        results = self.sarcasm_pipe(text)
        result = results[0] if isinstance(results, list) else results

        if isinstance(result, list):
            result = result[0]

        label = result["label"].lower()
        score = result["score"]

        # Handle various label formats
        is_sarcastic = any(
            kw in label
            for kw in ["sarcas", "ironi", "irony", "label_1", "1"]
        )

        return is_sarcastic, score if is_sarcastic else 1.0 - score

    @staticmethod
    def _heuristic_sarcasm(text: str) -> tuple[bool, float]:
        """Rule-based sarcasm detection as fallback.

        Looks for common sarcasm indicators:
            - Excessive punctuation (!!! ???)
            - All caps with positive words
            - Quotes around words ("great")
            - Common sarcastic phrases
            - Emoji contradictions
        """
        score = 0.0
        indicators_found = 0

        text_lower = text.lower()

        # Sarcastic phrases
        sarcastic_phrases = [
            "oh great", "yeah right", "sure thing", "oh wonderful",
            "oh perfect", "how lovely", "oh fantastic", "big surprise",
            "oh joy", "what a surprise", "oh really", "no kidding",
            "wow really", "thanks a lot", "oh how nice", "just perfect",
            "as if", "oh sure", "clearly", "obviously",
        ]
        for phrase in sarcastic_phrases:
            if phrase in text_lower:
                score += 0.3
                indicators_found += 1

        # Excessive punctuation
        if text.count("!") >= 3 or text.count("?") >= 3:
            score += 0.15
            indicators_found += 1

        # Air quotes
        if text.count('"') >= 2:
            score += 0.2
            indicators_found += 1

        # Ellipsis with positive words
        if "..." in text and any(w in text_lower for w in ["great", "nice", "wonderful", "perfect", "love"]):
            score += 0.2
            indicators_found += 1

        # ALL CAPS for emphasis
        words = text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if len(caps_words) >= 2:
            score += 0.15
            indicators_found += 1

        score = min(score, 1.0)
        is_sarcastic = score >= 0.35

        return is_sarcastic, score

    def _compute_escalation_score(
        self,
        emotion: str,
        sarcasm_confidence: float,
        lexical_features: dict[str, float],
        recent_emotions: list[str],
    ) -> float:
        """Compute escalation risk score for a turn.

        Args:
            emotion: Current (corrected) emotion.
            sarcasm_confidence: Sarcasm detection confidence.
            lexical_features: Hand-crafted lexical features.
            recent_emotions: Recent emotion history (last N turns).

        Returns:
            Escalation score 0.0 (safe) to 1.0 (alert).
        """
        from src.config import ESCALATION_CONFIG as ec

        # Sentiment component: negative emotions push score up
        negative_emotions = {"anger", "disgust", "fear", "sadness"}
        sentiment_score = 1.0 if emotion in negative_emotions else 0.0

        # Recent negativity trend
        if recent_emotions:
            recent_negative_ratio = sum(
                1 for e in recent_emotions if e in negative_emotions
            ) / len(recent_emotions)
            sentiment_score = (sentiment_score + recent_negative_ratio) / 2

        # Sarcasm component
        sarcasm_score = sarcasm_confidence

        # Lexical intensity component
        caps = lexical_features.get("caps_ratio", 0.0)
        excl = min(lexical_features.get("exclamation_count", 0) / 5, 1.0)
        repeated = min(lexical_features.get("repeated_punct", 0) / 3, 1.0)
        lexical_score = (caps + excl + repeated) / 3

        # Weighted combination
        escalation = (
            ec.sentiment_weight * sentiment_score
            + ec.sarcasm_weight * sarcasm_score
            + ec.lexical_weight * lexical_score
        )

        return min(max(escalation, 0.0), 1.0)

    @staticmethod
    def _escalation_level(score: float) -> str:
        """Convert escalation score to level label."""
        from src.config import ESCALATION_CONFIG as ec

        if score < ec.safe_threshold:
            return "safe"
        elif score < ec.caution_threshold:
            return "caution"
        return "alert"

    def analyze_conversation(
        self, conversation: ParsedConversation
    ) -> ConversationAnalysis:
        """Run full analysis on a parsed conversation.

        Args:
            conversation: ParsedConversation from any chat parser.

        Returns:
            ConversationAnalysis with per-turn predictions.
        """
        user_messages = conversation.user_messages()
        predictions: list[TurnPrediction] = []
        recent_emotions: list[str] = []

        for msg in user_messages:
            # Clean text
            cleaned = clean_chat_text(msg.text)
            if not cleaned:
                continue

            # Extract lexical features (from raw text)
            lexical = extract_lexical_features(msg.text)

            # Predict emotion
            emotion, emotion_conf, emotion_probs = self._predict_emotion(cleaned)

            # Predict sarcasm
            is_sarcastic, sarcasm_conf = self._predict_sarcasm(cleaned)

            # Apply sarcasm correction
            correction = correct_emotion_for_sarcasm(
                emotion_label=emotion,
                emotion_confidence=emotion_conf,
                is_sarcastic=is_sarcastic,
                sarcasm_confidence=sarcasm_conf,
            )

            # Compute escalation
            esc_score = self._compute_escalation_score(
                emotion=correction.corrected_emotion,
                sarcasm_confidence=sarcasm_conf,
                lexical_features=lexical,
                recent_emotions=recent_emotions[-5:],
            )

            prediction = TurnPrediction(
                turn_index=msg.turn_index,
                speaker=msg.speaker,
                text=msg.text,
                cleaned_text=cleaned,
                emotion=emotion,
                emotion_confidence=emotion_conf,
                emotion_probabilities=emotion_probs,
                is_sarcastic=is_sarcastic,
                sarcasm_confidence=sarcasm_conf,
                corrected_emotion=correction.corrected_emotion,
                was_corrected=correction.was_corrected,
                correction_explanation=correction.explanation,
                escalation_score=esc_score,
                escalation_level=self._escalation_level(esc_score),
                lexical_features=lexical,
                timestamp=msg.timestamp.isoformat() if msg.timestamp else None,
            )

            predictions.append(prediction)
            recent_emotions.append(correction.corrected_emotion)

        return ConversationAnalysis(
            source_format=conversation.source_format,
            num_turns=len(predictions),
            speakers=conversation.speakers,
            predictions=predictions,
            metadata=conversation.metadata,
        )

    def analyze_text_list(
        self,
        messages: list[dict[str, str]],
    ) -> ConversationAnalysis:
        """Analyze a simple list of messages (no file parsing needed).

        Args:
            messages: List of {"speaker": ..., "text": ...} dicts.

        Returns:
            ConversationAnalysis.
        """
        parsed_messages = [
            ParsedMessage(
                turn_index=i,
                speaker=msg.get("speaker", f"Speaker_{i % 2}"),
                text=msg["text"],
            )
            for i, msg in enumerate(messages)
        ]

        conversation = ParsedConversation(
            source_format="direct",
            source_file="",
            messages=parsed_messages,
            speakers=list({m.speaker for m in parsed_messages}),
        )

        return self.analyze_conversation(conversation)
