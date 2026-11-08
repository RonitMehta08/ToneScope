"""
Text preprocessing pipeline for ToneScope.

Cleans, normalizes, and prepares chat text for model consumption.
"""
from __future__ import annotations

import re
import emoji


def clean_chat_text(text: str) -> str:
    """Full cleaning pipeline for a single chat message.

    Steps:
        1. Convert emoji to text descriptions
        2. Remove URLs
        3. Normalize whitespace
        4. Remove special unicode characters
        5. Lowercase (preserving capitalization info as a feature)

    Args:
        text: Raw message text.

    Returns:
        Cleaned text suitable for tokenization.
    """
    if not text or not text.strip():
        return ""

    # Convert emoji to text descriptions (":grinning_face:" etc.)
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

    # Remove mentions (@username) — keep the name, drop the @
    text = re.sub(r"@(\w+)", r"\1", text)

    # Remove hashtag symbol, keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # Normalize repeated punctuation (!!!! → !!, ??? → ??)
    text = re.sub(r"([!?.])\1{2,}", r"\1\1", text)

    # Normalize repeated characters (heeeeello → heello)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    # Remove non-printable/special unicode (keep basic latin + common punctuation)
    text = re.sub(r"[^\x20-\x7E\u00C0-\u024F\u2000-\u206F]", " ", text)

    # Collapse whitespace
    text = " ".join(text.split())

    return text.strip()


def extract_lexical_features(text: str) -> dict[str, float]:
    """Extract hand-crafted lexical features for escalation detection.

    Args:
        text: Raw (pre-cleaning) message text.

    Returns:
        Dictionary of feature name → value.
    """
    if not text:
        return {
            "caps_ratio": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "word_count": 0,
            "avg_word_length": 0.0,
            "emoji_count": 0,
            "repeated_punct": 0,
        }

    words = text.split()
    alpha_chars = [c for c in text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]

    caps_ratio = len(upper_chars) / max(len(alpha_chars), 1)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    emoji_count = len([c for c in text if c in emoji.EMOJI_DATA])
    repeated_punct = len(re.findall(r"[!?]{2,}", text))

    return {
        "caps_ratio": round(caps_ratio, 3),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "word_count": word_count,
        "avg_word_length": round(avg_word_length, 2),
        "emoji_count": emoji_count,
        "repeated_punct": repeated_punct,
    }


def prepare_context_window(
    messages: list[str],
    current_index: int,
    window_size: int = 5,
    separator: str = " [SEP] ",
) -> str:
    """Build a context string from preceding messages for context-aware models.

    Args:
        messages: List of all message texts in the conversation.
        current_index: Index of the current message.
        window_size: Number of prior messages to include.
        separator: Token to separate context turns.

    Returns:
        Concatenated context string ending with the current message.
    """
    start = max(0, current_index - window_size)
    context_messages = messages[start : current_index + 1]
    return separator.join(context_messages)
