"""
Base classes and schemas for chat message parsing.

All platform-specific parsers normalize their output into these structures,
ensuring a uniform interface for downstream analysis regardless of source format.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ParsedMessage:
    """A single normalized chat message."""

    turn_index: int
    speaker: str
    text: str
    timestamp: datetime | None = None
    is_system_message: bool = False
    raw_line: str = ""

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_system_message": self.is_system_message,
        }


@dataclass
class ParsedConversation:
    """A fully parsed multi-turn conversation."""

    source_format: str
    source_file: str
    messages: list[ParsedMessage] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.messages)

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)

    def user_messages(self) -> list[ParsedMessage]:
        """Return only non-system messages."""
        return [m for m in self.messages if not m.is_system_message]

    def speaker_messages(self, speaker: str) -> list[ParsedMessage]:
        """Return messages from a specific speaker."""
        return [m for m in self.messages if m.speaker == speaker]

    def to_dicts(self) -> list[dict]:
        return [m.to_dict() for m in self.messages]


class BaseChatParser(abc.ABC):
    """Abstract base for all chat format parsers."""

    FORMAT_NAME: str = "unknown"

    @abc.abstractmethod
    def parse(self, file_path: str) -> ParsedConversation:
        """Parse a chat export file into a ParsedConversation.

        Args:
            file_path: Path to the exported chat file.

        Returns:
            ParsedConversation with all messages normalized.
        """
        ...

    def _extract_speakers(self, messages: list[ParsedMessage]) -> list[str]:
        """Extract unique speaker names preserving first-appearance order."""
        seen: set[str] = set()
        speakers: list[str] = []
        for msg in messages:
            if msg.speaker and msg.speaker not in seen and not msg.is_system_message:
                seen.add(msg.speaker)
                speakers.append(msg.speaker)
        return speakers

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic text cleaning — strip whitespace, collapse newlines."""
        text = text.strip()
        # Collapse multiple newlines to single space for analysis
        text = " ".join(text.split())
        return text
