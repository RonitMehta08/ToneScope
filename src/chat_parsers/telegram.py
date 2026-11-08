"""
Telegram chat export parser.

Handles JSON exports from Telegram Desktop's "Export Chat History" feature.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.chat_parsers.base import BaseChatParser, ParsedConversation, ParsedMessage


class TelegramParser(BaseChatParser):
    """Parses Telegram Desktop JSON exports."""

    FORMAT_NAME = "telegram"

    def parse(self, file_path: str) -> ParsedConversation:
        path = Path(file_path)
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))

        raw_messages = raw.get("messages", [])
        chat_name = raw.get("name", path.stem)

        messages: list[ParsedMessage] = []
        turn_index = 0

        for msg in raw_messages:
            msg_type = msg.get("type", "message")

            # Handle service messages (group joins, pins, etc.)
            if msg_type == "service":
                action = msg.get("action", "")
                actor = msg.get("actor", "SYSTEM")
                messages.append(
                    ParsedMessage(
                        turn_index=turn_index,
                        speaker="SYSTEM",
                        text=self._clean_text(f"{actor}: {action}"),
                        timestamp=self._parse_telegram_ts(msg.get("date")),
                        is_system_message=True,
                        raw_line=json.dumps(msg),
                    )
                )
                turn_index += 1
                continue

            # Extract text — Telegram stores text as either a string or a list
            # of text entities (for formatted text)
            raw_text = msg.get("text", "")
            text = self._flatten_text(raw_text)
            if not text:
                continue

            speaker = msg.get("from", "") or msg.get("actor", "") or "Unknown"
            timestamp = self._parse_telegram_ts(msg.get("date"))

            messages.append(
                ParsedMessage(
                    turn_index=turn_index,
                    speaker=speaker,
                    text=self._clean_text(text),
                    timestamp=timestamp,
                    is_system_message=False,
                    raw_line=json.dumps(msg),
                )
            )
            turn_index += 1

        speakers = self._extract_speakers(messages)

        return ParsedConversation(
            source_format=self.FORMAT_NAME,
            source_file=str(path),
            messages=messages,
            speakers=speakers,
            metadata={"chat_name": chat_name, "chat_type": raw.get("type", "unknown")},
        )

    @staticmethod
    def _flatten_text(text_field: str | list) -> str:
        """Flatten Telegram's text field which can be a string or list of entities."""
        if isinstance(text_field, str):
            return text_field

        if isinstance(text_field, list):
            parts: list[str] = []
            for part in text_field:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("text", ""))
            return "".join(parts)

        return ""

    @staticmethod
    def _parse_telegram_ts(date_str: str | None) -> datetime | None:
        """Parse Telegram's ISO-like date format."""
        if not date_str:
            return None

        for fmt in [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
        ]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
