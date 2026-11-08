"""
Discord chat export parser.

Handles exports from DiscordChatExporter (JSON or CSV format) and other
common Discord export tools.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from src.chat_parsers.base import BaseChatParser, ParsedConversation, ParsedMessage


class DiscordParser(BaseChatParser):
    """Parses Discord chat exports (JSON or CSV)."""

    FORMAT_NAME = "discord"

    def parse(self, file_path: str) -> ParsedConversation:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".json":
            return self._parse_json(path)
        elif suffix == ".csv":
            return self._parse_csv(path)
        else:
            raise ValueError(f"Unsupported Discord export format: {suffix}. Use .json or .csv")

    def _parse_json(self, path: Path) -> ParsedConversation:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))

        # DiscordChatExporter JSON format
        raw_messages = raw.get("messages", []) if isinstance(raw, dict) else raw

        messages: list[ParsedMessage] = []
        turn_index = 0

        for msg in raw_messages:
            content = msg.get("content", "").strip()
            if not content:
                continue

            # Extract author
            author = msg.get("author", {})
            if isinstance(author, dict):
                speaker = author.get("name") or author.get("nickname") or author.get("id", "Unknown")
            elif isinstance(author, str):
                speaker = author
            else:
                speaker = "Unknown"

            # Check for bot/system
            is_system = (
                msg.get("type", "") != "Default"
                and msg.get("type", "") != ""
            ) or (isinstance(author, dict) and author.get("isBot", False))

            timestamp = self._parse_discord_ts(msg.get("timestamp"))

            messages.append(
                ParsedMessage(
                    turn_index=turn_index,
                    speaker=speaker,
                    text=self._clean_text(content),
                    timestamp=timestamp,
                    is_system_message=is_system,
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
            metadata={"channel": raw.get("channel", {}).get("name", path.stem) if isinstance(raw, dict) else path.stem},
        )

    def _parse_csv(self, path: Path) -> ParsedConversation:
        messages: list[ParsedMessage] = []
        turn_index = 0

        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Common CSV column names from DiscordChatExporter
                content = (
                    row.get("Content", "")
                    or row.get("content", "")
                    or row.get("message", "")
                ).strip()

                if not content:
                    continue

                speaker = (
                    row.get("Author", "")
                    or row.get("author", "")
                    or row.get("username", "")
                    or "Unknown"
                )

                date_str = row.get("Date", "") or row.get("date", "") or row.get("timestamp", "")
                timestamp = self._parse_discord_ts(date_str) if date_str else None

                messages.append(
                    ParsedMessage(
                        turn_index=turn_index,
                        speaker=speaker,
                        text=self._clean_text(content),
                        timestamp=timestamp,
                        is_system_message=False,
                        raw_line=str(row),
                    )
                )
                turn_index += 1

        speakers = self._extract_speakers(messages)

        return ParsedConversation(
            source_format=self.FORMAT_NAME,
            source_file=str(path),
            messages=messages,
            speakers=speakers,
        )

    @staticmethod
    def _parse_discord_ts(ts_str: str | None) -> datetime | None:
        """Parse Discord ISO-8601 timestamp."""
        if not ts_str:
            return None

        # Remove trailing timezone Z or +00:00
        ts_str = ts_str.strip()

        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]:
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        return None
