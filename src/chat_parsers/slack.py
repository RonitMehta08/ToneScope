"""
Slack JSON export parser.

Handles Slack workspace exports (JSON files from channel history) and the
slack_export format where each channel is a folder of day-by-day JSON files.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.chat_parsers.base import BaseChatParser, ParsedConversation, ParsedMessage


class SlackParser(BaseChatParser):
    """Parses Slack JSON channel exports."""

    FORMAT_NAME = "slack"

    def parse(self, file_path: str) -> ParsedConversation:
        path = Path(file_path)

        if path.is_dir():
            return self._parse_directory(path)
        return self._parse_single_file(path)

    def _parse_single_file(self, path: Path) -> ParsedConversation:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))

        # Handle both list-of-messages and {"messages": [...]} formats
        if isinstance(raw, list):
            raw_messages = raw
        elif isinstance(raw, dict) and "messages" in raw:
            raw_messages = raw["messages"]
        else:
            raw_messages = []

        messages: list[ParsedMessage] = []
        turn_index = 0

        for msg in raw_messages:
            # Skip bot messages, channel joins, etc.
            subtype = msg.get("subtype", "")
            if subtype in ("channel_join", "channel_leave", "bot_message", "channel_topic", "channel_purpose"):
                messages.append(
                    ParsedMessage(
                        turn_index=turn_index,
                        speaker="SYSTEM",
                        text=self._clean_text(msg.get("text", "")),
                        timestamp=self._parse_slack_ts(msg.get("ts")),
                        is_system_message=True,
                        raw_line=json.dumps(msg),
                    )
                )
                turn_index += 1
                continue

            text = msg.get("text", "").strip()
            if not text:
                continue

            # Extract speaker: prefer real_name > user_profile.real_name > user
            speaker = (
                msg.get("user_profile", {}).get("real_name")
                or msg.get("user_profile", {}).get("display_name")
                or msg.get("user", "Unknown")
            )

            messages.append(
                ParsedMessage(
                    turn_index=turn_index,
                    speaker=speaker,
                    text=self._clean_text(text),
                    timestamp=self._parse_slack_ts(msg.get("ts")),
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
            metadata={"channel": path.stem},
        )

    def _parse_directory(self, dir_path: Path) -> ParsedConversation:
        """Parse a Slack export directory (day-by-day JSON files)."""
        json_files = sorted(dir_path.glob("*.json"))
        all_messages: list[ParsedMessage] = []

        for jf in json_files:
            conv = self._parse_single_file(jf)
            # Re-index turns for continuity
            offset = len(all_messages)
            for msg in conv.messages:
                msg.turn_index += offset
            all_messages.extend(conv.messages)

        speakers = self._extract_speakers(all_messages)

        return ParsedConversation(
            source_format=self.FORMAT_NAME,
            source_file=str(dir_path),
            messages=all_messages,
            speakers=speakers,
            metadata={"num_files": len(json_files)},
        )

    @staticmethod
    def _parse_slack_ts(ts_str: str | None) -> datetime | None:
        """Parse Slack's epoch timestamp string (e.g. '1609459200.000100')."""
        if not ts_str:
            return None
        try:
            epoch = float(ts_str)
            return datetime.fromtimestamp(epoch)
        except (ValueError, OSError):
            return None
