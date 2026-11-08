"""
WhatsApp chat export parser.

Handles the .txt export format produced by WhatsApp's "Export Chat" feature.
Supports multiple locale-specific date/time formats (DD/MM/YY, MM/DD/YY,
DD/MM/YYYY, 12h/24h clocks) via a bank of regex patterns.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from src.chat_parsers.base import BaseChatParser, ParsedConversation, ParsedMessage

# Regex patterns for WhatsApp message lines across different locales.
# Format: <date>, <time> - <sender>: <message>
# The date/time separators and formats vary by OS, language, and region.
_WHATSAPP_PATTERNS: list[re.Pattern] = [
    # DD/MM/YY, HH:MM (24h) — most common
    re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s+(.*?):\s(.+)$",
        re.DOTALL,
    ),
    # DD/MM/YY, HH:MM am/pm (12h)
    re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?\s*[AaPp][Mm])\s*-\s+(.*?):\s(.+)$",
        re.DOTALL,
    ),
    # [DD/MM/YY, HH:MM:SS] (bracket-wrapped variant)
    re.compile(
        r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\]\s+(.*?):\s(.+)$",
        re.DOTALL,
    ),
    # YYYY-MM-DD HH:MM (ISO-ish)
    re.compile(
        r"^(\d{4}-\d{2}-\d{2}),?\s+(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s+(.*?):\s(.+)$",
        re.DOTALL,
    ),
]

# System messages (no sender)
_SYSTEM_PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\s*-\s+(.+)$",
    re.DOTALL,
)

# Common date parse formats to try in order
_DATE_FORMATS: list[str] = [
    "%d/%m/%y",
    "%m/%d/%y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y-%m-%d",
]

_TIME_FORMATS: list[str] = [
    "%H:%M",
    "%H:%M:%S",
    "%I:%M %p",
    "%I:%M:%S %p",
    "%I:%M%p",
    "%I:%M:%S%p",
]


def _parse_datetime(date_str: str, time_str: str) -> datetime | None:
    """Try multiple format combos to parse the datetime."""
    time_str = time_str.strip()
    date_str = date_str.strip()

    for dfmt in _DATE_FORMATS:
        for tfmt in _TIME_FORMATS:
            try:
                return datetime.strptime(f"{date_str} {time_str}", f"{dfmt} {tfmt}")
            except ValueError:
                continue
    return None


class WhatsAppParser(BaseChatParser):
    """Parses WhatsApp .txt chat exports."""

    FORMAT_NAME = "whatsapp"

    def parse(self, file_path: str) -> ParsedConversation:
        path = Path(file_path)
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        lines = raw_text.splitlines()

        messages: list[ParsedMessage] = []
        turn_index = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            matched = False

            # Try each message pattern
            for pattern in _WHATSAPP_PATTERNS:
                match = pattern.match(line)
                if match:
                    date_str, time_str, sender, text = match.groups()
                    timestamp = _parse_datetime(date_str, time_str)

                    # Collect continuation lines (lines that don't start a new message)
                    j = i + 1
                    while j < len(lines) and not any(p.match(lines[j]) for p in _WHATSAPP_PATTERNS) and not _SYSTEM_PATTERN.match(lines[j]):
                        text += " " + lines[j]
                        j += 1

                    cleaned = self._clean_text(text)
                    if cleaned and cleaned != "<Media omitted>":
                        messages.append(
                            ParsedMessage(
                                turn_index=turn_index,
                                speaker=sender.strip(),
                                text=cleaned,
                                timestamp=timestamp,
                                is_system_message=False,
                                raw_line=line,
                            )
                        )
                        turn_index += 1

                    i = j
                    matched = True
                    break

            if not matched:
                # Check for system message
                sys_match = _SYSTEM_PATTERN.match(line)
                if sys_match:
                    date_str, time_str, text = sys_match.groups()
                    timestamp = _parse_datetime(date_str, time_str)
                    messages.append(
                        ParsedMessage(
                            turn_index=turn_index,
                            speaker="SYSTEM",
                            text=self._clean_text(text),
                            timestamp=timestamp,
                            is_system_message=True,
                            raw_line=line,
                        )
                    )
                    turn_index += 1
                i += 1

        speakers = self._extract_speakers(messages)

        return ParsedConversation(
            source_format=self.FORMAT_NAME,
            source_file=str(path),
            messages=messages,
            speakers=speakers,
            metadata={"total_lines": len(lines)},
        )
