"""
Tests for chat format parsers.

Uses real-format sample data to verify parsing correctness.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_parsers.base import ParsedConversation, ParsedMessage
from src.chat_parsers.whatsapp import WhatsAppParser
from src.chat_parsers.slack import SlackParser
from src.chat_parsers.discord import DiscordParser
from src.chat_parsers.telegram import TelegramParser
from src.chat_parsers import detect_and_parse


# ─── WhatsApp Tests ──────────────────────────────────────────────────

WHATSAPP_SAMPLE = """12/09/26, 14:30 - Alice: Hey everyone!
12/09/26, 14:31 - Bob: Hi Alice, how's it going?
12/09/26, 14:32 - Alice: Great! Just finished the project.
12/09/26, 14:33 - Bob: Oh wow, that's amazing! Congrats!
12/09/26, 14:34 - Charlie joined using this group's invite link
12/09/26, 14:35 - Charlie: Thanks for adding me!
"""


class TestWhatsAppParser:
    def test_parse_basic_messages(self, tmp_path: Path):
        file = tmp_path / "chat.txt"
        file.write_text(WHATSAPP_SAMPLE, encoding="utf-8")

        parser = WhatsAppParser()
        result = parser.parse(str(file))

        assert isinstance(result, ParsedConversation)
        assert result.source_format == "whatsapp"
        assert result.num_turns >= 4

    def test_extracts_speakers(self, tmp_path: Path):
        file = tmp_path / "chat.txt"
        file.write_text(WHATSAPP_SAMPLE, encoding="utf-8")

        result = WhatsAppParser().parse(str(file))
        user_messages = result.user_messages()

        speakers = {m.speaker for m in user_messages}
        assert "Alice" in speakers
        assert "Bob" in speakers

    def test_filters_system_messages(self, tmp_path: Path):
        file = tmp_path / "chat.txt"
        file.write_text(WHATSAPP_SAMPLE, encoding="utf-8")

        result = WhatsAppParser().parse(str(file))
        user_msgs = result.user_messages()

        # System messages like "joined using invite link" should be filtered
        for msg in user_msgs:
            assert not msg.is_system_message


# ─── Slack Tests ─────────────────────────────────────────────────────

SLACK_SAMPLE = json.dumps([
    {
        "user": "U123",
        "user_profile": {"real_name": "Alice"},
        "text": "Hello team!",
        "ts": "1695312000.000000",
    },
    {
        "user": "U456",
        "user_profile": {"real_name": "Bob"},
        "text": "Hey Alice!",
        "ts": "1695312060.000000",
    },
    {
        "subtype": "channel_join",
        "user": "U789",
        "text": "Charlie has joined the channel",
        "ts": "1695312120.000000",
    },
])


class TestSlackParser:
    def test_parse_json_messages(self, tmp_path: Path):
        file = tmp_path / "general.json"
        file.write_text(SLACK_SAMPLE, encoding="utf-8")

        result = SlackParser().parse(str(file))

        assert isinstance(result, ParsedConversation)
        assert result.source_format == "slack"

    def test_speaker_extraction(self, tmp_path: Path):
        file = tmp_path / "general.json"
        file.write_text(SLACK_SAMPLE, encoding="utf-8")

        result = SlackParser().parse(str(file))
        assert "Alice" in result.speakers

    def test_system_message_detection(self, tmp_path: Path):
        file = tmp_path / "general.json"
        file.write_text(SLACK_SAMPLE, encoding="utf-8")

        result = SlackParser().parse(str(file))
        system_msgs = [m for m in result.messages if m.is_system_message]
        assert len(system_msgs) >= 1


# ─── Discord Tests ───────────────────────────────────────────────────

DISCORD_SAMPLE = json.dumps({
    "channel": {"name": "general"},
    "messages": [
        {
            "author": {"name": "Alice", "isBot": False},
            "content": "Hey everyone!",
            "timestamp": "2026-09-12T14:30:00+00:00",
            "type": "Default",
        },
        {
            "author": {"name": "Bot", "isBot": True},
            "content": "Welcome!",
            "timestamp": "2026-09-12T14:30:01+00:00",
            "type": "Default",
        },
    ],
})


class TestDiscordParser:
    def test_parse_json_format(self, tmp_path: Path):
        file = tmp_path / "discord.json"
        file.write_text(DISCORD_SAMPLE, encoding="utf-8")

        result = DiscordParser().parse(str(file))
        assert result.source_format == "discord"
        assert result.num_turns >= 1


# ─── Telegram Tests ──────────────────────────────────────────────────

TELEGRAM_SAMPLE = json.dumps({
    "name": "Test Group",
    "type": "personal_chat",
    "messages": [
        {
            "type": "message",
            "date": "2026-09-12T14:30:00",
            "from": "Alice",
            "text": "Hello!",
        },
        {
            "type": "message",
            "date": "2026-09-12T14:31:00",
            "from": "Bob",
            "text": [
                "Check this ",
                {"type": "bold", "text": "important"},
                " update!",
            ],
        },
        {
            "type": "service",
            "date": "2026-09-12T14:32:00",
            "actor": "Charlie",
            "action": "joined group",
        },
    ],
})


class TestTelegramParser:
    def test_parse_json_format(self, tmp_path: Path):
        file = tmp_path / "telegram.json"
        file.write_text(TELEGRAM_SAMPLE, encoding="utf-8")

        result = TelegramParser().parse(str(file))
        assert result.source_format == "telegram"

    def test_flattens_rich_text(self, tmp_path: Path):
        file = tmp_path / "telegram.json"
        file.write_text(TELEGRAM_SAMPLE, encoding="utf-8")

        result = TelegramParser().parse(str(file))
        bob_msgs = result.speaker_messages("Bob")
        assert any("important" in m.text for m in bob_msgs)


# ─── Auto-detect Tests ──────────────────────────────────────────────

class TestAutoDetect:
    def test_detect_whatsapp_by_extension(self, tmp_path: Path):
        file = tmp_path / "chat.txt"
        file.write_text(WHATSAPP_SAMPLE, encoding="utf-8")
        result = detect_and_parse(str(file))
        assert result.source_format == "whatsapp"

    def test_detect_slack_by_content(self, tmp_path: Path):
        file = tmp_path / "export.json"
        file.write_text(SLACK_SAMPLE, encoding="utf-8")
        result = detect_and_parse(str(file))
        assert result.source_format == "slack"
