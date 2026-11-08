"""Chat parsers package — unified ingestion for multiple chat formats."""
from src.chat_parsers.base import ParsedMessage, ParsedConversation
from src.chat_parsers.whatsapp import WhatsAppParser
from src.chat_parsers.slack import SlackParser
from src.chat_parsers.discord import DiscordParser
from src.chat_parsers.telegram import TelegramParser

PARSERS = {
    "whatsapp": WhatsAppParser,
    "slack": SlackParser,
    "discord": DiscordParser,
    "telegram": TelegramParser,
}


def detect_and_parse(file_path: str, format_hint: str | None = None) -> ParsedConversation:
    """Auto-detect chat format and parse, or use the provided hint.

    Args:
        file_path: Path to the exported chat file.
        format_hint: One of 'whatsapp', 'slack', 'discord', 'telegram'.
                     If None, attempts auto-detection from file extension/content.

    Returns:
        ParsedConversation with normalized messages.

    Raises:
        ValueError: If format cannot be determined.
    """
    from pathlib import Path

    path = Path(file_path)

    if format_hint:
        parser_cls = PARSERS.get(format_hint.lower())
        if not parser_cls:
            raise ValueError(f"Unknown format: {format_hint}. Use one of {list(PARSERS.keys())}")
        return parser_cls().parse(file_path)

    # Auto-detect based on extension and content
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return WhatsAppParser().parse(file_path)
    elif suffix == ".json":
        content = path.read_text(encoding="utf-8", errors="replace")
        # Slack: look for user_profile, client_msg_id, or subtype fields
        if '"client_msg_id"' in content or '"subtype"' in content or '"user_profile"' in content:
            return SlackParser().parse(file_path)
        # Telegram: "type": "message" + "from" fields
        elif '"type": "message"' in content and '"from"' in content:
            return TelegramParser().parse(file_path)
        # Discord: "author" object
        elif '"author"' in content:
            return DiscordParser().parse(file_path)
        # Fallback: try Slack (bare list of messages with "ts")
        elif '"ts"' in content and '"text"' in content:
            return SlackParser().parse(file_path)
        else:
            return DiscordParser().parse(file_path)
    elif suffix == ".csv":
        return DiscordParser().parse(file_path)
    else:
        raise ValueError(
            f"Cannot auto-detect format for '{path.name}'. "
            f"Provide format_hint: {list(PARSERS.keys())}"
        )


__all__ = [
    "ParsedMessage",
    "ParsedConversation",
    "WhatsAppParser",
    "SlackParser",
    "DiscordParser",
    "TelegramParser",
    "detect_and_parse",
    "PARSERS",
]
