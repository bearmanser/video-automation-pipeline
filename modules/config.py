"""Channel configuration loader."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

DEFAULT_IMAGE_STYLE_GUIDANCE = (
    "unified modern financial visual style, consistent color palette of cool blues "
    "and soft neutrals, clean professional composition, subtle gradients, sharp "
    "clarity, cohesive lighting, refined minimal aesthetic, polished and "
    "trustworthy tone"
)
DEFAULT_SHORT_VIDEO_STYLE_GUIDANCE = (
    "cinematic storytelling, smooth camera presence, steady framing, confident yet "
    "friendly tone, balanced lighting, cohesive color palette, and clean modern "
    "motion pacing"
)
DEFAULT_VOICE_ID = "Wise_Woman"
DEFAULT_VOICE_EMOTION = "happy"
DEFAULT_BG_MUSIC = "assets/music/bg.mp3"

CONFIG_PATH = Path("channels.json")


@dataclass
class ChannelConfig:
    """Resolved configuration values for a channel."""

    name: str
    channel_description: str | None = None
    image_style_guidance: str = DEFAULT_IMAGE_STYLE_GUIDANCE
    short_video_style_guidance: str = DEFAULT_SHORT_VIDEO_STYLE_GUIDANCE
    avatar_path: str | None = None
    avatar_enabled: bool = False
    voice_id: str = DEFAULT_VOICE_ID
    voice_emotion: str = DEFAULT_VOICE_EMOTION
    bg_music: str = DEFAULT_BG_MUSIC
    token_path: str | None = None
    still_image_path: str | None = None

    @property
    def resolved_token_path(self) -> Path:
        path = Path(self.token_path) if self.token_path else Path("channel") / self.name / "token.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resolved_still_image_path(self) -> Path | None:
        return Path(self.still_image_path) if self.still_image_path else None


class ChannelConfigError(Exception):
    """Custom exception for configuration issues."""


def _coerce_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_channel(entry: dict) -> ChannelConfig:
    name = _coerce_string(entry.get("name"))
    if not name:
        raise ChannelConfigError("Channel entry is missing a name")

    avatar_path = _coerce_string(entry.get("avatar_path") or entry.get("avatar"))
    avatar_enabled_setting = entry.get("avatar_enabled")
    avatar_enabled = bool(avatar_enabled_setting) if avatar_enabled_setting is not None else bool(avatar_path)

    return ChannelConfig(
        name=name,
        channel_description=_coerce_string(entry.get("channel_description")),
        image_style_guidance=_coerce_string(entry.get("image_style_guidance"))
        or DEFAULT_IMAGE_STYLE_GUIDANCE,
        short_video_style_guidance=_coerce_string(entry.get("short_video_style_guidance"))
        or DEFAULT_SHORT_VIDEO_STYLE_GUIDANCE,
        avatar_path=avatar_path,
        avatar_enabled=avatar_enabled,
        voice_id=_coerce_string(entry.get("voice_id")) or DEFAULT_VOICE_ID,
        voice_emotion=_coerce_string(entry.get("voice_emotion"))
        or DEFAULT_VOICE_EMOTION,
        bg_music=_coerce_string(entry.get("bg_music")) or DEFAULT_BG_MUSIC,
        token_path=_coerce_string(entry.get("token_path")),
        still_image_path=_coerce_string(entry.get("still_image_path")),
    )


def load_channels() -> List[ChannelConfig]:
    if not CONFIG_PATH.exists():
        raise ChannelConfigError(f"Missing channel config file: {CONFIG_PATH}")

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    entries = payload.get("channels") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ChannelConfigError("channels.json must include a 'channels' array")

    channels = [_build_channel(entry) for entry in entries if isinstance(entry, dict)]
    if not channels:
        raise ChannelConfigError("No valid channels found in configuration")
    return channels


def get_channel_config(name: str) -> ChannelConfig:
    for channel in load_channels():
        if channel.name == name:
            return channel
    raise ChannelConfigError(f"Channel '{name}' not defined in configuration")


def resolve_channel(payload_channel: Any, provided_channel: str | None = None) -> ChannelConfig:
    """Resolve and validate a channel from payload or provided input.

    The channel name must be present in ``channels.json``; otherwise a
    ``ChannelConfigError`` is raised to halt the pipeline rather than falling
    back to an undefined default.
    """

    channel_name = _coerce_string(payload_channel) or _coerce_string(provided_channel)
    if not channel_name:
        raise ChannelConfigError(
            "Channel not provided. Specify channel_name and add it to channels.json."
        )

    return get_channel_config(channel_name)


__all__ = [
    "ChannelConfig",
    "ChannelConfigError",
    "DEFAULT_IMAGE_STYLE_GUIDANCE",
    "DEFAULT_SHORT_VIDEO_STYLE_GUIDANCE",
    "DEFAULT_VOICE_EMOTION",
    "get_channel_config",
    "load_channels",
    "resolve_channel",
]
