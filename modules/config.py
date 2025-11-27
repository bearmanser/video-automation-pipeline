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
DEFAULT_BG_MUSIC = "assets/music/bg.mp3"

CONFIG_PATH = Path("channels.json")


@dataclass
class ChannelConfig:
    """Resolved configuration values for a channel."""

    name: str
    image_style_guidance: str = DEFAULT_IMAGE_STYLE_GUIDANCE
    short_video_style_guidance: str = DEFAULT_SHORT_VIDEO_STYLE_GUIDANCE
    avatar_path: str | None = None
    avatar_enabled: bool = True
    voice_id: str = DEFAULT_VOICE_ID
    bg_music: str = DEFAULT_BG_MUSIC
    token_path: str | None = None

    @property
    def resolved_token_path(self) -> Path:
        path = Path(self.token_path) if self.token_path else Path("channel") / self.name / "token.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


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

    avatar_setting = entry.get("avatar", "auto")
    avatar_enabled = avatar_setting is not None
    avatar_path = None if avatar_setting in (None, "auto") else _coerce_string(avatar_setting)

    return ChannelConfig(
        name=name,
        image_style_guidance=_coerce_string(entry.get("image_style_guidance"))
        or DEFAULT_IMAGE_STYLE_GUIDANCE,
        short_video_style_guidance=_coerce_string(entry.get("short_video_style_guidance"))
        or DEFAULT_SHORT_VIDEO_STYLE_GUIDANCE,
        avatar_path=avatar_path,
        avatar_enabled=avatar_enabled,
        voice_id=_coerce_string(entry.get("voice_id")) or DEFAULT_VOICE_ID,
        bg_music=_coerce_string(entry.get("bg_music")) or DEFAULT_BG_MUSIC,
        token_path=_coerce_string(entry.get("token_path")),
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


__all__ = [
    "ChannelConfig",
    "ChannelConfigError",
    "DEFAULT_IMAGE_STYLE_GUIDANCE",
    "DEFAULT_SHORT_VIDEO_STYLE_GUIDANCE",
    "get_channel_config",
    "load_channels",
]
