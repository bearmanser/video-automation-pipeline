"""Voice generation module using Replicate's minimax speech model."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import replicate

MODEL_NAME = "minimax/speech-02-turbo"
DEFAULT_VOICE_ID = "Deep_Voice_Man"
DEFAULT_AUDIO_FORMAT = "mp3"


def _slugify(value: str) -> str:
    sanitized = value.strip().replace(" ", "-")
    cleaned = "".join(char for char in sanitized if char.isalnum() or char == "-")
    collapsed = "-".join(filter(None, cleaned.split("-")))
    return collapsed or "video"


def _generate_video_id() -> str:
    return uuid.uuid4().hex[:8]


def _prepare_output_path(video_title: str, video_id: str, audio_format: str) -> Path:
    safe_title = _slugify(video_title)
    output_dir = Path("channel") / f"{safe_title}-{video_id}" / "audios"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"voiceover.{audio_format}"


def generate_voiceover(
    *,
    script: str,
    video_title: str,
    video_id: Optional[str] = None,
    voice_id: str = DEFAULT_VOICE_ID,
    pitch: int = 0,
    speed: float = 1,
    volume: float = 1,
    bitrate: int = 128000,
    channel: str = "mono",
    emotion: str = "fluent",
    sample_rate: int = 32000,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    language_boost: str = "English",
    subtitle_enable: bool = False,
    english_normalization: bool = True,
) -> Path:
    """Generate a voiceover for the provided script and save it to disk."""

    resolved_video_id = video_id or _generate_video_id()
    output_path = _prepare_output_path(video_title, resolved_video_id, audio_format)

    response = replicate.run(
        MODEL_NAME,
        input={
            "text": script,
            "pitch": pitch,
            "speed": speed,
            "volume": volume,
            "bitrate": bitrate,
            "channel": channel,
            "emotion": emotion,
            "voice_id": voice_id,
            "sample_rate": sample_rate,
            "audio_format": audio_format,
            "language_boost": language_boost,
            "subtitle_enable": subtitle_enable,
            "english_normalization": english_normalization,
        },
    )

    with open(output_path, "wb") as file:
        file.write(response.read())

    return output_path


__all__ = [
    "generate_voiceover",
]
