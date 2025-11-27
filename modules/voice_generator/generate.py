"""Voice generation module using Replicate's minimax speech model."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

import replicate

from modules.config import resolve_channel

MODEL_NAME = "minimax/speech-02-turbo"
DEFAULT_VOICE_ID = "Wise_Woman"
DEFAULT_AUDIO_FORMAT = "mp3"
DEFAULT_PAUSE_SECONDS = 0.5


def _slugify(value: str) -> str:
    sanitized = value.strip().replace(" ", "-")
    cleaned = "".join(char for char in sanitized if char.isalnum() or char == "-")
    collapsed = "-".join(filter(None, cleaned.split("-")))
    return collapsed or "video"


def _generate_video_id() -> str:
    return uuid.uuid4().hex[:8]


def _prepare_output_dir(video_title: str, video_id: str, channel_name: str) -> Path:
    safe_title = _slugify(video_title)
    output_dir = Path("channel") / channel_name / f"{safe_title}-{video_id}" / "audios"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _extract_section(script: str, start: str, end: Optional[str]) -> str:
    pattern = (
        rf"\[{re.escape(start)}[^\]]*\]\s*(.*?)(?=\n\[{re.escape(end)}[^\]]*\]|\Z)"
        if end
        else rf"\[{re.escape(start)}[^\]]*\]\s*(.*)"
    )
    match = re.search(pattern, script, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing section [{start}] in script")
    return match.group(1).strip()


def _extract_all_scenes(script: str) -> List[str]:
    matches = re.findall(
        r"\[SCENE[^\]]*\]\s*(.*?)(?=\n\[SCENE|\n\[OUTRO|\Z)",
        script,
        flags=re.DOTALL | re.IGNORECASE,
    )
    scenes = [m.strip() for m in matches if m.strip()]
    if not scenes:
        raise ValueError("No SCENE sections found")
    return scenes


def _collect_sections(script: str) -> List[Tuple[str, str]]:
    hook = _extract_section(script, "HOOK", "INTRO")
    intro = _extract_section(script, "INTRO", "SCENE")
    scenes = _extract_all_scenes(script)
    outro = _extract_section(script, "OUTRO", None)

    ordered_sections: List[Tuple[str, str]] = [("hook", hook), ("intro", intro)]
    ordered_sections.extend(
        [(f"scene-{idx + 1}", scene) for idx, scene in enumerate(scenes)]
    )
    ordered_sections.append(("outro", outro))
    return ordered_sections


def _insert_pauses(text: str, pause_seconds: float = DEFAULT_PAUSE_SECONDS) -> str:
    pause_marker = f"<# {pause_seconds:.1f} #>".replace(" ", "")
    punctuated = re.sub(
        r"([.!?])\s+",
        lambda match: f"{match.group(1)} {pause_marker} ",
        text.strip(),
    )
    return f"{pause_marker} {punctuated} {pause_marker}".strip()


def _write_audio_response(
    output_dir: Path, name: str, audio_format: str, response: Any
) -> Path:
    output_path = output_dir / f"{name}.{audio_format}"
    with open(output_path, "wb") as file:
        file.write(response.read())
    return output_path


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
    emotion: str = "happy",
    sample_rate: int = 32000,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    language_boost: str = "English",
    subtitle_enable: bool = False,
    english_normalization: bool = True,
    channel_name: Optional[str] = None,
) -> list[Path]:
    """Generate a voiceover for each section of the provided script.

    Returns a list of file paths in the order: hook, intro, scenes, outro.
    """

    resolved_video_id = video_id or _generate_video_id()
    channel = resolve_channel(None, channel_name).name
    output_dir = _prepare_output_dir(video_title, resolved_video_id, channel)
    sections = _collect_sections(script)

    audio_paths: list[Path] = []
    for name, text in sections:
        spoken_text = _insert_pauses(text)
        response = replicate.run(
            MODEL_NAME,
            input={
                "text": spoken_text,
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

        audio_paths.append(
            _write_audio_response(output_dir, name, audio_format, response)
        )

    return audio_paths


__all__ = [
    "generate_voiceover",
]
