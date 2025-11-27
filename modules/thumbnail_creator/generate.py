"""Thumbnail creation module using Google's Imagen model.

This implementation uses the google/imagen-4-fast model via Replicate and builds
the prompt from the video title plus key visual cues pulled from the media plan,
with dedicated YouTube thumbnail style guidance.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional

import replicate

from modules.config import resolve_channel

MODEL_NAME = "google/imagen-4-fast"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_OUTPUT_FORMAT = "jpg"
DEFAULT_SAFETY_FILTER_LEVEL = "block_only_high"
THUMBNAIL_FILENAME = "thumbnail.jpg"
STYLE_GUIDANCE = (
    "high-impact YouTube thumbnail, cinematic depth, bold focal subject, dramatic "
    "lighting, clear contrast, vibrant yet professional palette, clean negative "
    "space for title placement, modern and trustworthy aesthetic"
)


def _slugify(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip())
    collapsed = re.sub(r"-+", "-", sanitized).strip("-")
    return collapsed or "video"


def _load_media_plan(media_plan_path: Path) -> dict:
    if not media_plan_path.exists():
        raise FileNotFoundError(f"Media plan not found: {media_plan_path}")

    payload = json.loads(media_plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Media plan must be a JSON object")

    return payload


def _prepare_output_dir(video_title: str, video_id: str, channel_name: str) -> Path:
    safe_title = _slugify(video_title)
    output_dir = Path("channel") / channel_name / f"{safe_title}-{video_id}" / "thumbnails"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _build_prompt(video_title: str, entries: list[dict] | None = None) -> str:
    lines: list[str] = [
        "Design an eye-catching YouTube thumbnail that instantly conveys the topic.",
        f"Video title: {video_title}.",
    ]

    cues: list[str] = []
    for entry in entries or []:
        image_prompt = str(entry.get("image_prompt", "")).strip()
        identifier = str(entry.get("identifier", "")).strip()
        if image_prompt:
            cues.append(f"{identifier + ': ' if identifier else ''}{image_prompt}")
        if len(cues) >= 3:
            break

    if cues:
        lines.append("Incorporate these visual cues from the media plan:")
        lines.extend(f"- {cue}" for cue in cues)

    lines.append("Keep clear negative space for title text placement.")
    lines.append(STYLE_GUIDANCE)
    return "\n".join(lines)


def _run_thumbnail_model(prompt: str):
    return replicate.run(
        MODEL_NAME,
        input={
            "prompt": prompt,
            "aspect_ratio": DEFAULT_ASPECT_RATIO,
            "output_format": DEFAULT_OUTPUT_FORMAT,
            "safety_filter_level": DEFAULT_SAFETY_FILTER_LEVEL,
        },
    )


def _collect_first_image(output_obj: Any) -> str:
    if hasattr(output_obj, "url"):
        return str(output_obj.url)

    if isinstance(output_obj, (str, Path)):
        return str(output_obj)

    if hasattr(output_obj, "read"):
        raise ValueError(
            "Thumbnail output is a file-like object; expected URL or string path"
        )

    if isinstance(output_obj, Iterable):
        for item in output_obj:
            if isinstance(item, str):
                return item
            if hasattr(item, "url"):
                return str(item.url)

    raise ValueError("Thumbnail generation did not return a usable URL")


def _persist_thumbnail(output_obj: Any, output_path: Path) -> Path:
    url = _collect_first_image(output_obj)
    url_str = str(url)
    if Path(url_str).exists():
        output_path.write_bytes(Path(url_str).read_bytes())
        return output_path

    with urllib.request.urlopen(url_str) as response:
        output_path.write_bytes(response.read())
    return output_path


def generate_thumbnail(
    media_plan_path: Path | str, *, channel_name: Optional[str] = None
) -> Path:
    """Generate a single thumbnail image based on the video title."""

    path = Path(media_plan_path)
    payload = _load_media_plan(path)

    video_title = str(payload.get("video_title", "video")).strip()
    video_id = str(payload.get("video_id", "")).strip()
    channel = resolve_channel(payload.get("channel_name"), channel_name).name
    if not video_id:
        raise ValueError("Media plan missing 'video_id'")

    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []

    prompt = _build_prompt(video_title or "YouTube video", entries)
    output_dir = _prepare_output_dir(video_title or "video", video_id, channel)
    output_path = output_dir / THUMBNAIL_FILENAME

    response = _run_thumbnail_model(prompt)
    return _persist_thumbnail(response, output_path)


__all__ = ["generate_thumbnail"]
