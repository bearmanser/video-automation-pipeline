"""Metadata generation module for YouTube uploads."""

from __future__ import annotations

import json
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import replicate

from modules.config import resolve_channel

MODEL_NAME = "openai/gpt-5"
METADATA_FORMAT_VERSION = "YOUTUBE_METADATA_V1"


def _slugify(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip())
    collapsed = re.sub(r"-+", "-", sanitized).strip("-")
    return collapsed or "video"


def _collect_response(chunks: Iterable[str]) -> str:
    if isinstance(chunks, str):
        return chunks
    return "".join(chunk for chunk in chunks if chunk)


def _load_media_plan(media_plan_path: Path) -> dict:
    if not media_plan_path.exists():
        raise FileNotFoundError(f"Media plan not found: {media_plan_path}")

    payload = json.loads(media_plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Media plan must be a JSON object")

    return payload


def _prepare_output_path(video_title: str, video_id: str, channel_name: str) -> Path:
    safe_title = _slugify(video_title)
    output_dir = Path("channel") / channel_name / f"{safe_title}-{video_id}" / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "metadata.json"


def _build_prompt(video_title: str, entries: list[dict], format_version: str) -> str:
    plan_summary_lines = []
    for entry in entries[:5]:
        identifier = str(entry.get("identifier", "")).strip()
        image_prompt = str(entry.get("image_prompt", "")).strip()
        if identifier and image_prompt:
            plan_summary_lines.append(f"- {identifier}: {image_prompt}")

    plan_summary = "\n".join(plan_summary_lines) or "- No media plan entries provided"

    return (
        "You are crafting metadata for a YouTube video. "
        "Return a compact JSON object with these keys: title, description and tags. "
        "Use the exact provided video title without changing it. "
        "The title should be punchy and under 70 characters. "
        "The description should be 2-3 sentences summarizing the video and inviting engagement. "
        "Provide 8-12 concise tags as a comma-separated string. "
        "Do not include any explanations or markdown—only return JSON.\n\n"
        f"VIDEO_TITLE: {video_title}\n"
        f"FORMAT: {format_version}\n"
        "MEDIA PLAN HIGHLIGHTS:\n"
        f"{plan_summary}\n\n"
        "JSON:"
    )


def generate_metadata(
    *,
    video_title: str,
    media_plan_path: Path | str,
    video_id: Optional[str] = None,
    channel_name: Optional[str] = None,
) -> Path:
    """Generate and save upload-ready metadata for YouTube."""

    path = Path(media_plan_path)
    payload = _load_media_plan(path)

    resolved_video_id = video_id or str(payload.get("video_id", ""))
    channel = resolve_channel(payload.get("channel_name"), channel_name).name
    if not resolved_video_id:
        raise ValueError("Video ID is required to generate metadata")

    entries = payload.get("entries") or []
    if not isinstance(entries, list):
        raise ValueError("Media plan entries must be a list")

    prompt = _build_prompt(video_title, entries, METADATA_FORMAT_VERSION)
    response = replicate.run(MODEL_NAME, input={"prompt": prompt})
    content = _collect_response(response)

    try:
        metadata = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Metadata response was not valid JSON") from exc

    if not isinstance(metadata, dict):
        raise ValueError("Metadata response must be a JSON object")

    metadata["title"] = video_title

    output_path = _prepare_output_path(video_title, resolved_video_id, channel)
    payload = {
        "video_title": video_title,
        "video_id": resolved_video_id,
        "channel_name": channel,
        "format": METADATA_FORMAT_VERSION,
        "metadata": metadata,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


__all__ = ["generate_metadata"]
