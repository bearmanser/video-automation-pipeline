"""Image generation module that consumes a media plan and saves rendered images."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, List

import replicate
import urllib.request

MODEL_NAME = "google/imagen-4-fast"
DEFAULT_ASPECT_RATIO = "16:9"
OUTPUT_FORMAT = "jpg"
SAFETY_FILTER_LEVEL = "block_only_high"
STYLE_GUIDANCE = (
    "unified modern financial visual style, consistent color palette of cool blues "
    "and soft neutrals, clean professional composition, subtle gradients, sharp "
    "clarity, cohesive lighting, refined minimal aesthetic, polished and "
    "trustworthy tone"
)


def _slugify(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip())
    collapsed = re.sub(r"-+", "-", sanitized).strip("-")
    return collapsed or "video"


def _is_valid_timestamp(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _load_media_plan(media_plan_path: Path) -> dict:
    if not media_plan_path.exists():
        raise FileNotFoundError(f"Media plan not found: {media_plan_path}")

    payload = json.loads(media_plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Media plan must be a JSON object")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Media plan entries must be a list")

    return payload


def _prepare_output_dir(video_title: str, video_id: str) -> Path:
    safe_title = _slugify(video_title)
    output_dir = Path("channel") / f"{safe_title}-{video_id}" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _download_image(url: str, output_path: Path) -> Path:
    with urllib.request.urlopen(url) as response:
        output_path.write_bytes(response.read())
    return output_path


def _run_image_model(prompt: str):
    styled_prompt = f"{prompt}\n\n{STYLE_GUIDANCE}" if prompt else STYLE_GUIDANCE
    return replicate.run(
        MODEL_NAME,
        input={
            "prompt": styled_prompt,
            "aspect_ratio": DEFAULT_ASPECT_RATIO,
            "output_format": OUTPUT_FORMAT,
            "safety_filter_level": SAFETY_FILTER_LEVEL,
        },
    )


def _persist_generated_image(output_obj: Any, output_path: Path) -> Path:
    """Write replicate output to disk, supporting both file objects and URLs."""

    if hasattr(output_obj, "read"):
        output_path.write_bytes(output_obj.read())
        return output_path

    if hasattr(output_obj, "url"):
        return _download_image(str(output_obj.url()), output_path)

    if isinstance(output_obj, str):
        return _download_image(output_obj, output_path)

    if isinstance(output_obj, Iterable):
        for item in output_obj:
            if isinstance(item, str):
                return _download_image(item, output_path)

    raise ValueError("Image generation did not return usable image data")


def _build_filename(index: int, timestamp: float) -> str:
    timestamp_part = f"{timestamp:.2f}".replace(".", "-")
    return f"{index:03d}-t{timestamp_part}.{OUTPUT_FORMAT}"


def generate_images(media_plan_path: Path | str) -> List[Path]:
    """Generate and save images for all media plan entries with valid timestamps."""

    path = Path(media_plan_path)
    payload = _load_media_plan(path)

    video_title = str(payload.get("video_title", "video"))
    video_id = str(payload.get("video_id", ""))
    if not video_id:
        raise ValueError("Media plan missing 'video_id'")

    output_dir = _prepare_output_dir(video_title, video_id)
    entries = payload.get("entries", [])

    image_paths: List[Path] = []
    for idx, entry in enumerate(entries, start=1):
        timestamp = entry.get("timestamp")
        prompt = str(entry.get("image_prompt", "")).strip()

        if not _is_valid_timestamp(timestamp) or not prompt:
            continue

        filename = _build_filename(idx, float(timestamp))
        output_obj = _run_image_model(prompt)
        output_path = _persist_generated_image(output_obj, output_dir / filename)
        image_paths.append(output_path)

    return image_paths


__all__ = ["generate_images"]
