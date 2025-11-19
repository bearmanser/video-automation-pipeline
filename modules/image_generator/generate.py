"""Image generation module that consumes a media plan and saves rendered images."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, List

import replicate
import urllib.request

MODEL_NAME = "black-forest-labs/flux-1.1-pro"


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


def _run_image_model(prompt: str) -> str:
    response = replicate.run(MODEL_NAME, input={"prompt": prompt})
    if isinstance(response, str):
        return response
    if isinstance(response, Iterable):
        items = list(response)
        if items:
            return str(items[0])
    raise ValueError("Image generation did not return a usable URL")


def _build_filename(index: int, timestamp: float) -> str:
    timestamp_part = f"{timestamp:.2f}".replace(".", "-")
    return f"{index:03d}-t{timestamp_part}.png"


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
        image_url = _run_image_model(prompt)
        output_path = _download_image(image_url, output_dir / filename)
        image_paths.append(output_path)

    return image_paths


__all__ = ["generate_images"]
