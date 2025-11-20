"""Short video generator using Bytedance's seedance model."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import replicate

MODEL_NAME = "bytedance/seedance-1-pro-fast"
DEFAULT_DURATION = 5
DEFAULT_FPS = 24
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_RESOLUTION = "720p"
DEFAULT_CAMERA_FIXED = False
OUTPUT_FILENAME = "short-video.mp4"


def _slugify(value: str) -> str:
    cleaned = value.strip().replace(" ", "-")
    sanitized = "".join(char for char in cleaned if char.isalnum() or char == "-")
    collapsed = "-".join(filter(None, sanitized.split("-")))
    return collapsed or "video"


def _load_media_plan(media_plan_path: Path) -> dict:
    if not media_plan_path.exists():
        raise FileNotFoundError(f"Media plan not found: {media_plan_path}")

    payload = json.loads(media_plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Media plan must be a JSON object")

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Media plan entries must be a non-empty list")

    return payload


def _prepare_output_dir(video_title: str, video_id: str) -> Path:
    safe_title = _slugify(video_title)
    output_dir = Path("channel") / f"{safe_title}-{video_id}" / "shorts"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _select_prompt(entries: list[dict]) -> str:
    if len(entries) < 2:
        raise ValueError(
            "Media plan must contain at least two entries to pick the second prompt"
        )

    prompt = str(entries[1].get("image_prompt", "")).strip()
    if not prompt:
        raise ValueError("Second media plan entry is missing an image_prompt")
    return prompt


def _collect_first_url(output_obj: Any) -> str:
    if hasattr(output_obj, "url"):
        return str(output_obj.url)

    if isinstance(output_obj, (str, Path)):
        return output_obj

    if hasattr(output_obj, "read"):
        raise ValueError(
            "Video output is a file-like object; expected URL or string path"
        )

    if isinstance(output_obj, Iterable):
        for item in output_obj:
            if isinstance(item, str):
                return item
            if hasattr(item, "url"):
                return str(item.url())

    raise ValueError("Video generation did not return a usable URL")


def _download_video(url: str | Path, output_path: Path) -> Path:
    url_str = str(url)
    if Path(url_str).exists():
        output_path.write_bytes(Path(url_str).read_bytes())
        return output_path

    with urllib.request.urlopen(url_str) as response:
        output_path.write_bytes(response.read())
    return output_path


def _persist_video(output_obj: Any, output_path: Path) -> Path:
    url = _collect_first_url(output_obj)
    return _download_video(url, output_path)


def _run_video_model(
    *,
    prompt: str,
    duration: int,
    fps: int,
    aspect_ratio: str,
    resolution: str,
    camera_fixed: bool,
):
    return replicate.run(
        MODEL_NAME,
        input={
            "fps": fps,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "camera_fixed": camera_fixed,
        },
    )


def generate_short_video(
    media_plan_path: Path | str,
    *,
    duration: int = DEFAULT_DURATION,
    fps: int = DEFAULT_FPS,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    resolution: str = DEFAULT_RESOLUTION,
    camera_fixed: bool = DEFAULT_CAMERA_FIXED,
) -> Path:
    """Generate a single short video using the second media plan entry."""

    path = Path(media_plan_path)
    payload = _load_media_plan(path)

    video_title = str(payload.get("video_title", "video"))
    video_id = str(payload.get("video_id", ""))
    if not video_id:
        raise ValueError("Media plan missing 'video_id'")

    prompt = _select_prompt(payload.get("entries", []))
    output_dir = _prepare_output_dir(video_title, video_id)
    output_path = output_dir / OUTPUT_FILENAME

    response = _run_video_model(
        prompt=prompt,
        duration=duration,
        fps=fps,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        camera_fixed=camera_fixed,
    )

    return _persist_video(response, output_path)


__all__ = ["generate_short_video"]
