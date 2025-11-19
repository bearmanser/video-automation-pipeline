"""Video composition module that stitches together generated assets."""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips

DEFAULT_VIDEO_FILENAME = "final_video.mp4"
DEFAULT_FPS = 30
DEFAULT_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"


def _slugify(value: str) -> str:
    cleaned = value.strip().replace(" ", "-")
    sanitized = "".join(char for char in cleaned if char.isalnum() or char == "-")
    collapsed = "-".join(filter(None, sanitized.split("-")))
    return collapsed or "video"


def _ensure_paths(paths: Sequence[Path | str], label: str) -> List[Path]:
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError(f"No {label} paths provided")
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {label} files: {', '.join(missing)}"
        )
    return resolved


def _prepare_output_dir(video_title: str, video_id: str) -> Path:
    if not video_id:
        raise ValueError("video_id is required to compose the video")
    safe_title = _slugify(video_title)
    output_dir = Path("channel") / f"{safe_title}-{video_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _pair_media(
    audio_paths: Sequence[Path], image_paths: Sequence[Path]
) -> Iterable[Tuple[Path, Path]]:
    last_image = image_paths[-1]
    for audio_path, image_path in zip_longest(audio_paths, image_paths):
        if audio_path is None:
            break
        yield audio_path, image_path or last_image


def compose_video(
    *,
    audio_paths: Sequence[Path | str],
    image_paths: Sequence[Path | str],
    video_title: str,
    video_id: str,
    fps: int = DEFAULT_FPS,
    codec: str = DEFAULT_CODEC,
    audio_codec: str = DEFAULT_AUDIO_CODEC,
    resolution: Tuple[int, int] | None = None,
) -> Path:
    """Compose the final video by pairing audio clips with generated images."""

    resolved_audio_paths = _ensure_paths(audio_paths, "audio")
    resolved_image_paths = _ensure_paths(image_paths, "image")
    output_dir = _prepare_output_dir(video_title, video_id)
    output_path = output_dir / DEFAULT_VIDEO_FILENAME

    clip_pairs: List[Tuple[ImageClip, AudioFileClip]] = []
    try:
        for audio_path, image_path in _pair_media(
            resolved_audio_paths, resolved_image_paths
        ):
            audio_clip = AudioFileClip(str(audio_path))
            image_clip = ImageClip(str(image_path)).set_duration(audio_clip.duration)
            if resolution:
                image_clip = image_clip.resize(newsize=resolution)
            image_clip = image_clip.set_audio(audio_clip)
            clip_pairs.append((image_clip, audio_clip))

        if not clip_pairs:
            raise ValueError("Unable to create any video segments")

        final_clip = concatenate_videoclips(
            [clip for clip, _ in clip_pairs], method="compose"
        )
        final_clip.write_videofile(
            str(output_path),
            fps=fps,
            codec=codec,
            audio_codec=audio_codec,
        )
        final_clip.close()
    finally:
        for clip, audio_clip in clip_pairs:
            clip.close()
            audio_clip.close()

    return output_path


__all__ = ["compose_video"]
