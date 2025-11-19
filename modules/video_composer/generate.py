"""Video composition module that stitches together generated assets."""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips, vfx

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS  # type: ignore[attr-defined, assignment]

DEFAULT_VIDEO_FILENAME = "final_video.mp4"
DEFAULT_FPS = 30
DEFAULT_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_TRANSITION_DURATION = 0.6
DEFAULT_ZOOM_FACTOR = 1.04


def _slugify(value: str) -> str:
    cleaned = value.strip().replace(" ", "-")
    sanitized = "".join(char for char in cleaned if char.isalnum() or char == "-")
    collapsed = "-".join(filter(None, sanitized.split("-")))
    return collapsed or "video"


def _apply_subtle_motion(clip: ImageClip, zoom_factor: float) -> ImageClip:
    if zoom_factor <= 1 or not clip.duration:
        return clip

    duration = clip.duration

    def _resize_factor(t: float) -> float:
        progress = 0.0 if duration <= 0 else min(max(t / duration, 0.0), 1.0)
        return 1 + (zoom_factor - 1) * progress

    return clip.resize(_resize_factor)


def _apply_transitions(clip: ImageClip, transition_duration: float) -> ImageClip:
    if transition_duration <= 0:
        return clip

    clip = clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration)
    return clip


def _ensure_paths(paths: Sequence[Path | str], label: str) -> List[Path]:
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError(f"No {label} paths provided")
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} files: {', '.join(missing)}")
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
    transition_duration: float = DEFAULT_TRANSITION_DURATION,
    zoom_factor: float = DEFAULT_ZOOM_FACTOR,
) -> Path:
    """Compose the final video by pairing audio clips with generated images.

    The composer adds a subtle Ken Burns-style zoom plus fade-in/fade-out
    transitions so each scene change feels smooth and highlights important beats.
    Control these effects with ``transition_duration`` and ``zoom_factor``.
    """

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

            effective_transition = (
                min(transition_duration, audio_clip.duration / 2)
                if audio_clip.duration and transition_duration
                else 0
            )
            motion_clip = _apply_subtle_motion(image_clip, zoom_factor)
            motion_clip = _apply_transitions(motion_clip, effective_transition)
            clip_pairs.append((motion_clip, audio_clip))

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
