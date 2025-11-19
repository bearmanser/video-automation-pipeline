"""Video composition module that stitches together generated assets."""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from moviepy.audio.fx.all import audio_loop
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
    vfx,
)
from PIL import Image, ImageOps

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS  # type: ignore[attr-defined, assignment]

DEFAULT_VIDEO_FILENAME = "final_video.mp4"
DEFAULT_FPS = 30
DEFAULT_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_TRANSITION_DURATION = 0.6
DEFAULT_ZOOM_FACTOR = 1
AVATAR_DIR = Path("assets/avatar")
BG_MUSIC = "assets/music/bg.mp3"
CASUAL_AVATAR_SEQUENCE = [
    AVATAR_DIR / "casual_1.png",
    AVATAR_DIR / "casual_2.png",
    AVATAR_DIR / "casual_3.png",
]
AVATAR_PRIORITY = {
    "hook": [AVATAR_DIR / "pointing_1.png"],
    "intro": [AVATAR_DIR / "casual_1.png", AVATAR_DIR / "waving_1.png"],
    "outro": [AVATAR_DIR / "waving_1.png", AVATAR_DIR / "casual_2.png"],
}


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


def _extract_section_name(audio_path: Path) -> str:
    return audio_path.stem.lower()


def _avatar_side_for_index(index: int) -> str:
    return "left" if index % 2 == 0 else "right"


def _resolve_avatar_path(preferences: Sequence[Path]) -> Path:
    for candidate in preferences:
        if candidate.exists():
            return candidate

    available = sorted(AVATAR_DIR.glob("*.png"))
    if available:
        return available[0]

    raise FileNotFoundError("No avatar images found in assets/avatar")


def _select_avatar_asset(section_name: str, section_index: int) -> Path:
    section_key = next(
        (key for key in AVATAR_PRIORITY if key in section_name.lower()), ""
    )

    if section_key:
        preferred = AVATAR_PRIORITY[section_key]
    else:
        preferred = [
            CASUAL_AVATAR_SEQUENCE[section_index % len(CASUAL_AVATAR_SEQUENCE)]
        ]

    preferred += CASUAL_AVATAR_SEQUENCE
    return _resolve_avatar_path(preferred)


def _scale_avatar(avatar_clip: ImageClip, base_size: Tuple[int, int]) -> ImageClip:
    base_width, base_height = base_size
    if not base_width or not base_height:
        return avatar_clip

    max_width = base_width * 0.28
    max_height = base_height * 0.8
    width_factor = max_width / avatar_clip.w if avatar_clip.w else 1
    height_factor = max_height / avatar_clip.h if avatar_clip.h else 1
    scale_factor = min(width_factor, height_factor, 1)

    return avatar_clip.resize(scale_factor)


def _position_avatar(
    avatar_clip: ImageClip, base_size: Tuple[int, int], side: str
) -> Tuple[float, float]:
    base_width, base_height = base_size
    margin_x = base_width * 0.04
    margin_y = base_height * 0.05

    x = margin_x if side == "left" else base_width - avatar_clip.w - margin_x
    y = base_height - avatar_clip.h - margin_y
    return x, y


def _load_avatar_image(path: Path, mirror: bool) -> np.ndarray:
    with Image.open(path).convert("RGBA") as img:
        if mirror:
            img = ImageOps.mirror(img)
        return np.array(img)


def _build_avatar_clip(
    *,
    section_name: str,
    section_index: int,
    duration: float,
    base_size: Tuple[int, int],
) -> ImageClip:
    avatar_path = _select_avatar_asset(section_name, section_index)
    side = _avatar_side_for_index(section_index)
    avatar_image = _load_avatar_image(avatar_path, mirror=side == "left")
    avatar_clip = ImageClip(avatar_image).set_duration(duration)

    avatar_clip = _scale_avatar(avatar_clip, base_size)
    position = _position_avatar(avatar_clip, base_size, side)
    return avatar_clip.set_position(position)


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
    short_video_path: Path | str | None = None,
    video_title: str,
    video_id: str,
    fps: int = DEFAULT_FPS,
    codec: str = DEFAULT_CODEC,
    audio_codec: str = DEFAULT_AUDIO_CODEC,
    resolution: Tuple[int, int] | None = None,
    transition_duration: float = DEFAULT_TRANSITION_DURATION,
    zoom_factor: float = DEFAULT_ZOOM_FACTOR,
    short_video_index: int = 1,
) -> Path:
    """Compose the final video by pairing audio clips with generated images.

    The composer adds a subtle Ken Burns-style zoom plus fade-in/fade-out
    transitions so each scene change feels smooth and highlights important beats.
    Control these effects with ``transition_duration`` and ``zoom_factor``.
    """

    resolved_audio_paths = _ensure_paths(audio_paths, "audio")
    resolved_image_paths = _ensure_paths(image_paths, "image")
    resolved_short_video = Path(short_video_path) if short_video_path else None
    if resolved_short_video and not resolved_short_video.exists():
        raise FileNotFoundError(
            f"Missing short video file: {resolved_short_video}"  # pragma: no cover
        )
    output_dir = _prepare_output_dir(video_title, video_id)
    output_path = output_dir / DEFAULT_VIDEO_FILENAME

    clip_pairs: List[Tuple[VideoClip, AudioFileClip]] = []
    try:
        for index, (audio_path, image_path) in enumerate(
            _pair_media(resolved_audio_paths, resolved_image_paths)
        ):
            audio_clip = AudioFileClip(str(audio_path))
            use_short_video = (
                resolved_short_video is not None and index == short_video_index
            )
            if use_short_video:
                visual_clip: VideoClip = VideoFileClip(
                    str(resolved_short_video)
                ).without_audio()
                target_duration = audio_clip.duration or visual_clip.duration
                if target_duration:
                    visual_clip = visual_clip.fx(vfx.loop, duration=target_duration)
                    visual_clip = visual_clip.subclip(0, target_duration)
                if resolution:
                    visual_clip = visual_clip.resize(newsize=resolution)
            else:
                image_clip = ImageClip(str(image_path)).set_duration(
                    audio_clip.duration
                )
                if resolution:
                    image_clip = image_clip.resize(newsize=resolution)
                visual_clip = _apply_subtle_motion(image_clip, zoom_factor)

            visual_clip = visual_clip.set_audio(audio_clip)

            effective_transition = (
                min(transition_duration, audio_clip.duration / 2)
                if audio_clip.duration and transition_duration
                else 0
            )

            section_name = _extract_section_name(audio_path)
            avatar_clip = _build_avatar_clip(
                section_name=section_name,
                section_index=index,
                duration=audio_clip.duration or 0,
                base_size=visual_clip.size,
            )

            composite_clip = CompositeVideoClip(
                [visual_clip, avatar_clip], size=visual_clip.size
            ).set_audio(audio_clip)

            composite_clip = _apply_transitions(composite_clip, effective_transition)
            clip_pairs.append((composite_clip, audio_clip))

        if not clip_pairs:
            raise ValueError("Unable to create any video segments")

        final_clip = concatenate_videoclips(
            [clip for clip, _ in clip_pairs], method="compose"
        )

        bg_music_base = AudioFileClip(BG_MUSIC)
        target_duration = final_clip.duration or bg_music_base.duration or 0
        bg_music = bg_music_base.fx(audio_loop, duration=target_duration).volumex(0.12)

        narration_audio = final_clip.audio
        composite_audio = (
            CompositeAudioClip([narration_audio, bg_music])
            if narration_audio
            else bg_music
        )

        final_clip = final_clip.set_audio(composite_audio)

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
