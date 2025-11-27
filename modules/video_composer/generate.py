"""Video composition module that stitches together generated assets."""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from moviepy.audio.fx.all import audio_loop
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
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


def _apply_transitions(clip: VideoClip, transition_duration: float) -> VideoClip:
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


def _scale_avatar_image(
    avatar_image: Image.Image, base_size: Tuple[int, int]
) -> Image.Image:
    base_width, base_height = base_size
    if not base_width or not base_height:
        return avatar_image

    max_width = base_width * 0.28
    max_height = base_height * 0.8
    width_factor = max_width / avatar_image.width if avatar_image.width else 1
    height_factor = max_height / avatar_image.height if avatar_image.height else 1
    scale_factor = min(width_factor, height_factor, 1)

    new_size = (
        max(1, int(avatar_image.width * scale_factor)),
        max(1, int(avatar_image.height * scale_factor)),
    )
    return avatar_image.resize(new_size, Image.ANTIALIAS)


def _position_avatar_image(
    avatar_size: Tuple[int, int], base_size: Tuple[int, int], side: str
) -> Tuple[int, int]:
    base_width, base_height = base_size
    avatar_width, avatar_height = avatar_size
    margin_x = int(base_width * 0.04)
    margin_y = int(base_height * 0.05)

    x = margin_x if side == "left" else base_width - avatar_width - margin_x
    y = base_height - avatar_height - margin_y
    return x, y


def _render_frame_with_avatar(
    *,
    image_path: Path,
    base_resolution: Tuple[int, int] | None,
    section_name: str,
    section_index: int,
) -> np.ndarray:
    avatar_path = _select_avatar_asset(section_name, section_index)
    side = _avatar_side_for_index(section_index)

    with Image.open(image_path).convert("RGBA") as base_image:
        if base_resolution:
            base_image = base_image.resize(base_resolution, Image.ANTIALIAS)

        with Image.open(avatar_path).convert("RGBA") as avatar_image:
            if side == "left":
                avatar_image = ImageOps.mirror(avatar_image)

            avatar_image = _scale_avatar_image(avatar_image, base_image.size)
            position = _position_avatar_image(
                avatar_image.size, base_image.size, side
            )

            base_image.paste(avatar_image, position, avatar_image)

        return np.array(base_image.convert("RGB"))


def _build_avatar_overlay(
    *,
    section_name: str,
    section_index: int,
    duration: float,
    base_size: Tuple[int, int],
) -> Optional[ImageClip]:
    base_width, base_height = int(base_size[0] or 0), int(base_size[1] or 0)
    if not base_width or not base_height:
        return None

    avatar_path = _select_avatar_asset(section_name, section_index)
    side = _avatar_side_for_index(section_index)

    with Image.open(avatar_path).convert("RGBA") as avatar_image:
        if side == "left":
            avatar_image = ImageOps.mirror(avatar_image)

        avatar_image = _scale_avatar_image(avatar_image, base_size)
        position = _position_avatar_image(avatar_image.size, base_size, side)
        avatar_array = np.array(avatar_image)

    return (
        ImageClip(avatar_array)
        .set_duration(duration)
        .set_position(position)
    )


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


def _determine_base_resolution(
    *,
    resolution: Tuple[int, int] | None,
    short_video_path: Path | None,
    image_paths: Sequence[Path],
) -> Tuple[int, int] | None:
    if resolution:
        return resolution

    if short_video_path:
        with VideoFileClip(str(short_video_path)) as clip:
            if clip.w and clip.h:
                return int(clip.w), int(clip.h)

    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
            if width and height:
                return int(width), int(height)

    return None


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
    short_video_index: int = 1,
) -> Path:
    """Compose the final video by pairing audio clips with generated images.

    Static visuals are precomposited with the avatar using Pillow before being
    passed to MoviePy to avoid frame-by-frame compositing and resizing. The
    composer adds fade-in/fade-out transitions so each scene change feels smooth
    and highlights important beats.

    If a ``resolution`` is not provided, the composer will infer a base resolution
    from the short video (when available) or the first generated image so every
    visual clip is scaled consistently.
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

    base_resolution = _determine_base_resolution(
        resolution=resolution,
        short_video_path=resolved_short_video,
        image_paths=resolved_image_paths,
    )

    clip_pairs: List[Tuple[VideoClip, AudioFileClip]] = []
    short_video_clip: VideoFileClip | None = None
    final_clip: VideoClip | None = None
    bg_music_base: AudioFileClip | None = None
    bg_music: AudioFileClip | None = None
    composite_audio: CompositeAudioClip | None = None

    try:
        if resolved_short_video is not None:
            short_video_clip = VideoFileClip(str(resolved_short_video)).without_audio()

        for index, (audio_path, image_path) in enumerate(
            _pair_media(resolved_audio_paths, resolved_image_paths)
        ):
            audio_clip = AudioFileClip(str(audio_path))
            section_name = _extract_section_name(audio_path)
            use_short_video = (
                short_video_clip is not None and index == short_video_index
            )
            if use_short_video:
                visual_clip: VideoClip = short_video_clip
                target_duration = audio_clip.duration or visual_clip.duration
                if target_duration:
                    visual_clip = visual_clip.fx(vfx.loop, duration=target_duration)
                    visual_clip = visual_clip.subclip(0, target_duration)
                if base_resolution:
                    visual_clip = visual_clip.resize(newsize=base_resolution)

                avatar_clip = _build_avatar_overlay(
                    section_name=section_name,
                    section_index=index,
                    duration=audio_clip.duration or visual_clip.duration or 0,
                    base_size=visual_clip.size,
                )
                if avatar_clip is not None:
                    visual_clip = CompositeVideoClip(
                        [visual_clip, avatar_clip], size=visual_clip.size
                    )
            else:
                composited_frame = _render_frame_with_avatar(
                    image_path=image_path,
                    base_resolution=base_resolution,
                    section_name=section_name,
                    section_index=index,
                )
                visual_clip = ImageClip(composited_frame).set_duration(
                    audio_clip.duration
                )

            visual_clip = visual_clip.set_audio(audio_clip)

            effective_transition = (
                min(transition_duration, audio_clip.duration / 2)
                if audio_clip.duration and transition_duration
                else 0
            )

            visual_clip = _apply_transitions(visual_clip, effective_transition)
            clip_pairs.append((visual_clip, audio_clip))

        if not clip_pairs:
            raise ValueError("Unable to create any video segments")

        final_clip = concatenate_videoclips(
            [clip for clip, _ in clip_pairs], method="compose"
        )

        bg_music_base = AudioFileClip(BG_MUSIC)
        target_duration = final_clip.duration or bg_music_base.duration or 0
        bg_music = bg_music_base.fx(audio_loop, duration=target_duration).volumex(0.5)

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
    finally:
        for clip, audio_clip in clip_pairs:
            try:
                clip.close()
            except Exception:
                pass
            try:
                audio_clip.close()
            except Exception:
                pass
        if final_clip is not None:
            try:
                final_clip.close()
            except Exception:
                pass
        if composite_audio is not None:
            try:
                composite_audio.close()
            except Exception:
                pass
        if bg_music is not None:
            try:
                bg_music.close()
            except Exception:
                pass
        if bg_music_base is not None:
            try:
                bg_music_base.close()
            except Exception:
                pass
        if short_video_clip is not None:
            try:
                short_video_clip.close()
            except Exception:
                pass

    return output_path


__all__ = ["compose_video"]
