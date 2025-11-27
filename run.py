"""Pipeline entrypoint that orchestrates the content pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from modules.config import ChannelConfigError, get_channel_config
from modules.image_generator import generate_images
from modules.media_planner.generate import generate_media_plan
from modules.metadata_generator.generate import generate_metadata
from modules.script_generator.generate import generate_and_save_script
from modules.short_video_generator.generate import generate_short_video
from modules.thumbnail_creator.generate import generate_thumbnail
from modules.uploader.generate import upload_video
from modules.video_composer import compose_video
from modules.voice_generator.generate import generate_voiceover

load_dotenv()

PROGRESS_FILE = Path("pipeline_progress.json")


def _load_progress() -> Dict[str, Any]:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_progress(progress: Dict[str, Any]) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _reset_progress() -> None:
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


def _paths_exist(paths: List[str] | None) -> bool:
    return bool(paths) and all(Path(path).exists() for path in paths)


def main() -> None:
    video_title = "What to do with your first paycheck"
    channel_name = "default"

    try:
        channel_config = get_channel_config(channel_name)
    except ChannelConfigError as exc:
        raise SystemExit(str(exc)) from exc

    progress = _load_progress()
    if (
        progress.get("video_title") != video_title
        or progress.get("channel_name") != channel_config.name
    ):
        _reset_progress()
        progress = {}

    progress["video_title"] = video_title
    progress["channel_name"] = channel_config.name
    _save_progress(progress)

    script_path_str = progress.get("script_path")
    video_id = progress.get("video_id")
    script_path = Path(script_path_str) if script_path_str else None
    if not script_path or not script_path.exists() or not video_id:
        script_path, video_id = generate_and_save_script(
            video_title, word_length=1000, channel_name=channel_config.name
        )
        progress.update({"script_path": str(script_path), "video_id": video_id})
        _save_progress(progress)
        print(f"Script saved to {script_path}")
    else:
        print(f"Using existing script at {script_path}")

    script_text = Path(progress["script_path"]).read_text(encoding="utf-8")

    audio_paths = progress.get("audio_paths")
    if not _paths_exist(audio_paths):
        audio_paths = generate_voiceover(
            script=script_text,
            video_title=video_title,
            video_id=video_id,
            voice_id=channel_config.voice_id,
            channel_name=channel_config.name,
        )
        progress["audio_paths"] = [str(path) for path in audio_paths]
        _save_progress(progress)
        for path in audio_paths:
            print(f"Voiceover saved to {path}")
    else:
        audio_paths = [str(path) for path in audio_paths]
        print("Using existing voiceover files")

    media_plan_path_str = progress.get("media_plan_path")
    media_plan_path = Path(media_plan_path_str) if media_plan_path_str else None
    if not media_plan_path or not media_plan_path.exists():
        media_plan_path, _ = generate_media_plan(
            script=script_text,
            audio_paths=audio_paths,
            video_title=video_title,
            video_id=video_id,
            channel_name=channel_config.name,
        )
        progress["media_plan_path"] = str(media_plan_path)
        _save_progress(progress)
        print(f"Media plan saved to {media_plan_path}")
    else:
        print(f"Using existing media plan at {media_plan_path}")

    image_paths = progress.get("image_paths")
    if not _paths_exist(image_paths):
        image_paths = generate_images(
            media_plan_path,
            style_guidance=channel_config.image_style_guidance,
            channel_name=channel_config.name,
        )
        progress["image_paths"] = [str(path) for path in image_paths]
        _save_progress(progress)
        for path in image_paths:
            print(f"Image saved to {path}")
    else:
        image_paths = [str(path) for path in image_paths]
        print("Using existing images")

    short_video_path_str = progress.get("short_video_path")
    short_video_path = Path(short_video_path_str) if short_video_path_str else None
    if not short_video_path or not short_video_path.exists():
        short_video_path = generate_short_video(
            media_plan_path,
            style_guidance=channel_config.short_video_style_guidance,
            channel_name=channel_config.name,
        )
        progress["short_video_path"] = str(short_video_path)
        _save_progress(progress)
        print(f"Short video saved to {short_video_path}")
    else:
        print(f"Using existing short video at {short_video_path}")

    video_path_str = progress.get("video_path")
    video_path = Path(video_path_str) if video_path_str else None
    if not video_path or not video_path.exists():
        video_path = compose_video(
            audio_paths=audio_paths,
            image_paths=image_paths,
            short_video_path=short_video_path,
            video_title=video_title,
            video_id=video_id,
            avatar_path=channel_config.avatar_path,
            avatar_enabled=channel_config.avatar_enabled,
            bg_music_path=channel_config.bg_music,
            channel_name=channel_config.name,
        )
        progress["video_path"] = str(video_path)
        _save_progress(progress)
        print(f"Video saved to {video_path}")
    else:
        print(f"Using existing composed video at {video_path}")

    thumbnail_path_str = progress.get("thumbnail_path")
    thumbnail_path = Path(thumbnail_path_str) if thumbnail_path_str else None
    if not thumbnail_path or not thumbnail_path.exists():
        thumbnail_path = generate_thumbnail(
            media_plan_path, channel_name=channel_config.name
        )
        progress["thumbnail_path"] = str(thumbnail_path)
        _save_progress(progress)
        print(f"Thumbnail saved to {thumbnail_path}")
    else:
        print(f"Using existing thumbnail at {thumbnail_path}")

    metadata_path_str = progress.get("metadata_path")
    metadata_path = Path(metadata_path_str) if metadata_path_str else None
    if not metadata_path or not metadata_path.exists():
        metadata_path = generate_metadata(
            video_title=video_title,
            media_plan_path=media_plan_path,
            video_id=video_id,
            channel_name=channel_config.name,
        )
        progress["metadata_path"] = str(metadata_path)
        _save_progress(progress)
        print(f"Metadata saved to {metadata_path}")
    else:
        print(f"Using existing metadata at {metadata_path}")

    upload_response = upload_video(
        video_path=video_path,
        metadata_path=metadata_path,
        thumbnail_path=thumbnail_path,
        token_path=channel_config.resolved_token_path,
    )
    print(f"Upload complete: {upload_response}")

    _reset_progress()


if __name__ == "__main__":
    main()
