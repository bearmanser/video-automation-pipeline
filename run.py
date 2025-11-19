"""Pipeline entrypoint that triggers script generation."""

from __future__ import annotations

from dotenv import load_dotenv

from modules.image_generator import generate_images
from modules.media_planner.generate import generate_media_plan
from modules.metadata_generator.generate import generate_metadata
from modules.script_generator.generate import generate_and_save_script
from modules.short_video_generator.generate import generate_short_video
from modules.thumbnail_creator.generate import generate_thumbnail
from modules.video_composer import compose_video
from modules.voice_generator.generate import generate_voiceover

load_dotenv()


def main() -> None:
    video_title = "How Inflation Works"
    topic = "Inflation basics"

    script_path, video_id = generate_and_save_script(
        video_title, topic=topic, word_length=1000
    )
    print(f"Script saved to {script_path}")

    script_text = script_path.read_text(encoding="utf-8")
    audio_paths = generate_voiceover(
        script=script_text, video_title=video_title, video_id=video_id
    )
    for path in audio_paths:
        print(f"Voiceover saved to {path}")

    media_plan_path, _ = generate_media_plan(
        script=script_text,
        audio_paths=audio_paths,
        video_title=video_title,
        video_id=video_id,
    )
    print(f"Media plan saved to {media_plan_path}")

    image_paths = generate_images(media_plan_path)
    for path in image_paths:
        print(f"Image saved to {path}")

    video_path = compose_video(
        audio_paths=audio_paths,
        image_paths=image_paths,
        video_title=video_title,
        video_id=video_id,
    )
    print(f"Video saved to {video_path}")

    thumbnail_path = generate_thumbnail(media_plan_path)
    print(f"Thumbnail saved to {thumbnail_path}")

    short_video_path = generate_short_video(media_plan_path)
    print(f"Short video saved to {short_video_path}")

    metadata_path = generate_metadata(
        video_title=video_title,
        media_plan_path=media_plan_path,
        video_id=video_id,
    )
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
