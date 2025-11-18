"""Pipeline entrypoint that triggers script generation."""

from __future__ import annotations

from dotenv import load_dotenv

from modules.media_planner import generate_media_plan
from modules.script_generator.generate import generate_and_save_script
from modules.voice_generator import generate_voiceover

load_dotenv()


def main() -> None:
    script_path, video_id = generate_and_save_script(
        "How Inflation Works", topic="Inflation basics", word_length=200
    )
    print(f"Script saved to {script_path}")

    script_text = script_path.read_text(encoding="utf-8")
    audio_paths = generate_voiceover(
        script=script_text, video_title="How Inflation Works", video_id=video_id
    )
    for path in audio_paths:
        print(f"Voiceover saved to {path}")

    media_plan_path, _ = generate_media_plan(
        script=script_text,
        audio_paths=audio_paths,
        video_title="How Inflation Works",
        video_id=video_id,
    )
    print(f"Media plan saved to {media_plan_path}")


if __name__ == "__main__":
    main()
