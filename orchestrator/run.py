"""Pipeline entrypoint that triggers script generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from modules.script_generator.generate import generate_and_save_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a YouTube-ready script.")
    parser.add_argument("title", help="Video title to include in the script header.")
    parser.add_argument("video_id", help="Unique video identifier used for folder naming.")
    parser.add_argument(
        "--topic",
        dest="topic",
        help="Optional topic or outline to steer the script content.",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path: Path = generate_and_save_script(args.title, args.video_id, args.topic)
    print(f"Script saved to {script_path}")


if __name__ == "__main__":
    main()
