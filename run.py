"""Pipeline entrypoint that triggers script generation."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from modules.script_generator.generate import generate_and_save_script

load_dotenv()


def main() -> None:
    script_path: Path = generate_and_save_script(
        "How Inflation Works", topic="Inflation basics", word_length=220
    )
    print(f"Script saved to {script_path}")


if __name__ == "__main__":
    main()
