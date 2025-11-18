"""Script generation module using Replicate's OpenAI GPT-5 model."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Iterable, Optional

import replicate

MODEL_NAME = "openai/gpt-5"
SCRIPT_FORMAT_VERSION = "YOUTUBE_SCRIPT_V1"


def _slugify(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip())
    collapsed = re.sub(r"-+", "-", sanitized).strip("-")
    return collapsed or "video"


def _collect_response_chunks(chunks: Iterable[str]) -> str:
    if isinstance(chunks, str):
        return chunks

    return "".join(chunk for chunk in chunks if chunk)


def _build_prompt(
    video_title: str, video_id: str, topic: Optional[str], word_length: Optional[int]
) -> str:
    script_format = f"""
VIDEO_TITLE: {video_title}
VIDEO_ID: {video_id}
FORMAT: {SCRIPT_FORMAT_VERSION}

[HOOK]
- one to two short lines that grab attention.

[INTRO]
- a concise setup that frames the topic.

[SCENES]
1. Scene title
   Narration: 2-4 sentences of dialogue.
   Visuals: 1-2 sentences that describe imagery.
2. Scene title
   Narration: 2-4 sentences of dialogue.
   Visuals: 1-2 sentences that describe imagery.
3. Scene title
   Narration: 2-4 sentences of dialogue.
   Visuals: 1-2 sentences that describe imagery.

[OUTRO]
- a brief takeaway and call to action.
"""

    topic_context = topic.strip() if topic else "a compelling YouTube topic"
    word_count_guidance = (
        f" Keep the overall length close to {word_length} words." if word_length else ""
    )
    return (
        "You are a professional YouTube script writer. "
        "Create a concise script following the exact format below. "
        "Use vivid language suitable for voiceover and visuals. "
        f"Aim for a compact script that can be delivered quickly.{word_count_guidance} "
        f"The script should focus on {topic_context}.\n\n"
        "Return only the script using the template. Do not include commentary.\n\n"
        f"Template:\n{script_format}"
    )


def _validate_script(script: str, video_title: str, video_id: str) -> None:
    errors = []

    if not re.search(rf"^VIDEO_TITLE:\s*{re.escape(video_title)}\s*$", script, re.MULTILINE):
        errors.append("Missing or incorrect VIDEO_TITLE header.")

    if not re.search(rf"^VIDEO_ID:\s*{re.escape(video_id)}\s*$", script, re.MULTILINE):
        errors.append("Missing or incorrect VIDEO_ID header.")

    if not re.search(rf"^FORMAT:\s*{SCRIPT_FORMAT_VERSION}\s*$", script, re.MULTILINE):
        errors.append("Missing or incorrect FORMAT header.")

    required_sections = ["[HOOK]", "[INTRO]", "[SCENES]", "[OUTRO]"]
    for section in required_sections:
        if section not in script:
            errors.append(f"Missing required section {section}.")

    scene_matches = re.findall(r"\n\d+\.\s", script)
    if len(scene_matches) < 3:
        errors.append("At least three numbered scenes are required.")

    if "<" in script or ">" in script:
        errors.append("Script contains placeholder brackets.")

    if errors:
        raise ValueError("Invalid script generated: " + " ".join(errors))


def _save_script(video_title: str, video_id: str, content: str) -> Path:
    safe_title = _slugify(video_title)
    base_dir = Path("channel") / f"{safe_title}-{video_id}" / "scripts"
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path = base_dir / f"script-{SCRIPT_FORMAT_VERSION.lower()}.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def _generate_video_id() -> str:
    return uuid.uuid4().hex[:8]


def generate_script(
    video_title: str,
    video_id: Optional[str] = None,
    topic: Optional[str] = None,
    word_length: Optional[int] = None,
) -> str:
    resolved_video_id = video_id or _generate_video_id()
    prompt = _build_prompt(video_title, resolved_video_id, topic, word_length)
    response = replicate.run(MODEL_NAME, input={"prompt": prompt})
    script = _collect_response_chunks(response)
    _validate_script(script, video_title, resolved_video_id)
    return script


def generate_and_save_script(
    video_title: str,
    video_id: Optional[str] = None,
    topic: Optional[str] = None,
    word_length: Optional[int] = None,
) -> Path:
    resolved_video_id = video_id or _generate_video_id()
    script = generate_script(video_title, resolved_video_id, topic, word_length)
    return _save_script(video_title, resolved_video_id, script)


__all__ = [
    "generate_script",
    "generate_and_save_script",
]
