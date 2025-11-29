"""Script generation module using Replicate's OpenAI GPT-5 model."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Iterable, Optional, TYPE_CHECKING

import replicate

MODEL_NAME = "openai/gpt-5"
SCRIPT_FORMAT_VERSION = "YOUTUBE_SCRIPT_V2"


if TYPE_CHECKING:
    from modules.config import ChannelConfig


def _slugify(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip())
    collapsed = re.sub(r"-+", "-", sanitized).strip("-")
    return collapsed or "video"


def _collect_response_chunks(chunks: Iterable[str]) -> str:
    if isinstance(chunks, str):
        return chunks

    return "".join(chunk for chunk in chunks if chunk)


def _build_prompt(
    video_title: str,
    video_id: str,
    word_length: Optional[int],
    channel_name: str | None = None,
    channel_description: str | None = None,
) -> str:
    script_format = f"""
VIDEO_TITLE: {video_title}
VIDEO_ID: {video_id}
FORMAT: {SCRIPT_FORMAT_VERSION}

[HOOK]
A brief opening that grabs attention and introduces the central question or tension.

[INTRO]
A short setup that frames the topic and leads naturally into the first scene.

[SCENE — Title]
Begin with a line that connects smoothly to the intro or previous section.
Develop the idea in a natural, conversational way.
End with a transition that points gently toward the next scene.

[SCENE — Title]
Open with a sentence that continues the flow from the prior scene’s transition.
Explore the next part of the story or explanation.
Close with a forward-moving line that sets up what follows.

[SCENE — Title]
Start with a link to the previous scene’s final thought.
Unfold the next segment of the narrative or concept.
Finish with a soft bridge that hints at the next direction.

(Repeat for as many scenes as needed.)

[OUTRO]
A concise reflection that ties back to the hook and wraps the topic with a sense of completion.
"""

    word_count_guidance = (
        f" Keep the overall length close to {word_length} words." if word_length else ""
    )
    channel_guidance = ""
    if channel_name or channel_description:
        guidance_lines = []
        if channel_name:
            guidance_lines.append(f"Channel: {channel_name}")
        if channel_description:
            guidance_lines.append(f"Channel description: {channel_description}")
        channel_guidance = (
            "\nUse the channel context below to match tone, audience, and expectations:\n"
            + "\n".join(guidance_lines)
            + "\n"
        )
    return (
        "You are a professional YouTube script writer. "
        "Create a concise script following the exact format below. "
        "Write narration that feels natural, conversational, and paced for voiceover delivery. "
        "Make sure every section flows into the next with gentle transitions. "
        "Avoid describing specific visuals or camera directions because another module handles them. "
        "Keep each scene focused on the spoken story only—no extra labels or visual instructions. "
        f"Aim for a compact script that can be delivered quickly.{word_count_guidance} "
        f"The script should focus on {video_title}.{channel_guidance}\n"
        "Return only the script using the template. Do not include commentary.\n\n"
        f"Template:\n{script_format}"
    )


def _validate_script(script: str, video_title: str, video_id: str) -> None:
    errors = []

    if not re.search(
        rf"^VIDEO_TITLE:\s*{re.escape(video_title)}\s*$", script, re.MULTILINE
    ):
        errors.append("Missing or incorrect VIDEO_TITLE header.")

    if not re.search(rf"^VIDEO_ID:\s*{re.escape(video_id)}\s*$", script, re.MULTILINE):
        errors.append("Missing or incorrect VIDEO_ID header.")

    if not re.search(rf"^FORMAT:\s*{SCRIPT_FORMAT_VERSION}\s*$", script, re.MULTILINE):
        errors.append("Missing or incorrect FORMAT header.")

    required_sections = ["[HOOK]", "[INTRO]", "[OUTRO]"]
    for section in required_sections:
        if section not in script:
            errors.append(f"Missing required section {section}.")

    scene_matches = re.findall(r"\n\[SCENE — .*?\]\s", script)
    if len(scene_matches) < 3:
        errors.append("At least three scenes using the [SCENE — Title] format are required.")

    if "<" in script or ">" in script:
        errors.append("Script contains placeholder brackets.")

    if errors:
        raise ValueError("Invalid script generated: " + " ".join(errors))


def _save_script(
    video_title: str, video_id: str, content: str, channel_name: str
) -> Path:
    safe_title = _slugify(video_title)
    base_dir = Path("channel") / channel_name / f"{safe_title}-{video_id}" / "scripts"
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path = base_dir / f"script-{SCRIPT_FORMAT_VERSION.lower()}.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def _generate_video_id() -> str:
    return uuid.uuid4().hex[:8]


def generate_script(
    video_title: str,
    video_id: Optional[str] = None,
    word_length: Optional[int] = None,
    channel: "ChannelConfig | None" = None,
) -> str:
    resolved_video_id = video_id or _generate_video_id()
    prompt = _build_prompt(
        video_title,
        resolved_video_id,
        word_length,
        channel_name=channel.name if channel else None,
        channel_description=channel.channel_description if channel else None,
    )
    response = replicate.run(MODEL_NAME, input={"prompt": prompt})
    script = _collect_response_chunks(response)
    _validate_script(script, video_title, resolved_video_id)
    return script


def generate_and_save_script(
    video_title: str,
    video_id: Optional[str] = None,
    word_length: Optional[int] = None,
    channel_name: Optional[str] = None,
) -> tuple[Path, str]:
    resolved_video_id = video_id or _generate_video_id()
    channel_config = resolve_channel(None, channel_name)
    script = generate_script(
        video_title, resolved_video_id, word_length, channel=channel_config
    )
    script_path = _save_script(
        video_title, resolved_video_id, script, channel_config.name
    )
    return script_path, resolved_video_id


__all__ = [
    "generate_script",
    "generate_and_save_script",
]
from modules.config import resolve_channel

