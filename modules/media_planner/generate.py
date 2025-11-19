"""Media planning module using GPT-5 and word-level timestamps."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Iterable, Optional, Sequence

import replicate

MODEL_NAME_PLANNER = "openai/gpt-5"
MODEL_NAME_TRANSCRIBE = (
    "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c"
)
PLAN_FORMAT_VERSION = "MEDIA_PLAN_V1"


def _slugify(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip())
    collapsed = re.sub(r"-+", "-", sanitized).strip("-")
    return collapsed or "video"


def _collect_response_chunks(chunks: Iterable[str]) -> str:
    if isinstance(chunks, str):
        return chunks
    return "".join(chunk for chunk in chunks if chunk)


def _generate_video_id() -> str:
    return uuid.uuid4().hex[:8]


def _decode_unicode_escapes(value: str) -> str:
    def _replace(match: re.Match) -> str:
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)

    return re.sub(r"\\u([0-9a-fA-F]{4})", _replace, value)


def _build_prompt(script: str) -> str:
    instruction = (
        "You are a media planner. Read the provided narration script and decide when a new "
        "image should appear. Identify concise snippets of the narration (6-10 words) that "
        "anchor where the image should change. For each cue, provide an expressive image "
        "generation prompt. Return a JSON array where each item has the fields "
        "'identifier' (the exact narration snippet) and 'image_prompt'. Do not include any "
        "explanations outside the JSON."
    )
    return f"{instruction}\n\nSCRIPT:\n{script.strip()}\n\nJSON:".strip()


def _request_plan(script: str) -> list[dict]:
    prompt = _build_prompt(script)
    response = replicate.run(MODEL_NAME_PLANNER, input={"prompt": prompt})
    content = _collect_response_chunks(response)
    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Planner response was not valid JSON") from exc

    if not isinstance(plan, list):
        raise ValueError("Planner response must be a JSON array")

    normalized_plan: list[dict] = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        identifier = _decode_unicode_escapes(str(item.get("identifier", "")).strip())
        image_prompt = str(item.get("image_prompt", "")).strip()
        if identifier and image_prompt:
            normalized_plan.append(
                {
                    "identifier": identifier,
                    "image_prompt": image_prompt,
                }
            )
    if not normalized_plan:
        raise ValueError("Planner returned no usable items")
    return normalized_plan


def _normalize_word(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _transcribe_audio_file(audio_path: Path) -> list[dict]:
    with open(audio_path, "rb") as file:
        response = replicate.run(
            MODEL_NAME_TRANSCRIBE,
            input={
                "audio": file,
                "task": "transcribe",
                "timestamp": "word",
                "batch_size": 64,
                "language": "None",
                "diarise_audio": False,
            },
        )

    if not isinstance(response, dict):
        return []

    output_section = response.get("output", {}) if isinstance(response.get("output", {}), dict) else {}
    chunks = (
        output_section.get("chunks")
        or response.get("chunks")
        or response.get("words")
        or []
    )

    words: list[dict] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        timestamp = chunk.get("timestamp") or chunk.get("timestamps")
        if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
            start = float(timestamp[0])
            end = float(timestamp[1])
        else:
            start = float(chunk.get("start", 0.0))
            end = float(chunk.get("end", start))
        if text:
            words.append({"word": text, "start": start, "end": end})

    if words:
        return words

    segments = response.get("segments", []) or output_section.get("segments", [])
    for segment in segments:
        for word in segment.get("words", []) or []:
            start = float(word.get("start", 0.0))
            end = float(word.get("end", start))
            words.append({"word": word.get("word", ""), "start": start, "end": end})
    return words


def _collect_transcripts(audio_paths: Sequence[Path]) -> list[dict]:
    timeline: list[dict] = []
    offset = 0.0
    for path in audio_paths:
        words = _transcribe_audio_file(path)
        for word in words:
            timeline.append(
                {
                    "word": word.get("word", ""),
                    "start": offset + float(word.get("start", 0.0)),
                    "end": offset + float(word.get("end", word.get("start", 0.0))),
                }
            )
        if words:
            offset = timeline[-1]["end"]
    return timeline


def _find_timestamp(identifier: str, transcript_words: Sequence[dict]) -> Optional[float]:
    tokens = [
        normalized
        for token in identifier.split()
        if token.strip()
        for normalized in [_normalize_word(token)]
        if normalized
    ]
    if not tokens:
        return None

    normalized_words = [_normalize_word(item.get("word", "")) for item in transcript_words]
    for idx in range(len(normalized_words) - len(tokens) + 1):
        window = normalized_words[idx : idx + len(tokens)]
        if window == tokens:
            return float(transcript_words[idx].get("start", 0.0))
    return None


def _attach_timestamps(plan: list[dict], transcript_words: Sequence[dict]) -> list[dict]:
    enriched: list[dict] = []
    for item in plan:
        timestamp = _find_timestamp(item["identifier"], transcript_words)
        enriched.append(
            {
                "identifier": item["identifier"],
                "image_prompt": item["image_prompt"],
                "timestamp": timestamp,
            }
        )
    return enriched


def _save_plan(video_title: str, video_id: str, plan: list[dict]) -> Path:
    safe_title = _slugify(video_title)
    base_dir = Path("channel") / f"{safe_title}-{video_id}" / "media-plans"
    base_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_dir / "media-plan.json"
    payload = {
        "video_title": video_title,
        "video_id": video_id,
        "format": PLAN_FORMAT_VERSION,
        "entries": plan,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def generate_media_plan(
    *,
    script: str,
    audio_paths: Sequence[Path],
    video_title: str,
    video_id: Optional[str] = None,
) -> tuple[Path, list[dict]]:
    """Generate and save a media plan mapping image prompts to timestamps."""

    resolved_video_id = video_id or _generate_video_id()
    plan = _request_plan(script)
    transcript_words = _collect_transcripts(audio_paths)
    enriched_plan = _attach_timestamps(plan, transcript_words)
    output_path = _save_plan(video_title, resolved_video_id, enriched_plan)
    return output_path, enriched_plan


__all__ = ["generate_media_plan"]
