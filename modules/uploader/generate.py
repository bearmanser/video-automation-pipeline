"""Upload module for sending rendered videos and thumbnails to YouTube."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
PRIVACY_STATUS = "private"


def _load_metadata(metadata_path: Path | str) -> Tuple[Dict, Dict]:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Metadata payload must be a JSON object")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Metadata payload missing 'metadata' object")

    return payload, metadata


def _normalize_tags(raw_tags) -> List[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, list):
        return [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    if isinstance(raw_tags, str):
        return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    return [str(raw_tags).strip()] if str(raw_tags).strip() else []


def get_credentials(token_path: str = "token.json", client_secret_path: str = "client_secret.json") -> Credentials:
    creds: Credentials | None = None
    token_file = Path(token_path)

    if token_file.exists():
        with token_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        creds = Credentials.from_authorized_user_info(data, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with token_file.open("w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return creds


def upload_video(*, video_path: Path | str, metadata_path: Path | str, thumbnail_path: Path | str) -> Dict:
    video_file = Path(video_path)
    thumb_file = Path(thumbnail_path)

    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")
    if not thumb_file.exists():
        raise FileNotFoundError(f"Thumbnail not found: {thumb_file}")

    payload, metadata = _load_metadata(metadata_path)
    video_id = str(payload.get("video_id", "")).strip()

    snippet = {
        "title": metadata.get("title") or payload.get("video_title") or video_file.stem,
        "description": metadata.get("description", ""),
        "tags": _normalize_tags(metadata.get("tags")),
    }

    creds = get_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    request = youtube.videos().insert(
        part="snippet,status",
        body={"snippet": snippet, "status": {"privacyStatus": PRIVACY_STATUS}},
        media_body=MediaFileUpload(str(video_file), resumable=True),
    )

    response = request.execute()
    uploaded_video_id = response.get("id", video_id)

    if uploaded_video_id:
        thumb_request = youtube.thumbnails().set(
            videoId=uploaded_video_id, media_body=MediaFileUpload(str(thumb_file))
        )
        thumb_request.execute()

    return {"video": response, "video_id": uploaded_video_id}


__all__ = ["upload_video", "get_credentials"]
