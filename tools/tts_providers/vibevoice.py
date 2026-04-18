from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger("hermes.tts.vibevoice")

SIDECAR_URL = os.environ.get("VIBEVOICE_URL", "http://127.0.0.1:7843")
TIMEOUT = httpx.Timeout(60.0, connect=3.0)


class VibevoiceError(RuntimeError):
    pass


async def _post(url: str, json: dict):
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        return await c.post(url, json=json)


async def _get(url: str):
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        return await c.get(url)


async def synthesize_vibevoice(text: str, voice: str = "helen") -> bytes:
    """Return WAV bytes from the local VibeVoice sidecar.

    Raises VibevoiceError on any network or HTTP failure so the caller's
    fallback-to-ElevenLabs branch in tts_tool.py fires correctly.
    """
    try:
        r = await _post(f"{SIDECAR_URL}/synthesize", json={"text": text, "voice": voice, "format": "wav"})
    except httpx.HTTPError as e:
        raise VibevoiceError(f"sidecar unreachable: {e.__class__.__name__}: {e}") from e
    if r.status_code != 200:
        body = getattr(r, "text", "") or str(r.json())
        raise VibevoiceError(f"sidecar {r.status_code}: {body}")
    try:
        path = r.json()["path"]
    except (KeyError, ValueError) as e:
        raise VibevoiceError(f"sidecar returned malformed JSON: {e}") from e
    filename = path.rsplit("/", 1)[-1]
    try:
        audio = await _get(f"{SIDECAR_URL}/audio/{filename}")
    except httpx.HTTPError as e:
        raise VibevoiceError(f"audio fetch failed: {e.__class__.__name__}: {e}") from e
    if audio.status_code != 200:
        raise VibevoiceError(f"audio fetch {audio.status_code}")
    return audio.content
