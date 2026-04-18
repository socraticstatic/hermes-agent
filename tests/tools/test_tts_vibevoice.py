"""Tests for the VibeVoice TTS provider (local sidecar client)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest


@pytest.mark.asyncio
async def test_happy_path_returns_bytes(tmp_path):
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF....fake-wav")

    async def fake_post(url, json):
        class R:
            status_code = 200

            def json(self):
                return {"path": str(wav), "duration_seconds": 1.1, "elapsed_seconds": 0.3}

        return R()

    async def fake_get(url):
        class R:
            status_code = 200
            content = b"RIFF....fake-wav"

        return R()

    with patch(
        "tools.tts_providers.vibevoice._post", new=AsyncMock(side_effect=fake_post)
    ), patch(
        "tools.tts_providers.vibevoice._get", new=AsyncMock(side_effect=fake_get)
    ):
        from tools.tts_providers.vibevoice import synthesize_vibevoice

        data = await synthesize_vibevoice("hello", voice="helen")
        assert data.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_http_error_raises_vibevoice_error():
    async def fail(url, json=None):
        class R:
            status_code = 500
            text = "model OOM"

            def json(self):
                return {}

        return R()

    with patch(
        "tools.tts_providers.vibevoice._post", new=AsyncMock(side_effect=fail)
    ):
        from tools.tts_providers.vibevoice import VibevoiceError, synthesize_vibevoice

        with pytest.raises(VibevoiceError) as ei:
            await synthesize_vibevoice("hello")
        assert "500" in str(ei.value)


@pytest.mark.asyncio
async def test_connect_error_raises_vibevoice_error():
    """When sidecar is down (httpx.ConnectError), raise VibevoiceError so the
    dispatch branch's fallback-to-ElevenLabs fires instead of propagating raw httpx."""
    async def connect_refused(url, json=None):
        raise httpx.ConnectError("connection refused")

    with patch("tools.tts_providers.vibevoice._post", new=AsyncMock(side_effect=connect_refused)):
        from tools.tts_providers.vibevoice import VibevoiceError, synthesize_vibevoice

        with pytest.raises(VibevoiceError) as ei:
            await synthesize_vibevoice("hello")
        assert "unreachable" in str(ei.value).lower() or "connect" in str(ei.value).lower()


@pytest.mark.asyncio
async def test_timeout_raises_vibevoice_error():
    """ReadTimeout (slow model load on first request) also becomes VibevoiceError."""
    async def timeout(url, json=None):
        raise httpx.ReadTimeout("timeout reading /synthesize")

    with patch("tools.tts_providers.vibevoice._post", new=AsyncMock(side_effect=timeout)):
        from tools.tts_providers.vibevoice import VibevoiceError, synthesize_vibevoice

        with pytest.raises(VibevoiceError):
            await synthesize_vibevoice("hello")
