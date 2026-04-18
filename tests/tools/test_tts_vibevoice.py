"""Tests for the VibeVoice TTS provider (local sidecar client)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

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
