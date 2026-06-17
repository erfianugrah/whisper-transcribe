"""Unit tests for StreamingTranscriber + server. Stubs faster_whisper so no GPU needed."""
import asyncio
import types
import sys

# ── stub faster_whisper ────────────────────────────────────────────────────────
_seg = types.SimpleNamespace(text=" hello world", start=0.0, end=1.5)
_info = types.SimpleNamespace(language="en", duration=1.5)

fw = types.ModuleType("faster_whisper")

class _FakeModel:
    def __init__(self, *a, **k): pass
    def transcribe(self, audio, **k):
        return [_seg], _info

fw.WhisperModel = _FakeModel
sys.modules["faster_whisper"] = fw
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
from transcriber import StreamingTranscriber


def _sine_pcm(seconds: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    arr = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return arr.tobytes()


def test_transcribe_chunk_returns_segments():
    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._model = _FakeModel()
    tr._lock = asyncio.Lock()
    segs = asyncio.run(tr.transcribe_chunk(_sine_pcm(1.0), context=""))
    assert isinstance(segs, list)
    assert len(segs) == 1
    assert segs[0]["text"] == "hello world"
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 1.5


def test_transcribe_chunk_strips_empty_segments():
    empty_seg = types.SimpleNamespace(text="   ", start=0.0, end=0.5)

    class _EmptyModel:
        def transcribe(self, audio, **k):
            return [empty_seg], _info

    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._lock = asyncio.Lock()
    tr._model = _EmptyModel()
    segs = asyncio.run(tr.transcribe_chunk(_sine_pcm(), context=""))
    assert segs == []


def test_transcribe_chunk_passes_context_as_initial_prompt():
    calls = {}

    class _TrackModel:
        def transcribe(self, audio, **k):
            calls.update(k)
            return [_seg], _info

    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._lock = asyncio.Lock()
    tr._model = _TrackModel()
    asyncio.run(tr.transcribe_chunk(_sine_pcm(), context="previous text"))
    assert calls.get("initial_prompt") == "previous text"


# ── server integration tests ───────────────────────────────────────────────────
import os
os.environ.setdefault("LIVE_MODEL", "stub")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("LIVE_MAX_STREAMS", "2")


def _make_app():
    import server as srv
    stub_tr = StreamingTranscriber.__new__(StreamingTranscriber)
    stub_tr._lock = asyncio.Lock()
    stub_tr._model = _FakeModel()
    srv._transcriber = stub_tr
    return srv.app


def test_health_endpoint_returns_ok():
    from starlette.testclient import TestClient
    client = TestClient(_make_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "active_streams" in body


def test_transcribe_chunk_endpoint():
    from starlette.testclient import TestClient
    client = TestClient(_make_app())
    resp = client.post(
        "/transcribe-chunk",
        content=_sine_pcm(1.0),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    assert resp.json()["segments"][0]["text"] == "hello world"


def test_transcribe_chunk_rejects_empty_body():
    from starlette.testclient import TestClient
    client = TestClient(_make_app())
    resp = client.post(
        "/transcribe-chunk",
        content=b"",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 400


def test_probe_rejects_missing_url():
    from starlette.testclient import TestClient
    client = TestClient(_make_app())
    resp = client.get("/probe")
    assert resp.status_code == 400


def test_ws_url_route_registered():
    """The /ws-url WebSocket route must be registered (full behaviour is
    covered by the Task 8 smoke test — subprocess can't be unit-tested here)."""
    app = _make_app()
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/ws-url" in paths


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
