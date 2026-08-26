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


# ── LocalAgreement streaming tests ──────────────────────────────────────────────
from transcriber import HypothesisBuffer, OnlineSession, SAMPLE_RATE


def _w(start, end, text):
    return types.SimpleNamespace(start=start, end=end, word=text)


class _ScriptedModel:
    """Returns a queued word-list per transcribe() call, as faster-whisper
    would: one segment carrying `.words`."""

    def __init__(self, scripts):
        self._scripts = list(scripts)

    def transcribe(self, audio, **k):
        words = self._scripts.pop(0) if self._scripts else []
        seg = types.SimpleNamespace(
            start=words[0].start if words else 0.0,
            end=words[-1].end if words else 0.0,
            text="".join(w.word for w in words),
            words=words,
        )
        return [seg], _info


def _session(scripts, **kw):
    return OnlineSession(_ScriptedModel(scripts), **kw)


# Non-silent (sine) so the tail-silence finalizer doesn't fire — these tests
# exercise the LocalAgreement prefix logic specifically.
_PCM_2S = _sine_pcm(2.0)
_SILENT_2S = (np.zeros(SAMPLE_RATE * 2, dtype=np.int16)).tobytes()


def test_hypothesis_commits_only_agreed_prefix():
    """Two passes: stable prefix commits, divergent tail is held back."""
    sess = _session(
        [
            [_w(0.0, 0.5, " hello"), _w(0.5, 1.0, " world"), _w(1.0, 1.5, " foo")],
            [_w(0.0, 0.5, " hello"), _w(0.5, 1.0, " world"), _w(1.0, 1.6, " bar")],
        ]
    )
    sess.insert_audio(_PCM_2S)
    c1, _, _ = sess.process()  # first pass: nothing agreed yet
    assert c1 == ""
    sess.insert_audio(_PCM_2S)
    c2, partial, _ = sess.process()  # second pass: prefix now agreed
    assert c2 == "hello world"
    assert "bar" in partial  # divergent tail still provisional


def test_transient_hallucination_never_commits():
    """A word present in one pass but gone the next is never committed."""
    sess = _session(
        [
            [_w(0.0, 0.5, " testing"), _w(0.5, 1.0, " Bye-bye")],
            [_w(0.0, 0.5, " testing")],
            [_w(0.0, 0.5, " testing")],
        ]
    )
    for _ in range(3):
        sess.insert_audio(_PCM_2S)
        sess.process()
    committed = "".join(w[2] for w in sess.committed).strip()
    assert "testing" in committed
    assert "Bye-bye" not in committed


def test_process_waits_for_min_chunk():
    """Below the min-chunk threshold, no inference / no commit."""
    sess = _session([[_w(0.0, 0.5, " hi")]], min_chunk_s=1.0)
    sess.insert_audio((np.zeros(SAMPLE_RATE // 2, dtype=np.int16)).tobytes())  # 0.5s
    c, p, _ = sess.process()
    assert c == "" and p == ""


def test_tail_silence_finalizes_utterance():
    """A trailing-silence pause commits the unconfirmed tail immediately
    (end-of-utterance), without waiting for a second agreeing pass."""
    sess = _session([[_w(0.0, 0.5, " hello"), _w(0.5, 1.0, " world")]])
    sess.insert_audio(_SILENT_2S)
    committed, _, eou = sess.process()
    assert committed == "hello world"
    assert eou is True  # trailing silence closed the utterance


def test_finish_flushes_unconfirmed_tail():
    sess = _session([[_w(0.0, 0.5, " hello"), _w(0.5, 1.0, " there")]])
    sess.insert_audio(_PCM_2S)
    sess.process()  # 'hello there' staged in buffer, not yet agreed
    final = sess.finish()
    assert final == "hello there"


# ── server integration tests ────────────────────────────────────────────────────
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
    assert "/ws-stream" in paths


# ── language pinning + diagnostics (added after the live test session) ─────────
class _RecordingModel:
    """Captures the kwargs of the last transcribe() call."""
    def __init__(self):
        self.last_kwargs = None
    def transcribe(self, audio, **k):
        self.last_kwargs = k
        return [_seg], _info


def test_session_pins_language_into_transcribe():
    """OnlineSession(language='en') must pass language='en' to model.transcribe
    instead of leaving it None (per-pass auto-detect)."""
    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._model = _RecordingModel()
    sess = tr.new_session(language="en", min_chunk_s=0.1)
    sess.insert_audio(_sine_pcm(0.5))
    sess.process()
    assert tr._model.last_kwargs is not None
    assert tr._model.last_kwargs.get("language") == "en"


def test_session_language_defaults_to_auto():
    """No language given → None (auto-detect), not an empty string."""
    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._model = _RecordingModel()
    sess = tr.new_session(min_chunk_s=0.1)
    sess.insert_audio(_sine_pcm(0.5))
    sess.process()
    assert tr._model.last_kwargs.get("language") is None


def test_chunk_threads_language():
    """transcribe_chunk(language=...) must reach the model."""
    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._model = _RecordingModel()
    tr._lock = asyncio.Lock()
    asyncio.run(tr.transcribe_chunk(_sine_pcm(0.3), language="de"))
    assert tr._model.last_kwargs.get("language") == "de"


def test_session_level_distinguishes_silence_from_signal():
    """level() ~0 on silence, clearly higher on a tone — the signal the server
    logs as 'SILENT' vs a real level."""
    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._model = _RecordingModel()
    sess = tr.new_session(min_chunk_s=0.1)
    sess.insert_audio((np.zeros(16000, dtype=np.int16)).tobytes())
    sess._drain()
    assert sess.level() < 0.005  # silence
    sess2 = tr.new_session(min_chunk_s=0.1)
    sess2.insert_audio(_sine_pcm(0.5))
    sess2._drain()
    assert sess2.level() > 0.05  # audible tone


def test_session_tracks_received_samples():
    """_received_samples accumulates ingested audio (drives the recv=Ns log)."""
    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._model = _RecordingModel()
    sess = tr.new_session(min_chunk_s=0.1)
    assert sess._received_samples == 0
    sess.insert_audio(_sine_pcm(1.0))
    sess._drain()
    assert sess._received_samples == 16000


def test_session_translate_sets_task():
    """translate=True → model.transcribe(task='translate'); default transcribe."""
    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._model = _RecordingModel()
    sess = tr.new_session(translate=True, min_chunk_s=0.1)
    sess.insert_audio(_sine_pcm(0.5)); sess.process()
    assert tr._model.last_kwargs.get("task") == "translate"
    tr2 = StreamingTranscriber.__new__(StreamingTranscriber)
    tr2._model = _RecordingModel()
    s2 = tr2.new_session(min_chunk_s=0.1)
    s2.insert_audio(_sine_pcm(0.5)); s2.process()
    assert tr2._model.last_kwargs.get("task") == "transcribe"


def test_ws_stream_handshake_applies_language_and_translate():
    """A JSON config frame sent first must pin language + translate on the
    session; binary-first clients keep working (covered by other tests)."""
    import json
    import time
    import server as srv
    rec = _RecordingModel()
    stub_tr = StreamingTranscriber.__new__(StreamingTranscriber)
    stub_tr._lock = asyncio.Lock()
    stub_tr._model = rec
    srv._transcriber = stub_tr
    srv.PROCESS_INTERVAL = 0.01  # force fast inference passes for the test
    from starlette.testclient import TestClient
    client = TestClient(srv.app)
    with client.websocket_connect("/ws-stream") as ws:
        ws.send_text(json.dumps({"language": "de", "translate": True}))
        ws.send_bytes(_sine_pcm(2.0))
        time.sleep(0.15)  # let the processor run ≥ 1 pass
        ws.send_text("done")
        try:
            for _ in range(50):
                if json.loads(ws.receive_text()).get("type") == "done":
                    break
        except Exception:
            pass
    assert rec.last_kwargs is not None, "model.transcribe never ran"
    assert rec.last_kwargs.get("language") == "de"
    assert rec.last_kwargs.get("task") == "translate"


# ── Lazy load + idle unload (VRAM sharing with llama-server) ────────────────
# The live service used to load large-v3 at boot and never release it, holding
# ~4 GB of a single shared GPU even when nobody streamed. That silently capped
# llama-server's usable context. These lock in load-on-demand + release-on-idle.

def test_constructor_does_not_load_model():
    """Construction must NOT touch the GPU - the model loads on first use."""
    tr = StreamingTranscriber("stub-model", "cuda", "float16")
    assert tr.loaded is False
    assert tr._model is None


def test_ensure_model_loads_once_and_is_idempotent():
    tr = StreamingTranscriber("stub-model", "cuda", "float16")
    m1 = tr.ensure_model()
    assert tr.loaded is True
    assert m1 is tr.ensure_model(), "ensure_model must reuse the loaded model"


def test_unload_frees_and_reports():
    tr = StreamingTranscriber("stub-model", "cuda", "float16")
    tr.ensure_model()
    assert asyncio.run(tr.unload()) is True
    assert tr.loaded is False
    # Second unload is a no-op, not an error.
    assert asyncio.run(tr.unload()) is False


def test_reload_after_unload():
    """An idle-unloaded service must serve the next request, not 500."""
    tr = StreamingTranscriber("stub-model", "cuda", "float16")
    tr.ensure_model()
    asyncio.run(tr.unload())
    segs = asyncio.run(tr.transcribe_chunk(_sine_pcm(1.0), context=""))
    assert tr.loaded is True
    assert segs[0]["text"] == "hello world"


def test_new_session_reloads_after_unload():
    """new_session must not hand OnlineSession a None model post-unload."""
    tr = StreamingTranscriber("stub-model", "cuda", "float16")
    asyncio.run(tr.unload())
    sess = tr.new_session()
    assert tr.loaded is True
    assert sess is not None


def test_unload_serialises_against_inference():
    """unload() must take the inference lock, so it cannot drop the model
    while the stateless transcribe_chunk path is mid-call."""
    tr = StreamingTranscriber("stub-model", "cuda", "float16")
    tr.ensure_model()

    async def scenario():
        await tr._lock.acquire()          # simulate an in-flight inference
        task = asyncio.ensure_future(tr.unload())
        await asyncio.sleep(0.05)
        still_loaded = tr.loaded          # must NOT have unloaded yet
        tr._lock.release()
        freed = await task
        return still_loaded, freed

    still_loaded, freed = asyncio.run(scenario())
    assert still_loaded is True, "unload raced an in-flight inference call"
    assert freed is True


def test_server_idle_defaults_are_share_friendly():
    """Default config must free VRAM when unused: idle unload ON, preload OFF."""
    import importlib
    srv = importlib.import_module("server")
    assert srv.LIVE_MODEL_IDLE_TIMEOUT > 0, "idle unload must be on by default"
    assert srv.LIVE_PRELOAD is False, "must not preload the model at boot"


def test_health_reports_model_loaded_state():
    """/health must expose whether the model is resident, so VRAM state is
    observable without nvidia-smi."""
    import importlib
    from starlette.testclient import TestClient
    srv = importlib.import_module("server")
    tr = StreamingTranscriber("stub-model", "cuda", "float16")
    srv._transcriber = tr
    client = TestClient(srv.app)
    body = client.get("/health").json()
    assert body["model_loaded"] is False
    assert body["idle_timeout"] == srv.LIVE_MODEL_IDLE_TIMEOUT
    tr.ensure_model()
    assert client.get("/health").json()["model_loaded"] is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
