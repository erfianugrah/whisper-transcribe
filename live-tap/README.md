# live-tap — standalone audio → transcript

`desktop_tap.py` captures **OBS / Windows-sink / mic** audio, streams raw
16 kHz mono PCM to the `whisper-live` `/ws-stream` endpoint, and prints the
committed transcript to **stdout**. No Discord bot, no browser, no SPA —
just a pipe:

```bash
python desktop_tap.py | your-llm-research-tool
```

It reuses the already-deployed `whisper-live` service (LocalAgreement
streaming, commits only words confirmed across passes). Bring it up with
`make up` (or `make up-standalone` for transcription-only, no llm-compose).

## Why a separate capturer

`whisper-live` runs in Docker and **cannot see Windows audio devices**, so
capture happens in this script via an ffmpeg subprocess. Docker Desktop
forwards container port `7861` to `localhost` on both Windows and WSL, so the
default `--url ws://localhost:7861/ws-stream` works from either side. Run the
script on Windows, or in WSL with `--ffmpeg ffmpeg.exe` (the Windows binary,
which can see the audio devices).

## One-time: route OBS / desktop audio to a capturable device

Windows has no loopback device ffmpeg can grab directly. Pick one:

- **VB-Audio Virtual Cable** (simplest). Set Windows default playback (or
  OBS → Settings → Audio → *Monitoring Device*) to **CABLE Input**. Capture
  **CABLE Output**:
  ```
  --device 'audio=CABLE Output (VB-Audio Virtual Cable)'
  ```
- **VoiceMeeter** — same idea with more routing flexibility (keep hearing the
  audio while also capturing it).
- **Plain mic** — no virtual cable needed:
  ```
  --device 'audio=Microphone (Your Mic Name)'
  ```

Find the exact device names ffmpeg sees:
```bash
python desktop_tap.py --list-devices
```

## Usage

```bash
# Verify connectivity to whisper-live (5 s sine tone, no audio hardware)
python desktop_tap.py --self-test

# Capture the VB-Cable loopback (OBS / desktop audio), print transcript
python desktop_tap.py --device 'audio=CABLE Output (VB-Audio Virtual Cable)'

# From WSL using the Windows ffmpeg, see provisional words on stderr
python desktop_tap.py --ffmpeg ffmpeg.exe --partials

# Pipe live transcript into an LLM / research step (stdout = committed text,
# one line per end-of-utterance)
python desktop_tap.py | llm 'summarise and fact-check the running transcript'

# Linux/PulseAudio capture instead of Windows dshow
python desktop_tap.py --input-format pulse --device default
```

## Output contract

- **stdout** — committed transcript. Words are appended as they stabilise; a
  newline is emitted on each end-of-utterance (`eou`) pause. Line-buffered, so
  downstream pipes get text in near-real-time.
- **stderr** — status (`[tap] …`), ffmpeg errors, and (with `--partials`)
  provisional words. Keeping these off stdout means the pipe stays clean for
  an LLM consumer.

## Dependencies

`pip install websockets` and an `ffmpeg` binary. Nothing else — the script is
self-contained (`desktop_tap.py`).
