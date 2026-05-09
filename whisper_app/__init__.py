"""whisper_app — pure-function helpers extracted from app.py.

Stateful pieces (model loading, transcription generator, history, Gradio UI,
HTTP API) remain in app.py. This package collects the parts that are safe to
test and reuse independently:

- formatting:    timestamp + HTML/plain rendering
- segmentation:  word-list to segment, sentence/speaker splits, CJK handling
- media:         /media filesystem scan, yt-dlp helpers (auth, error classify)
- audio_analysis: pitch-based speaker gender estimation
"""
