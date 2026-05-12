# Design: multilingual / code-switched audio support

Status: **Proposed** — research complete, decisions made, implementation deferred.

## Problem

Whisper locks to one language detected from the first ~30 seconds and forces every subsequent segment through that language's tokenizer. Real-world CS videos (Indonesian/English, Cantonese/English, Spanish/English, etc.) come out with the non-dominant language as garbled transliterations, hallucinations, or silent skips.

The current `app.py` pipeline (whisperx 3.8.5 → load_model with `language=None` → transcribe) reproduces this default. Users summarising a YouTube interview where speakers code-switch get a transcript that's largely wrong on the switched-in language.

## Research summary

### Whisper's actual LID behaviour

- Detection is a **single 30s mel-spectrogram window**, argmax over 99/100 language token logits.
- `whisper/decoding.py:detect_language()` is the canonical entry point. Once set, language token is fixed in the decoder prompt for every subsequent 30s chunk.
- `large-v3` and `turbo` support 100 languages including `yue` (Cantonese). Pre-v3 has 99 and misclassifies Cantonese as Mandarin.

### faster-whisper has a hidden `multilingual=True` (since v1.1.0)

`BatchedInferencePipeline.generate_segment_batched()` in `faster_whisper/transcribe.py`:

```python
if options.multilingual:
    language_tokens = [
        tokenizer.tokenizer.token_to_id(segment_langs[0][0])
        for segment_langs in self.model.model.detect_language(encoder_output)
    ]
    # ... swap language token per chunk before decode
```

Per-VAD-chunk LID via Whisper's own first-token logits. Free (no extra model). Maintainer (SYSTRAN) calls it "a bit hacky" — language can flap between adjacent chunks on ambiguous audio. Works best with VAD chunks ≥10s.

**whisperX 3.8.5 does NOT pass `multilingual=True` through to faster-whisper.** To use it we'd bypass `whisperx.transcribe()` and call `BatchedInferencePipeline` directly. faster-whisper is already in our dep tree.

### `task="translate"` is genuinely good for CS audio

CS-FLEURS paper (Yan et al., Interspeech 2025, [arXiv:2509.14161](https://arxiv.org/abs/2509.14161)):

| Task | CS vs monolingual baseline |
|------|---------------------------|
| Transcribe (ASR) | **CER 2× worse** on code-switched audio |
| Translate to English (ST) | **Minimal BLEU degradation**, sometimes better |

The translation decoder leans on English regardless of source-language phonetics. Cross-referenced in Whisper paper §3.5 + 3 separate community benchmarks.

For our use case (LLM-summarises-the-transcript), translate is the killer finding.

### Alignment is broken for CS audio

whisperX's `DEFAULT_ALIGN_MODELS_TORCH` / `_HF` dicts in `whisperx/alignment.py` hard-code one wav2vec2 model per language:

- Other-language phonemes in audio → aligner falls through wildcard path
- Same-script pairs (Indonesian/English): word boundaries drift hundreds of ms
- Different-script pairs (Mandarin/English): uniform time interpolation, timestamps meaningless
- **Cantonese (`yue`) is not in the dict at all** — whisperx throws `ValueError: No default align-model for language: yue`. Latent bug.

Multilingual aligner exists: **MMS-FA** (Meta), one model, 1130 languages, ~1.2GB VRAM. HF port `MahmoudAshraf/mms-300m-1130-forced-aligner` has 3.5M downloads/month. Not first-class in whisperX. Open whisperX discussion [#1353](https://github.com/m-bain/whisperX/discussions/1353).

For `task="translate"` mode the answer is simpler: **skip alignment entirely**. English transcript vs non-English audio → phoneme model can't align them. Word-level timestamps don't survive translate mode.

### What NOT to do

1. Don't fine-tune Whisper for CS without 100h+ paired CS data (Yang et al. Interspeech 2025 — LoRA adaptation needs SEAME-scale data).
2. Don't binary-search switch boundaries by Whisper confidence threshold (Bhargava Medium post — prototype-grade, multiple passes per audio).
3. Don't run wav2vec2 alignment in translate mode (garbage timestamps).
4. Don't keep N language-specific aligners hot (VRAM bloat; use MMS-FA or skip).
5. Don't set `condition_on_previous_text=True` across language switches (primes decoder with old language).
6. Don't use `medium` or smaller for CS — quality cliff on non-dominant language.
7. Don't expect stock whisperX to handle `yue` — fix the latent bug.

## Decisions

1. **Auto-translate when source ≠ English** is the default for `translate="auto"` jobs.
2. **Ship Tier 1 first.** Review research before implementing. Tier 2 follows in a separate session.
3. **Bot exposes the override** via slash command options (`translate: bool` on `/transcribe`, `/summarize`).

## Implementation plan

### Tier 1 — auto-translate (~1-2h)

Server (`app.py`):

- New payload field on `/api/jobs`: `translate: "auto" | true | false` (default `"auto"`).
- New helper `_quick_detect_language(file_path)` — runs faster-whisper's LID on the first 30s only. Cheap (single encoder pass, no decoding). Returns `(lang_code, confidence)`.
- In `_execute_transcription`:
  - If `translate == True` → force `task="translate"`.
  - If `translate == False` → force `task="transcribe"` (current behaviour).
  - If `translate == "auto"`:
    - If `language` was already explicitly set (not "Auto-detect") → respect it, no translate.
    - Otherwise: quick-LID → if `en` (high confidence) → transcribe. Else → translate.
- Thread `task` through `_run_transcription` → `_transcribe_inner` → `whisperx.transcribe(..., task=task)`.
- **Skip alignment when `task == "translate"`** — short-circuit the align block in `_transcribe_inner`. Add a log line so it's not surprising.
- **Fix the latent `yue` bug** — if Whisper detects a language not in the alignment dict, log + skip alignment (don't raise). Segment timestamps still come from Whisper directly.

Cache key (`_transcript_cache_key`):

- Add `task` to the key components. `task=translate` and `task=transcribe` produce different transcripts — must not collide.
- New shape: `transcripts:{sha1}:{model}:{language}:{diarize}:{task}`. Old keys auto-expire via TTL; no migration step needed.

Bot (`bot/main.py`):

- Add `Job.translate: Literal["auto", True, False]` field, default `"auto"`.
- Pass `translate` through the `/api/jobs` payload.
- Surface as an option on `/transcribe` and `/summarize` slash commands — bool with `Auto` as the default (Discord doesn't have tri-state booleans cleanly; use a string choice: `auto | translate | native`).
- Embed footer / status reaction notes when translate was applied so users know.

MCP server (`~/llm-compose/mcp/whisper-server.py`):

- Add `translate` arg to `_do_yt_transcribe`, `_do_transcribe`, `_do_yt_transcribe_playlist`. Pass through to `/api/jobs`. Default `"auto"`.
- Tool schemas document the option.

Tests:

- `_quick_detect_language` returns sane values for an English clip vs Indonesian clip (use checked-in tiny WAVs or mock the faster-whisper call).
- `translate="auto"` + Indonesian → `task=translate` in the call site.
- `translate="auto"` + English → `task=transcribe`.
- `translate=false` always uses `task=transcribe`.
- Cache key includes task.
- Alignment skipped in translate mode (source-string check on the new code path).
- `yue`-detection no longer throws (graceful skip).

### Tier 2 — true CS transcripts (~1-2 days, separate session)

For users who want non-translated CS output (Tier 1 always translates non-English to English).

- New payload field: `multilingual: bool` (default `false`).
- When `true`, bypass `whisperx.transcribe()` and call faster-whisper's `BatchedInferencePipeline.transcribe(audio, multilingual=True, vad_filter=True)` directly.
- Reshape output into whisperX-compatible segment dicts so downstream (`_transcribe_inner` continuation) doesn't change.
- Alignment: either skip OR integrate MMS-FA via `MahmoudAshraf/ctc-forced-aligner`. **Recommend skip in v1**, add MMS-FA in a follow-up.
- Cache key gains `multilingual` axis.
- Bot/MCP get a `multilingual` option similarly to `translate`.

Open Tier 2 risks:

- Bypassing whisperX means we lose its VAD integration; faster-whisper's `vad_filter=True` uses Silero VAD instead of pyannote. Different segmentation. Need to validate quality on real CS audio before shipping.
- Per-chunk LID flapping (maintainer's "hacky" caveat) might produce ugly transcripts where the language switches every few seconds.

### Why not Tier 3 (explicit VAD → VoxLingua107 LID → per-group)

- Complexity: model swaps, hot-cache management, per-language alignment dispatch.
- Maintenance cost: every new language needs an alignment model decision.
- VRAM pressure under concurrent jobs.
- Diminishing returns over Tier 2: `multilingual=True` already does per-chunk LID via Whisper's own logits. A dedicated LID model improves chunk-boundary precision but the listener can't tell.
- Punt indefinitely.

## Quick-LID implementation sketch (for Tier 1)

```python
# In app.py near _transcript_cache_key

async def _quick_detect_language(file_path: str) -> tuple[str, float]:
    """30s LID pre-pass via faster-whisper. Returns (lang_code, confidence).

    Cheap — single encoder forward pass, no decoding. Used by the
    translate=auto heuristic to decide between task=transcribe (English
    source) and task=translate (non-English source, including CS audio
    where the dominant language is non-English).
    """
    def _detect() -> tuple[str, float]:
        # Reuse the whisper model we already have loaded.
        from faster_whisper import WhisperModel
        # whisper_model is the whisperx-wrapped instance; the underlying
        # faster-whisper model is whisper_model.model.
        if whisper_model is None:
            load_whisper("turbo")
        fw = whisper_model.model  # the faster-whisper WhisperModel
        # detect_language on a 30s audio chunk — returns (lang, prob)
        audio = whisperx.load_audio(file_path)[:30 * 16000]  # 30s @ 16kHz
        lang, prob, _all_probs = fw.detect_language(audio)
        return (lang, prob)
    return await asyncio.to_thread(_detect)
```

Pseudocode for the auto-translate decision:

```python
# In _execute_transcription, before the asyncio.to_thread(_run_transcription, ...)

translate_mode = payload.get("translate", "auto")
explicit_lang = payload.get("language", "Auto-detect")

if translate_mode is True:
    task = "translate"
elif translate_mode is False:
    task = "transcribe"
elif explicit_lang and explicit_lang != "Auto-detect":
    # User explicitly set a language → respect it, no translate.
    task = "transcribe"
else:
    lang, conf = await _quick_detect_language(file_path)
    if lang == "en":
        task = "transcribe"
    elif conf < 0.5:
        # Low confidence often indicates mixed audio. Translate is safer.
        task = "translate"
    else:
        # High-confidence non-English → translate (summarisation default).
        task = "translate"

# task threads through to _run_transcription → whisperx.transcribe(task=task)
```

## Effort estimate

| Tier | Scope | Estimate | Ship priority |
|------|-------|----------|---------------|
| 1 | Auto-translate + alignment-skip + `yue` fix + cache key + bot slash command + MCP option + tests | 2-3 hours | Next session |
| 2 | `multilingual=true` via faster-whisper BatchedInferencePipeline + optional MMS-FA | 1-2 days | Sprint after Tier 1 |
| 3 | Explicit VAD/LID/group pipeline | 1 week | Skip indefinitely |

## Open questions for a future iteration

1. Should `translate=auto` track per-channel preferences in the bot? Some Discord servers might want native transcripts even for non-English content.
2. Cache: should we store both `task=transcribe` and `task=translate` results when we know the user might want either? Today they're keyed separately; only one is computed per request.
3. Streaming consumers (e.g. Gradio live UI): does the auto path add unacceptable latency from the 30s pre-pass? Could short-circuit to `transcribe` for streaming mode.
4. Quality validation: we should curate a small CS test set (Indonesian/English, Cantonese/English, Spanish/English) and benchmark BLEU/CER before declaring Tier 1 done.

## Key citations

- faster-whisper `multilingual` param: `faster_whisper/transcribe.py:BatchedInferencePipeline.generate_segment_batched`
- whisperX alignment models: `whisperx/alignment.py` `DEFAULT_ALIGN_MODELS_TORCH` (~L25), `DEFAULT_ALIGN_MODELS_HF` (~L33)
- CS-FLEURS benchmark: [arXiv:2509.14161](https://arxiv.org/abs/2509.14161) (Yan et al., Interspeech 2025)
- Whisper CS adaptation SOTA: [arXiv:2412.16507](https://arxiv.org/abs/2412.16507)
- MMS-FA: `torchaudio.pipelines.MMS_FA`; HF port `MahmoudAshraf/mms-300m-1130-forced-aligner`
- Open issues confirming the gap: [whisperX#466](https://github.com/m-bain/whisperX/issues/466), [whisperX#271](https://github.com/m-bain/whisperX/issues/271), [faster-whisper#918](https://github.com/SYSTRAN/faster-whisper/issues/918), [whisperX#1353](https://github.com/m-bain/whisperX/discussions/1353)
