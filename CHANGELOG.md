# Changelog

All notable changes to asr-bench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Multiple TTS engines as a first-class benchmark axis.** `--tts` now takes `say`, `piper` or `kokoro` — three unrelated synthesis lineages (Apple system voices, VITS, StyleTTS2) — and the engine is part of the session directory name (`standup-de__piper__clean`). This is not cosmetic: on `support-call-en__clean` the same three models score 9.3 / 11.8 / 11.4 % WER under `say` and 13.5 / 11.8 / 12.7 % under `piper`, which *inverts* the ranking between best and worst model. A single-engine benchmark reports the wrong answer confidently.
- Piper and Kokoro voices are auto-assigned per language from a curated pool, one distinct voice per speaker, downloading on first use. Piper previously required an explicit `.onnx` path per speaker, which made it unusable as a comparison axis.
- Optional `gender:` on a speaker, used only to pick a plausible voice from an engine's catalogue. Distinctness still wins over matching, and it has no effect on scoring.
- Eight models: `parakeet-ctc-1.1b`, `canary-qwen-2.5b` (SALM, LLM decoder), `moonshine-base`, `kyutai-stt-2.6b`, `granite-speech-4.1`, `voxtral-mini-3b`, `qwen2-audio-7b` and `phi-4-multimodal`. Measured on `support-call-en__say__clean`: kyutai 2.5 %, parakeet-ctc 5.5 %, canary-qwen 9.7 %, granite 9.7 %, moonshine 11.4 % WER.
- `hf_runners.py`: the `transformers`-based models, one runner per family. The HuggingFace side is not uniform the way NeMo and faster-whisper are — an audio LLM wants a chat template, an encoder-decoder wants a feature tensor, Voxtral wants its own request builder.
- `--timeout` on `bench.py` (default 900 s). Not every failure raises: some models deadlock inside a Metal command buffer that never completes, which presents as a very slow run and holds the machine indefinitely. A timeout records the cell as failed and lets the matrix finish.
- `sortformer-streaming` diarisation backend (nvidia/diar_streaming_sortformer_4spk-v2, 2025) — the streaming successor to Sortformer v1. Cheaper (3.2 s vs 9.9 s on a 71 s session) and more consistent across sessions, though v1 remains stronger on overlapping speech.
- Overlap-specific diarisation metrics: `diarization_error_rate` now reports a separate `overlap` block — seconds, share of speech, DER/miss/false-alarm/confusion restricted to frames with two or more simultaneous reference speakers, plus `detection_recall`. The 0.25 s collar is deliberately not applied there, since overlap sits on the boundaries a collar would mask. Surfaced by `compare.py` as its own table.
- `conversations/crosstalk-de.yaml`: four speakers, 9.3 % overlapping speech, built to exercise crosstalk. The existing scripts have 0.6 % overlap, which is no test at all.
- Tunable `titanet` parameters exposed as CLI flags — `--window`, `--hop`, `--cluster-threshold`, `--min-turn`, `--vad-db` — and `--sweep` / `--sweep-sessions` for a coordinate-descent search over them. The sweep scores every configuration across several sessions and optimises mean DER, counting a failed session as DER 1.0; single-session tuning overfits badly (27.3 % → 3.6 % on the tuned session, 58.8 % on another).
- `diarize.py`: speaker diarisation scored as DER against the session's speaker-labelled ground truth. Two backends — `sortformer` (nvidia/diar_sortformer_4spk-v1, end-to-end, max 4 speakers) and `titanet` (energy VAD → TitaNet embeddings → agglomerative clustering, no speaker limit). The clustering is implemented in-repo, so diarisation needs no extra dependency.
- `score.py` gained `diarization_error_rate`: optimal speaker-label mapping, 0.25 s collar, overlap-aware, reported split into missed speech / false alarm / speaker confusion.
- More models, including current ones: `parakeet-tdt-v3` and `parakeet-tdt-v2` (2025), `canary-1b-v2` (2025), `canary-180m-flash`, `whisper-large-v3-turbo` and `distil-whisper-large-v3`.
- Models declare the languages they support; `bench.py` skips pairings that cannot work (English-only model on a German session) unless `--ignore-language-support` is passed.
- `compare.py` reports the diarisation ranking when `diarize.py` wrote results into the same run directory.
- `envfile.py`: the `.env` loading that has to happen before torch/NeMo import, shared by every entry point instead of copied into each.
- `synth.py`: conversation scripts (YAML) are rendered to session audio via TTS — one voice per speaker, laid out on a timeline with configurable gaps and genuine overlaps, written as per-speaker isolated tracks plus a mixed channel. Because the text is known up front, every session ships a word-exact reference transcript.
- `degrade.py`: reproducible degradation profiles (`clean`, `noisy`, `very-noisy`, `phone`, `voip`, `farfield`, `clipped`, `worst-case`) built from noise, gain, clipping, reverb and codec round-trip steps, seeded so a given seed reproduces byte-identical audio.
- `score.py`: dependency-free WER/CER with Levenshtein alignment and German/English number normalisation, so `3712` and `dreitausendsiebenhundertzwölf` count as a match. Uses `jiwer` / `num2words` when installed.
- `audio_io.py`: shared decode/encode helpers, so synthesis and benchmarking cannot drift on sample rate or channel layout.
- Example conversation scripts: `conversations/standup-de.yaml` and `conversations/support-call-en.yaml`.
- `bench.py` scores every transcript against the reference and ranks results by WER; runs write `run.json` alongside `summary.json`.
- `compare.py` leads with an accuracy ranking and shows the ground truth beside each model's transcript.

### Changed

- **Session directories now carry the TTS engine**: `standup-de__clean` becomes `standup-de__say__clean`. Two sessions from the same script and profile but different engines are different test data, and their numbers are not interchangeable — burying that in a manifest field invites comparing what cannot be compared.
- No CPU fallback for the `transformers` models. A 3B audio LLM on CPU is slower than realtime by a wide margin, which makes the RTF column meaningless and ties up the machine for nothing; a model that will not run on Metal is a result in itself. `ASR_BENCH_DEVICE` overrides this for debugging.
- Voxtral loads with eager attention. Its Ministral decoder's fused SDPA path submits a Metal command buffer that never completes on MPS — the run does not fail, it parks forever.
- Renamed the project to `asr-bench`; it is no longer tied to any particular recording app.
- Sessions are now described by a `session.json` manifest with arbitrarily named channels, replacing the fixed mic/system layout.
- Transcription language is pinned from the session manifest for every model that accepts a hint, Whisper included — auto-detection can lose a whole session to one bad guess, which measures language ID rather than transcription.
- Default `titanet` VAD threshold from −33 dB to −40 dB. The only sweep result that held up across sessions; the window and hop defaults were already optimal, and the honest cross-session gain from the whole search is 12.4 % → 11.7 % DER.

### Removed

- The `parakeet-live` entry, which read a recording app's own `transcript.live.jsonl`. Sessions are synthetic now, so there is nothing for it to read.

### Fixed

- **A speaker's `rate:` was only applied by `say`.** Piper ignored it and Kokoro had `speed=1.0` hard-coded, so the per-speaker tempo declared by all nine speakers across the three scripts silently vanished on two engines out of three. The value is words per minute — `say`'s unit, default 175 — and is now converted to Piper's `length_scale` and Kokoro's `speed` against that baseline. The engines do not respond identically (a 140→220 wpm range yields a duration ratio of 1.36 / 1.29 / 1.50 against an ideal 1.57), so tempo is comparable in direction and roughly in magnitude, not exactly.

  This mattered more than a missing knob suggests: it meant a `say` ↔ `piper` comparison varied voice *and* tempo together. **The ranking inversion previously documented as this repo's headline finding does not survive the fix** — with tempo controlled, `parakeet-tdt-v2` wins on all three engines (9.3 / 11.4 / 9.3 % WER). What remains is a large but order-preserving effect: up to 6.4 points of spread, distributed unequally across models (`say`→`piper` costs Parakeet 2.1 points and Moonshine 5.1). The claim is corrected in the README rather than deleted, since it is a clean example of the failure mode the TTS axis exists to detect.
- TTS cache keys carry a version, so a change that alters the waveform behind an existing key invalidates it instead of serving stale audio.
- Each model now runs in its own subprocess, so `Peak RSS` is attributable per model. `ru_maxrss` is a monotonic per-process high-water mark, so the previous single-process matrix reported the largest model seen so far for every model after the first — with all six German models in one process every one of them read 7559 MB, against the 2.5–9.4 GB spread the isolated runs show. `--in-process` restores the old behaviour.
- NeMo windows are cut near the quietest frame around each boundary instead of on a fixed 30 s grid, padded with silence, and re-split when a window decodes to an empty string. Canary was losing whole utterances to abrupt window edges — WER 37.6 % before, 20.3 % after.
- Sortformer speaker labels are normalised from NeMo's `speaker_0` to the same `spk0` form the clustering backend emits, so the two backends' output can be compared directly.

## [0.1.0] - 2026-04-30

### Added

- Side-by-side ASR benchmark harness for real recorded session audio with mic and system channels.
- Initial model matrix for Parakeet, Canary, and Whisper Large-v3.
- Run output directories with per-model transcripts, metrics, and machine-readable summary data.
- Comparison report generation with performance tables, transcript previews, and hallucination heuristics.
- GitHub Actions secret scan workflow and local guardrail compatibility.

### Changed

- Routed Canary through German language hints and generalized NeMo transcription keyword arguments.
- Swapped NeMo Parakeet execution to live transcript handling and chunked NeMo inference.

### Fixed

- Added ffmpeg fallback for CAF audio files that libsndfile cannot parse.
- Skipped private hosted secret scan jobs by default unless explicitly enabled.

[Unreleased]: https://github.com/trsdn/asr-bench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/trsdn/asr-bench/releases/tag/v0.1.0