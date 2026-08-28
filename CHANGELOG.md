# Changelog

All notable changes to asr-bench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

- Renamed the project to `asr-bench`; it is no longer tied to any particular recording app.
- Sessions are now described by a `session.json` manifest with arbitrarily named channels, replacing the fixed mic/system layout.
- Transcription language is pinned from the session manifest for every model that accepts a hint, Whisper included — auto-detection can lose a whole session to one bad guess, which measures language ID rather than transcription.

### Removed

- The `parakeet-live` entry, which read a recording app's own `transcript.live.jsonl`. Sessions are synthetic now, so there is nothing for it to read.

### Fixed

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