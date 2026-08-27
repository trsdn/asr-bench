# Changelog

All notable changes to asr-bench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
- Canary's language is taken from the session manifest rather than hardcoded.

### Fixed

- NeMo windows are cut near the quietest frame around each boundary instead of on a fixed 30 s grid, padded with silence, and re-split when a window decodes to an empty string. Canary was losing whole utterances to abrupt window edges — WER 37.6 % before, 20.3 % after.

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