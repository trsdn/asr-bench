# Changelog

All notable changes to openoats-asr-bench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-04-30

### Added

- Side-by-side ASR benchmark harness for real OpenOats session audio with mic and system channels.
- Initial model matrix for Parakeet, Canary, and Whisper Large-v3.
- Run output directories with per-model transcripts, metrics, and machine-readable summary data.
- Comparison report generation with performance tables, transcript previews, and hallucination heuristics.
- GitHub Actions secret scan workflow and local guardrail compatibility.

### Changed

- Routed Canary through German language hints and generalized NeMo transcription keyword arguments.
- Swapped NeMo Parakeet execution to OpenOats live transcript handling and chunked NeMo inference.

### Fixed

- Added ffmpeg fallback for CAF audio files that libsndfile cannot parse.
- Skipped private hosted secret scan jobs by default unless explicitly enabled.

[Unreleased]: https://github.com/trsdn/openoats-asr-bench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/trsdn/openoats-asr-bench/releases/tag/v0.1.0