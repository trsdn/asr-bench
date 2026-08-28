"""
asr-bench — side-by-side ASR model comparison on multi-speaker session audio.

Works on two kinds of session:

  * synthetic sessions built by `synth.py`, which ship a word-exact
    reference transcript — these give real WER/CER numbers;
  * any recorded session directory with an `audio/` folder — these give
    speed/RAM numbers and side-by-side transcripts, but no error rate.

Usage:

    uv run python synth.py --script conversations/standup-de.yaml --degrade phone
    uv run python bench.py --session sessions/standup-de__phone \\
        --models canary whisper-large-v3 \\
        --run-name phone-test

Each model runs once per channel; transcript + metrics (wall-clock, RTF,
peak RAM, WER/CER when a reference exists) land under
`runs/<run-name>/<model>/`.

Design notes
------------
- All HF / NeMo / torch caches are redirected via the `.env` file next to
  this script so `~/` doesn't fill up with model weights.
- Each model runs in its own function, isolated so a failure in one
  doesn't kill the rest of the matrix.
- We avoid a shared abstraction because the runtimes (NeMo,
  faster-whisper) have different load / infer signatures and smashing
  them into one interface obscures more than it saves.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Load .env (or fall back to .env.example) before anything that might
# touch HF / NeMo / Torch caches. `.env` is per-machine and gitignored;
# `.env.example` is the committed template.
from envfile import load_env  # noqa: E402

load_env()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from audio_io import TARGET_SR, load_mono_16k  # noqa: E402
from score import score as score_transcript  # noqa: E402

# Silence padded around every NeMo window. Attention decoders are prone
# to emitting EOS straight away when a window starts or ends abruptly.
NEMO_PAD_SECONDS = 0.3
# How often an empty window may be split in half before we accept the
# empty result. 3 takes a 15 s window down to ~2 s.
NEMO_MAX_RETRY_DEPTH = 3

console = Console()


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────


@dataclass
class ModelRun:
    model_id: str
    channel: str         # whatever channels the session provides
    audio_seconds: float
    wall_seconds: float
    rtf: float           # wall / audio (< 1 = faster than realtime)
    peak_rss_mb: float
    text: str
    error: str | None = None
    # Populated only for sessions that carry a reference transcript
    # (i.e. anything produced by synth.py).
    accuracy: dict | None = None

    @property
    def wer(self) -> float | None:
        return self.accuracy["wer"] if self.accuracy else None

    def to_json(self) -> dict:
        return asdict(self)


def peak_rss_mb() -> float:
    """Peak resident set size since process start, in MB (Darwin reports
    ru_maxrss in bytes; Linux reports kB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


# ──────────────────────────────────────────────
# Model runners
# ──────────────────────────────────────────────


def run_faster_whisper(
    audio: np.ndarray,
    model_name: str = "large-v3",
    language: str | None = None,
) -> str:
    from faster_whisper import WhisperModel
    # int8 for Apple Silicon — fast enough, small footprint. float16 is
    # not supported on CPU; the Metal backend is experimental in CT2.
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,  # trims long silence → fewer hallucinations
        # Pinned to the session language, same as Canary. Auto-detection
        # is a separate capability and letting it run here means a model
        # can lose a whole session to one bad guess on the first few
        # seconds — that measures language ID, not transcription.
        language=language,
    )
    chunks = [seg.text.strip() for seg in segments]
    del model
    gc.collect()
    return " ".join(chunks).strip()


def quietest_frame(
    audio: np.ndarray,
    target: int,
    search_samples: int,
    frame_ms: float = 20.0,
) -> int:
    """Return the sample offset near `target` that sits in the quietest
    short frame within +/- `search_samples` — i.e. the least damaging
    place to cut. Falls back to `target` when the window is too small."""
    frame = max(1, int(frame_ms / 1000.0 * TARGET_SR))
    lo = max(0, target - search_samples)
    hi = min(len(audio) - frame, target + search_samples)
    if hi - lo < frame:
        return target
    window = audio[lo:hi]
    # Mean |x| per frame; the minimum is the closest thing to a pause in
    # this neighbourhood.
    usable = (window.size // frame) * frame
    frames = np.abs(window[:usable]).reshape(-1, frame).mean(axis=1)
    return lo + int(np.argmin(frames)) * frame + frame // 2


def silence_aware_chunks(
    audio: np.ndarray,
    chunk_seconds: float = 30.0,
    search_seconds: float = 4.0,
    frame_ms: float = 20.0,
) -> list[tuple[int, int]]:
    """Split long audio into ~`chunk_seconds` windows that end in the
    quietest spot nearby, and return (start, end) sample offsets.

    Cutting on a fixed grid slices through the middle of utterances, and
    NeMo simply drops the fragments on both sides — worth several words
    per boundary, which is enough to flip a model ranking. Searching a
    few seconds around each boundary for the lowest-energy frame moves
    the cut into a pause instead."""
    n = len(audio)
    chunk = int(chunk_seconds * TARGET_SR)
    if n <= chunk:
        return [(0, n)]

    frame = max(1, int(frame_ms / 1000.0 * TARGET_SR))
    search = int(search_seconds * TARGET_SR)

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n:
        target = start + chunk
        if target >= n - frame:
            bounds.append((start, n))
            break

        lo = max(start + chunk // 2, target - search)
        hi = min(n - frame, target + search)
        window = audio[lo:hi]
        if window.size >= frame:
            usable = (window.size // frame) * frame
            frames = np.abs(window[:usable]).reshape(-1, frame).mean(axis=1)
            cut = lo + int(np.argmin(frames)) * frame + frame // 2
        else:
            cut = target

        bounds.append((start, cut))
        start = cut
    return bounds


def run_nemo_asr(
    audio: np.ndarray,
    model_id: str,
    extra_transcribe_kwargs: dict | None = None,
    chunk_seconds: float = 30.0,
) -> str:
    """Unified runner for Parakeet / Canary. Chunks the audio into short
    windows and transcribes each one individually — NeMo's default
    Lhotse dataloader fails on long audio with macOS's spawn-based
    multiprocessing (the 8 default workers silently exit before producing
    a single batch, leaving "Transcribing: 0it" in the log). Passing each
    window as its own one-shot transcribe call sidesteps the dataloader
    path entirely and keeps peak RAM bounded.

    `extra_transcribe_kwargs` is merged into `transcribe()` — used for
    Canary's `source_lang`/`target_lang`/`pnc`/`task` hints."""
    from nemo.collections.asr.models import ASRModel
    import tempfile

    extra_kwargs = dict(extra_transcribe_kwargs or {})
    model = ASRModel.from_pretrained(model_id)
    # Belt-and-braces: even if NeMo's transcribe() internally builds a
    # dataloader, make sure any worker count is 0.
    try:
        model._cfg.test_ds.num_workers = 0
    except Exception:
        pass
    try:
        model._cfg.validation_ds.num_workers = 0
    except Exception:
        pass

    bounds = silence_aware_chunks(audio, chunk_seconds=chunk_seconds)

    texts: list[str] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="nemo-bench-"))
    counter = itertools.count()
    pad = np.zeros(int(NEMO_PAD_SECONDS * TARGET_SR), dtype=np.float32)

    def transcribe_segment(segment: np.ndarray, depth: int = 0) -> str:
        """Transcribe one window, retrying on an empty decode.

        Canary is an attention-encoder-decoder model and occasionally
        emits EOS immediately, returning an empty string for a window
        that is plainly speech — a 10 s window came back empty while
        both 8 s and 12 s of the same audio transcribed fine. Padding
        the window with silence fixes some of those; splitting it in
        half fixes most of the rest. Without this the empty windows
        score as bulk deletions and we would be measuring our own
        chunking rather than the model."""
        if len(segment) < TARGET_SR // 10:
            # Anything shorter than 100 ms carries no words and NeMo
            # occasionally errors on such micro-chunks.
            return ""

        wav_path = tmpdir / f"chunk_{next(counter):05d}.wav"
        sf.write(
            str(wav_path),
            np.concatenate([pad, segment, pad]),
            TARGET_SR,
            subtype="PCM_16",
        )
        try:
            result = model.transcribe(
                [str(wav_path)],
                batch_size=1,
                num_workers=0,
                verbose=False,
                **extra_kwargs,
            )
        except TypeError:
            # Older NeMo versions don't accept num_workers/verbose kwargs.
            # Still try with the model-specific extras (Canary's
            # source_lang/target_lang are required, not optional).
            result = model.transcribe(
                [str(wav_path)],
                batch_size=1,
                **extra_kwargs,
            )
        finally:
            wav_path.unlink(missing_ok=True)

        # NeMo returns list[str] or list[Hypothesis] depending on
        # the model family + version. Normalise.
        parts: list[str] = []
        for r in result:
            if isinstance(r, str):
                parts.append(r)
            elif hasattr(r, "text"):
                parts.append(str(r.text))
            elif isinstance(r, list) and r and hasattr(r[0], "text"):
                parts.append(str(r[0].text))
            else:
                parts.append(str(r))
        text = " ".join(p.strip() for p in parts).strip()

        if text or depth >= NEMO_MAX_RETRY_DEPTH:
            return text

        # Empty decode: split at the quietest frame near the middle so
        # neither half starts or ends mid-word, and try the halves.
        middle = len(segment) // 2
        split = quietest_frame(
            segment,
            middle,
            search_samples=min(middle, int(1.0 * TARGET_SR)),
        )
        left = transcribe_segment(segment[:split], depth + 1)
        right = transcribe_segment(segment[split:], depth + 1)
        return " ".join(p for p in (left, right) if p).strip()

    try:
        for start, end in bounds:
            text = transcribe_segment(audio[start:end])
            if text:
                texts.append(text)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        del model
        gc.collect()

    return " ".join(t.strip() for t in texts).strip()


def run_salm(
    audio: np.ndarray,
    model_id: str,
    chunk_seconds: float = 30.0,
) -> str:
    """Canary-Qwen and friends: a speech encoder feeding an LLM decoder.

    NeMo exposes these through `SALM`, not `ASRModel` — loading one with
    `ASRModel.from_pretrained` fails looking for a `model_config.yaml`
    that a SALM checkpoint does not have. The task comes from a prompt
    rather than the architecture, which is the whole point of the design
    and also its risk: an LLM decoder can produce fluent text that owes
    nothing to the audio."""
    from nemo.collections.speechlm2.models import SALM
    import tempfile

    model = SALM.from_pretrained(model_id)
    model.eval()

    bounds = silence_aware_chunks(audio, chunk_seconds=chunk_seconds)
    tmpdir = Path(tempfile.mkdtemp(prefix="salm-bench-"))
    pad = np.zeros(int(NEMO_PAD_SECONDS * TARGET_SR), dtype=np.float32)

    texts: list[str] = []
    try:
        for index, (start, end) in enumerate(bounds):
            segment = np.concatenate([pad, audio[start:end], pad])
            wav = tmpdir / f"chunk-{index}.wav"
            sf.write(str(wav), segment, TARGET_SR, subtype="PCM_16")
            answer = model.generate(
                prompts=[
                    [
                        {
                            "role": "user",
                            "content": (
                                f"Transcribe the following: "
                                f"{model.audio_locator_tag}"
                            ),
                            "audio": [str(wav)],
                        }
                    ]
                ],
                max_new_tokens=256,
            )
            texts.append(model.tokenizer.ids_to_text(answer[0].cpu()).strip())
            wav.unlink(missing_ok=True)
    finally:
        try:
            tmpdir.rmdir()
        except OSError:
            pass
        del model
        gc.collect()

    return " ".join(t for t in texts if t).strip()


def run_hf_model(
    audio: np.ndarray,
    model_id: str,
    family: str,
    language: str = "en",
    chunk_seconds: float = 30.0,
    device: str | None = None,
) -> str:
    """Bridge to `hf_runners`. Chunking stays here so every model in the
    matrix sees the same windowing policy."""
    from hf_runners import run_hf

    bounds = silence_aware_chunks(audio, chunk_seconds=chunk_seconds)
    pad = np.zeros(int(NEMO_PAD_SECONDS * TARGET_SR), dtype=np.float32)
    chunks = [
        np.concatenate([pad, audio[start:end], pad]) for start, end in bounds
    ]
    texts = run_hf(chunks, model_id, family, language=language, device=device)
    return " ".join(t for t in texts if t).strip()


MODEL_REGISTRY: dict[str, dict] = {
    # `languages` lists the languages a model actually supports. English-only
    # models scored against a German session produce nonsense that looks like
    # a catastrophic model failure rather than what it is — the wrong tool for
    # the job — so bench.py skips those pairings unless you force them.
    # `languages: None` means "unrestricted".
    "canary": {
        "kind": "nemo",
        "nemo_id": "nvidia/canary-1b-flash",
        "label": "NVIDIA Canary-1B-Flash (en/de/fr/es)",
        "languages": {"en", "de", "fr", "es"},
        # Canary silently drops material inside long windows — on a 29 s
        # chunk it returned ~75% of the words, with whole utterances
        # missing from the middle rather than the edges. Shorter windows
        # trade a little context for output that actually covers the
        # audio, without which its WER measures our chunking rather than
        # the model.
        "chunk_seconds": 15.0,
        # Canary is a Multi-Task model: without explicit language hints
        # it auto-translates to English, which would score as a total
        # miss against a German reference. We pin source == target to the
        # session language (transcribe, don't translate) and request
        # punctuation + capitalisation. `{lang}` is substituted from
        # session.json at run time; it falls back to English for recorded
        # sessions that carry no manifest.
        "nemo_transcribe_kwargs": {
            "source_lang": "{lang}",
            "target_lang": "{lang}",
            "pnc": "yes",
            "task": "asr",
        },
    },
    "canary-1b-v2": {
        "kind": "nemo",
        "nemo_id": "nvidia/canary-1b-v2",
        "label": "NVIDIA Canary-1B v2 (2025, 25 European languages)",
        "languages": None,
        "chunk_seconds": 15.0,
        "nemo_transcribe_kwargs": {
            "source_lang": "{lang}",
            "target_lang": "{lang}",
            "pnc": "yes",
        },
    },
    "canary-180m-flash": {
        "kind": "nemo",
        "nemo_id": "nvidia/canary-180m-flash",
        "label": "NVIDIA Canary-180M-Flash (small, en/de/fr/es)",
        "languages": {"en", "de", "fr", "es"},
        "chunk_seconds": 15.0,
        "nemo_transcribe_kwargs": {
            "source_lang": "{lang}",
            "target_lang": "{lang}",
            "pnc": "yes",
            "task": "asr",
        },
    },
    "parakeet-tdt-v3": {
        "kind": "nemo",
        "nemo_id": "nvidia/parakeet-tdt-0.6b-v3",
        "label": "NVIDIA Parakeet-TDT 0.6B v3 (2025, 25 European languages)",
        "languages": None,
        # TDT is a transducer: frame-synchronous and alignment-forced, so
        # it has no EOS to emit early and tolerates longer windows than
        # the attention-decoder models.
        "chunk_seconds": 30.0,
    },
    "parakeet-tdt-v2": {
        "kind": "nemo",
        "nemo_id": "nvidia/parakeet-tdt-0.6b-v2",
        "label": "NVIDIA Parakeet-TDT 0.6B v2 (2025, English only)",
        "languages": {"en"},
        "chunk_seconds": 30.0,
    },
    "parakeet-ctc-1.1b": {
        "kind": "nemo",
        "nemo_id": "nvidia/parakeet-ctc-1.1b",
        # Same family as the TDT models but a plain CTC head instead of a
        # transducer, which isolates the decoder choice: CTC assumes
        # conditional independence between output frames, so it has no
        # internal language model to lean on — and no way to loop either.
        "label": "NVIDIA Parakeet-CTC 1.1B (CTC head, English only)",
        "languages": {"en"},
        "chunk_seconds": 30.0,
    },
    "canary-qwen-2.5b": {
        "kind": "salm",
        "nemo_id": "nvidia/canary-qwen-2.5b",
        # A speech-augmented language model: Canary encoder bolted onto a
        # Qwen LLM decoder. Worth having because it is the one NeMo model
        # here whose decoder is a general-purpose LLM, which is exactly
        # the design that can hallucinate fluent text over bad audio.
        "label": "NVIDIA Canary-Qwen 2.5B (SALM, LLM decoder, English only)",
        "languages": {"en"},
        "chunk_seconds": 30.0,
    },
    "moonshine-base": {
        "kind": "hf",
        "hf_id": "UsefulSensors/moonshine-base",
        "hf_family": "seq2seq",
        # 200 MB against Whisper-large's 3 GB. Here to establish the
        # floor: how much accuracy does the smallest credible model
        # actually give up?
        "label": "Moonshine Base (61M params, English only)",
        "languages": {"en"},
        "chunk_seconds": 30.0,
    },
    "kyutai-stt-2.6b": {
        "kind": "hf",
        "hf_id": "kyutai/stt-2.6b-en-trfs",
        "hf_family": "seq2seq",
        "label": "Kyutai STT 2.6B (streaming architecture, English only)",
        "languages": {"en"},
        "chunk_seconds": 30.0,
    },
    "granite-speech-4.1": {
        "kind": "hf",
        "hf_id": "ibm-granite/granite-speech-4.1-2b",
        "hf_family": "audio-llm",
        "label": "IBM Granite Speech 4.1 2B (2026)",
        "languages": {"en"},
        "chunk_seconds": 30.0,
    },
    "qwen2-audio-7b": {
        "kind": "hf",
        "hf_id": "Qwen/Qwen2-Audio-7B-Instruct",
        "hf_family": "audio-llm",
        "label": "Qwen2-Audio 7B Instruct (audio LLM)",
        "languages": None,
        "chunk_seconds": 30.0,
    },
    "voxtral-mini-3b": {
        "kind": "hf",
        "hf_id": "mistralai/Voxtral-Mini-3B-2507",
        "hf_family": "voxtral",
        "label": "Mistral Voxtral Mini 3B (2025)",
        "languages": None,
        "chunk_seconds": 30.0,
        # MPS deadlocks on this one: the process parks forever inside a
        # Metal command buffer instead of raising, so it looks like a very
        # slow run rather than a failure. CPU is slower but finishes.
        "device": "cpu",
    },
    "phi-4-multimodal": {
        "kind": "hf",
        "hf_id": "microsoft/Phi-4-multimodal-instruct",
        "hf_family": "phi4",
        "label": "Microsoft Phi-4 Multimodal (audio + text)",
        "languages": None,
        "chunk_seconds": 30.0,
    },
    "whisper-large-v3": {
        "kind": "whisper",
        "fw_id": "large-v3",
        "label": "OpenAI Whisper Large-v3 (via faster-whisper)",
        "languages": None,
    },
    "whisper-large-v3-turbo": {
        "kind": "whisper",
        "fw_id": "large-v3-turbo",
        "label": "OpenAI Whisper Large-v3-Turbo (4-layer decoder)",
        "languages": None,
    },
    "distil-whisper-large-v3": {
        "kind": "whisper",
        "fw_id": "distil-large-v3",
        "label": "Distil-Whisper Large-v3 (English only)",
        "languages": {"en"},
    },
}


def supports_language(model_key: str, language: str) -> bool:
    """Whether a model claims to handle the session language."""
    languages = MODEL_REGISTRY[model_key].get("languages")
    if not languages:
        return True
    return (language or "en").split("-")[0].split("_")[0].lower() in languages


def resolve_kwargs(kwargs: dict | None, language: str) -> dict | None:
    """Substitute `{lang}` placeholders in a model's transcribe kwargs with
    the session language, so one registry entry works for any language."""
    if not kwargs:
        return kwargs
    lang = (language or "en").split("-")[0].split("_")[0].lower()
    return {
        k: (v.format(lang=lang) if isinstance(v, str) else v)
        for k, v in kwargs.items()
    }


def run_model(
    model_key: str,
    audio: np.ndarray,
    language: str = "en",
) -> tuple[str, str | None]:
    """Dispatch on the registry. Returns (text, error). Any exception is
    caught so one broken runtime doesn't kill the rest of the matrix."""
    cfg = MODEL_REGISTRY[model_key]
    lang = (language or "en").split("-")[0].split("_")[0].lower()
    try:
        if cfg["kind"] == "whisper":
            return run_faster_whisper(audio, cfg["fw_id"], language=lang), None
        elif cfg["kind"] == "nemo":
            return run_nemo_asr(
                audio,
                cfg["nemo_id"],
                extra_transcribe_kwargs=resolve_kwargs(
                    cfg.get("nemo_transcribe_kwargs"), language
                ),
                chunk_seconds=cfg.get("chunk_seconds", 30.0),
            ), None
        elif cfg["kind"] == "salm":
            return run_salm(
                audio,
                cfg["nemo_id"],
                chunk_seconds=cfg.get("chunk_seconds", 30.0),
            ), None
        elif cfg["kind"] == "hf":
            return run_hf_model(
                audio,
                cfg["hf_id"],
                cfg["hf_family"],
                language=lang,
                chunk_seconds=cfg.get("chunk_seconds", 30.0),
                device=cfg.get("device"),
            ), None
        else:
            raise ValueError(f"Unknown kind: {cfg['kind']}")
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


# ──────────────────────────────────────────────
# Session loading
# ──────────────────────────────────────────────

AUDIO_EXTS = (".wav", ".caf", ".flac", ".m4a", ".mp3", ".aiff", ".aif", ".ogg", ".opus")


@dataclass
class Session:
    """A directory of audio to benchmark, plus whatever ground truth it
    happens to carry. Synthetic sessions (synth.py) have references;
    recorded ones generally don't, and everything downstream is written
    to degrade gracefully in that case."""

    path: Path
    language: str
    channels: dict[str, Path]           # channel name → audio file
    references: dict[str, str]          # channel name → reference text
    manifest: dict

    @property
    def has_reference(self) -> bool:
        return bool(self.references)


def load_session(session_dir: Path) -> Session:
    """Discover channels and ground truth in a session directory.

    A `session.json` manifest (written by synth.py) is authoritative for
    language and channel order; without one we fall back to globbing
    `audio/`, which keeps plain recorded sessions working."""
    audio_dir = session_dir / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"No audio/ directory under {session_dir}")

    manifest: dict = {}
    manifest_path = session_dir / "session.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    channels: dict[str, Path] = {}
    for f in sorted(audio_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            channels[f.stem] = f

    if not channels:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    references: dict[str, str] = {}
    ref_dir = session_dir / "reference"
    if ref_dir.is_dir():
        for name in channels:
            ref_file = ref_dir / f"{name}.txt"
            if ref_file.exists():
                references[name] = ref_file.read_text(encoding="utf-8").strip()

    return Session(
        path=session_dir,
        language=str(manifest.get("language", "en")),
        channels=channels,
        references=references,
        manifest=manifest,
    )


# ──────────────────────────────────────────────
# One benchmark cell (model × channel)
# ──────────────────────────────────────────────
#
# Every cell runs in its own subprocess. That costs a few seconds of
# interpreter and model start-up, but it buys two things that matter more:
#
#   * Honest peak-RSS numbers. `ru_maxrss` is a monotonic high-water mark
#     for the whole process, so running several models in one process gives
#     every model after the first the peak of the largest one seen so far.
#     A fresh process per cell is the only way to attribute memory.
#   * Isolation. NeMo in particular does not reliably free model memory,
#     so a long matrix in one process slowly starves the machine.


def bench_cell(
    session: "Session",
    model_key: str,
    channel: str,
    language: str,
    run_dir: Path,
) -> "ModelRun":
    """Transcribe one channel with one model and score it."""
    audio = load_mono_16k(session.channels[channel])
    start = time.perf_counter()
    text, err = run_model(model_key, audio, language)
    wall = time.perf_counter() - start

    audio_seconds = len(audio) / TARGET_SR
    accuracy = None
    if channel in session.references and not err:
        accuracy = score_transcript(text, session.references[channel], language)

    run = ModelRun(
        model_id=model_key,
        channel=channel,
        audio_seconds=audio_seconds,
        wall_seconds=wall,
        rtf=wall / audio_seconds if audio_seconds > 0 else 0.0,
        peak_rss_mb=peak_rss_mb(),
        text=text,
        error=err,
        accuracy=accuracy,
    )
    write_cell(run, run_dir)
    return run


def cell_paths(run_dir: Path, model_key: str, channel: str) -> tuple[Path, Path]:
    model_dir = run_dir / model_key
    return model_dir / f"{channel}.txt", model_dir / f"{channel}.metrics.json"


def write_cell(run: "ModelRun", run_dir: Path) -> None:
    out_path, metrics_path = cell_paths(run_dir, run.model_id, run.channel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(run.text or "", encoding="utf-8")
    metrics_path.write_text(
        json.dumps(run.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def bench_cell_subprocess(
    session: "Session",
    model_key: str,
    channel: str,
    language: str,
    run_dir: Path,
) -> "ModelRun":
    """Run one cell in a fresh interpreter, then read back what it wrote.

    The child writes the same files the in-process path does, so it is the
    single source of truth for the result; we only parse them back so the
    summary table can be printed here.
    """
    _, metrics_path = cell_paths(run_dir, model_key, channel)
    metrics_path.unlink(missing_ok=True)

    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--session", str(session.path),
        "--worker", model_key, channel,
        "--language", language,
        "--run-name", run_dir.name,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if not metrics_path.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit code {proc.returncode}"
        return ModelRun(
            model_id=model_key, channel=channel, audio_seconds=0.0,
            wall_seconds=0.0, rtf=0.0, peak_rss_mb=0.0, text="",
            error=f"worker failed: {detail}", accuracy=None,
        )

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    text_path, _ = cell_paths(run_dir, model_key, channel)
    return ModelRun(
        model_id=data["model_id"],
        channel=data["channel"],
        audio_seconds=data["audio_seconds"],
        wall_seconds=data["wall_seconds"],
        rtf=data["rtf"],
        peak_rss_mb=data["peak_rss_mb"],
        text=text_path.read_text(encoding="utf-8") if text_path.exists() else "",
        error=data.get("error"),
        accuracy=data.get("accuracy"),
    )


def worker_main(args) -> int:
    """Internal entry point: run exactly one cell and write its files."""
    model_key, channel = args.worker
    session = load_session(args.session.expanduser().resolve())
    run_dir = Path(__file__).resolve().parent / "runs" / args.run_name
    bench_cell(session, model_key, channel, args.language or session.language, run_dir)
    return 0


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Session directory containing an audio/ folder. Synthetic "
             "sessions from synth.py also carry reference/ for WER scoring.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY),
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to run. Default: all of them. Models that "
             "don't support the session language are skipped.",
    )
    ap.add_argument(
        "--ignore-language-support",
        action="store_true",
        help="Run models even on languages they don't claim to support. "
             "Useful to see what an English-only model does with German, "
             "but the resulting WER says nothing about model quality.",
    )
    ap.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channel names to transcribe (file stems under audio/). "
             "Default: every channel in the session.",
    )
    ap.add_argument(
        "--language",
        default=None,
        help="Override the session language hint (affects Canary and WER "
             "number normalisation).",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Sub-directory under runs/. Default: <session>_<timestamp>.",
    )
    ap.add_argument(
        "--in-process",
        action="store_true",
        help="Run models in this process instead of one subprocess each. "
             "Faster for a single model, but peak-RSS figures become "
             "meaningless from the second model onwards (see --worker).",
    )
    ap.add_argument(
        "--worker",
        nargs=2,
        metavar=("MODEL", "CHANNEL"),
        default=None,
        help=argparse.SUPPRESS,  # internal: run one cell, emit JSON
    )
    args = ap.parse_args()

    if args.worker:
        return worker_main(args)

    session_dir: Path = args.session.expanduser().resolve()
    try:
        session = load_session(session_dir)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    language = args.language or session.language
    run_name = args.run_name or f"{session_dir.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = Path(__file__).resolve().parent / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    selected = args.channels or list(session.channels)
    unknown = [c for c in selected if c not in session.channels]
    if unknown:
        console.print(
            f"[red]Unknown channel(s): {', '.join(unknown)}. "
            f"Available: {', '.join(session.channels)}[/red]"
        )
        return 1

    console.print(
        f"[bold]Session[/bold] {session_dir.name}  "
        f"(lang={language}, channels={', '.join(selected)}, "
        f"reference={'yes' if session.has_reference else 'no'})"
    )

    models = list(args.models)
    if not args.ignore_language_support:
        skipped = [m for m in models if not supports_language(m, language)]
        models = [m for m in models if supports_language(m, language)]
        for m in skipped:
            console.print(
                f"[yellow]skip {m}: no support for '{language}' "
                f"(--ignore-language-support to run anyway)[/yellow]"
            )
    if not models:
        console.print(f"[red]No selected model supports '{language}'.[/red]")
        return 1

    # Audio is loaded inside each cell rather than once up front: with
    # subprocess isolation the parent never touches it, and a 16 kHz decode
    # is negligible next to model start-up.
    results: list[ModelRun] = []

    for model_key in models:
        for ch in selected:
            console.print(f"\n[bold magenta]▶ {model_key} / {ch}[/bold magenta]")
            if args.in_process:
                run = bench_cell(session, model_key, ch, language, run_dir)
            else:
                run = bench_cell_subprocess(session, model_key, ch, language, run_dir)
            results.append(run)

            if run.error:
                console.print(f"[red]  {run.error}[/red]")
            else:
                acc = run.accuracy
                wer_note = f" · WER {acc['wer']:.1%}" if acc else ""
                console.print(
                    f"  [green]done[/green] in {run.wall_seconds:.1f}s  "
                    f"(RTF {run.rtf:.2f} · "
                    f"{len(run.text.split())} words · "
                    f"peak RSS {run.peak_rss_mb:.0f} MB{wer_note})"
                )

    # Summary table — WER columns only appear when the session has ground
    # truth, so recorded sessions don't get a wall of dashes.
    console.print()
    table = Table(title=f"Bench summary — run {run_name}")
    table.add_column("Model")
    table.add_column("Channel")
    table.add_column("Audio", justify="right")
    table.add_column("Wall", justify="right")
    table.add_column("RTF", justify="right")
    if session.has_reference:
        table.add_column("WER", justify="right")
        table.add_column("CER", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Peak RSS", justify="right")
    table.add_column("Error")

    # Best WER first: the ranking is the point of the whole exercise.
    ordered = sorted(
        results,
        key=lambda r: (r.channel, r.wer if r.wer is not None else float("inf")),
    ) if session.has_reference else results

    for r in ordered:
        row = [
            r.model_id,
            r.channel,
            f"{r.audio_seconds:.0f}s",
            f"{r.wall_seconds:.1f}s",
            f"{r.rtf:.2f}",
        ]
        if session.has_reference:
            row += [
                f"{r.accuracy['wer']:.1%}" if r.accuracy else "—",
                f"{r.accuracy['cer']:.1%}" if r.accuracy else "—",
            ]
        row += [
            str(len((r.text or "").split())),
            f"{r.peak_rss_mb:.0f} MB",
            r.error or "",
        ]
        table.add_row(*row)
    console.print(table)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps([r.to_json() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Snapshot what was benchmarked so a run directory stays interpretable
    # after the session is regenerated or deleted.
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "session": str(session_dir),
                "language": language,
                "models": models,
                "model_labels": {
                    m: MODEL_REGISTRY[m]["label"] for m in models
                },
                "channels": selected,
                "has_reference": session.has_reference,
                "session_manifest": session.manifest,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    console.print(f"\nOutputs in [bold]{run_dir}[/bold]")
    console.print(f"Report:  uv run python compare.py --run-name {run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
