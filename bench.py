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
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Load .env (or fall back to .env.example) before anything that might
# touch HF / NeMo / Torch caches. `.env` is per-machine and gitignored;
# `.env.example` is the committed template.
_repo_dir = Path(__file__).resolve().parent
_env_file = _repo_dir / ".env"
if not _env_file.exists():
    _example = _repo_dir / ".env.example"
    if _example.exists():
        print(
            f"[asr-bench] no .env found — falling back to {_example.name}. "
            f"Copy it to .env and edit paths for your machine.",
            file=sys.stderr,
        )
        _env_file = _example
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

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


def run_faster_whisper(audio: np.ndarray, model_name: str = "large-v3") -> str:
    from faster_whisper import WhisperModel
    # int8 for Apple Silicon — fast enough, small footprint. float16 is
    # not supported on CPU; the Metal backend is experimental in CT2.
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,  # trims long silence → fewer hallucinations
        language=None,    # auto-detect
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


MODEL_REGISTRY: dict[str, dict] = {
    "parakeet-live": {
        # Reads the recording app's own transcript.live.jsonl — the Parakeet v3
        # output the user actually sees in production. Avoids having to
        # reproduce the FluidAudio / Swift pipeline in Python (NeMo's
        # EncDecRNNTBPE wrapper for parakeet-tdt-0.6b-v3 multilingual
        # decoded to 0 tokens on our test audio). This is also a
        # "truer" comparison target: it's literally the output the
        # user is comparing Whisper / Canary against.
        "kind": "live_transcript",
        "label": "Live Parakeet-TDT v3 (as produced by the recording app)",
    },
    "canary": {
        "kind": "nemo",
        "nemo_id": "nvidia/canary-1b-flash",
        "label": "NVIDIA Canary-1B-Flash (multilingual: en/de/fr/es)",
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
    "whisper-large-v3": {
        "kind": "whisper",
        "fw_id": "large-v3",
        "label": "OpenAI Whisper Large-v3 (via faster-whisper)",
    },
}


def run_live_transcript(session_dir: Path, channel: str) -> str:
    """Extract the Parakeet-v3 output from the recording app's own
    `transcript.live.jsonl` and split it by speaker so we can line it up
    against model output for the `mic` vs `sys` channels separately.

    `.you` speaker records map to the mic channel, everything else
    (`.them` / `.remote(…)`) maps to sys. If the preferred file name is
    missing we fall back to the `.pre-cleanup.bak` copy the app writes
    at finalize time."""
    import json

    candidates = [
        session_dir / "transcript.live.jsonl",
        session_dir / "transcript.live.jsonl.pre-cleanup.bak",
    ]
    jsonl_path = next((p for p in candidates if p.exists()), None)
    if jsonl_path is None:
        raise FileNotFoundError(
            f"No transcript.live.jsonl (or .bak) under {session_dir}"
        )

    mic_parts: list[str] = []
    sys_parts: list[str] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        speaker = rec.get("speaker")
        # Speaker is serialised as either "you" or a dict like
        # {"them": null} / {"remote": 1} depending on app version.
        if speaker == "you" or (isinstance(speaker, dict) and "you" in speaker):
            is_mic = True
        else:
            is_mic = False
        # Prefer the cleaned / refined text (`refinedText`) over the raw
        # `text` — that's what the UI shows.
        text = rec.get("refinedText") or rec.get("text") or ""
        text = text.strip()
        if not text:
            continue
        (mic_parts if is_mic else sys_parts).append(text)

    return " ".join(mic_parts if channel == "mic" else sys_parts).strip()


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
    session_dir: Path,
    channel: str,
    language: str = "en",
) -> tuple[str, str | None]:
    """Dispatch on the registry. Returns (text, error). Any exception is
    caught so one broken runtime doesn't kill the rest of the matrix."""
    cfg = MODEL_REGISTRY[model_key]
    try:
        if cfg["kind"] == "whisper":
            return run_faster_whisper(audio, cfg["fw_id"]), None
        elif cfg["kind"] == "nemo":
            return run_nemo_asr(
                audio,
                cfg["nemo_id"],
                extra_transcribe_kwargs=resolve_kwargs(
                    cfg.get("nemo_transcribe_kwargs"), language
                ),
                chunk_seconds=cfg.get("chunk_seconds", 30.0),
            ), None
        elif cfg["kind"] == "live_transcript":
            return run_live_transcript(session_dir, channel), None
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
        default=[k for k in MODEL_REGISTRY if k != "parakeet-live"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to run. Default: all real ASR models "
             "(`parakeet-live` only applies to app-recorded sessions).",
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
    args = ap.parse_args()

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

    # Pre-load every channel once so we amortise the decode cost across
    # all models in the matrix.
    channel_audio: dict[str, np.ndarray] = {}
    for ch in selected:
        path = session.channels[ch]
        console.print(f"[cyan]Loading[/cyan] {path.name} ({path.stat().st_size / 1e6:.1f} MB)…")
        channel_audio[ch] = load_mono_16k(path)
        console.print(f"  → {len(channel_audio[ch]) / TARGET_SR:.1f}s at 16 kHz mono")

    results: list[ModelRun] = []

    for model_key in args.models:
        for ch, audio in channel_audio.items():
            model_dir = run_dir / model_key
            model_dir.mkdir(exist_ok=True)
            out_path = model_dir / f"{ch}.txt"
            metrics_path = model_dir / f"{ch}.metrics.json"

            console.print(f"\n[bold magenta]▶ {model_key} / {ch}[/bold magenta]")
            start = time.perf_counter()
            rss_before = peak_rss_mb()
            text, err = run_model(model_key, audio, session_dir, ch, language)
            wall = time.perf_counter() - start
            rss_after = peak_rss_mb()

            audio_seconds = len(audio) / TARGET_SR
            accuracy = None
            if ch in session.references and not err:
                accuracy = score_transcript(text, session.references[ch], language)

            run = ModelRun(
                model_id=model_key,
                channel=ch,
                audio_seconds=audio_seconds,
                wall_seconds=wall,
                rtf=wall / audio_seconds if audio_seconds > 0 else 0.0,
                peak_rss_mb=max(rss_before, rss_after),
                text=text,
                error=err,
                accuracy=accuracy,
            )
            results.append(run)

            out_path.write_text(text or "", encoding="utf-8")
            metrics_path.write_text(
                json.dumps(run.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
            )

            if err:
                console.print(f"[red]  {err}[/red]")
            else:
                wer_note = f" · WER {accuracy['wer']:.1%}" if accuracy else ""
                console.print(
                    f"  [green]done[/green] in {wall:.1f}s  "
                    f"(RTF {run.rtf:.2f} · "
                    f"{len(text.split())} words · "
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
                "models": list(args.models),
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
