"""
diarize.py — speaker diarisation ("who spoke when") on a session, scored
against the synthetic ground truth.

The synthetic sessions already know who says what and when, so
diarisation can be measured the same way transcription is: run a system,
compare to `reference.json`, report DER.

Two backends, both from NeMo:

  * `sortformer` — nvidia/diar_sortformer_4spk-v1, an end-to-end
    diarisation model (2025). One forward pass produces speaker-labelled
    time spans; no VAD, embeddings or clustering to tune. Hard limit of
    4 speakers.
  * `titanet`    — the classic pipeline: energy VAD → TitaNet speaker
    embeddings on sliding windows → agglomerative clustering. No speaker
    limit, and it can be told the true speaker count, but every stage is
    a hyperparameter you can accidentally tune to your own test set.

Both are worth having: `sortformer` shows what current end-to-end models
do out of the box, `titanet` is the baseline that has to be beaten, and
disagreement between them on the same audio is a useful signal that a
session is genuinely hard rather than that one model is broken.

Usage:

    uv run python diarize.py --session sessions/standup-de__clean
    uv run python diarize.py --session sessions/standup-de__phone \\
        --backends sortformer titanet --num-speakers 3

Writes `runs/<run-name>/diarization/<backend>.<channel>.json` with the
hypothesis turns and the DER breakdown.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from envfile import load_env

load_env()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from audio_io import TARGET_SR, load_mono_16k  # noqa: E402
from score import diarization_error_rate  # noqa: E402

console = Console()

# Sliding window for the embedding pipeline. 1.5 s is the usual
# compromise: long enough for a stable speaker embedding, short enough
# that a window rarely straddles a speaker change.
WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.75
# Turns shorter than this are dropped — below ~0.3 s a "turn" is almost
# always a clustering artefact rather than someone speaking.
MIN_TURN_SECONDS = 0.3
SORTFORMER_MAX_SPEAKERS = 4


@dataclass
class Turn:
    speaker: str
    start: float
    end: float

    def to_json(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }


@dataclass
class DiarizationRun:
    backend: str
    channel: str
    label: str
    audio_seconds: float
    wall_seconds: float
    peak_rss_mb: float
    speakers_found: int
    turns: list[Turn]
    accuracy: dict | None
    error: str | None = None

    @property
    def der(self) -> float | None:
        return self.accuracy["der"] if self.accuracy else None


# ──────────────────────────────────────────────
# Voice activity detection
# ──────────────────────────────────────────────


def energy_vad(
    audio: np.ndarray,
    frame_seconds: float = 0.02,
    threshold_db: float = -33.0,
    min_speech: float = 0.20,
    min_silence: float = 0.20,
) -> list[tuple[float, float]]:
    """Return speech regions as (start, end) seconds.

    The threshold is relative to the loudest frame rather than absolute,
    so it survives the degradation profiles that change overall level
    (`farfield` is deliberately quiet). Short gaps are bridged and short
    islands dropped, because a raw per-frame decision produces hundreds
    of 20 ms fragments that mean nothing on their own."""
    frame = max(1, int(frame_seconds * TARGET_SR))
    usable = (len(audio) // frame) * frame
    if usable == 0:
        return []
    frames = np.abs(audio[:usable]).reshape(-1, frame).mean(axis=1)
    peak = float(frames.max())
    if peak <= 0:
        return []
    db = 20 * np.log10(np.maximum(frames, 1e-10) / peak)
    active = db > threshold_db

    regions: list[list[float]] = []
    for i, is_speech in enumerate(active):
        if not is_speech:
            continue
        start, end = i * frame_seconds, (i + 1) * frame_seconds
        if regions and start - regions[-1][1] <= min_silence:
            regions[-1][1] = end
        else:
            regions.append([start, end])
    return [(s, e) for s, e in regions if e - s >= min_speech]


# ──────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────


def cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    normed = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9
    )
    return np.clip(1.0 - normed @ normed.T, 0.0, 2.0)


def agglomerative(
    distances: np.ndarray,
    num_clusters: int | None,
    threshold: float = 0.55,
) -> list[int]:
    """Average-linkage agglomerative clustering on a distance matrix.

    Implemented here rather than pulled from scikit-learn so diarisation
    has the same "runs offline, no extra dependency" property as the WER
    scoring. n is the number of windows in a session — a few hundred —
    so the naive O(n^3) merge loop is irrelevant next to the cost of
    computing the embeddings in the first place.

    With `num_clusters` set, merging stops at that many clusters; without
    it, merging stops when the closest pair is further apart than
    `threshold`, which is what decides the speaker count."""
    n = distances.shape[0]
    if n == 0:
        return []
    clusters = {i: [i] for i in range(n)}

    while len(clusters) > 1:
        if num_clusters is not None and len(clusters) <= num_clusters:
            break

        keys = list(clusters)
        best = None
        for a_idx in range(len(keys)):
            for b_idx in range(a_idx + 1, len(keys)):
                a, b = keys[a_idx], keys[b_idx]
                d = float(
                    distances[np.ix_(clusters[a], clusters[b])].mean()
                )
                if best is None or d < best[0]:
                    best = (d, a, b)

        if best is None:
            break
        distance, a, b = best
        if num_clusters is None and distance > threshold:
            break
        clusters[a] = clusters[a] + clusters.pop(b)

    labels = [0] * n
    for label, members in enumerate(clusters.values()):
        for i in members:
            labels[i] = label
    return labels


def windows_to_turns(
    windows: list[tuple[float, float]], labels: list[int]
) -> list[Turn]:
    """Merge consecutive same-speaker windows into turns."""
    turns: list[Turn] = []
    for (start, end), label in zip(windows, labels):
        name = f"spk{label}"
        if turns and turns[-1].speaker == name and start <= turns[-1].end + 0.01:
            turns[-1].end = max(turns[-1].end, end)
        else:
            turns.append(Turn(name, start, end))
    return [t for t in turns if t.end - t.start >= MIN_TURN_SECONDS]


# ──────────────────────────────────────────────
# Backends
# ──────────────────────────────────────────────


def run_titanet(
    audio: np.ndarray,
    num_speakers: int | None = None,
    model_id: str = "titanet_large",
) -> list[Turn]:
    """VAD → TitaNet embeddings → clustering."""
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    import torch

    regions = energy_vad(audio)
    if not regions:
        return []

    # Slide a window across each speech region. Windows are clipped to
    # the region so a window never spans a silence and picks up two
    # speakers either side of it.
    windows: list[tuple[float, float]] = []
    for start, end in regions:
        if end - start <= WINDOW_SECONDS:
            windows.append((start, end))
            continue
        t = start
        while t + WINDOW_SECONDS <= end:
            windows.append((t, t + WINDOW_SECONDS))
            t += HOP_SECONDS
        if end - t > MIN_TURN_SECONDS:
            windows.append((max(start, end - WINDOW_SECONDS), end))

    model = EncDecSpeakerLabelModel.from_pretrained(model_id)
    model.eval()
    try:
        vectors = []
        with torch.no_grad():
            for start, end in windows:
                chunk = audio[int(start * TARGET_SR):int(end * TARGET_SR)]
                signal = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
                length = torch.tensor([signal.shape[1]])
                _logits, emb = model.forward(
                    input_signal=signal, input_signal_length=length
                )
                vectors.append(emb.squeeze().cpu().numpy())
    finally:
        del model
        gc.collect()

    if not vectors:
        return []

    embeddings = np.vstack(vectors)
    labels = agglomerative(cosine_distances(embeddings), num_speakers)
    return windows_to_turns(windows, labels)


def run_sortformer(
    audio: np.ndarray,
    model_id: str = "nvidia/diar_sortformer_4spk-v1",
) -> list[Turn]:
    """End-to-end diarisation: audio in, speaker-labelled spans out."""
    from nemo.collections.asr.models import SortformerEncLabelModel
    import tempfile

    model = SortformerEncLabelModel.from_pretrained(model_id)
    model.eval()
    tmpdir = Path(tempfile.mkdtemp(prefix="diar-"))
    wav_path = tmpdir / "input.wav"
    sf.write(str(wav_path), audio, TARGET_SR, subtype="PCM_16")
    try:
        predictions = model.diarize(audio=[str(wav_path)], batch_size=1)
    finally:
        wav_path.unlink(missing_ok=True)
        tmpdir.rmdir()
        del model
        gc.collect()

    # NeMo returns one entry per input file, each a list of
    # "start end speaker_id" strings.
    entries = predictions[0] if predictions else []
    turns: list[Turn] = []
    for entry in entries:
        parts = str(entry).split()
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        # NeMo labels them "speaker_0"; normalise to the "spk0" form the
        # clustering backend uses so both are comparable at a glance.
        label = parts[2].removeprefix("speaker_")
        turns.append(Turn(f"spk{label}", start, end))
    return [t for t in turns if t.end - t.start >= MIN_TURN_SECONDS]


BACKENDS: dict[str, dict] = {
    "sortformer": {
        "label": "NVIDIA Sortformer 4-spk v1 (end-to-end, 2025)",
        "runner": run_sortformer,
        "max_speakers": SORTFORMER_MAX_SPEAKERS,
        "uses_speaker_count": False,
    },
    "titanet": {
        "label": "TitaNet-Large embeddings + agglomerative clustering",
        "runner": run_titanet,
        "max_speakers": None,
        "uses_speaker_count": True,
    },
}


# ──────────────────────────────────────────────
# Session plumbing
# ──────────────────────────────────────────────


def load_reference_turns(session_dir: Path, channel: str) -> list[dict] | None:
    """Speaker-labelled ground-truth turns for one channel."""
    path = session_dir / "reference.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("channels", {}).get(channel)
    if not segments:
        return None
    return [
        {"speaker": s["speaker"], "start": s["start"], "end": s["end"]}
        for s in segments
    ]


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def diarize_channel(
    backend: str,
    session_dir: Path,
    channel: str,
    num_speakers: int | None,
) -> DiarizationRun:
    cfg = BACKENDS[backend]
    audio = load_mono_16k(session_dir / "audio" / f"{channel}.wav")
    reference = load_reference_turns(session_dir, channel)

    started = time.perf_counter()
    error = None
    turns: list[Turn] = []
    try:
        if cfg["uses_speaker_count"]:
            turns = cfg["runner"](audio, num_speakers)
        else:
            turns = cfg["runner"](audio)
    except Exception as exc:  # one broken runtime must not kill the run
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started

    accuracy = None
    if reference and turns:
        accuracy = diarization_error_rate(
            reference, [t.to_json() for t in turns]
        )

    return DiarizationRun(
        backend=backend,
        channel=channel,
        label=cfg["label"],
        audio_seconds=round(len(audio) / TARGET_SR, 2),
        wall_seconds=round(wall, 2),
        peak_rss_mb=round(peak_rss_mb(), 1),
        speakers_found=len({t.speaker for t in turns}),
        turns=turns,
        accuracy=accuracy,
        error=error,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument(
        "--backends", nargs="+", default=["sortformer", "titanet"],
        choices=sorted(BACKENDS),
    )
    ap.add_argument(
        "--channels", nargs="+", default=["mixed"],
        help="Channels to diarise. Only the mixed channel is a real "
             "diarisation task — the isolated per-speaker tracks contain "
             "one voice each and exist as a sanity check.",
    )
    ap.add_argument(
        "--num-speakers", type=int, default=None,
        help="Speaker count hint for backends that accept one. Default: "
             "read from session.json. Pass 0 to force auto-detection, "
             "which is the harder and more realistic setting.",
    )
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()

    session_dir = args.session
    if not session_dir.exists():
        console.print(f"[red]No such session: {session_dir}[/red]")
        return 1

    manifest = {}
    manifest_path = session_dir / "session.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.num_speakers is None:
        num_speakers = len(manifest.get("speakers", [])) or None
    elif args.num_speakers <= 0:
        num_speakers = None
    else:
        num_speakers = args.num_speakers

    run_name = args.run_name or (
        f"{session_dir.name}_diar_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    out_dir = args.runs_dir / run_name / "diarization"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Session:[/bold] {session_dir}")
    console.print(
        f"[bold]Speakers:[/bold] "
        f"{num_speakers if num_speakers else 'auto-detect'}"
    )

    runs: list[DiarizationRun] = []
    for channel in args.channels:
        if not (session_dir / "audio" / f"{channel}.wav").exists():
            console.print(f"[yellow]skip {channel}: no audio[/yellow]")
            continue
        for backend in args.backends:
            limit = BACKENDS[backend]["max_speakers"]
            if limit and num_speakers and num_speakers > limit:
                console.print(
                    f"[yellow]skip {backend} on {channel}: session has "
                    f"{num_speakers} speakers, model handles {limit}[/yellow]"
                )
                continue
            console.print(f"[cyan]{backend} · {channel}[/cyan]")
            run = diarize_channel(backend, session_dir, channel, num_speakers)
            runs.append(run)

            payload = asdict(run)
            payload["turns"] = [t.to_json() for t in run.turns]
            (out_dir / f"{backend}.{channel}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            if run.error:
                console.print(f"  [red]{run.error}[/red]")
            else:
                der = f"{run.der * 100:.1f}%" if run.der is not None else "n/a"
                console.print(
                    f"  done in {run.wall_seconds:.1f}s  "
                    f"({len(run.turns)} turns · {run.speakers_found} speakers "
                    f"· DER {der})"
                )

    if not runs:
        console.print("[red]Nothing ran.[/red]")
        return 1

    table = Table(title="Diarisation", show_lines=False)
    for column in (
        "Backend", "Channel", "Wall", "Speakers", "DER",
        "Miss", "False alarm", "Confusion", "Error",
    ):
        table.add_column(column)
    for run in sorted(
        runs, key=lambda r: (r.der if r.der is not None else 9.9)
    ):
        acc = run.accuracy or {}
        pct = lambda key: (  # noqa: E731
            f"{acc[key] * 100:.1f}%" if key in acc else "—"
        )
        table.add_row(
            run.backend,
            run.channel,
            f"{run.wall_seconds:.1f}s",
            f"{run.speakers_found}",
            pct("der"),
            pct("miss"),
            pct("false_alarm"),
            pct("confusion"),
            (run.error or "")[:40],
        )
    console.print(table)

    summary = {
        "session": str(session_dir),
        "num_speakers_hint": num_speakers,
        "runs": [
            {
                k: v for k, v in asdict(run).items() if k != "turns"
            }
            for run in runs
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"\n[green]Wrote {out_dir}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
