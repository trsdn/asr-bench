"""
diarize.py — speaker diarisation ("who spoke when") on a session, scored
against the synthetic ground truth.

The synthetic sessions already know who says what and when, so
diarisation can be measured the same way transcription is: run a system,
compare to `reference.json`, report DER.

Three backends, all from NeMo:

  * `sortformer` — nvidia/diar_sortformer_4spk-v1, an end-to-end
    diarisation model (2025). One forward pass produces speaker-labelled
    time spans; no VAD, embeddings or clustering to tune. Hard limit of
    4 speakers.
  * `sortformer-streaming` — nvidia/diar_streaming_sortformer_4spk-v2,
    the streaming successor. Same interface and same 4-speaker limit, a
    fraction of the cost, and it does not rank the same way as v1: on
    these sessions it is steadier across recordings but weaker on heavily
    overlapping speech.
  * `titanet`    — the classic pipeline: energy VAD → TitaNet speaker
    embeddings on sliding windows → agglomerative clustering. No speaker
    limit, and it can be told the true speaker count, but every stage is
    a hyperparameter you can accidentally tune to your own test set.

All three are worth having: the Sortformer pair shows what current
end-to-end models do out of the box, `titanet` is the baseline that has
to be beaten, and disagreement between them on the same audio is a useful
signal that a session is genuinely hard rather than that one model is
broken. Only the end-to-end models can report two people speaking at
once; clustering assigns one label per window and cannot.

Usage:

    uv run python diarize.py --session sessions/standup-de__clean
    uv run python diarize.py --session sessions/standup-de__phone \\
        --backends sortformer titanet --num-speakers 3

Writes `runs/<run-name>/diarization/<backend>.<channel>.json` with the
hypothesis turns and the DER breakdown.
"""

from __future__ import annotations

import argparse
import functools
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

# ──────────────────────────────────────────────
# Tunable defaults
# ──────────────────────────────────────────────
#
# Only the `titanet` pipeline has knobs — `sortformer` is end-to-end and
# takes nothing but audio. Every value here is exposed on the CLI, and
# `--sweep` searches them, so these are starting points rather than
# settled truths. See the Tuning section of the README for what each one
# actually buys on the bundled sessions.

# Sliding window for the embedding pipeline. 1.5 s is the usual
# compromise: long enough for a stable speaker embedding, short enough
# that a window rarely straddles a speaker change.
WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.75
# Turns shorter than this are dropped — below ~0.3 s a "turn" is almost
# always a clustering artefact rather than someone speaking.
MIN_TURN_SECONDS = 0.3
# Only consulted when the speaker count is unknown: merging stops once
# the closest pair of clusters is further apart than this.
CLUSTER_THRESHOLD = 0.55
# Energy VAD threshold, relative to the loudest frame in the recording.
# -40 rather than a tighter value because a multi-session sweep showed it
# is the one parameter here that pays: it recovers quiet speech that -33
# clips off, worth ~0.7 DER points on average, and never hurt a session.
VAD_THRESHOLD_DB = -40.0
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


_TITANET_CACHE: dict[str, object] = {}


def _load_titanet(model_id: str, cache: bool):
    """Load TitaNet, optionally keeping it around.

    A sweep evaluates dozens of parameter sets against the same audio;
    reloading a 100 MB model each time would dominate the wall clock and
    measure disk rather than the parameters. Normal runs still drop the
    model afterwards so a long matrix doesn't accumulate memory."""
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    if cache and model_id in _TITANET_CACHE:
        return _TITANET_CACHE[model_id]
    model = EncDecSpeakerLabelModel.from_pretrained(model_id)
    model.eval()
    if cache:
        _TITANET_CACHE[model_id] = model
    return model


def windows_to_turns(
    windows: list[tuple[float, float]],
    labels: list[int],
    min_turn: float = MIN_TURN_SECONDS,
) -> list[Turn]:
    """Merge consecutive same-speaker windows into turns."""
    turns: list[Turn] = []
    for (start, end), label in zip(windows, labels):
        name = f"spk{label}"
        if turns and turns[-1].speaker == name and start <= turns[-1].end + 0.01:
            turns[-1].end = max(turns[-1].end, end)
        else:
            turns.append(Turn(name, start, end))
    return [t for t in turns if t.end - t.start >= min_turn]


# ──────────────────────────────────────────────
# Backends
# ──────────────────────────────────────────────


def run_titanet(
    audio: np.ndarray,
    num_speakers: int | None = None,
    model_id: str = "titanet_large",
    window: float = WINDOW_SECONDS,
    hop: float = HOP_SECONDS,
    threshold: float = CLUSTER_THRESHOLD,
    min_turn: float = MIN_TURN_SECONDS,
    vad_db: float = VAD_THRESHOLD_DB,
    cache_model: bool = False,
) -> list[Turn]:
    """VAD → TitaNet embeddings → clustering."""
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    import torch

    regions = energy_vad(audio, threshold_db=vad_db)
    if not regions:
        return []

    # Slide a window across each speech region. Windows are clipped to
    # the region so a window never spans a silence and picks up two
    # speakers either side of it.
    windows: list[tuple[float, float]] = []
    for start, end in regions:
        if end - start <= window:
            windows.append((start, end))
            continue
        t = start
        while t + window <= end:
            windows.append((t, t + window))
            t += hop
        if end - t > min_turn:
            windows.append((max(start, end - window), end))

    model = _load_titanet(model_id, cache=cache_model)
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
        if not cache_model:
            del model
            gc.collect()

    if not vectors:
        return []

    embeddings = np.vstack(vectors)
    labels = agglomerative(cosine_distances(embeddings), num_speakers, threshold)
    return windows_to_turns(windows, labels, min_turn)


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
    "sortformer-streaming": {
        "label": "NVIDIA Streaming Sortformer 4-spk v2 (end-to-end, 2025)",
        "runner": functools.partial(
            run_sortformer, model_id="nvidia/diar_streaming_sortformer_4spk-v2"
        ),
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
    params: dict | None = None,
    audio: np.ndarray | None = None,
) -> DiarizationRun:
    cfg = BACKENDS[backend]
    if audio is None:
        audio = load_mono_16k(session_dir / "audio" / f"{channel}.wav")
    reference = load_reference_turns(session_dir, channel)

    started = time.perf_counter()
    error = None
    turns: list[Turn] = []
    try:
        if cfg["uses_speaker_count"]:
            turns = cfg["runner"](audio, num_speakers, **(params or {}))
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


SWEEP_GRID = {
    "window": [0.75, 1.0, 1.5, 2.0, 3.0],
    "hop": [0.25, 0.5, 0.75],
    "min_turn": [0.2, 0.3, 0.5],
    "vad_db": [-40.0, -33.0, -25.0],
    # Only meaningful when the speaker count is unknown; skipped otherwise.
    "threshold": [0.4, 0.5, 0.55, 0.6, 0.7, 0.8],
}


def run_sweep(
    session_dirs: list[Path],
    channel: str,
    num_speakers: int | None,
    base: dict,
) -> int:
    """Coordinate-descent search over the titanet parameters.

    Scores every configuration on *all* the given sessions and optimises
    the mean DER. Tuning against a single session does not generalise:
    the values this search picks on `standup-de__clean` alone score 3.6 %
    there and 58.8 % on `standup-de__phone`, because a setting that
    happens to split one recording's speakers cleanly can shatter another
    into thirty clusters. Pass several degradation profiles.

    A full grid over five parameters is hundreds of configurations per
    session. Coordinate descent sweeps one parameter at a time, keeping
    the best value so far, covering the same axes in a fraction of the
    runs. It can settle in a local optimum, but for parameters this
    loosely coupled that is a fair trade — and every value tried is
    printed, so the shape of each axis stays visible."""
    loaded = []
    for session_dir in session_dirs:
        reference = load_reference_turns(session_dir, channel)
        if not reference:
            console.print(f"[yellow]skip {session_dir.name}: no reference[/yellow]")
            continue
        loaded.append(
            (
                session_dir.name,
                load_mono_16k(session_dir / "audio" / f"{channel}.wav"),
                reference,
            )
        )
    if not loaded:
        console.print("[red]Sweep needs at least one session with a reference.[/red]")
        return 1

    current = dict(base)
    current["cache_model"] = True
    history: list[tuple[dict, float, list[dict]]] = []

    def evaluate(params: dict) -> tuple[float, list[dict]] | None:
        """Mean DER across sessions. A session that yields no turns counts
        as a total failure rather than being skipped, so a configuration
        cannot win by collapsing on the sessions it finds hard."""
        per_session = []
        for _name, audio, reference in loaded:
            turns = run_titanet(audio, num_speakers, **params)
            metrics = (
                diarization_error_rate(reference, [t.to_json() for t in turns])
                if turns
                else None
            )
            per_session.append(metrics or {"der": 1.0, "miss": 1.0,
                                           "false_alarm": 0.0, "confusion": 0.0})
        return sum(m["der"] for m in per_session) / len(per_session), per_session

    console.print(
        f"[bold]Sweep[/bold] {channel} · "
        f"{num_speakers if num_speakers else 'auto-detect'} speakers · "
        f"{len(loaded)} session(s): {', '.join(n for n, _, _ in loaded)}"
    )

    baseline_der, baseline_per = evaluate(current)
    history.append((dict(current), baseline_der, baseline_per))
    console.print(f"baseline mean DER {baseline_der:.1%}")

    for name, values in SWEEP_GRID.items():
        if name == "threshold" and num_speakers is not None:
            console.print(
                "[dim]skipping threshold: it only decides the speaker count, "
                "and the count is given (--num-speakers 0 to search it)[/dim]"
            )
            continue
        best_value, best_der = current[name], None
        for value in values:
            trial = dict(current)
            trial[name] = value
            mean_der, per_session = evaluate(trial)
            history.append((trial, mean_der, per_session))
            marker = ""
            if best_der is None or mean_der < best_der:
                best_der, best_value, marker = mean_der, value, " ←"
            spread = " ".join(f"{m['der']:.0%}" for m in per_session)
            console.print(
                f"  {name}={value}: mean DER {mean_der:.1%}  [{spread}]{marker}"
            )
        current[name] = best_value
        console.print(f"[green]→ {name}={best_value}[/green]")

    best_params, best_der, best_per = min(history, key=lambda h: h[1])
    table = Table(title="Sweep result")
    table.add_column("Parameter")
    table.add_column("Default", justify="right")
    table.add_column("Best", justify="right")
    for name in SWEEP_GRID:
        if name == "threshold" and num_speakers is not None:
            continue
        table.add_row(name, str(base.get(name)), str(best_params.get(name)))
    console.print()
    console.print(table)

    per_table = Table(title="Per-session DER at the chosen values")
    per_table.add_column("Session")
    per_table.add_column("Baseline", justify="right")
    per_table.add_column("Tuned", justify="right")
    for (name, _, _), before, after in zip(loaded, baseline_per, best_per):
        per_table.add_row(name, f"{before['der']:.1%}", f"{after['der']:.1%}")
    console.print(per_table)
    console.print(
        f"[bold]mean DER {baseline_der:.1%} → {best_der:.1%}[/bold]"
    )
    console.print(
        "[dim]Tuned on the sessions listed above. A gain that appears on one "
        "session and not the others is noise, not a better setting.[/dim]"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument(
        "--backends", nargs="+",
        default=["sortformer", "sortformer-streaming", "titanet"],
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

    tune = ap.add_argument_group(
        "titanet tuning",
        "Only affects the `titanet` pipeline; `sortformer` is end-to-end "
        "and takes no parameters.",
    )
    tune.add_argument(
        "--window", type=float, default=WINDOW_SECONDS,
        help=f"Embedding window in seconds (default {WINDOW_SECONDS}). "
             "Shorter resolves speaker changes more precisely but gives "
             "the embedder less evidence per decision.",
    )
    tune.add_argument(
        "--hop", type=float, default=HOP_SECONDS,
        help=f"Window hop in seconds (default {HOP_SECONDS}). Smaller is "
             "finer-grained and proportionally slower.",
    )
    tune.add_argument(
        "--cluster-threshold", type=float, default=CLUSTER_THRESHOLD,
        help=f"Stop merging clusters beyond this cosine distance "
             f"(default {CLUSTER_THRESHOLD}). Only consulted when the "
             "speaker count is unknown — with --num-speakers set, the "
             "count decides when merging stops and this is ignored.",
    )
    tune.add_argument(
        "--min-turn", type=float, default=MIN_TURN_SECONDS,
        help=f"Drop turns shorter than this (default {MIN_TURN_SECONDS}).",
    )
    tune.add_argument(
        "--vad-db", type=float, default=VAD_THRESHOLD_DB,
        help=f"VAD threshold in dB below the loudest frame "
             f"(default {VAD_THRESHOLD_DB}).",
    )
    tune.add_argument(
        "--sweep", action="store_true",
        help="Grid-search the tuning parameters against the reference and "
             "print a ranking instead of writing a run. Loads the model "
             "once and reuses it. Give several sessions via "
             "--sweep-sessions — tuning on one does not generalise.",
    )
    tune.add_argument(
        "--sweep-sessions", nargs="+", type=Path, default=[],
        help="Extra sessions to include in the sweep, scored alongside "
             "--session. The search optimises the mean DER across all of "
             "them, which is the only way to tell a real improvement from "
             "one that fits a single recording.",
    )
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

    titanet_params = {
        "window": args.window,
        "hop": args.hop,
        "threshold": args.cluster_threshold,
        "min_turn": args.min_turn,
        "vad_db": args.vad_db,
    }

    if args.sweep:
        sweep_base = dict(titanet_params)
        return run_sweep(
            [session_dir, *args.sweep_sessions],
            args.channels[0],
            num_speakers,
            sweep_base,
        )

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
            run = diarize_channel(
                backend, session_dir, channel, num_speakers,
                params=titanet_params if BACKENDS[backend]["uses_speaker_count"] else None,
            )
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
