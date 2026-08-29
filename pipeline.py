"""
pipeline.py — run a full speaker-attributed transcription pipeline and score it.

    uv run python pipeline.py --session sessions/crosstalk-de__piper__clean \
        --diarizer sortformer --asr parakeet-tdt-v3 --num-speakers 4

`bench.py` answers "which model transcribes best" and `diarize.py` answers
"which backend separates speakers best". Neither answers the question a
meeting transcript is judged on, which is whether the right words end up
under the right person — a configuration can win both halves separately and
still produce unusable minutes. This runs the halves as one pipeline and
scores the result with cpWER.

The pipeline is diarise-first: speakers are found, then audio is
transcribed per speaker. That ordering is deliberate. The alternative --
transcribe first, then attribute words by timestamp -- needs word-level
timestamps from every backend, which not all of them expose and none of
them expose the same way. Diarise-first needs nothing from the ASR model
except that it accepts an array of samples, so all thirteen models in the
registry are usable on day one.

What the caller knows matters and is part of the configuration. In practice
you usually do know how many people are in the room and what language they
are speaking, and a backend given that information is a different algorithm
from one guessing it. `--num-speakers` and `--language` are therefore knobs
to be searched over, not fixed facts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import envfile  # noqa: F401  — loads .env before torch/NeMo import

import score
from audio_io import load_mono_16k

TARGET_SR = 16_000

# Silence inserted between two non-adjacent pieces of one speaker's audio.
# Splicing them flush together creates a hard transition the model reads as
# a plosive and sometimes transcribes as a word; a short gap costs nothing
# and removes the artefact.
JOIN_SILENCE_SECONDS = 0.25

# Speech shorter than this is not transcribed on its own. Every ASR model
# here degrades badly on sub-second input, and diarisation fragments of a
# few hundred milliseconds are usually errors rather than utterances --
# this repo's own diarisation sweep found a 0.4 s phantom cluster to be the
# single most common failure. In `speaker` mode the fragment still joins
# its speaker's stream, so nothing is discarded; only isolated decoding of
# it is skipped.
MIN_SEGMENT_SECONDS = 0.4


@dataclass
class PipelineConfig:
    """One point in the configuration space.

    Everything here is something a deployment actually chooses, and the
    point of the harness is to find which combination wins under which
    conditions rather than to assume any of them."""

    diarizer: str = "sortformer"
    asr: list[str] = field(default_factory=lambda: ["parakeet-tdt-v3"])
    attribution: str = "speaker"     # speaker | segment
    num_speakers: int | None = None  # None = let the backend decide
    language: str | None = None      # None = take it from session.json
    diarizer_params: dict = field(default_factory=dict)

    def key(self) -> str:
        n = "auto" if self.num_speakers is None else str(self.num_speakers)
        return f"{self.diarizer}+{'+'.join(self.asr)}@{self.attribution}/n={n}"


def merge_turns(turns: list[dict], max_gap: float = 0.5) -> list[dict]:
    """Merge consecutive turns from the same speaker separated by less than
    `max_gap`.

    Diarisation backends emit one turn per analysis window, so a single
    sentence arrives as a dozen fragments. Transcribing those individually
    throws away the context the model needs and multiplies the per-call
    overhead, and the gaps between them are pauses inside an utterance
    rather than turn boundaries."""
    merged: list[dict] = []
    for turn in sorted(turns, key=lambda t: float(t["start"])):
        if (
            merged
            and merged[-1]["speaker"] == turn["speaker"]
            and float(turn["start"]) - float(merged[-1]["end"]) <= max_gap
        ):
            merged[-1]["end"] = max(float(merged[-1]["end"]), float(turn["end"]))
            continue
        merged.append({
            "speaker": turn["speaker"],
            "start": float(turn["start"]),
            "end": float(turn["end"]),
        })
    return merged


def slice_audio(audio: np.ndarray, start: float, end: float,
                pad: float = 0.1) -> np.ndarray:
    lo = max(0, int((start - pad) * TARGET_SR))
    hi = min(len(audio), int((end + pad) * TARGET_SR))
    return audio[lo:hi] if hi > lo else np.zeros(0, dtype=audio.dtype)


def build_speaker_streams(
    audio: np.ndarray, turns: list[dict]
) -> dict[str, np.ndarray]:
    """Concatenate each speaker's audio into one stream, separated by short
    silences.

    This is what makes the pipeline affordable: every ASR runner in the
    registry loads its weights on each call, so decoding forty segments
    individually means forty model loads. Per speaker it is three to six.
    The cost is the artificial join between non-adjacent speech, which the
    silence gap softens but does not remove -- `--attribution segment`
    exists for when that matters more than the wall clock."""
    silence = np.zeros(int(JOIN_SILENCE_SECONDS * TARGET_SR), dtype=audio.dtype)
    streams: dict[str, list[np.ndarray]] = {}
    for turn in turns:
        piece = slice_audio(audio, turn["start"], turn["end"])
        if piece.size == 0:
            continue
        streams.setdefault(turn["speaker"], []).extend([piece, silence])
    return {
        spk: np.concatenate(parts) if parts else np.zeros(0, dtype=audio.dtype)
        for spk, parts in streams.items()
    }


def transcribe_by_speaker(
    audio: np.ndarray, turns: list[dict], model_key: str, language: str
) -> tuple[list[dict], dict]:
    """One ASR call per speaker. Returns attributed turns and per-call cost."""
    from bench import run_model

    streams = build_speaker_streams(audio, turns)
    attributed, calls, errors = [], [], []
    for speaker in sorted(streams):
        stream = streams[speaker]
        if stream.size < MIN_SEGMENT_SECONDS * TARGET_SR:
            continue
        started = time.perf_counter()
        text, error = run_model(model_key, stream, language)
        calls.append(round(time.perf_counter() - started, 2))
        if error:
            errors.append(f"{speaker}: {error}")
            continue
        attributed.append({"speaker": speaker, "text": text})
    return attributed, {"asr_calls": len(calls), "asr_seconds": sum(calls),
                        "errors": errors}


def transcribe_by_segment(
    audio: np.ndarray, turns: list[dict], model_key: str, language: str
) -> tuple[list[dict], dict]:
    """One ASR call per diarised turn. Slower by the number of turns, but
    it never splices non-adjacent speech and it keeps the timing, so the
    output can be read as a real transcript rather than only scored."""
    from bench import run_model

    attributed, calls, errors = [], [], []
    for turn in turns:
        piece = slice_audio(audio, turn["start"], turn["end"])
        if piece.size < MIN_SEGMENT_SECONDS * TARGET_SR:
            continue
        started = time.perf_counter()
        text, error = run_model(model_key, piece, language)
        calls.append(round(time.perf_counter() - started, 2))
        if error:
            errors.append(f"{turn['start']:.1f}s: {error}")
            continue
        if text.strip():
            attributed.append({
                "speaker": turn["speaker"],
                "start": turn["start"],
                "end": turn["end"],
                "text": text,
            })
    return attributed, {"asr_calls": len(calls), "asr_seconds": sum(calls),
                        "errors": errors}


def run_pipeline(
    session_dir: Path, config: PipelineConfig, channel: str = "mixed",
    baseline_wer: bool = True,
) -> dict:
    import diarize

    session = json.loads((session_dir / "session.json").read_text())
    language = config.language or session.get("language", "en")
    audio = load_mono_16k(session_dir / "audio" / f"{channel}.wav")

    started = time.perf_counter()
    diar = diarize.diarize_channel(
        config.diarizer, session_dir, channel,
        num_speakers=config.num_speakers,
        params=config.diarizer_params or None,
        audio=audio,
    )
    diar_seconds = round(time.perf_counter() - started, 2)

    if diar.error or not diar.turns:
        return {
            "config": config.key(),
            "error": diar.error or "diarisation produced no turns",
        }

    turns = merge_turns([t.to_json() for t in diar.turns])
    transcribe = (transcribe_by_speaker if config.attribution == "speaker"
                  else transcribe_by_segment)
    attributed, asr_cost = transcribe(audio, turns, config.asr[0], language)

    reference = json.loads((session_dir / "reference.json").read_text())
    ref_turns = reference["channels"][channel]

    cp = score.cp_wer(ref_turns, attributed, language)

    # Baseline: the same model on the undiarised audio, which is what
    # bench.py measures. Comparing cpWER against it answers the question
    # the pipeline exists to ask -- does separating speakers first help or
    # hurt the words? -- and it is not obvious either way: each speaker's
    # stream is cleaner than the mix, but splicing non-adjacent speech
    # costs context, and any word the diarizer drops is gone for good.
    baseline = None
    if baseline_wer:
        from bench import run_model

        started = time.perf_counter()
        flat_text, flat_error = run_model(config.asr[0], audio, language)
        if not flat_error:
            flat = score.score(
                flat_text, " ".join(t["text"] for t in ref_turns), language
            )
            baseline = {
                "wer": flat["wer"] if flat else None,
                "wall_seconds": round(time.perf_counter() - started, 2),
            }

    return {
        "config": config.key(),
        "session": str(session_dir),
        "channel": channel,
        "language": language,
        "diarizer": config.diarizer,
        "asr": config.asr,
        "attribution": config.attribution,
        "num_speakers_hint": config.num_speakers,
        "audio_seconds": round(len(audio) / TARGET_SR, 2),
        "diarization_seconds": diar_seconds,
        "speakers_found": diar.speakers_found,
        "reference_speakers": len({t["speaker"] for t in ref_turns}),
        "der": diar.accuracy.get("der") if diar.accuracy else None,
        "cpwer": cp["cpwer"] if cp else None,
        "baseline_wer": baseline["wer"] if baseline else None,
        "vs_baseline": (round(cp["cpwer"] - baseline["wer"], 4)
                        if cp and baseline and baseline["wer"] is not None
                        else None),
        "wall_seconds": round(diar_seconds + asr_cost["asr_seconds"], 2),
        "peak_rss_mb": diar.peak_rss_mb,
        **asr_cost,
        "cp_detail": cp,
        "transcript": attributed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--channel", default="mixed")
    ap.add_argument("--diarizer", default="sortformer")
    ap.add_argument("--asr", default="parakeet-tdt-v3",
                    help="comma-separated; only the first is used for now")
    ap.add_argument("--attribution", choices=["speaker", "segment"],
                    default="speaker")
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="tell the diarizer how many people are present; "
                         "omit to make it guess")
    ap.add_argument("--language", default=None)
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the undiarised reference run (one ASR call)")
    ap.add_argument("--run-name", default=None,
                    help="write the result under runs/<name>/pipeline/")
    args = ap.parse_args()

    config = PipelineConfig(
        diarizer=args.diarizer,
        asr=[m.strip() for m in args.asr.split(",") if m.strip()],
        attribution=args.attribution,
        num_speakers=args.num_speakers,
        language=args.language,
    )
    result = run_pipeline(args.session, config, args.channel,
                          baseline_wer=not args.no_baseline)

    if result.get("error"):
        print(f"FAILED  {result['config']}: {result['error']}")
        return 1

    print(f"{result['config']}")
    print(f"  cpWER      {result['cpwer']:.1%}")
    if result.get("baseline_wer") is not None:
        print(f"  baseline   {result['baseline_wer']:.1%} WER "
              f"(same model, no diarisation) -> pipeline is "
              f"{result['vs_baseline']:+.1%}")
    print(f"  DER        {result['der']:.1%}" if result["der"] is not None
          else "  DER        n/a")
    print(f"  speakers   {result['speakers_found']} found, "
          f"{result['reference_speakers']} real")
    print(f"  cost       {result['wall_seconds']:.1f} s "
          f"({result['asr_calls']} ASR calls), {result['peak_rss_mb']:.0f} MB")
    for err in result.get("errors", []):
        print(f"  ! {err}")

    if args.run_name:
        out = Path("runs") / args.run_name / "pipeline"
        out.mkdir(parents=True, exist_ok=True)
        # The session belongs in the filename: without it, benchmarking two
        # sessions under one run name silently overwrites the first result.
        session_tag = Path(result["session"]).name
        name = f"{session_tag}__{result['config']}".replace("/", "_").replace("+", "_")
        (out / f"{name}.json").write_text(json.dumps(result, indent=2))
        print(f"  → {out / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
