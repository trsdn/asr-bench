"""
Build a side-by-side Markdown report from a bench run.

    uv run python compare.py --run-name standup-de__phone_2026-08-28

Reads `runs/<run-name>/*/<channel>.txt` + `*.metrics.json`, writes
`runs/<run-name>/comparison.md` with an accuracy ranking (when the
session had a reference transcript), a speed/RAM summary, the
diarisation ranking if `diarize.py` wrote one into the same run, a
hallucination heuristic for known failure modes (repeated phrases,
Whisper's signature "Thank you for watching" ghosts), and side-by-side
transcript previews against the ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Known phrases that Whisper regularly hallucinates when audio is silent
# or unclear. English + German variants. Case-insensitive match.
WHISPER_GHOST_PHRASES = [
    r"thank you for watching",
    r"please subscribe",
    r"bitte abonnieren",
    r"untertitel (von|im auftrag)",
    r"untertitelung des zdf",
    r"vielen dank fürs zuschauen",
    r"thanks for watching",
    r"like and subscribe",
]


def count_ghosts(text: str) -> dict[str, int]:
    """Count occurrences of each known Whisper ghost phrase."""
    hits: dict[str, int] = {}
    for pattern in WHISPER_GHOST_PHRASES:
        n = len(re.findall(pattern, text, flags=re.IGNORECASE))
        if n:
            hits[pattern] = n
    return hits


def longest_repeated_ngram(text: str, n: int = 5) -> tuple[str, int]:
    """Return the (tokens, repeat-count) of the most-repeated n-gram with
    length >= n words. Low baseline for normal speech; high counts indicate
    loop / stuttering failures."""
    words = text.split()
    if len(words) < n:
        return "", 0
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    counter = Counter(ngrams)
    if not counter:
        return "", 0
    top, count = counter.most_common(1)[0]
    return top, count


def load_run_info(run_dir: Path) -> dict:
    """Read the run manifest bench.py writes. Older runs predate it, so
    every consumer treats it as optional."""
    info = run_dir / "run.json"
    if info.exists():
        return json.loads(info.read_text(encoding="utf-8"))
    return {}


def load_reference(run_info: dict, channel: str) -> str:
    """Fetch the ground-truth text for a channel, if the session that
    produced this run is still on disk."""
    session = run_info.get("session")
    if not session:
        return ""
    ref = Path(session) / "reference" / f"{channel}.txt"
    return ref.read_text(encoding="utf-8").strip() if ref.exists() else ""


def load_metrics(run_dir: Path) -> list[dict]:
    summary = run_dir / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))
    # Fallback: glob model dirs
    out: list[dict] = []
    for mdir in sorted(run_dir.iterdir()):
        if not mdir.is_dir():
            continue
        for mfile in mdir.glob("*.metrics.json"):
            out.append(json.loads(mfile.read_text(encoding="utf-8")))
    return out


def load_diarization(run_dir: Path) -> list[dict]:
    """Read diarize.py's summary from the same run directory, if it ran."""
    summary = run_dir / "diarization" / "summary.json"
    if not summary.exists():
        return []
    return json.loads(summary.read_text(encoding="utf-8")).get("runs", [])


def build_report(run_dir: Path) -> str:
    records = load_metrics(run_dir)
    if not records:
        return "_No metrics found in this run._\n"

    by_model: dict[str, dict[str, dict]] = {}
    for r in records:
        by_model.setdefault(r["model_id"], {})[r["channel"]] = r

    run_info = load_run_info(run_dir)
    scored = [r for r in records if r.get("accuracy")]

    lines: list[str] = []
    lines.append(f"# ASR bench — {run_dir.name}\n")

    manifest = run_info.get("session_manifest") or {}
    if manifest:
        deg = (manifest.get("degradation") or {}).get("profile", "—")
        lines.append(
            f"Session `{manifest.get('name', '?')}` · "
            f"language **{manifest.get('language', '?')}** · "
            f"degradation **{deg}** · "
            f"{manifest.get('duration_seconds', 0):.0f}s · "
            f"{len(manifest.get('speakers') or [])} speakers · "
            f"{manifest.get('word_count', 0)} reference words\n"
        )

    # ──── Accuracy ranking ────
    if scored:
        lines.append("## Accuracy (vs. ground truth)\n")
        lines.append(
            "Ranked by WER, best first. `WER raw` scores punctuation- and "
            "case-normalised text only; `WER` additionally normalises "
            "numbers, so the gap between the two is formatting rather than "
            "misrecognition.\n"
        )
        lines.append("| Model | Channel | WER | CER | WER raw | Sub | Del | Ins | Ref words |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(scored, key=lambda x: (x["channel"], x["accuracy"]["wer"])):
            a = r["accuracy"]
            lines.append(
                f"| `{r['model_id']}` | {r['channel']} | "
                f"**{a['wer']:.1%}** | {a['cer']:.1%} | {a['wer_raw']:.1%} | "
                f"{a['substitutions']} | {a['deletions']} | {a['insertions']} | "
                f"{a['reference_words']} |"
            )
        lines.append("")

    # ──── Performance table ────
    lines.append("## Speed & memory\n")
    lines.append("| Model | Channel | Audio | Wall | RTF | Words | Peak RSS | Error |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for model_id in sorted(by_model):
        for ch in sorted(by_model[model_id]):
            r = by_model[model_id][ch]
            lines.append(
                f"| `{model_id}` | {ch} | "
                f"{r['audio_seconds']:.0f}s | "
                f"{r['wall_seconds']:.1f}s | "
                f"{r['rtf']:.2f} | "
                f"{len((r['text'] or '').split())} | "
                f"{r['peak_rss_mb']:.0f} MB | "
                f"{(r.get('error') or '') or '—'} |"
            )
    lines.append("")

    # ──── Diarisation ────
    diarization = load_diarization(run_dir)
    if diarization:
        lines.append("## Diarisation (who spoke when)\n")
        lines.append(
            "Ranked by DER, best first. DER counts missed speech, false "
            "alarms and speaker confusion as a fraction of reference "
            "speech, after mapping the system's arbitrary speaker labels "
            "onto the real ones and excluding a 0.25 s collar around each "
            "reference boundary.\n"
        )
        lines.append(
            "| Backend | Channel | Speakers | DER | Miss | False alarm | "
            "Confusion | Wall | Error |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

        def pct(acc: dict | None, key: str) -> str:
            return f"{acc[key]:.1%}" if acc and key in acc else "—"

        for d in sorted(
            diarization,
            key=lambda x: (
                x["channel"],
                (x.get("accuracy") or {}).get("der", 9.9),
            ),
        ):
            acc = d.get("accuracy")
            found = d.get("speakers_found", 0)
            expected = (acc or {}).get("reference_speakers")
            speakers = f"{found}" + (f" / {expected}" if expected else "")
            lines.append(
                f"| `{d['backend']}` | {d['channel']} | {speakers} | "
                f"**{pct(acc, 'der')}** | {pct(acc, 'miss')} | "
                f"{pct(acc, 'false_alarm')} | {pct(acc, 'confusion')} | "
                f"{d['wall_seconds']:.1f}s | {(d.get('error') or '—')} |"
            )
        lines.append("")

        overlapping = [
            d for d in diarization if (d.get("accuracy") or {}).get("overlap")
        ]
        if overlapping:
            share = (overlapping[0]["accuracy"]["overlap"] or {}).get(
                "share_of_speech"
            )
            lines.append("### Overlapping speech\n")
            lines.append(
                "Restricted to frames where two or more people speak at once "
                f"({share:.1%} of reference speech here). The collar is not "
                "applied: overlap sits on speaker boundaries, so a collar "
                "would mask exactly what is being measured. `Recall` is the "
                "share of overlapped seconds where the system reported more "
                "than one speaker at all — a clustering backend assigns one "
                "label per window and so cannot represent simultaneity by "
                "construction.\n"
            )
            lines.append(
                "| Backend | Channel | Overlap | DER | Miss | Confusion | "
                "Recall |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for d in sorted(
                overlapping,
                key=lambda x: (x["channel"], x["accuracy"]["overlap"]["der"]),
            ):
                ov = d["accuracy"]["overlap"]
                lines.append(
                    f"| `{d['backend']}` | {d['channel']} | "
                    f"{ov['seconds']:.1f}s | **{pct(ov, 'der')}** | "
                    f"{pct(ov, 'miss')} | {pct(ov, 'confusion')} | "
                    f"{pct(ov, 'detection_recall')} |"
                )
            lines.append("")

    # ──── Hallucination heuristics ────
    lines.append("## Hallucination / degeneration heuristics\n")
    lines.append("| Model | Channel | Whisper ghosts | Top-repeated 5-gram | × |")
    lines.append("|---|---|---|---|---:|")
    for model_id in sorted(by_model):
        for ch in sorted(by_model[model_id]):
            r = by_model[model_id][ch]
            text = r.get("text") or ""
            ghosts = count_ghosts(text)
            gtxt = ", ".join(f"`{p}`×{n}" for p, n in ghosts.items()) or "—"
            ngram, count = longest_repeated_ngram(text)
            ntxt = f"`{ngram}`" if ngram and count > 1 else "—"
            lines.append(f"| `{model_id}` | {ch} | {gtxt} | {ntxt} | {count} |")
    lines.append("")

    # ──── Side-by-side transcripts (first N chars) ────
    preview_chars = 4000
    lines.append(f"## Transcript previews (first ~{preview_chars} chars per cell)\n")

    channels = sorted({ch for m in by_model.values() for ch in m})
    def cell(text: str) -> str:
        text = (text or "").strip()
        snippet = text[:preview_chars]
        if len(text) > preview_chars:
            snippet += f"… _(+{len(text) - preview_chars} more chars)_"
        # Escape pipes to avoid breaking markdown tables.
        return snippet.replace("|", "\\|").replace("\n", " ") or "_(empty)_"

    for ch in channels:
        lines.append(f"### Channel: `{ch}`\n")
        header_models = [m for m in sorted(by_model) if ch in by_model[m]]
        reference = load_reference(run_info, ch)

        headers = (["**reference**"] if reference else []) + header_models
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        cells = [cell(reference)] if reference else []
        cells += [cell(by_model[m][ch].get("text") or "") for m in header_models]
        lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("---")
    lines.append(
        "_Full transcripts are in each model's subdirectory "
        "(`<model>/<channel>.txt`). Metrics in `summary.json`._"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-name",
        required=True,
        help="Sub-directory under runs/ produced by bench.py",
    )
    args = ap.parse_args()

    run_dir = Path(__file__).resolve().parent / "runs" / args.run_name
    if not run_dir.is_dir():
        print(f"Run dir not found: {run_dir}")
        return 1

    md = build_report(run_dir)
    out_path = run_dir / "comparison.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
