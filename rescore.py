"""
Re-score stored transcripts against their references, without re-running models.

    uv run python rescore.py --dry-run
    uv run python rescore.py

Every run keeps the decoded text in `summary.json`, so a change to the
normalisation rules in `score.py` can be applied to the whole result history
for free -- no GPU, no model loads, seconds instead of hours. This matters
because scoring bugs are not hypothetical here: two published rankings turned
out to be artefacts of normalisation, and both were corrected with this script.

Rewrites `accuracy` in place and reports which cells moved. Run `--dry-run`
first; the diff is the evidence that a scoring change did what you intended.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import score


def rescore_run(run_dir: Path, apply: bool) -> list[tuple[str, str, str, float, float]]:
    run_json, summary_json = run_dir / "run.json", run_dir / "summary.json"
    if not (run_json.exists() and summary_json.exists()):
        return []

    meta = json.loads(run_json.read_text())
    session = Path(meta["session"])
    lang = meta.get("language", "en")
    rows = json.loads(summary_json.read_text())
    if not isinstance(rows, list):
        return []

    changed, dirty = [], False
    for row in rows:
        if not row.get("text") or not row.get("accuracy"):
            continue
        ref_file = session / "reference" / f"{row['channel']}.txt"
        if not ref_file.exists():
            continue

        new = score.score(row["text"], ref_file.read_text(), lang)
        if not new:
            continue
        if abs(new["wer"] - row["accuracy"]["wer"]) > 1e-9:
            changed.append((run_dir.name, row["model_id"], row["channel"],
                            row["accuracy"]["wer"] * 100, new["wer"] * 100))
            dirty = True
        row["accuracy"] = new

    if dirty and apply:
        summary_json.write_text(json.dumps(rows, indent=2))
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--limit", type=int, default=60,
                    help="how many changed cells to print (0 for all)")
    args = ap.parse_args()

    changed = []
    for run_dir in sorted(args.runs_dir.iterdir()):
        if run_dir.is_dir():
            changed += rescore_run(run_dir, apply=not args.dry_run)

    verb = "would change" if args.dry_run else "changed"
    print(f"{len(changed)} cells {verb}")
    shown = changed if args.limit == 0 else changed[:args.limit]
    for run_name, model_id, channel, old, new in shown:
        print(f"  {run_name:20s} {model_id:24s} {channel:10s} "
              f"{old:6.1f}% -> {new:6.1f}%")
    if len(shown) < len(changed):
        print(f"  ... {len(changed) - len(shown)} more")


if __name__ == "__main__":
    main()
