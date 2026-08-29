#!/usr/bin/env python3
"""Search the pipeline configuration space and report what wins where.

Two questions, and they are not the same question:

  1. Which single configuration is best overall?
  2. Given what a caller knows before the audio is decoded -- roughly how
     many speakers, which language, how clean the recording is -- which
     configuration is best *for that case*?

The repo's own results say the second question is the interesting one.
The best diarizer flips with speaker count (end-to-end wins at four
speakers, clustering wins at six), and telling the backend the true
speaker count helps on one session and costs twenty points on another.
No single row of a results table survives contact with all conditions.

Two design decisions follow from findings already in the README, and both
are about not fooling ourselves:

**Held-out evaluation is mandatory, not a nicety.** The TTS engine
reorders the German model ranking outright, and diarisation collapse
profiles are *disjoint* between `say` and `piper` -- a profile that fails
on one is near-perfect on the other. A config picked on the sessions it
was measured on is therefore fitted to the synthesiser, not to speech.
`--holdout` splits by session or by engine, picks on the development
split, and reports the winner's score on data it never saw. The gap
between those two numbers is the honest measure of how much of the
"winner" is real.

**Search is cheap only if diarisation is not repeated.** Diarisation does
not depend on which ASR model follows it, so sweeping models re-runs the
same diarisation dozens of times. Caching it is the difference between a
sweep that runs over lunch and one that does not run at all.

This is deliberately not reinforcement learning. There is no state that
the agent's actions carry forward and no delayed reward -- every
configuration is evaluated independently and scored immediately. That
makes it a contextual bandit at most, and in practice a budgeted search:
random sampling with successive halving over sessions, which spends its
compute on the configurations that are still plausible.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import pipeline
from pipeline import PipelineConfig, run_pipeline

# The axes a deployment actually chooses between. Kept small on purpose:
# the point is to find a defensible answer within a machine-hour, not to
# enumerate everything that could vary.
SPACE: dict[str, list] = {
    "diarizer": ["sortformer", "sortformer-streaming", "titanet"],
    "asr": [["parakeet-tdt-v3"], ["whisper-large-v3"], ["canary-180m-flash"]],
    "attribution": ["speaker"],
    # None means "let the backend decide"; "known" is replaced at
    # evaluation time by the session's true speaker count, which is what a
    # caller who knows the meeting size would pass.
    "num_speakers": [None, "known"],
}


def enumerate_space(space: dict[str, list]) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*space.values())]


def sample_space(space: dict[str, list], n: int, seed: int) -> list[dict]:
    """Random search, deduplicated.

    Random beats grid whenever some axes matter far more than others,
    which is the normal case: a grid spends its budget resolving the axis
    that turns out to be flat."""
    grid = enumerate_space(space)
    if n >= len(grid):
        return grid
    rng = random.Random(seed)
    return rng.sample(grid, n)


def session_facts(session_dir: Path) -> dict:
    """What a caller plausibly knows before decoding: language, speaker
    count, and how the audio was produced."""
    session = json.loads((session_dir / "session.json").read_text())
    reference = json.loads((session_dir / "reference.json").read_text())
    turns = reference["channels"]["mixed"]
    name = session_dir.name
    parts = name.split("__")
    return {
        "session": name,
        "engine": parts[1] if len(parts) > 2 else "unknown",
        "profile": parts[2] if len(parts) > 2 else "clean",
        "language": session.get("language", "en"),
        "speakers": len({t["speaker"] for t in turns}),
    }


def materialise(point: dict, facts: dict) -> PipelineConfig:
    """Turn a search point into a runnable config for one session."""
    n = point["num_speakers"]
    return PipelineConfig(
        diarizer=point["diarizer"],
        asr=list(point["asr"]),
        attribution=point["attribution"],
        num_speakers=facts["speakers"] if n == "known" else None,
    )


def point_key(point: dict) -> str:
    n = point["num_speakers"] or "auto"
    return (f"{point['diarizer']}+{'+'.join(point['asr'])}"
            f"@{point['attribution']}/n={n}")


def evaluate(point: dict, session_dir: Path, facts: dict) -> dict:
    """One configuration on one session. Failures score 1.0 rather than
    being dropped: a configuration that crashes on a condition is worse
    than one that handles it badly, and silently skipping it would let a
    fragile config win by never being measured where it breaks."""
    config = materialise(point, facts)
    try:
        result = run_pipeline(session_dir, config, baseline_wer=False,
                              cache_diarization=True)
    except Exception as exc:                                # noqa: BLE001
        return {"cpwer": 1.0, "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": 0.0}
    if result.get("error") or result.get("cpwer") is None:
        return {"cpwer": 1.0, "error": result.get("error", "no cpWER"),
                "wall_seconds": result.get("wall_seconds", 0.0)}
    return {
        "cpwer": result["cpwer"],
        "der": result.get("der"),
        "speakers_found": result.get("speakers_found"),
        "wall_seconds": result.get("wall_seconds", 0.0),
        "peak_rss_mb": result.get("peak_rss_mb"),
    }


def condition(facts: dict) -> str:
    """The bucket a session belongs to, in terms a caller knows in advance."""
    return f"{facts['language']}/{facts['speakers']}spk"


def successive_halving(points: list[dict], sessions: list[Path],
                       keep: float = 0.5, log=print) -> dict[str, dict]:
    """Evaluate every point on the first session, drop the worst half, and
    repeat on progressively more sessions.

    **This is only sound when the sessions share a condition**, and the
    first real run of this harness proved why. With one 4-speaker and two
    6-speaker sessions, round one eliminated TitaNet -- correctly, it is
    the worst backend at four speakers -- and TitaNet is the *winner* at
    six by 26 points. Halving assumes a configuration that loses on the
    first session loses everywhere, which is exactly the assumption this
    repo's results refute. The caller is warned and the policy is
    disabled when the sessions span buckets; see `main`."""
    alive = list(points)
    scores: dict[str, dict] = {pt_key: {} for pt_key in map(point_key, points)}
    facts = {s: session_facts(s) for s in sessions}

    for round_no, session in enumerate(sessions, start=1):
        log(f"\n--- round {round_no}: {session.name} "
            f"({facts[session]['speakers']} speakers, "
            f"{facts[session]['language']}) — {len(alive)} configs")
        for point in alive:
            key = point_key(point)
            started = time.perf_counter()
            outcome = evaluate(point, session, facts[session])
            scores[key][session.name] = outcome
            flag = " FAILED" if outcome.get("error") else ""
            log(f"  {outcome['cpwer']*100:6.1f}%  {key:52s} "
                f"{time.perf_counter() - started:5.1f}s{flag}")

        if round_no == len(sessions) or len(alive) <= 2:
            continue
        ranked = sorted(alive, key=lambda p: statistics.mean(
            scores[point_key(p)][s.name]["cpwer"]
            for s in sessions[:round_no]))
        cut = max(2, int(len(ranked) * keep))
        dropped = [point_key(p) for p in ranked[cut:]]
        alive = ranked[:cut]
        log(f"  dropped {len(dropped)}: {', '.join(dropped[:4])}"
            f"{' …' if len(dropped) > 4 else ''}")

    return scores


def full_sweep(points: list[dict], sessions: list[Path], log=print) -> dict[str, dict]:
    scores: dict[str, dict] = {point_key(p): {} for p in points}
    facts = {s: session_facts(s) for s in sessions}
    for session in sessions:
        log(f"\n--- {session.name} ({facts[session]['speakers']} speakers, "
            f"{facts[session]['language']})")
        for point in points:
            started = time.perf_counter()
            outcome = evaluate(point, session, facts[session])
            scores[point_key(point)][session.name] = outcome
            flag = " FAILED" if outcome.get("error") else ""
            log(f"  {outcome['cpwer']*100:6.1f}%  {point_key(point):52s} "
                f"{time.perf_counter() - started:5.1f}s{flag}")
    return scores


def summarise(scores: dict[str, dict], facts: dict[str, dict]) -> dict:
    """Best overall, and best per condition.

    'Best overall' is reported only over configurations measured on every
    session -- successive halving means the others have no comparable
    mean, and ranking a config that ran once against one that ran four
    times would reward being eliminated early."""
    full = {k: v for k, v in scores.items()
            if len(v) == len(facts) and v}
    overall = sorted(
        ((statistics.mean(r["cpwer"] for r in v.values()), k)
         for k, v in full.items()))

    by_condition: dict[str, list] = {}
    for key, per_session in scores.items():
        for session_name, outcome in per_session.items():
            fact = facts[session_name]
            bucket = condition(fact)
            by_condition.setdefault(bucket, []).append(
                (outcome["cpwer"], key, session_name))
    for bucket in by_condition:
        by_condition[bucket].sort()

    return {
        "overall": [{"cpwer": round(c, 4), "config": k} for c, k in overall],
        "by_condition": {
            bucket: [{"cpwer": round(c, 4), "config": k, "session": s}
                     for c, k, s in rows]
            for bucket, rows in by_condition.items()
        },
    }


def split_sessions(sessions: list[Path], mode: str) -> tuple[list[Path], list[Path]]:
    """Development / held-out split.

    `engine` is the strictest split available and the one the repo's own
    data argues for: the synthesiser moves absolute WER by up to 6.4
    points and reorders models outright, so a config chosen on `piper` and
    validated on `piper` has proved nothing about speech."""
    facts = {s: session_facts(s) for s in sessions}
    if mode == "engine":
        engines = sorted({facts[s]["engine"] for s in sessions})
        if len(engines) < 2:
            return sessions, []
        held = engines[-1]
        return ([s for s in sessions if facts[s]["engine"] != held],
                [s for s in sessions if facts[s]["engine"] == held])
    if mode == "session":
        if len(sessions) < 2:
            return sessions, []
        return sessions[:-1], sessions[-1:]
    return sessions, []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True,
                    help="session directories to search over")
    ap.add_argument("--samples", type=int, default=0,
                    help="random-search points (0 = full grid)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", choices=["none", "session", "engine"],
                    default="engine",
                    help="how to split development from held-out sessions")
    ap.add_argument("--halving", action="store_true",
                    help="drop the worst half after each session; only sound "
                         "when the sessions share a condition")
    ap.add_argument("--run-name", help="write results under runs/<name>/search")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and its cost estimate, run nothing")
    args = ap.parse_args()

    sessions = [Path(s) for s in args.sessions]
    missing = [s for s in sessions if not (s / "session.json").exists()]
    if missing:
        print(f"not a session directory: {', '.join(map(str, missing))}")
        return 1

    points = (sample_space(SPACE, args.samples, args.seed) if args.samples
              else enumerate_space(SPACE))
    dev, held = split_sessions(sessions, args.holdout)
    if not dev:
        print("holdout left no development sessions")
        return 1

    print(f"{len(points)} configs × {len(dev)} dev sessions"
          f"{f' (holding out {len(held)})' if held else ''}")
    for session in sessions:
        fact = session_facts(session)
        role = "held out" if session in held else "dev"
        print(f"  {role:9s} {fact['session']:32s} {fact['speakers']} speakers, "
              f"{fact['language']}, {fact['engine']}")

    dev_conditions = {condition(session_facts(s)) for s in dev}
    halving = args.halving
    if halving and len(dev_conditions) > 1:
        print(f"\n!  --halving disabled: the development sessions span "
              f"{len(dev_conditions)} conditions ({', '.join(sorted(dev_conditions))}).")
        print("   Halving eliminates a config on the first session it loses, "
              "but the best diarizer flips with speaker count, so a "
              "condition specialist would be dropped before it is ever "
              "measured where it wins. Use one condition per search, or "
              "accept the full sweep.")
        halving = False

    if args.dry_run:
        evaluations = (sum(max(2, int(len(points) * 0.5 ** i))
                           for i in range(len(dev))) if halving
                       else len(points) * len(dev))
        print(f"\nwould run ~{evaluations} evaluations on dev"
              f" + {len(held)} held-out validations")
        return 0

    scores = (successive_halving(points, dev) if halving
              else full_sweep(points, dev))

    all_facts = {session_facts(s)["session"]: session_facts(s) for s in dev}
    summary = summarise(scores, all_facts)

    print("\n=== best overall (mean cpWER across all dev sessions)")
    if not summary["overall"]:
        print("  no config survived every session")
    for row in summary["overall"][:5]:
        print(f"  {row['cpwer']*100:6.1f}%  {row['config']}")

    print("\n=== best per condition")
    for bucket, rows in sorted(summary["by_condition"].items()):
        print(f"  {bucket:12s} {rows[0]['cpwer']*100:6.1f}%  {rows[0]['config']}")

    validation = []
    if held and summary["by_condition"]:
        print("\n=== held-out validation")
        print("  Each held-out session is checked against the config that won "
              "its *own* condition on dev. Comparing a 6-speaker dev mean to "
              "a 4-speaker held-out session measures difficulty, not "
              "generalisation.")
        for session in held:
            fact = session_facts(session)
            bucket = condition(fact)
            rows = summary["by_condition"].get(bucket)
            if not rows:
                print(f"\n  {fact['session']}: condition {bucket} never "
                      f"appears in dev — nothing to compare against.")
                continue
            winner_key = rows[0]["config"]
            dev_score = rows[0]["cpwer"]
            winner = next(p for p in points if point_key(p) == winner_key)
            outcome = evaluate(winner, session, fact)
            gap = outcome["cpwer"] - dev_score
            validation.append({"session": fact["session"], "condition": bucket,
                               "config": winner_key, "dev_cpwer": dev_score,
                               **outcome})
            print(f"\n  {bucket}: {winner_key}")
            print(f"    dev {dev_score*100:.1f}% ({rows[0]['session']}) "
                  f"→ held out {outcome['cpwer']*100:.1f}% "
                  f"({fact['session']}) — {gap*100:+.1f} points")
        if validation:
            print("\n  A large positive gap means the winner is fitted to the "
                  "development synthesiser, not to speech.")

    if args.run_name:
        out = Path("runs") / args.run_name / "search"
        out.mkdir(parents=True, exist_ok=True)
        (out / "search.json").write_text(json.dumps({
            "space": {k: [list(v) if isinstance(v, list) else v for v in vals]
                      for k, vals in SPACE.items()},
            "dev_sessions": [s.name for s in dev],
            "holdout_sessions": [s.name for s in held],
            "holdout_mode": args.holdout,
            "halving": halving,
            "scores": scores,
            "summary": summary,
            "validation": validation,
        }, indent=2, default=str))
        print(f"\n→ {out / 'search.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
