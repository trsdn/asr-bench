"""Scoring regression tests.

Two published rankings in this repo turned out to be artefacts of how
numbers were normalised rather than facts about the models. These cases
pin down the behaviour that fixed them, so the next change to `score.py`
has to break a test rather than quietly re-order the README.

Runs without pytest — `python test_score.py` — because the repo has no
test dependency and scoring must stay usable without one.
"""

import score

# (hypothesis, reference, language, expected WER or None for "just show it")
CASES = [
    # A quantity written as digits against a spelled-out cardinal.
    ("etwa 120 Millisekunden", "etwa hundertzwanzig Millisekunden", "de", 0.0),
    ("about 120 milliseconds", "about one hundred twenty milliseconds", "en", 0.0),

    # An identifier written as digits against a digit-by-digit reading.
    # Expanding this as a cardinal ("one thousand two hundred twenty seven")
    # cost every digit-emitting model ~6 WER points and invented the
    # English ranking.
    ("ticket 1227", "ticket one two two seven", "en", 0.0),

    # Spoken punctuation inside an identifier. The hyphen is stripped as
    # punctuation while the word "dash" survives in the reference, so the
    # word became a deletion nobody earned.
    ("case 4821-773", "case four eight two one, dash, seven seven three", "en", 0.0),
    ("version 2.4.1", "version two point four point one", "en", 0.0),
    ("Version 2.4.1", "Version zwei Punkt vier Punkt eins", "de", 0.0),

    # The fixes must not swallow real errors.
    ("ticket 1338", "ticket one two two seven", "en", 0.6),
    ("about 130 milliseconds", "about one hundred twenty milliseconds", "en", 0.25),
    # "point" away from any number is an ordinary word and stays scored.
    ("that is a good", "that is a good point", "en", 0.2),

    ("the cat sat", "the cat sat", "en", 0.0),
    ("", "the cat sat", "en", 1.0),
]


def main() -> int:
    failures = 0
    for hyp, ref, lang, expected in CASES:
        got = score.score(hyp, ref, lang)["wer"]
        ok = expected is None or abs(got - expected) < 1e-6
        failures += not ok
        mark = "ok  " if ok else "FAIL"
        print(f"{mark} [{lang}] {hyp!r:42s} wer={got:.3f}"
              + ("" if ok else f" expected {expected:.3f}"))

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
