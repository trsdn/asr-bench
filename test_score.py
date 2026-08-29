"""Scoring regression tests.

Two published rankings in this repo turned out to be artefacts of how
numbers were normalised rather than facts about the models. These cases
pin down the behaviour that fixed them, so the next change to `score.py`
has to break a test rather than quietly re-order the README.

Runs without pytest — `python test_score.py` — because the repo has no
test dependency and scoring must stay usable without one.
"""

import cleanup
import fuse
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


# Speaker-attributed cases. A meeting transcript is judged on whether the
# right words are filed under the right person, and flat WER cannot see
# that at all — the first case below is word-perfect (WER 0.000) with one
# turn misfiled, which is exactly the failure that makes minutes unusable.
_REF = [
    {"speaker": "a", "text": "we should ship on friday"},
    {"speaker": "b", "text": "i disagree the tests are red"},
    {"speaker": "a", "text": "then we fix them first"},
]

CP_CASES = [
    # Labels are arbitrary and matched by content, not by name.
    ("perfect, relabelled", [
        {"speaker": "spk1", "text": "we should ship on friday"},
        {"speaker": "spk0", "text": "i disagree the tests are red"},
        {"speaker": "spk1", "text": "then we fix them first"},
    ], 0.0),

    # Every word correct, one turn attributed to the wrong speaker.
    # Flat WER on the same text is 0.000.
    ("word-perfect, one turn misfiled", [
        {"speaker": "spk1", "text": "we should ship on friday"},
        {"speaker": "spk0", "text": "i disagree the tests are red"},
        {"speaker": "spk0", "text": "then we fix them first"},
    ], 0.625),

    # The collapse this repo measured 14 times out of 15 diarisation
    # failures: two real speakers merged into one.
    ("all speech merged into one speaker", [
        {"speaker": "spk0", "text": "we should ship on friday i disagree "
                                    "the tests are red then we fix them first"},
    ], 0.75),

    # An invented speaker must cost something, or a system can hedge by
    # splitting everyone in two.
    ("phantom third speaker", [
        {"speaker": "spk1", "text": "we should ship on friday"},
        {"speaker": "spk0", "text": "i disagree the tests are red"},
        {"speaker": "spk2", "text": "then we fix them first"},
    ], 0.625),

    # Nothing recognised at all is a total loss, not a crash.
    ("empty hypothesis", [], 1.0),
]


# Fusion. ROVER can only overrule the pivot when a majority disagrees, so
# the two-voter no-op is a property worth pinning: paying for a second
# model buys nothing on its own.
FUSE_CASES = [
    ("majority overrules a pivot neighbour", ["a b c", "a x c", "a b c"], "a b c"),
    ("majority overrules the pivot",         ["a x c", "a b c", "a b c"], "a b c"),
    ("two voters leave the pivot alone",     ["a b c", "a x c"],          "a b c"),
    ("majority deletes a word",              ["a b c", "a b", "a b"],     "a b"),
    ("majority inserts a word",              ["a b c", "a b b c", "a b b c"], "a b b c"),
    ("empty input",                          [""],                        ""),
]

AGREEMENT_CASES = [
    ("identical",       ["a b c d", "a b c d"], 1.0),
    ("one word differs", ["a b c d", "a x c d"], 0.75),
]


# The cleanup guard. Half of these must be *accepted*: a guard that vetoes
# everything is trivially safe and useless, so the tests pin the permissive
# side as hard as the restrictive one.
GUARD_CASES = [
    ("punctuation and casing only",
     "der deployment lief durch die latenz liegt bei 120 millisekunden",
     "Der Deployment lief durch. Die Latenz liegt bei 120 Millisekunden.",
     "de", True),
    ("spoken number rewritten to digits",
     "Ticket vier null neun strich zwei eins ist noch offen",
     "Ticket 409-21 ist noch offen.",
     "de", True),
    ("digits rewritten to words",
     "wir sind bei Version 2.4.1",
     "Wir sind bei Version zwei Punkt vier Punkt eins.",
     "de", True),
    ("filler words removed",
     "also ähm ich glaube ähm das passt so",
     "Ich glaube, das passt so.",
     "de", True),
    ("acronym capitalised",
     "das api gateway war kurz weg",
     "Das API Gateway war kurz weg.",
     "en", True),
    ("digit changed",
     "Ticket vier null neun strich zwei eins ist offen",
     "Ticket 409-12 ist offen.",
     "de", False),
    ("version truncated",
     "wir sind bei Version 2.4.1",
     "Wir sind bei Version 2.4.",
     "de", False),
    ("number invented",
     "die Latenz ist gestiegen",
     "Die Latenz ist um 30 Prozent gestiegen.",
     "de", False),
    ("number dropped",
     "die Latenz liegt bei 120 Millisekunden",
     "Die Latenz ist hoch.",
     "de", False),
    ("identifier mangled",
     "wir migrieren auf postgres16 nächste Woche",
     "Wir migrieren auf Postgres nächste Woche.",
     "de", False),
    ("summarised instead of cleaned",
     "also ich habe gestern das deployment durchgezogen und danach noch die "
     "logs geprüft und da war alles ruhig soweit ich das sehen konnte",
     "Deployment erledigt, Logs unauffällig.",
     "de", False),
    ("empty output",
     "die Latenz ist hoch",
     "",
     "de", False),
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

    print()
    for name, hyp, expected in CP_CASES:
        result = score.cp_wer(_REF, hyp, "en")
        got = 1.0 if result is None else result["cpwer"]
        ok = abs(got - expected) < 1e-6
        failures += not ok
        mark = "ok  " if ok else "FAIL"
        print(f"{mark} [cp] {name:42s} cpwer={got:.3f}"
              + ("" if ok else f" expected {expected:.3f}"))

    print()
    for name, hyps, expected in FUSE_CASES:
        got = fuse.rover(hyps)
        ok = got == expected
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} [fuse] {name:42s} -> {got!r}"
              + ("" if ok else f" expected {expected!r}"))

    for name, hyps, expected in AGREEMENT_CASES:
        got = fuse.agreement(hyps)
        ok = abs(got - expected) < 1e-9
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} [fuse] agreement {name:32s} -> {got:.3f}"
              + ("" if ok else f" expected {expected:.3f}"))

    print()
    for name, raw, cleaned, lang, expected in GUARD_CASES:
        if cleaned.strip():
            got = cleanup.verdict(raw, cleaned, lang).ok
            why = cleanup.verdict(raw, cleaned, lang).reason
        else:
            got = cleanup.clean(cleaned or raw, lambda t, l: "", lang)[2].ok
            why = "empty"
        ok = got == expected
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} [guard] {name:36s} "
              f"{'accept' if got else 'veto'}"
              + ("" if ok else f" expected {'accept' if expected else 'veto'}")
              + ("" if got else f"  ({why})"))

    total = (len(CASES) + len(CP_CASES) + len(FUSE_CASES)
             + len(AGREEMENT_CASES) + len(GUARD_CASES))
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
