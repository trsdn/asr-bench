#!/usr/bin/env python3
"""Combine several ASR hypotheses, and decide when a third opinion is worth
paying for.

Two related ideas, both cheap to state and neither obviously worth it:

**Fusion.** Run more than one model on the same audio and vote word by word
(ROVER, Fiscus 1997). Different architectures make different mistakes, so a
majority vote can beat every voter. It can also be worse than the best
voter, when two weak models outvote a strong one -- so this has to be
measured, not assumed.

**Escalation.** Voting requires running every model everywhere, which costs
what it costs. But the disagreement between two cheap models is a
*reference-free* signal available at inference time, and it is concentrated:
most of a transcript is easy and every model gets it right. So run two cheap
models, and send only the passages where they disagree to the expensive one.
If a fifth of the audio is contested, three-model quality costs about 1.2x
one model rather than 3x.

The honest caveat, and it is a large one: this repo's config search found
that swapping the ASR model moves cpWER by 2-4 points while swapping the
diarizer moves it by up to 55. Fusion and escalation both operate on the
small half of the problem. They are worth measuring precisely because the
prior is unfavourable.
"""

from __future__ import annotations

import difflib


def align_to_pivot(pivot: list[str], other: list[str]) -> list[list[str]]:
    """For each pivot slot, the words of `other` aligned to it.

    A slot can hold zero words (the other hypothesis deleted it) or several
    (it split one word into two). Insertions that fall between slots are
    attached to the following slot, which keeps the slot count equal to the
    pivot length and makes voting a simple per-slot majority."""
    slots: list[list[str]] = [[] for _ in pivot]
    if not pivot:
        return slots
    matcher = difflib.SequenceMatcher(a=pivot, b=other, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                slots[i1 + offset].append(other[j1 + offset])
        elif tag == "replace":
            # Spread the replacement across the pivot slots it covers; any
            # surplus words pile onto the last one.
            span = i2 - i1
            words = other[j1:j2]
            for offset in range(span):
                lo = (len(words) * offset) // span
                hi = (len(words) * (offset + 1)) // span
                if offset == span - 1:
                    hi = len(words)
                slots[i1 + offset].extend(words[lo:hi])
        elif tag == "delete":
            pass                      # other has nothing here
        elif tag == "insert":
            target = min(i1, len(pivot) - 1)
            slots[target].extend(other[j1:j2])
    return slots


def rover(hypotheses: list[str]) -> str:
    """Majority vote over word slots, with the first hypothesis as pivot.

    The pivot should be the model you would have used alone: ties break in
    its favour, so fusion can only overrule it when a majority actually
    disagrees. With two hypotheses there is never a majority, so the result
    is the pivot -- fusion needs three voters to do anything at all, which
    is worth knowing before paying for two."""
    hyps = [h.split() for h in hypotheses if h and h.strip()]
    if not hyps:
        return ""
    if len(hyps) == 1:
        return " ".join(hyps[0])

    pivot = hyps[0]
    aligned = [align_to_pivot(pivot, other) for other in hyps[1:]]

    out: list[str] = []
    for index, pivot_word in enumerate(pivot):
        votes: dict[str, int] = {pivot_word: 1}
        for slots in aligned:
            candidate = " ".join(slots[index])
            votes[candidate] = votes.get(candidate, 0) + 1
        best = max(votes.items(), key=lambda kv: (kv[1], kv[0] == pivot_word))
        if best[0]:
            out.append(best[0])
    return " ".join(out)


def agreement(hypotheses: list[str]) -> float:
    """Fraction of pivot slots where every hypothesis says the same thing.

    This is the escalation trigger, and its whole value is that it needs no
    reference: it is computable on real audio at inference time, which a WER
    is not."""
    hyps = [h.split() for h in hypotheses if h and h.strip()]
    if len(hyps) < 2:
        return 1.0
    pivot = hyps[0]
    if not pivot:
        return 0.0
    aligned = [align_to_pivot(pivot, other) for other in hyps[1:]]
    agreed = sum(
        1 for index, word in enumerate(pivot)
        if all(" ".join(slots[index]) == word for slots in aligned)
    )
    return agreed / len(pivot)
