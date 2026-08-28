"""
score.py — transcription (WER/CER) and diarisation (DER) metrics.

Scoring transcripts fairly is mostly a normalisation problem. A model
that writes "120 Millisekunden" is not wrong just because the reference
says "hundertzwanzig Millisekunden", and one that omits a comma is not
wrong at all. So we score on normalised text and report two levels:

  * `wer_raw`  — casing and punctuation stripped, nothing else. Strict.
  * `wer`      — additionally spells out digits and collapses the common
                 German/English number-word variants, so the metric
                 measures recognition rather than formatting.

`wer` is the headline number; `wer_raw` shows how much of a model's
score is formatting choices rather than real errors.

Caveat worth knowing: how a spoken digit string *should* be written is
genuinely ambiguous ("4821" vs. "four eight two one"). Normalisation
narrows that gap but cannot close it, which is exactly why both numbers
are reported side by side instead of one authoritative score.

No hard third-party dependency: `jiwer` and `num2words` are used when
installed, otherwise the equivalent logic here takes over, so the
benchmark still produces error rates on a machine with no network.

`diarization_error_rate` scores "who spoke when" the same way, against
the speaker-labelled turns in a session's `reference.json`.
"""

from __future__ import annotations

import re
import unicodedata

# German number words have several accepted spellings for the same value
# ("einhundertzwanzig" / "hundertzwanzig"). Collapsing them keeps the
# metric focused on whether the model heard the number at all.
_VARIANTS = [
    (r"\beinhundert", "hundert"),
    (r"\beintausend", "tausend"),
    (r"\bone hundred\b", "hundred"),
    (r"\bone thousand\b", "thousand"),
    (r"\bdreissig\b", "dreißig"),
]

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


# ──────────────────────────────────────────────
# Number spelling (fallback for num2words)
# ──────────────────────────────────────────────

_DE_ONES = ["null", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
_DE_TEENS = ["zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn",
             "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
_DE_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig",
            "sechzig", "siebzig", "achtzig", "neunzig"]

_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_EN_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
             "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"]


def _de_below_1000(n: int) -> str:
    """German numbers are written as a single compound word, which is why
    this concatenates without separators: 712 → 'siebenhundertzwölf'."""
    if n == 0:
        return ""
    out = ""
    hundreds, rest = divmod(n, 100)
    if hundreds:
        out += ("" if hundreds == 1 else _DE_ONES[hundreds]) + "hundert"
    if rest == 0:
        return out
    if rest < 10:
        return out + ("eins" if rest == 1 and not out else _DE_ONES[rest])
    if rest < 20:
        return out + _DE_TEENS[rest - 10]
    tens, ones = divmod(rest, 10)
    if ones:
        return out + _DE_ONES[ones] + "und" + _DE_TENS[tens]
    return out + _DE_TENS[tens]


def _en_below_1000(n: int) -> str:
    if n == 0:
        return ""
    parts: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        parts += [_EN_ONES[hundreds], "hundred"]
    if rest:
        if rest < 10:
            parts.append(_EN_ONES[rest])
        elif rest < 20:
            parts.append(_EN_TEENS[rest - 10])
        else:
            tens, ones = divmod(rest, 10)
            parts.append(_EN_TENS[tens] + ("-" + _EN_ONES[ones] if ones else ""))
    return " ".join(parts)


def spell_number(value: int, lang: str) -> str:
    """Spell an integer up to 999 999 999. Beyond that the digits are
    returned unchanged — no benchmark utterance needs billions, and a
    wrong spelling would silently corrupt the reference."""
    if value < 0 or value > 999_999_999:
        return str(value)
    german = lang.startswith("de")
    below = _de_below_1000 if german else _en_below_1000
    join = "" if german else " "

    millions, rest = divmod(value, 1_000_000)
    thousands, small = divmod(rest, 1_000)

    parts: list[str] = []
    if millions:
        unit = ("millionen" if millions > 1 else "million") if german else "million"
        parts.append(below(millions) + join + unit)
    if thousands:
        parts.append(below(thousands) + join + ("tausend" if german else "thousand"))
    if small or not parts:
        parts.append(below(small) or ("null" if german else "zero"))
    return join.join(p for p in parts if p)


def _digits_to_words(text: str, lang: str) -> str:
    """Replace digit runs with their spelled-out form so "2 4 1" and
    "zwei vier eins" compare equal. Prefers num2words when installed."""
    try:
        from num2words import num2words
    except ImportError:
        num2words = None

    def repl(m: re.Match[str]) -> str:
        value = int(m.group(0))
        if num2words is not None:
            try:
                return " " + num2words(value, lang=lang) + " "
            except (NotImplementedError, OverflowError):
                pass
        return " " + spell_number(value, lang) + " "

    return _DIGITS.sub(repl, text)


# ──────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────


def _strip_punct(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("’", "'").replace("–", " ").replace("—", " ")
    text = _PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def normalize(text: str, lang: str = "en", numbers: bool = True) -> str:
    """Normalise a transcript for scoring. `numbers=False` gives the strict
    variant behind `wer_raw`."""
    lang = (lang or "en").split("-")[0].split("_")[0].lower()
    out = text or ""
    if numbers:
        # Digits are spelled out before punctuation is stripped, so a
        # version string stays one number per component rather than
        # dissolving into unrelated digits.
        out = _digits_to_words(out, lang)
    out = _strip_punct(out)
    if numbers:
        for pattern, replacement in _VARIANTS:
            out = re.sub(pattern, replacement, out)
        out = _WS.sub(" ", out).strip()
    return out


# ──────────────────────────────────────────────
# Alignment
# ──────────────────────────────────────────────


def edit_ops(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """Levenshtein alignment returning (hits, substitutions, deletions,
    insertions). Implemented here so error rates don't depend on an
    optional package; jiwer is used instead when it's installed."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, 0, m
    if m == 0:
        return 0, 0, n, 0

    # Full DP table: benchmark transcripts are a few hundred tokens, so
    # O(n·m) memory is nothing, and keeping the table lets us backtrace
    # exact S/D/I counts rather than just a distance.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ref_i = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_i == hyp[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)

    hits = subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            hits += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return hits, subs, dels, ins


def _word_rates(ref: str, hyp: str) -> tuple[float, int, int, int, int]:
    ref_w, hyp_w = ref.split(), hyp.split()
    hits, subs, dels, ins = edit_ops(ref_w, hyp_w)
    denom = len(ref_w) or 1
    return (subs + dels + ins) / denom, hits, subs, dels, ins


def _char_rate(ref: str, hyp: str) -> float:
    _, subs, dels, ins = edit_ops(list(ref), list(hyp))
    return (subs + dels + ins) / (len(ref) or 1)


def score(hypothesis: str, reference: str, lang: str = "en") -> dict | None:
    """Return WER/CER metrics for one transcript, or None if there is no
    usable reference.

    WER above 1.0 is possible and meaningful — it means the model inserted
    more than it got right (typically a hallucination loop), so the value
    is deliberately not clamped."""
    if not (reference or "").strip():
        return None

    ref = normalize(reference, lang, numbers=True)
    hyp = normalize(hypothesis, lang, numbers=True)
    ref_raw = normalize(reference, lang, numbers=False)
    hyp_raw = normalize(hypothesis, lang, numbers=False)
    if not ref:
        return None

    # An empty hypothesis is a total miss, not a crash: every reference
    # word counts as a deletion.
    if not hyp:
        return {
            "wer": 1.0, "cer": 1.0, "wer_raw": 1.0,
            "hits": 0, "substitutions": 0,
            "deletions": len(ref.split()), "insertions": 0,
            "reference_words": len(ref.split()), "hypothesis_words": 0,
        }

    try:
        import jiwer

        out = jiwer.process_words(ref, hyp)
        wer = out.wer
        hits, subs, dels, ins = out.hits, out.substitutions, out.deletions, out.insertions
        cer = jiwer.cer(ref, hyp)
        wer_raw = jiwer.wer(ref_raw, hyp_raw)
    except ImportError:
        wer, hits, subs, dels, ins = _word_rates(ref, hyp)
        cer = _char_rate(ref, hyp)
        wer_raw = _word_rates(ref_raw, hyp_raw)[0]

    return {
        "wer": round(float(wer), 4),
        "cer": round(float(cer), 4),
        "wer_raw": round(float(wer_raw), 4),
        "hits": int(hits),
        "substitutions": int(subs),
        "deletions": int(dels),
        "insertions": int(ins),
        "reference_words": len(ref.split()),
        "hypothesis_words": len(hyp.split()),
    }


# ──────────────────────────────────────────────
# Diarisation: DER
# ──────────────────────────────────────────────
#
# Diarisation output is a set of (start, end, speaker) turns, and the
# speaker labels are arbitrary — a system that segments perfectly but
# calls the speakers "3, 1, 2" instead of "a, b, c" is not wrong. So the
# metric first finds the best one-to-one mapping between hypothesis and
# reference labels, then counts errors on a fixed frame grid:
#
#   DER = (missed speech + false alarm + speaker confusion) / total
#         reference speech
#
# Frames within a `collar` of a reference boundary are excluded. That is
# the NIST convention and it exists because the "true" moment a word
# starts is not annotatable to better than ~100 ms — without it every
# system is punished for its boundaries being a few frames off rather
# than for getting speakers wrong.
#
# Reference frames may hold more than one speaker (our conversations
# contain deliberate overlaps), and DER counts those: a system that
# outputs one speaker where two are talking takes a miss for the second.

MAX_EXHAUSTIVE_LABELS = 7


def _frame_labels(
    segments: list[dict], frames: int, frame_seconds: float
) -> list[set]:
    """Turn (start, end, speaker) turns into a per-frame set of speakers."""
    grid: list[set] = [set() for _ in range(frames)]
    for seg in segments:
        start = max(0, int(float(seg["start"]) / frame_seconds))
        end = min(frames, int(round(float(seg["end"]) / frame_seconds)))
        speaker = seg["speaker"]
        for i in range(start, end):
            grid[i].add(speaker)
    return grid


def _scored_mask(
    reference: list[dict], frames: int, frame_seconds: float, collar: float
) -> list[bool]:
    """False for frames sitting within `collar` of a reference boundary."""
    mask = [True] * frames
    if collar <= 0:
        return mask
    pad = int(round(collar / frame_seconds))
    for seg in reference:
        for boundary in (float(seg["start"]), float(seg["end"])):
            centre = int(boundary / frame_seconds)
            for i in range(max(0, centre - pad), min(frames, centre + pad + 1)):
                mask[i] = False
    return mask


def _best_mapping(
    ref_grid: list[set],
    hyp_grid: list[set],
    mask: list[bool],
    ref_labels: list,
    hyp_labels: list,
) -> dict:
    """One-to-one hypothesis→reference label mapping maximising the number
    of correctly attributed frames.

    Exhaustive for the handful of speakers a conversation actually has;
    greedy beyond that, because the number of injective mappings grows
    factorially and no realistic session needs it."""
    import itertools

    # Frames where each (hyp, ref) pair co-occurs — the overlap matrix
    # the assignment is chosen from.
    overlap: dict[tuple, int] = {}
    for i, keep in enumerate(mask):
        if not keep:
            continue
        for h in hyp_grid[i]:
            for r in ref_grid[i]:
                overlap[(h, r)] = overlap.get((h, r), 0) + 1

    if not overlap:
        return {}

    if max(len(ref_labels), len(hyp_labels)) <= MAX_EXHAUSTIVE_LABELS:
        best_score, best_map = -1, {}
        # Map the shorter list onto permutations of the longer one so
        # every candidate mapping stays one-to-one.
        if len(hyp_labels) <= len(ref_labels):
            for perm in itertools.permutations(ref_labels, len(hyp_labels)):
                mapping = dict(zip(hyp_labels, perm))
                sc = sum(overlap.get((h, r), 0) for h, r in mapping.items())
                if sc > best_score:
                    best_score, best_map = sc, mapping
        else:
            for perm in itertools.permutations(hyp_labels, len(ref_labels)):
                mapping = {h: r for h, r in zip(perm, ref_labels)}
                sc = sum(overlap.get((h, r), 0) for h, r in mapping.items())
                if sc > best_score:
                    best_score, best_map = sc, mapping
        return best_map

    mapping, used_ref, used_hyp = {}, set(), set()
    for (h, r), _count in sorted(overlap.items(), key=lambda kv: -kv[1]):
        if h in used_hyp or r in used_ref:
            continue
        mapping[h] = r
        used_hyp.add(h)
        used_ref.add(r)
    return mapping


def _count_errors(
    ref_grid: list[set],
    hyp_grid: list[set],
    mapping: dict,
    selected: list[bool],
) -> tuple[int, int, int, int]:
    """Sum reference speaker-seconds and the three DER error types over
    the frames flagged in `selected`. Counts are in speaker-frames, so a
    frame with two active speakers contributes two."""
    total = miss = false_alarm = confusion = 0
    for i, keep in enumerate(selected):
        if not keep:
            continue
        ref_set = ref_grid[i]
        hyp_set = hyp_grid[i]
        total += len(ref_set)
        if not ref_set and not hyp_set:
            continue
        mapped = {mapping.get(h) for h in hyp_set} - {None}
        correct = len(mapped & ref_set)
        miss += max(0, len(ref_set) - len(hyp_set))
        false_alarm += max(0, len(hyp_set) - len(ref_set))
        confusion += min(len(ref_set), len(hyp_set)) - correct
    return total, miss, false_alarm, confusion


def diarization_error_rate(
    reference: list[dict],
    hypothesis: list[dict],
    collar: float = 0.25,
    frame_seconds: float = 0.01,
) -> dict | None:
    """Score a diarisation hypothesis against reference turns.

    Both arguments are lists of dicts with `speaker`, `start`, `end`.
    Returns DER plus its three components, all as fractions of reference
    speech, or None when the reference contains no speech.

    When the reference has overlapping speech, an `overlap` block reports
    the same breakdown restricted to frames where two or more reference
    speakers are active. That sub-score deliberately ignores the collar:
    the collar masks a window either side of every reference boundary,
    and overlap regions sit exactly on those boundaries, so applying it
    would erase most of the very frames the sub-score exists to measure.
    """
    if not reference:
        return None

    end = max(
        max(float(s["end"]) for s in reference),
        max((float(s["end"]) for s in hypothesis), default=0.0),
    )
    frames = int(round(end / frame_seconds)) + 1

    ref_grid = _frame_labels(reference, frames, frame_seconds)
    hyp_grid = _frame_labels(hypothesis, frames, frame_seconds)
    mask = _scored_mask(reference, frames, frame_seconds, collar)

    ref_labels = sorted({s["speaker"] for s in reference}, key=str)
    hyp_labels = sorted({s["speaker"] for s in hypothesis}, key=str)
    mapping = _best_mapping(ref_grid, hyp_grid, mask, ref_labels, hyp_labels)

    total, miss, false_alarm, confusion = _count_errors(
        ref_grid, hyp_grid, mapping, mask
    )
    if total == 0:
        return None

    errors = miss + false_alarm + confusion
    result = {
        "der": round(errors / total, 4),
        "miss": round(miss / total, 4),
        "false_alarm": round(false_alarm / total, 4),
        "confusion": round(confusion / total, 4),
        "reference_speakers": len(ref_labels),
        "hypothesis_speakers": len(hyp_labels),
        "scored_seconds": round(sum(1 for k in mask if k) * frame_seconds, 2),
        "reference_speech_seconds": round(total * frame_seconds, 2),
        "collar": collar,
        "mapping": {str(h): str(r) for h, r in sorted(mapping.items(), key=str)},
    }

    overlap_frames = [len(s) >= 2 for s in ref_grid]
    overlap_count = sum(overlap_frames)
    if overlap_count:
        o_total, o_miss, o_fa, o_conf = _count_errors(
            ref_grid, hyp_grid, mapping, overlap_frames
        )
        # How much of the overlap did the system even represent as
        # simultaneous speech? A clustering backend assigns one speaker
        # per window, so this is structurally 0 for it however good its
        # boundaries are — which is the point of reporting it separately.
        detected = sum(
            1 for i, is_ov in enumerate(overlap_frames) if is_ov and len(hyp_grid[i]) >= 2
        )
        result["overlap"] = {
            "seconds": round(overlap_count * frame_seconds, 2),
            "share_of_speech": round(overlap_count / max(1, sum(1 for s in ref_grid if s)), 4),
            "der": round((o_miss + o_fa + o_conf) / o_total, 4) if o_total else None,
            "miss": round(o_miss / o_total, 4) if o_total else None,
            "false_alarm": round(o_fa / o_total, 4) if o_total else None,
            "confusion": round(o_conf / o_total, 4) if o_total else None,
            "detected_seconds": round(detected * frame_seconds, 2),
            "detection_recall": round(detected / overlap_count, 4),
        }

    return result
