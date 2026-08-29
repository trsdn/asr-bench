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
    # "one hundred and twenty" and "one hundred twenty" are both ordinary
    # English. num2words emits the first, most references write the second,
    # and the difference is a word neither speaker chose.
    (r"\b(hundred|thousand|million) and\b", r"\1"),
]

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")

# Spoken names for the punctuation inside an identifier. A reference that
# says "four eight two one dash seven seven three" and a model that writes
# "4821-773" agree; the hyphen is stripped as punctuation while the word
# survives, so the word becomes a deletion the model did not earn. These
# are only removed next to a number, so an ordinary "that's a good point"
# is still scored.
_SEP_WORDS = {
    "dash", "point", "dot", "hyphen", "slash",
    "punkt", "strich", "bindestrich", "schrägstrich",
}

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million",
    "null", "eins", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
    "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn",
    "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn", "zwanzig",
    "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig",
    "neunzig", "hundert", "tausend",
}


def _drop_separators(text: str) -> str:
    """Remove spoken punctuation words that sit next to a number."""
    words = text.split()
    keep = []
    for i, w in enumerate(words):
        if w in _SEP_WORDS:
            prev = words[i - 1] if i else ""
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if prev in _NUMBER_WORDS or nxt in _NUMBER_WORDS:
                continue
        keep.append(w)
    return " ".join(keep)


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


def _digits_to_words(text: str, lang: str, digitwise: bool = False) -> str:
    """Replace digit runs with their spelled-out form so "2 4 1" and
    "zwei vier eins" compare equal. Prefers num2words when installed.

    `digitwise` spells each digit separately — "1227" becomes "one two two
    seven" rather than "one thousand two hundred twenty seven". Both
    readings occur in real speech and neither is wrong: a quantity is read
    as a cardinal, an identifier (ticket, phone, version) digit by digit.
    A model that writes `1227` has not told us which one it heard, so
    scoring tries both and keeps the better — see `score()`."""
    try:
        from num2words import num2words
    except ImportError:
        num2words = None

    def repl(m: re.Match[str]) -> str:
        run = m.group(0)
        if digitwise:
            return " " + " ".join(spell_number(int(d), lang) for d in run) + " "
        value = int(run)
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


def normalize(
    text: str, lang: str = "en", numbers: bool = True, digitwise: bool = False
) -> str:
    """Normalise a transcript for scoring. `numbers=False` gives the strict
    variant behind `wer_raw`; `digitwise` spells digit runs one digit at a
    time instead of as a cardinal."""
    lang = (lang or "en").split("-")[0].split("_")[0].lower()
    out = text or ""
    if numbers:
        # Digits are spelled out before punctuation is stripped, so a
        # version string stays one number per component rather than
        # dissolving into unrelated digits.
        out = _digits_to_words(out, lang, digitwise=digitwise)

    out = _strip_punct(out)
    if numbers:
        for pattern, replacement in _VARIANTS:
            out = re.sub(pattern, replacement, out)
        out = _drop_separators(out)
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

    # A digit run does not say how it was spoken. "1227" is a cardinal if it
    # is a quantity and a digit sequence if it is a ticket number, and a
    # model that emits digits has discarded that distinction — while one
    # that spells the words out has kept it. Scoring only the cardinal
    # reading therefore charges digit-emitting models for a decision the
    # reference made, not for anything they misheard: on
    # support-call-en it cost parakeet-tdt-v2 5.2 WER points and moved it
    # from 7th place to 2nd once removed. So both readings are tried, with
    # the same rule applied to reference and hypothesis, and the better one
    # is kept. Numeral formatting is not what this benchmark measures.
    alt_ref = normalize(reference, lang, numbers=True, digitwise=True)
    alt_hyp = normalize(hypothesis, lang, numbers=True, digitwise=True)

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

        if jiwer.wer(alt_ref, alt_hyp) < jiwer.wer(ref, hyp):
            ref, hyp = alt_ref, alt_hyp
        out = jiwer.process_words(ref, hyp)
        wer = out.wer
        hits, subs, dels, ins = out.hits, out.substitutions, out.deletions, out.insertions
        cer = jiwer.cer(ref, hyp)
        wer_raw = jiwer.wer(ref_raw, hyp_raw)
    except ImportError:
        if _word_rates(alt_ref, alt_hyp)[0] < _word_rates(ref, hyp)[0]:
            ref, hyp = alt_ref, alt_hyp
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


# ──────────────────────────────────────────────
# Speaker-attributed transcription: cpWER
# ──────────────────────────────────────────────
#
# WER and DER measure different halves of the same job and neither one
# answers the question a meeting transcript is actually judged on: are the
# right words attributed to the right person? A pipeline can score 5 % WER
# and 20 % DER and still produce minutes nobody can use, because the
# sentence is correct and filed under the wrong name.
#
# cpWER (concatenated minimum-permutation WER, from CHiME-6/7 DASR) closes
# that gap. Concatenate every speaker's utterances into one stream per
# speaker on both sides, find the speaker mapping that minimises total
# errors, and report WER over the whole thing. A speaker the system never
# found costs all their words as deletions; a speaker it invented costs all
# of theirs as insertions. There is no collar and no free pass: attribution
# errors show up as substitutions in the stream they landed in *and* as
# deletions in the one they left.
#
# It is deliberately harsh, and that harshness is the point — it is the
# only number here that a pipeline configuration can be selected on,
# because it is the only one that both halves can lose.


def _speaker_streams(turns: list[dict]) -> dict[str, str]:
    """Concatenate each speaker's text in the order given.

    Order matters: cpWER compares streams, so shuffling a speaker's turns
    changes the alignment. Callers pass turns in time order, which is what
    both `reference.json` and the pipeline output already do."""
    streams: dict[str, list[str]] = {}
    for turn in turns:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        streams.setdefault(str(turn["speaker"]), []).append(text)
    return {spk: " ".join(parts) for spk, parts in streams.items()}


def _pair_errors(hyp_text: str, ref_text: str, lang: str) -> tuple[int, int]:
    """(errors, reference words) for one hypothesis stream against one
    reference stream, using the same normalisation as `score()` so a
    numeral-formatting difference does not count here either."""
    if not ref_text.strip():
        # Nothing to match: every hypothesis word is an insertion.
        return len(normalize(hyp_text, lang, numbers=True).split()), 0
    result = score(hyp_text, ref_text, lang)
    if result is None:
        return 0, 0
    errors = result["substitutions"] + result["deletions"] + result["insertions"]
    return errors, result["reference_words"]


def cp_wer(
    reference: list[dict],
    hypothesis: list[dict],
    lang: str = "en",
) -> dict | None:
    """Score a speaker-attributed transcript.

    Both arguments are lists of dicts with `speaker` and `text`, in time
    order — the shape `reference.json` already stores and the shape a
    pipeline emits. Speaker labels are arbitrary on the hypothesis side and
    are matched by content, not by name.

    Returns cpWER plus the per-speaker breakdown and the mapping that was
    chosen, or None when the reference has no words.
    """
    ref_streams = _speaker_streams(reference)
    hyp_streams = _speaker_streams(hypothesis)
    if not ref_streams:
        return None

    ref_labels = sorted(ref_streams)
    hyp_labels = sorted(hyp_streams)

    # Cost matrix of hypothesis stream × reference stream. Total errors
    # decompose as a sum over assigned pairs plus the unassigned ones, and
    # the denominator is fixed, so minimising the assignment cost minimises
    # cpWER exactly — this is not an approximation.
    cost: dict[tuple[str, str], int] = {}
    ref_words: dict[str, int] = {}
    for h in hyp_labels:
        for r in ref_labels:
            errs, nwords = _pair_errors(hyp_streams[h], ref_streams[r], lang)
            cost[(h, r)] = errs
            ref_words[r] = nwords

    # An unmatched reference speaker costs all their words as deletions; an
    # unmatched hypothesis speaker costs all of theirs as insertions.
    miss_cost = {r: ref_words.get(r, 0) for r in ref_labels}
    extra_cost = {
        h: len(normalize(hyp_streams[h], lang, numbers=True).split())
        for h in hyp_labels
    }

    mapping = _assign_speakers(hyp_labels, ref_labels, cost, miss_cost, extra_cost)

    total_ref = sum(ref_words.get(r, 0) for r in ref_labels)
    if total_ref == 0:
        return None

    errors = 0
    per_speaker = {}
    for r in ref_labels:
        h = next((hh for hh, rr in mapping.items() if rr == r), None)
        if h is None:
            errors += miss_cost[r]
            per_speaker[r] = {
                "matched": None,
                "errors": miss_cost[r],
                "reference_words": ref_words.get(r, 0),
                "wer": 1.0 if ref_words.get(r, 0) else None,
            }
            continue
        e = cost[(h, r)]
        errors += e
        nw = ref_words.get(r, 0)
        per_speaker[r] = {
            "matched": h,
            "errors": e,
            "reference_words": nw,
            "wer": round(e / nw, 4) if nw else None,
        }
    for h in hyp_labels:
        if h not in mapping:
            errors += extra_cost[h]

    return {
        "cpwer": round(errors / total_ref, 4),
        "errors": int(errors),
        "reference_words": int(total_ref),
        "reference_speakers": len(ref_labels),
        "hypothesis_speakers": len(hyp_labels),
        "missed_speakers": sum(1 for r in ref_labels
                               if not any(rr == r for rr in mapping.values())),
        "extra_speakers": sum(1 for h in hyp_labels if h not in mapping),
        "mapping": {str(h): str(r) for h, r in sorted(mapping.items())},
        "per_speaker": per_speaker,
    }


def _assign_speakers(
    hyp_labels: list[str],
    ref_labels: list[str],
    cost: dict[tuple[str, str], int],
    miss_cost: dict[str, int],
    extra_cost: dict[str, int],
) -> dict[str, str]:
    """Hypothesis→reference assignment minimising total errors.

    Padded to a square matrix so leaving a speaker unassigned is a choice
    the solver can make and price, rather than something forced by the
    shapes. That matters when a system splits one person into two: pairing
    both fragments is impossible, and the cheaper fragment should be
    allowed to go unmatched rather than displace a real speaker."""
    if not hyp_labels or not ref_labels:
        return {}

    n = max(len(hyp_labels), len(ref_labels))
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i < len(hyp_labels) and j < len(ref_labels):
                    matrix[i, j] = cost[(hyp_labels[i], ref_labels[j])]
                elif i < len(hyp_labels):
                    # Hypothesis speaker matched to nothing: all insertions.
                    matrix[i, j] = extra_cost[hyp_labels[i]]
                elif j < len(ref_labels):
                    # Reference speaker matched to nothing: all deletions.
                    matrix[i, j] = miss_cost[ref_labels[j]]
        rows, cols = linear_sum_assignment(matrix)
        return {
            hyp_labels[i]: ref_labels[j]
            for i, j in zip(rows, cols)
            if i < len(hyp_labels) and j < len(ref_labels)
        }
    except ImportError:
        pass

    import itertools

    if n <= MAX_EXHAUSTIVE_LABELS:
        best, best_map = None, {}
        shorter, longer = (hyp_labels, ref_labels) if len(hyp_labels) <= len(ref_labels) \
            else (ref_labels, hyp_labels)
        for perm in itertools.permutations(longer, len(shorter)):
            pairs = list(zip(shorter, perm)) if shorter is hyp_labels \
                else [(h, r) for r, h in zip(shorter, perm)]
            total = sum(cost[(h, r)] for h, r in pairs)
            total += sum(miss_cost[r] for r in ref_labels
                         if r not in {r for _, r in pairs})
            total += sum(extra_cost[h] for h in hyp_labels
                         if h not in {h for h, _ in pairs})
            if best is None or total < best:
                best, best_map = total, dict(pairs)
        return best_map

    mapping, used_ref, used_hyp = {}, set(), set()
    for (h, r), _c in sorted(cost.items(), key=lambda kv: kv[1]):
        if h in used_hyp or r in used_ref:
            continue
        mapping[h] = r
        used_hyp.add(h)
        used_ref.add(r)
    return mapping
