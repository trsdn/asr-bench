"""
score.py — WER / CER against the synthetic ground truth.

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
