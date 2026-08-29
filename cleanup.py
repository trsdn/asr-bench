"""LLM cleanup for ASR transcripts, with a guard that can veto it.

An ASR transcript of a meeting is not what anyone wants to read. It has no
sentence boundaries a human would choose, it spells numbers the way they
were spoken, and it carries every stumble the speaker made. Handing it to a
language model to tidy up is the obvious last step, and it is the step where
this pipeline can lose more than it gains.

The failure mode is specific. A cleanup model rewrites fluently, so its
mistakes are fluent too: `Ticket vier null neun strich zwei eins` becomes
`Ticket 409-12`, `Version 2.4.1` becomes `Version 2.4`, `120 Millisekunden`
becomes `100 Millisekunden`. None of that looks wrong. WER may even improve,
because the model fixed six disfluencies while breaking one number. For
meeting minutes that trade is catastrophic in a way no average error rate
shows: a transcript with a wrong ticket number is worse than a transcript
with none, because it will be believed.

So the cleanup here is not trusted, it is *checked*. The model proposes, and
a reference-free guard decides whether the proposal may be accepted. The
guard protects the classes where hallucination is both likely and expensive:

* **numbers** — compared as a digit signature, so that a cleanup rewriting
  `vier null neun` to `409` is recognised as faithful while one rewriting it
  to `410` is not. This is the whole reason the comparison cannot be done on
  strings: the *desired* behaviour changes the surface form of exactly the
  tokens that must not change in value.
* **identifiers and acronyms** — anything mixing letters and digits, and
  any run of capitals. These are the tokens a model is most tempted to
  "correct" into a word it knows.
* **wholesale divergence** — a similarity floor against the raw text. Asked
  to tidy a transcript, models summarise it instead. That is not a cleanup
  and the guard treats it as a failure regardless of how good the prose is.

What the guard does not protect is proper nouns, and that is a real gap
rather than an oversight: German capitalises every noun, so capitalisation
cannot identify a name, and a reference-free check has nothing else to go
on. Names are the second most likely thing a cleanup damages. Detecting
that needs either a lexicon or the reference, and the reference is not
available at inference time.

The guard is deliberately conservative in one direction only. It vetoes and
falls back to the raw text; it never edits the model's output. A silent
partial repair would be a third version of the transcript that neither the
model nor the ASR produced, and nobody could reason about it.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import score

# ──────────────────────────────────────────────
# Canonicalising numbers
# ──────────────────────────────────────────────

# Both spellings and both languages map onto one value, built from score.py's
# own speller so the cleanup and the metric never disagree about what a
# number is called.
_MAX_PHRASE = 4


def _build_number_phrases() -> dict[str, dict[tuple[str, ...], int]]:
    table: dict[str, dict[tuple[str, ...], int]] = {"de": {}, "en": {}}
    for lang in ("de", "en"):
        for value in range(0, 1000):
            for variant in {score.spell_number(value, lang)}:
                words = tuple(variant.split())
                if len(words) <= _MAX_PHRASE:
                    table[lang].setdefault(words, value)
        # "einhundert" and "one hundred twenty" style variants that score.py
        # collapses during normalisation, registered here so a raw
        # transcript that was never normalised still resolves.
        table[lang].setdefault(("hundert",), 100)
        table[lang].setdefault(("tausend",), 1000)
        table[lang].setdefault(("hundred",), 100)
        table[lang].setdefault(("thousand",), 1000)
    return table


_PHRASES = _build_number_phrases()
_TOKEN = re.compile(r"\d+|[^\W\d_]+", flags=re.UNICODE)


def digit_signature(text: str, lang: str = "en") -> str:
    """Every number in `text`, reduced to the digits it denotes.

    `Ticket 409-21` and `Ticket vier null neun strich zwei eins` both give
    `40921`; `Version 2.4.1` and `Version zwei punkt vier punkt eins` both
    give `241`. A cleanup is free to change how a number is written and not
    free to change which number it is, and this is the only comparison that
    expresses that distinction.
    """
    phrases = _PHRASES.get(lang, _PHRASES["en"])
    words = [w.lower() for w in _TOKEN.findall(text)]
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.isdigit():
            out.append(w)
            i += 1
            continue
        matched = False
        for n in range(min(_MAX_PHRASE, len(words) - i), 0, -1):
            phrase = tuple(words[i:i + n])
            if phrase in phrases:
                out.append(str(phrases[phrase]))
                i += n
                matched = True
                break
        if not matched:
            i += 1
    return "".join(out)


# ──────────────────────────────────────────────
# Identifiers and acronyms
# ──────────────────────────────────────────────

_ID = re.compile(r"\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]{2,}\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")


def identifiers(text: str) -> set[str]:
    """Tokens mixing letters and digits — `v2`, `Postgres16`, `ABC123`.

    Unlike numbers these survive a cleanup verbatim or not at all, so they
    can be compared as strings. Case-folded, because a model is entitled to
    fix casing.
    """
    return {m.group(0).lower() for m in _ID.finditer(text)}


def acronyms(text: str) -> set[str]:
    """Runs of capitals, case-folded.

    These need a weaker test than identifiers do. ASR output is frequently
    uncased, so a cleanup turning `api gateway` into `API Gateway` is doing
    precisely its job — treating the new capitals as an invented entity
    would veto the desired behaviour. So an acronym is only required to
    survive as a *token*; whether it is capitalised is not the guard's
    business.
    """
    return {m.group(0).lower() for m in _ACRONYM.finditer(text)}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text)}


# ──────────────────────────────────────────────
# The guard
# ──────────────────────────────────────────────


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    digits_before: str = ""
    digits_after: str = ""
    lost_ids: list[str] = field(default_factory=list)
    invented_ids: list[str] = field(default_factory=list)
    similarity: float = 1.0

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "digits_before": self.digits_before,
            "digits_after": self.digits_after,
            "lost_ids": self.lost_ids,
            "invented_ids": self.invented_ids,
            "similarity": round(self.similarity, 4),
        }


def _mask_numbers(text: str, lang: str) -> str:
    """Replace every number with a single placeholder.

    The similarity floor and the digit signature have to look at disjoint
    parts of the text, or they contradict each other. Rewriting `vier null
    neun strich zwei eins` as `409-21` is the behaviour the cleanup exists
    for, and the digit check passes it — but it deletes five words out of
    nine, which a naive similarity floor reads as the model having thrown
    the sentence away. Masking numbers first means the floor only ever
    judges the prose, which is the only thing left for it to judge.
    """
    phrases = _PHRASES.get(lang, _PHRASES["en"])
    words = [w.lower() for w in _TOKEN.findall(text)]
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.isdigit():
            if not out or out[-1] != "\x00":
                out.append("\x00")
            i += 1
            continue
        matched = False
        for n in range(min(_MAX_PHRASE, len(words) - i), 0, -1):
            if tuple(words[i:i + n]) in phrases:
                if not out or out[-1] != "\x00":
                    out.append("\x00")
                i += n
                matched = True
                break
        if not matched:
            out.append(w)
            i += 1
    return " ".join(out)


def similarity(raw: str, cleaned: str, lang: str = "en") -> float:
    """Word-level similarity of the prose, numbers masked.

    Normalised, so the floor is not tripped by the punctuation and casing a
    cleanup is supposed to change, and number-masked so it does not
    double-punish the number rewriting the digit check already approved.
    """
    a = _mask_numbers(score.normalize(raw, lang, numbers=False), lang).split()
    b = _mask_numbers(score.normalize(cleaned, lang, numbers=False), lang).split()
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def verdict(
    raw: str, cleaned: str, lang: str = "en", min_similarity: float = 0.6
) -> Verdict:
    """Decide whether `cleaned` may replace `raw`.

    Checked in order of how badly the failure would mislead a reader: a
    changed number first, then a lost identifier, then wholesale rewriting.
    The first failure wins, because reporting three symptoms of one runaway
    generation is noise.
    """
    before = digit_signature(raw, lang)
    after = digit_signature(cleaned, lang)
    sim = similarity(raw, cleaned, lang)

    common = dict(
        digits_before=before, digits_after=after, similarity=sim
    )

    if before != after:
        return Verdict(
            ok=False,
            reason=f"numbers changed: {before or '(none)'} -> {after or '(none)'}",
            **common,
        )

    ids_before = identifiers(raw)
    ids_after = identifiers(cleaned)
    lost = sorted(ids_before - ids_after)
    invented = sorted(ids_after - ids_before)
    if lost:
        return Verdict(
            ok=False,
            reason=f"identifiers lost: {', '.join(lost)}",
            lost_ids=lost,
            invented_ids=invented,
            **common,
        )
    if invented:
        return Verdict(
            ok=False,
            reason=f"identifiers invented: {', '.join(invented)}",
            lost_ids=lost,
            invented_ids=invented,
            **common,
        )

    dropped = sorted(acronyms(raw) - _tokens(cleaned))
    if dropped:
        return Verdict(
            ok=False,
            reason=f"acronyms dropped: {', '.join(dropped)}",
            lost_ids=dropped,
            **common,
        )

    if sim < min_similarity:
        return Verdict(
            ok=False,
            reason=f"rewritten beyond recognition: similarity {sim:.2f} < {min_similarity:.2f}",
            **common,
        )

    return Verdict(ok=True, reason="accepted", **common)


# ──────────────────────────────────────────────
# Backends
# ──────────────────────────────────────────────

_PROMPT = {
    "de": (
        "Du bereinigst ein automatisch erzeugtes Transkript. Korrigiere "
        "Zeichensetzung, Groß- und Kleinschreibung und offensichtliche "
        "Erkennungsfehler. Entferne Füllwörter und Wiederholungen. "
        "Ändere KEINE Zahlen, Ticketnummern, Versionen oder Namen. "
        "Fasse nichts zusammen und ergänze nichts. Gib ausschließlich das "
        "bereinigte Transkript zurück.\n\nTranskript:\n"
    ),
    "en": (
        "You are cleaning up an automatically generated transcript. Fix "
        "punctuation, capitalisation and obvious recognition errors. Remove "
        "filler words and repetitions. Do NOT change any numbers, ticket "
        "identifiers, versions or names. Do not summarise and do not add "
        "anything. Return only the cleaned transcript.\n\nTranscript:\n"
    ),
}


class EchoBackend:
    """Returns the input unchanged. The honest baseline: a cleanup that does
    nothing is guaranteed not to damage anything, so any real backend has to
    beat zero rather than beat nothing."""

    name = "echo"

    def __call__(self, text: str, lang: str) -> str:
        return text


class ChatBackend:
    """Any instruct-tuned causal LM with a chat template.

    The first choice was Phi-4-multimodal, since the benchmark already has
    it cached and adding a multi-gigabyte download to a speech benchmark for
    a text step is hard to justify. It does not load in this environment:
    the checkpoint attaches its speech and vision LoRA adapters through
    `peft`, and `peft` 0.19 reaches for `prepare_inputs_for_generation` on
    the base model, which `transformers` 4.57 no longer puts there. That is
    a load-time failure with no argument that avoids it, and it is the same
    class of version conflict as [#9]. Fixing it would mean pinning
    `transformers` down, which the NeMo models in this repo need pinned up.

    So the cleanup uses a small separate text model instead. It is a
    deliberate choice of the *smallest* thing that can do the job: if
    cleanup only pays off with a large model, the wall-clock argument
    changes completely, and that is worth knowing.
    """

    name = "chat"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_id = model_id
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        import hf_runners
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, dtype = hf_runners._device_and_dtype(None, None)
        self._device = device
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype
        ).to(device)
        self._model.eval()

    def __call__(self, text: str, lang: str) -> str:
        self._load()
        import torch

        messages = [
            {"role": "system", "content": _PROMPT.get(lang, _PROMPT["en"])},
            {"role": "user", "content": text},
        ]
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                # Generous enough that a faithful cleanup is never cut off,
                # tight enough that a runaway generation stops. A truncated
                # output is rejected by the guard anyway, as lost numbers.
                max_new_tokens=min(2048, int(len(text.split()) * 2.0) + 128),
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(
            ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


BACKENDS = {"echo": EchoBackend, "chat": ChatBackend}


# ──────────────────────────────────────────────
# Applying it
# ──────────────────────────────────────────────


def clean(
    text: str, backend, lang: str = "en", min_similarity: float = 0.6
) -> tuple[str, str, Verdict]:
    """Propose a cleanup and accept it only if the guard agrees.

    Returns the accepted text, the raw proposal, and the verdict. The
    proposal is kept even when it is rejected, so that the cost of running
    without a guard can be scored from the same generation rather than from
    a second, differently-sampled one.

    On a veto the original is returned unchanged — never a partial repair.
    A silent partial repair would be a third version of the transcript that
    neither the model nor the ASR produced, and nobody could reason about
    it.
    """
    try:
        proposed = backend(text, lang)
    except Exception as exc:  # a failed generation is a veto, not a crash
        return text, "", Verdict(ok=False, reason=f"backend failed: {exc}")
    if not proposed.strip():
        return text, proposed, Verdict(ok=False, reason="backend returned nothing")
    v = verdict(text, proposed, lang, min_similarity)
    return (proposed if v.ok else text), proposed, v


def clean_transcript(
    transcript: list[dict], backend, lang: str, min_similarity: float = 0.6
) -> tuple[list[dict], list[dict], list[Verdict]]:
    """Returns the guarded transcript, the unguarded one (every proposal
    accepted), and the verdicts."""
    guarded, unguarded, verdicts = [], [], []
    for seg in transcript:
        accepted, proposed, v = clean(seg["text"], backend, lang, min_similarity)
        guarded.append({**seg, "text": accepted})
        unguarded.append({**seg, "text": proposed or seg["text"]})
        verdicts.append(v)
    return guarded, unguarded, verdicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--result", required=True, help="a pipeline.py result JSON")
    ap.add_argument("--backend", default="chat", choices=sorted(BACKENDS))
    ap.add_argument("--min-similarity", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = json.loads(Path(args.result).read_text())
    lang = result.get("language", "en")
    backend = BACKENDS[args.backend]()

    cleaned, unguarded, verdicts = clean_transcript(
        result["transcript"], backend, lang, args.min_similarity
    )

    reference = json.loads(
        (Path(result["session"]) / "reference.json").read_text()
    )["channels"]["mixed"]
    ref_turns = [
        {"speaker": t["speaker"], "text": t["text"]} for t in reference
    ]

    before = score.cp_wer(ref_turns, result["transcript"], lang)
    after = score.cp_wer(ref_turns, cleaned, lang)
    naive = score.cp_wer(ref_turns, unguarded, lang)

    accepted = sum(1 for v in verdicts if v.ok)
    print(f"\n{args.backend} cleanup on {Path(result['session']).name}")
    print(f"  cpWER      {before['cpwer']:.1%} raw")
    print(f"             {after['cpwer']:.1%} guarded "
          f"({accepted} of {len(verdicts)} proposals accepted)")
    print(f"             {naive['cpwer']:.1%} unguarded "
          f"(every proposal accepted)")
    for i, v in enumerate(verdicts):
        if not v.ok:
            print(f"    segment {i} vetoed: {v.reason}")

    out = Path(args.out or Path(args.result).with_suffix(".cleanup.json"))
    out.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "min_similarity": args.min_similarity,
                "cpwer_raw": before["cpwer"],
                "cpwer_guarded": after["cpwer"],
                "cpwer_unguarded": naive["cpwer"],
                "accepted": accepted,
                "segments": len(verdicts),
                "verdicts": [v.as_dict() for v in verdicts],
                "transcript": cleaned,
                "proposals": unguarded,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
