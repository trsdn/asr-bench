"""
synth.py — build synthetic multi-speaker conversations with known ground truth.

Recording real meetings gives you audio but no reference transcript, so
you can only ever eyeball model outputs against each other. Synthesising
the conversation from a script inverts that: we *start* from the text, so
every session ships a word-exact reference and the benchmark can report
real WER instead of vibes.

Pipeline
--------
    conversation script (YAML)
      → one TTS render per turn (macOS `say`, or Piper)
      → laid out on a timeline with pauses / overlaps
      → per-speaker isolated channels + one mixed conversation channel
      → optional degradation profile (see degrade.py)
      → session dir with audio/, reference/, reference.json, session.json

Usage
-----
    # one session per difficulty level, from the same script
    uv run python synth.py --script conversations/standup-de.yaml \\
        --degrade clean phone farfield

    uv run python bench.py --session sessions/standup-de__phone \\
        --models whisper-large-v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audio_io import TARGET_SR, load_mono_16k, write_wav
from degrade import PROFILES, apply_profile

REPO_DIR = Path(__file__).resolve().parent
TTS_CACHE = REPO_DIR / ".tts-cache"

DEFAULT_GAP = 0.35        # seconds of silence between turns

# macOS `say` measures speaking rate in words per minute and defaults to
# 175 (verified: `-r 175` and no `-r` produce byte-identical durations).
# A speaker's `rate:` is written in those units. Piper and Kokoro have no
# notion of wpm, so the script value is converted into each engine's own
# speed control against this same baseline — otherwise the per-speaker
# tempo silently applies on one engine out of three, and a cross-engine
# delta measures tempo as much as synthesiser fingerprint.
SAY_DEFAULT_WPM = 175

# Bumped when a change alters the waveform a cache key already maps to.
# Rate support for Piper and Kokoro is exactly that: same key, different
# audio, so old entries have to be treated as misses rather than served.
TTS_CACHE_VERSION = 2

# Level increase applied to a turn that begins while the previous speaker
# is still talking — the Lombard effect, which makes people talk louder
# against competing speech. 3 dB sits in the middle of the range reported
# for competing-talker conditions. This is deliberately a crude model:
# real Lombard speech also raises F0 and tilts the spectrum upward, and
# no TTS engine used here will do that on request, so overlapped speech
# is still easier to separate than the real thing.
LOMBARD_GAIN_DB = 3.0
LEAD_IN = 0.5             # silence before the first turn
TAIL = 0.8                # silence after the last turn

# macOS ships a pile of "novelty" voices (Bells, Zarvox, …) that are
# singing or robotic. They'd make the benchmark measure the wrong thing,
# so auto-assignment skips them; an explicit `voice:` in the script can
# still select one deliberately.
NOVELTY_VOICES = {
    "Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos",
    "Good News", "Jester", "Organ", "Superstar", "Trinoids", "Whisper",
    "Wobble", "Zarvox", "Hysterical", "Deranged", "Bruce", "Junior",
    "Kathy", "Princess", "Ralph", "Fred", "Agnes", "Bad Nachricht",
}

# Preferred natural voices per language, in assignment order. Anything
# not installed on this machine is skipped silently.
PREFERRED_VOICES = {
    "de": ["Anna", "Markus", "Petra", "Yannick", "Helena", "Martin", "Viktor"],
    "en": ["Samantha", "Daniel", "Karen", "Alex", "Moira", "Tom", "Fiona", "Rishi"],
    "fr": ["Thomas", "Amelie", "Audrey", "Aurelie"],
    "es": ["Monica", "Jorge", "Paulina", "Diego"],
    "it": ["Alice", "Luca", "Federica"],
}

# Every engine here is a different synthesis lineage — Apple's `say`,
# Piper (VITS) and Kokoro (StyleTTS2). That is the whole point: a model
# scored against one engine's output tells you as much about that engine
# as about the model. Rendering the same script through several engines
# turns "is this model good" into "is this model good regardless of who
# is speaking", which is the question a benchmark should answer.
TTS_BACKENDS = ("say", "piper", "kokoro")

# Piper voices as (name, gender), one entry per distinct *speaker*, best
# quality available for that speaker. Gender is only used to match a
# script's `gender:` field where it declares one; distinctness always
# wins over matching, because two speakers sharing a voice would make
# diarisation numbers meaningless while a mismatched voice would not.
#
# Caveat worth knowing before reading German cross-engine numbers: the
# German Piper voices are unevenly trained (only Thorsten reaches
# `medium`), so a German delta between `say` and `piper` mixes model
# behaviour with voice quality. The English voices are all medium/high
# and do not have this problem.
PIPER_VOICES = {
    "de": [
        ("de_DE-thorsten-medium", "m"),
        ("de_DE-kerstin-low", "f"),
        ("de_DE-karlsson-low", "m"),
        ("de_DE-eva_k-x_low", "f"),
        ("de_DE-pavoque-low", "m"),
        ("de_DE-ramona-low", "f"),
    ],
    "en": [
        ("en_US-ryan-high", "m"),
        ("en_US-lessac-high", "f"),
        ("en_US-joe-medium", "m"),
        ("en_US-amy-medium", "f"),
        ("en_US-hfc_male-medium", "m"),
        ("en_US-kristin-medium", "f"),
        ("en_GB-alan-medium", "m"),
        ("en_GB-cori-high", "f"),
    ],
}

# Kokoro ships American and British English plus a handful of other
# languages, but no German — hence English-only here. Names encode
# accent and gender: a=American, b=British, f=female, m=male.
KOKORO_VOICES = {
    "en": [
        ("am_michael", "m"), ("af_heart", "f"),
        ("am_fenrir", "m"), ("af_bella", "f"),
        ("am_puck", "m"), ("af_nicole", "f"),
        ("bm_george", "m"), ("bf_emma", "f"),
    ],
}

# Kokoro's language codes are single letters rather than ISO codes.
KOKORO_LANG_CODES = {"en": "a"}

# Where Piper voice models are downloaded to. Kept beside the audio cache
# rather than in the repo: these are weights, not source.
PIPER_VOICE_DIR = Path(
    os.environ.get("ASR_BENCH_PIPER_DIR", REPO_DIR / ".piper-voices")
)


# ──────────────────────────────────────────────
# Script model
# ──────────────────────────────────────────────


@dataclass
class Speaker:
    id: str
    name: str
    voice: str | None = None
    rate: int | None = None       # words per minute; mapped to each engine's own control
    backend: str | None = None    # override the global TTS backend
    # "f" / "m", optional. Only used to pick a plausible voice from an
    # engine's catalogue; it has no effect on scoring.
    gender: str | None = None


@dataclass
class Turn:
    speaker: str
    text: str
    gap: float = DEFAULT_GAP      # silence before this turn; negative = overlap


@dataclass
class Conversation:
    name: str
    language: str
    speakers: list[Speaker]
    turns: list[Turn]
    notes: str = ""
    _by_id: dict[str, Speaker] = field(default_factory=dict, repr=False)

    def speaker(self, sid: str) -> Speaker:
        return self._by_id[sid]


def load_script(path: Path) -> Conversation:
    """Parse a conversation YAML into a validated `Conversation`.

    Validation is strict and up-front: a typo'd speaker id in turn 40 of a
    long script should fail before we spend a minute on TTS."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    speakers = [
        Speaker(
            id=str(s["id"]),
            name=str(s.get("name", s["id"])),
            voice=s.get("voice"),
            rate=s.get("rate"),
            backend=s.get("backend"),
            gender=(str(s["gender"]).strip().lower()[:1] or None)
            if s.get("gender") else None,
        )
        for s in raw.get("speakers", [])
    ]
    if not speakers:
        raise ValueError(f"{path}: no speakers defined")

    by_id = {s.id: s for s in speakers}
    if len(by_id) != len(speakers):
        raise ValueError(f"{path}: duplicate speaker ids")

    turns: list[Turn] = []
    for i, t in enumerate(raw.get("turns", [])):
        sid = str(t["speaker"])
        if sid not in by_id:
            raise ValueError(
                f"{path}: turn {i} references unknown speaker {sid!r} "
                f"(known: {', '.join(by_id)})"
            )
        text = str(t["text"]).strip()
        if not text:
            raise ValueError(f"{path}: turn {i} has empty text")
        turns.append(Turn(speaker=sid, text=text, gap=float(t.get("gap", DEFAULT_GAP))))

    if not turns:
        raise ValueError(f"{path}: no turns defined")

    return Conversation(
        name=str(raw.get("name", path.stem)),
        language=str(raw.get("language", "en")),
        speakers=speakers,
        turns=turns,
        notes=str(raw.get("notes", "")),
        _by_id=by_id,
    )


# ──────────────────────────────────────────────
# TTS backends
# ──────────────────────────────────────────────


def list_say_voices() -> list[tuple[str, str]]:
    """Return [(voice_name, locale)] from `say -v '?'`. Empty list if we
    aren't on macOS or `say` is unavailable."""
    if shutil.which("say") is None:
        return []
    try:
        out = subprocess.run(
            ["say", "-v", "?"], capture_output=True, check=True, text=True
        ).stdout
    except subprocess.CalledProcessError:
        return []

    voices: list[tuple[str, str]] = []
    for line in out.splitlines():
        # Column widths vary: padded for short names, a single space for
        # long locale-qualified ones, so we anchor on the locale + '#'.
        m = re.match(r"^(.+?)\s+([a-z]{2}(?:_[A-Z]{2})?)\s+#", line)
        if m:
            voices.append((m.group(1).strip(), m.group(2)))
    return voices


def _assign_from_pool(
    conv: Conversation,
    pool: list[tuple[str, str]],
    backend: str,
    lang: str,
) -> dict[str, str]:
    """Hand out distinct voices from a fixed pool.

    Shared by Piper and Kokoro, which unlike `say` have a known voice
    catalogue rather than whatever happens to be installed. A speaker's
    declared `gender:` is honoured when a matching voice is still free,
    and quietly ignored when it is not — a mismatched voice costs nothing
    but a raised eyebrow on playback, whereas failing to synthesise would
    cost a session."""
    taken = {s.voice for s in conv.speakers if s.voice}
    assigned: dict[str, str] = {}
    for spk in conv.speakers:
        if spk.voice:
            assigned[spk.id] = spk.voice
            continue
        free = [name for name, _ in pool if name not in taken]
        preferred = [
            name for name, gender in pool
            if name not in taken and spk.gender and gender == spk.gender
        ]
        candidate = next(iter(preferred or free), None)
        if candidate is None:
            raise RuntimeError(
                f"{backend} has only {len(pool)} distinct {lang!r} voices, "
                f"but the script has {len(conv.speakers)} speakers. Reuse "
                f"would make diarisation numbers meaningless, so set "
                f"`voice:` explicitly for the extra speakers."
            )
        assigned[spk.id] = candidate
        taken.add(candidate)
    return assigned


def assign_voices(conv: Conversation, backend: str) -> dict[str, str]:
    """Pick a distinct voice per speaker.

    Distinctness matters more than realism here: if two speakers share a
    voice, diarisation and speaker-attribution numbers become meaningless.
    Explicit `voice:` entries in the script always win."""
    lang = conv.language.split("-")[0].split("_")[0].lower()

    if backend == "piper":
        pool = PIPER_VOICES.get(lang)
        if not pool:
            raise RuntimeError(
                f"No Piper voices configured for {lang!r}. Add them to "
                f"PIPER_VOICES or set `voice:` per speaker."
            )
        return _assign_from_pool(conv, pool, "Piper", lang)

    if backend == "kokoro":
        pool = KOKORO_VOICES.get(lang)
        if not pool:
            raise RuntimeError(
                f"Kokoro has no {lang!r} voices — it covers English plus a "
                f"few other languages, but not German. Use --tts say or "
                f"--tts piper for this script."
            )
        return _assign_from_pool(conv, pool, "Kokoro", lang)

    available = list_say_voices()
    if not available:
        raise RuntimeError(
            "macOS `say` not available. Use --tts piper with explicit voice "
            "models, or run on macOS."
        )

    lang = conv.language.split("-")[0].split("_")[0].lower()
    by_name = {name: locale for name, locale in available}
    def base_name(voice: str) -> str:
        """'Eddy (Deutsch (Deutschland))' → 'Eddy'. Recent macOS releases
        expose most voices only in this locale-qualified form."""
        return voice.split(" (", 1)[0].strip()

    # Candidate pool in tiers: curated names first, then plain names, then
    # locale-qualified ones. The tiers matter because the first speakers
    # in a script get the most natural-sounding voices, and a benchmark
    # should not be harder for speaker C than for speaker A by accident.
    pool = [v for v in PREFERRED_VOICES.get(lang, []) if v in by_name]
    for qualified in (False, True):
        for name, locale in available:
            if not locale.lower().startswith(lang):
                continue
            if ("(" in name) != qualified:
                continue
            if base_name(name) in NOVELTY_VOICES or name in pool:
                continue
            pool.append(name)

    taken = {s.voice for s in conv.speakers if s.voice}
    assigned: dict[str, str] = {}
    for spk in conv.speakers:
        if spk.voice:
            if spk.voice not in by_name:
                raise ValueError(
                    f"Voice {spk.voice!r} for speaker {spk.id!r} is not installed. "
                    f"Run `say -v '?'` to list available voices."
                )
            assigned[spk.id] = spk.voice
            continue
        candidate = next((v for v in pool if v not in taken), None)
        if candidate is None:
            raise RuntimeError(
                f"Not enough distinct {lang!r} voices installed for "
                f"{len(conv.speakers)} speakers. Install more in System "
                f"Settings → Accessibility → Spoken Content, or set `voice:` "
                f"explicitly in the script."
            )
        assigned[spk.id] = candidate
        taken.add(candidate)
    return assigned


_PIPER_CACHE: dict[str, object] = {}
_KOKORO_CACHE: dict[str, object] = {}


def _piper_voice(name: str):
    """Load a Piper voice, downloading it on first use.

    Cached per process: a voice is a few tens of MB of ONNX weights, and
    a script re-uses the same speaker across dozens of turns."""
    if name in _PIPER_CACHE:
        return _PIPER_CACHE[name]
    from piper import PiperVoice
    from piper.download_voices import download_voice

    PIPER_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    model = PIPER_VOICE_DIR / f"{name}.onnx"
    if not model.exists():
        print(f"  downloading Piper voice {name} …")
        download_voice(name, PIPER_VOICE_DIR)
    _PIPER_CACHE[name] = PiperVoice.load(model)
    return _PIPER_CACHE[name]


def _kokoro_pipeline(lang_code: str):
    """Kokoro on the GPU. It is 82M parameters, so this costs little, but
    it is still a torch model and reloading it per turn would dominate
    synthesis time."""
    if lang_code in _KOKORO_CACHE:
        return _KOKORO_CACHE[lang_code]
    import torch
    from kokoro import KPipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _KOKORO_CACHE[lang_code] = KPipeline(lang_code=lang_code, device=device)
    return _KOKORO_CACHE[lang_code]


def synth_turn(
    text: str,
    voice: str,
    backend: str,
    rate: int | None,
    sr: int = TARGET_SR,
    language: str = "en",
) -> np.ndarray:
    """Render one utterance to mono float32 at `sr`, with an on-disk cache.

    The cache key covers everything that affects the waveform, so
    regenerating the same script at three degradation levels costs one TTS
    pass, not three."""
    key_src = json.dumps(
        {
            "t": text, "v": voice, "b": backend, "r": rate, "sr": sr,
            "l": language, "cv": TTS_CACHE_VERSION,
        },
        sort_keys=True,
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:20]
    cached = TTS_CACHE / f"{key}.wav"
    if cached.exists():
        return load_mono_16k(cached, sr)

    TTS_CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        if backend == "say":
            raw = tmpdir / "turn.aiff"
            # Text goes through a file, not argv: utterances contain
            # quotes, dashes and other characters `say` would otherwise
            # interpret or the shell would mangle.
            txt_file = tmpdir / "turn.txt"
            txt_file.write_text(text, encoding="utf-8")
            cmd = ["say", "-v", voice, "-o", str(raw), "-f", str(txt_file)]
            if rate:
                cmd += ["-r", str(rate)]
            subprocess.run(cmd, capture_output=True, check=True)
        elif backend == "piper":
            raw = tmpdir / "turn.wav"
            import wave

            from piper import SynthesisConfig

            piper_voice = _piper_voice(voice)
            # `length_scale` stretches the predicted duration, so it is
            # the inverse of a rate: a faster speaker needs a shorter
            # utterance. Left at the voice's own default when the script
            # does not ask for a tempo.
            syn_config = (
                SynthesisConfig(length_scale=SAY_DEFAULT_WPM / rate) if rate else None
            )
            with wave.open(str(raw), "wb") as wav:
                piper_voice.synthesize_wav(text, wav, syn_config=syn_config)
        elif backend == "kokoro":
            lang_code = KOKORO_LANG_CODES.get(
                language.split("-")[0].split("_")[0].lower(), "a"
            )
            pipeline = _kokoro_pipeline(lang_code)
            # Kokoro's `speed` is a direct multiplier on the reference
            # tempo, so it maps to the script's wpm the other way round
            # from Piper's length_scale.
            speed = rate / SAY_DEFAULT_WPM if rate else 1.0
            # Kokoro splits on its own and yields one result per chunk;
            # a turn is short, but joining is still the correct thing to
            # do rather than taking the first result.
            parts = [
                r.audio.detach().cpu().numpy()
                for r in pipeline(text, voice=voice, speed=speed)
                if r.audio is not None
            ]
            if not parts:
                raise RuntimeError(f"Kokoro produced no audio for: {text[:60]!r}")
            audio_24k = np.concatenate(parts).astype(np.float32)
            raw = tmpdir / "turn.wav"
            write_wav(raw, audio_24k, 24_000)
        else:
            raise ValueError(f"Unknown TTS backend: {backend}")

        audio = load_mono_16k(raw, sr)

    audio = trim_silence(audio, sr)
    write_wav(cached, audio, sr)
    return audio


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -45.0) -> np.ndarray:
    """Trim leading/trailing silence so the script's `gap` values are the
    only thing controlling timing. TTS engines pad utterances by varying
    amounts; without this the ground-truth timestamps drift."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    thresh = peak * (10.0 ** (threshold_db / 20.0))
    above = np.flatnonzero(np.abs(audio) > thresh)
    if above.size == 0:
        return audio
    pad = int(0.02 * sr)  # keep 20 ms so plosives aren't clipped off
    start = max(0, int(above[0]) - pad)
    end = min(audio.size, int(above[-1]) + pad)
    return audio[start:end]


# ──────────────────────────────────────────────
# Timeline assembly
# ──────────────────────────────────────────────


@dataclass
class Segment:
    speaker: str
    speaker_name: str
    start: float
    end: float
    text: str

    def to_json(self) -> dict:
        return {
            "speaker": self.speaker,
            "speaker_name": self.speaker_name,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


def _gain_db(audio: np.ndarray, db: float) -> np.ndarray:
    return (audio * (10.0 ** (db / 20.0))).astype(np.float32)


def build_timeline(
    conv: Conversation,
    voices: dict[str, str],
    backend: str,
    sr: int = TARGET_SR,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], list[Segment]]:
    """Render every turn and lay it out on a shared timeline.

    Returns per-speaker isolated tracks (full length, silent when that
    speaker isn't talking) plus the ground-truth segments. Mixing is left
    to the caller so it can also emit the isolated channels."""
    rendered: list[tuple[Turn, np.ndarray]] = []
    for i, turn in enumerate(conv.turns):
        spk = conv.speaker(turn.speaker)
        audio = synth_turn(
            turn.text,
            voices[turn.speaker],
            spk.backend or backend,
            spk.rate,
            sr,
            language=conv.language,
        )
        rendered.append((turn, audio))
        if verbose:
            print(
                f"  [{i + 1}/{len(conv.turns)}] {spk.name} "
                f"({voices[turn.speaker]}): {len(audio) / sr:5.2f}s  "
                f"{turn.text[:60]}{'…' if len(turn.text) > 60 else ''}"
            )

    # First pass: absolute sample offsets. A negative `gap` pulls the turn
    # back into the previous one, producing real overlapped speech.
    cursor = LEAD_IN
    placements: list[tuple[Turn, np.ndarray, int]] = []
    segments: list[Segment] = []
    for turn, audio in rendered:
        start = max(0.0, cursor + turn.gap)
        # A speaker who cuts in while someone else still holds the floor
        # raises their voice. Summing both turns at their original level
        # models two independent recordings rather than a contested
        # floor, and makes the separation easier than it is in reality.
        if start < cursor - 1e-6:
            audio = _gain_db(audio, LOMBARD_GAIN_DB)
        offset = int(round(start * sr))
        duration = len(audio) / sr
        placements.append((turn, audio, offset))
        segments.append(
            Segment(
                speaker=turn.speaker,
                speaker_name=conv.speaker(turn.speaker).name,
                start=start,
                end=start + duration,
                text=turn.text,
            )
        )
        cursor = start + duration

    total = int(round((cursor + TAIL) * sr))
    tracks = {s.id: np.zeros(total, dtype=np.float32) for s in conv.speakers}
    for turn, audio, offset in placements:
        end = min(total, offset + len(audio))
        tracks[turn.speaker][offset:end] += audio[: end - offset]

    return tracks, segments


def overlap_seconds(segments: list["Segment"]) -> float:
    """Total time where more than one speaker is talking.

    Computed as (sum of segment durations − length of their union) rather
    than from the `gap` values, which only describe the intended offset
    and say nothing about how long the rendered audio actually turned
    out."""
    if not segments:
        return 0.0
    total = sum(s.end - s.start for s in segments)

    union = 0.0
    ordered = sorted(segments, key=lambda s: s.start)
    cur_start, cur_end = ordered[0].start, ordered[0].end
    for seg in ordered[1:]:
        if seg.start > cur_end:
            union += cur_end - cur_start
            cur_start, cur_end = seg.start, seg.end
        else:
            cur_end = max(cur_end, seg.end)
    union += cur_end - cur_start
    return max(0.0, total - union)


def normalise_peak(audio: np.ndarray, target: float = 0.89) -> np.ndarray:
    """Scale to a fixed peak so degradation SNRs mean the same thing across
    sessions regardless of how loud a given voice renders."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return audio
    return (audio * (target / peak)).astype(np.float32)


# ──────────────────────────────────────────────
# Session writing
# ──────────────────────────────────────────────


def write_session(
    out_dir: Path,
    conv: Conversation,
    voices: dict[str, str],
    backend: str,
    tracks: dict[str, np.ndarray],
    segments: list[Segment],
    profile: str,
    seed: int,
    isolated: bool,
    sr: int = TARGET_SR,
) -> Path:
    """Write one session directory: audio, ground truth, and manifest."""
    audio_dir = out_dir / "audio"
    ref_dir = out_dir / "reference"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    mixed = normalise_peak(np.sum(list(tracks.values()), axis=0))

    channels: dict[str, np.ndarray] = {"mixed": mixed}
    if isolated:
        for sid, track in tracks.items():
            channels[f"spk-{sid}"] = normalise_peak(track)

    # Distinct noise realisation per channel (seed + index) so isolated
    # tracks aren't degraded with a correlated copy of the same noise —
    # that would be an unrealistically easy denoising target.
    channel_segments: dict[str, list[Segment]] = {}
    for idx, (name, audio) in enumerate(sorted(channels.items())):
        degraded = apply_profile(audio, sr, profile, seed=seed + idx)
        write_wav(audio_dir / f"{name}.wav", degraded, sr)

        if name == "mixed":
            segs = segments
        else:
            sid = name.removeprefix("spk-")
            segs = [s for s in segments if s.speaker == sid]
        channel_segments[name] = segs
        (ref_dir / f"{name}.txt").write_text(
            " ".join(s.text for s in segs) + "\n", encoding="utf-8"
        )

    reference = {
        "channels": {
            name: [s.to_json() for s in segs]
            for name, segs in channel_segments.items()
        }
    }
    (out_dir / "reference.json").write_text(
        json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    duration = len(mixed) / sr
    speech = sum(s.end - s.start for s in segments)
    manifest = {
        "name": out_dir.name,
        "source": "synthetic",
        "script": conv.name,
        "language": conv.language,
        "notes": conv.notes,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_rate": sr,
        "duration_seconds": round(duration, 2),
        "speech_seconds": round(speech, 2),
        "overlap_seconds": round(overlap_seconds(segments), 2),
        "channels": sorted(channels),
        "reference": True,
        "degradation": {"profile": profile, "seed": seed},
        "tts": {"backend": backend},
        "speakers": [
            {
                "id": s.id,
                "name": s.name,
                "voice": voices[s.id],
                "channel": f"spk-{s.id}" if isolated else None,
                "turns": sum(1 for t in conv.turns if t.speaker == s.id),
            }
            for s in conv.speakers
        ],
        "word_count": sum(len(s.text.split()) for s in segments),
    }
    (out_dir / "session.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_dir


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--script", type=Path, required=True,
        help="Conversation YAML (see conversations/ for examples).",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Session directory. Default: sessions/<script>__<tts>__<profile>.",
    )
    ap.add_argument(
        "--degrade", nargs="+", default=["clean"], choices=sorted(PROFILES),
        help="Degradation profile(s). One session is written per profile, "
             "which is the intended way to build a difficulty ladder.",
    )
    ap.add_argument(
        "--tts", default="say", choices=list(TTS_BACKENDS),
        help="TTS backend. Default: macOS `say`. Rendering the same "
             "script through several backends is what keeps the benchmark "
             "from measuring one synthesiser instead of the models.",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="Noise seed — same seed gives byte-identical audio.",
    )
    ap.add_argument(
        "--no-isolated", action="store_true",
        help="Only write the mixed conversation channel.",
    )
    ap.add_argument(
        "--list-voices", action="store_true",
        help="Print installed `say` voices for the script language and exit.",
    )
    args = ap.parse_args()

    conv = load_script(args.script.expanduser().resolve())

    if args.list_voices:
        lang = conv.language.split("-")[0].split("_")[0].lower()
        for name, locale in list_say_voices():
            if locale.lower().startswith(lang):
                flag = " (novelty)" if name in NOVELTY_VOICES else ""
                print(f"{name:32} {locale}{flag}")
        return 0

    voices = assign_voices(conv, args.tts)
    print(f"Script '{conv.name}' — {len(conv.speakers)} speakers, {len(conv.turns)} turns")
    for spk in conv.speakers:
        print(f"  {spk.id}: {spk.name} → voice {voices[spk.id]}")

    print("\nSynthesising turns…")
    tracks, segments = build_timeline(conv, voices, args.tts)

    written: list[Path] = []
    for profile in args.degrade:
        if args.out and len(args.degrade) == 1:
            out_dir = args.out.expanduser().resolve()
        else:
            base = args.out or (REPO_DIR / "sessions")
            # The TTS engine is part of the session identity, not a
            # footnote in the manifest: two sessions from the same script
            # and profile but different engines are different test data,
            # and burying that in a JSON field invites comparing numbers
            # that are not comparable.
            out_dir = (
                Path(base).expanduser().resolve()
                / f"{conv.name}__{args.tts}__{profile}"
            )
        print(f"\nWriting [{profile}] → {out_dir}")
        write_session(
            out_dir, conv, voices, args.tts, tracks, segments,
            profile=profile, seed=args.seed, isolated=not args.no_isolated,
        )
        written.append(out_dir)

    print("\nDone. Benchmark with:")
    for out_dir in written:
        try:
            shown = out_dir.relative_to(Path.cwd())
        except ValueError:
            shown = out_dir
        print(f"  uv run python bench.py --session {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
