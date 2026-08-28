"""
Shared audio I/O for the benchmark harness.

Everything in this repo works on the same in-memory representation:
mono `float32` in [-1, 1] at `TARGET_SR`. Keeping that conversion in one
place means `synth.py` (which writes audio) and `bench.py` (which reads
it) can never drift apart on sample rate or channel layout.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

TARGET_SR = 16_000


def load_mono_16k(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read `path`, downmix to mono, resample to `target_sr`. Returns
    float32 in [-1, 1].

    First tries libsndfile (via soundfile); some container/codec variants
    (notably CAF written by AVAudioFile) aren't parseable there, so we
    fall back to ffmpeg which handles everything in the wild. ffmpeg is a
    hard runtime dependency — install via `brew install ffmpeg` on macOS.
    """
    import soundfile as sf

    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0]
    except Exception as libsnd_err:
        audio, sr = ffmpeg_decode(path, target_sr)
        if audio is None:
            raise RuntimeError(
                f"Could not decode {path} via soundfile or ffmpeg: {libsnd_err}"
            )
    if sr != target_sr:
        import librosa  # lazy — librosa pulls in scipy etc.

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32, copy=False)


def ffmpeg_decode(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray | None, int]:
    """Decode `path` to mono float32 at `target_sr` via a single ffmpeg
    subprocess call. Returns (samples, sample_rate) or (None, 0)."""
    ffmpeg = require_ffmpeg(hard=False)
    if ffmpeg is None:
        return None, 0

    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-ac", "1",              # downmix to mono
        "-ar", str(target_sr),   # resample
        "-f", "f32le",           # raw float32 little-endian
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[ffmpeg] failed: {exc.stderr.decode(errors='replace')}", file=sys.stderr)
        return None, 0
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    return samples, target_sr


def write_wav(path: Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write mono float32 audio as 16-bit PCM WAV. 16-bit rather than
    float32 because every ASR runtime reads it without complaint, and the
    quantisation floor is far below anything the models resolve."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def require_ffmpeg(hard: bool = True) -> str | None:
    """Locate the ffmpeg binary. With `hard=True` a missing binary is a
    fatal error, since audio synthesis cannot proceed without it."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and hard:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (macOS: `brew install ffmpeg`)."
        )
    return ffmpeg


def ffmpeg_filter(
    audio: np.ndarray,
    sr: int,
    filter_chain: str,
    out_sr: int | None = None,
) -> np.ndarray:
    """Pipe `audio` through an ffmpeg `-af` filter chain and read it back.

    Used by the degradation profiles so we get battle-tested DSP (biquads,
    reverb, dynamics) instead of hand-rolled filters."""
    ffmpeg = require_ffmpeg()
    out_sr = out_sr or sr
    cmd = [
        ffmpeg,
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
        "-af", filter_chain,
        "-f", "f32le", "-ar", str(out_sr), "-ac", "1", "pipe:1",
    ]
    proc = subprocess.run(
        cmd, input=audio.astype(np.float32).tobytes(), capture_output=True, check=True
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def ffmpeg_codec_roundtrip(
    audio: np.ndarray,
    sr: int,
    codec: str,
    bitrate: str,
    codec_sr: int,
) -> np.ndarray:
    """Encode `audio` with a lossy codec and decode it back at `sr`.

    This is the honest way to simulate telephony / VoIP: the artefacts a
    real Opus-at-12kbps link produces are not reproducible with EQ alone.
    Returns the input unchanged if the encoder isn't in this ffmpeg build.
    """
    import tempfile

    ffmpeg = require_ffmpeg()
    suffix = {"libopus": ".opus", "libmp3lame": ".mp3", "aac": ".m4a"}.get(codec, ".mka")

    with tempfile.TemporaryDirectory() as tmp:
        encoded = Path(tmp) / f"degraded{suffix}"
        enc = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            "-c:a", codec, "-b:a", bitrate, "-ar", str(codec_sr),
            str(encoded),
        ]
        try:
            subprocess.run(
                enc, input=audio.astype(np.float32).tobytes(),
                capture_output=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"[degrade] codec {codec} unavailable in this ffmpeg build, "
                f"skipping roundtrip: {exc.stderr.decode(errors='replace')[:200]}",
                file=sys.stderr,
            )
            return audio
        decoded, _ = ffmpeg_decode(encoded, sr)
        return audio if decoded is None else decoded


def audio_duration(path: Path, target_sr: int = TARGET_SR) -> float:
    """Duration in seconds.

    Tries the WAV header first: sessions are 16 kHz PCM WAV, and reading
    44 bytes to size a timeout is better than decoding a minute of audio
    to do it. Falls back to a full decode for anything else, and to 0.0
    if even that fails -- a duration probe must not be the thing that
    breaks a run.
    """
    import wave
    import contextlib

    with contextlib.suppress(Exception):
        with wave.open(str(path), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
            if rate:
                return frames / rate
    with contextlib.suppress(Exception):
        return len(load_mono_16k(path, target_sr)) / target_sr
    return 0.0
