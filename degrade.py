"""
Degradation profiles — make clean synthetic speech realistically hard.

TTS output is unnaturally pristine: no room, no codec, no background.
Every model looks good on it, which makes the benchmark useless for
ranking. These profiles push the audio through the failure modes that
actually break ASR in production — narrowband telephony codecs, room
reverberation, background noise at a controlled SNR, and clipping from
a mic with the gain cranked.

Because the reference transcript is known regardless of what we do to
the waveform, degradation gives us a *difficulty axis*: run the same
conversation at `clean`, `phone` and `farfield` and read off which model
degrades most gracefully.

Usage:

    from degrade import apply_profile, PROFILES
    noisy = apply_profile(audio, sr=16_000, profile="phone", seed=0)
"""

from __future__ import annotations

import numpy as np

from audio_io import ffmpeg_codec_roundtrip, ffmpeg_filter

# Each profile is an ordered list of (step, params). Steps are applied in
# sequence; order matters (band-limit before codec mirrors a real phone
# path, noise before reverb would sound like noise in the room).
PROFILES: dict[str, list[tuple[str, dict]]] = {
    "clean": [],
    "noisy": [
        ("noise", {"snr_db": 10.0, "color": "pink"}),
    ],
    "very-noisy": [
        ("noise", {"snr_db": 3.0, "color": "pink"}),
    ],
    "phone": [
        # Classic narrowband path: 300–3400 Hz then a low-bitrate codec.
        ("ffmpeg", {"chain": "highpass=f=300,lowpass=f=3400"}),
        ("codec", {"codec": "libopus", "bitrate": "12k", "codec_sr": 16_000}),
        ("noise", {"snr_db": 20.0, "color": "white"}),
    ],
    "voip": [
        ("codec", {"codec": "libopus", "bitrate": "24k", "codec_sr": 16_000}),
        ("noise", {"snr_db": 25.0, "color": "white"}),
    ],
    "farfield": [
        # Speaker across the room from a laptop mic: reverb, then level
        # drop, then ambient noise at the *reduced* level — which is why
        # the noise step comes after the gain step.
        ("reverb", {"amount": "medium"}),
        ("gain", {"db": -12.0}),
        ("noise", {"snr_db": 12.0, "color": "pink"}),
    ],
    "clipped": [
        ("gain", {"db": 14.0}),
        ("clip", {"threshold": 0.85}),
    ],
    "worst-case": [
        ("reverb", {"amount": "large"}),
        ("ffmpeg", {"chain": "highpass=f=300,lowpass=f=3400"}),
        ("codec", {"codec": "libopus", "bitrate": "8k", "codec_sr": 16_000}),
        ("noise", {"snr_db": 5.0, "color": "pink"}),
    ],
}

_REVERB_CHAINS = {
    # aecho taps approximate small/medium/large rooms. Not a measured
    # impulse response, but enough smearing to hurt an ASR frontend.
    "small": "aecho=0.8:0.7:20|32:0.25|0.15",
    "medium": "aecho=0.8:0.85:40|65|95:0.35|0.25|0.15",
    "large": "aecho=0.9:0.9:80|130|190|260:0.45|0.35|0.25|0.18",
}


def speech_rms(audio: np.ndarray) -> float:
    """RMS of the speech-active part of the signal.

    Synthetic conversations contain long gaps; a global RMS would count
    that silence as signal and make every requested SNR far too noisy.
    We take the frames above 1% of peak as "active" and measure those."""
    if audio.size == 0:
        return 0.0
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return 0.0
    active = audio[np.abs(audio) > 0.01 * peak]
    if active.size == 0:
        active = audio
    return float(np.sqrt(np.mean(np.square(active, dtype=np.float64))))


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise via spectral shaping of white noise. Pink is a
    much better stand-in for room/office background than white, which is
    unnaturally hiss-heavy."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spectrum * scale, n=n)
    peak = np.max(np.abs(shaped))
    return (shaped / peak if peak > 0 else shaped).astype(np.float32)


def add_noise(
    audio: np.ndarray,
    snr_db: float,
    color: str = "pink",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix in broadband noise at a target SNR, measured against speech-
    active level (see `speech_rms`)."""
    rng = rng or np.random.default_rng(0)
    noise = _pink(audio.size, rng) if color == "pink" else rng.standard_normal(audio.size).astype(np.float32)

    sig_rms = speech_rms(audio)
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    if sig_rms <= 0 or noise_rms <= 0:
        return audio
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    return (audio + noise * (target_noise_rms / noise_rms)).astype(np.float32)


def apply_profile(
    audio: np.ndarray,
    sr: int,
    profile: str,
    seed: int = 0,
) -> np.ndarray:
    """Apply the named degradation profile. Deterministic for a given
    `seed`, so a re-run of the same session is byte-identical and model
    comparisons stay apples-to-apples."""
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown degradation profile {profile!r}. "
            f"Available: {', '.join(sorted(PROFILES))}"
        )
    rng = np.random.default_rng(seed)
    out = audio.astype(np.float32, copy=True)

    for step, params in PROFILES[profile]:
        if step == "noise":
            out = add_noise(out, params["snr_db"], params.get("color", "pink"), rng)
        elif step == "gain":
            out = (out * (10.0 ** (params["db"] / 20.0))).astype(np.float32)
        elif step == "clip":
            t = params["threshold"]
            out = np.clip(out, -t, t).astype(np.float32)
        elif step == "reverb":
            out = ffmpeg_filter(out, sr, _REVERB_CHAINS[params["amount"]])
        elif step == "ffmpeg":
            out = ffmpeg_filter(out, sr, params["chain"])
        elif step == "codec":
            out = ffmpeg_codec_roundtrip(
                out, sr, params["codec"], params["bitrate"], params["codec_sr"]
            )
        else:
            raise ValueError(f"Unknown degradation step: {step}")

    # Guard against the gain/reverb stages pushing us past full scale;
    # only normalise when we actually exceeded it, so quiet profiles stay
    # quiet (level is part of the difficulty).
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = (out / peak).astype(np.float32)
    return out
