"""Transcription runners for models that load through `transformers`.

NeMo and faster-whisper each expose one uniform API, which is why their
runners live in `bench.py` as a single function apiece. The HuggingFace
side is not uniform at all: an audio LLM wants a chat template with an
audio placeholder, an encoder-decoder wants a feature tensor, and Voxtral
wants its own bespoke request builder. Pretending otherwise would mean a
runner full of special cases, so each family gets its own small function
and they share only the chunk loop.

Every runner has the same shape:

    fn(chunks: list[np.ndarray], model_id: str, language: str) -> list[str]

The caller does the chunking, so windowing policy stays in one place and
this module never imports `bench`.

Long audio has to be chunked here for the same reason it is chunked for
NeMo: these are all 30-second-context encoders at heart, and an audio LLM
asked to transcribe two minutes in one pass will summarise rather than
transcribe.
"""

from __future__ import annotations

import gc

import numpy as np

SAMPLE_RATE = 16_000

# Audio LLMs are told what to do in words rather than by architecture, so
# the instruction is part of the configuration. Kept deliberately blunt:
# anything resembling "summarise" or "answer" invites the model to do
# that instead, which scores as a catastrophic deletion rate.
TRANSCRIBE_PROMPT = (
    "Transcribe this audio exactly as spoken. Output only the transcript, "
    "with no commentary, translation or summary."
)


def _torch():
    import torch

    return torch


def _drop_remote_import(package: str) -> None:
    """Stop `trust_remote_code` from demanding a package it never uses.

    Transformers scans a remote modeling file for imports and refuses to
    load if one is missing. The scan only understands `try/except
    ImportError` as "optional", so an import guarded by a runtime `if`
    still counts as required. Phi-4-multimodal imports flash_attn under
    `if is_flash_attn_2_available()`, which is False on Metal — the
    branch never runs, but the check fails anyway, and flash_attn has no
    Metal build to install. Filtering the name out of the scan is the
    difference between running the model and not having the row."""
    from transformers import dynamic_module_utils

    original = dynamic_module_utils.get_imports

    def patched(filename):
        return [i for i in original(filename) if i != package]

    dynamic_module_utils.get_imports = patched


def _device_and_dtype(override: str | None = None, dtype: str | None = None):
    """Apple Silicon: MPS, float32 by default.

    float32 because half precision is patchy across these architectures
    on Metal and a wrong dtype shows up as silent garbage output rather
    than an exception.

    `dtype` overrides that per model, and exists because the default is
    not always affordable: a 7B audio LLM in float32 wants ~29 GiB of
    weights and dies allocating its KV cache on a 32 GB machine. Halving
    the weights is the difference between a number and no number at all.
    It does make dtype a variable across the table, so a model that
    carries an override says so in the registry, and its output is worth
    reading rather than trusting -- half-precision failure on Metal is
    silent garbage, not an exception.

    There is deliberately no CPU fallback for a working GPU. A 3B audio
    LLM on CPU is slower than realtime by a wide margin, which makes the
    RTF column meaningless and ties up the machine for nothing — a model
    that will not run on the GPU is a result in itself and belongs in the
    results table as such, not as a number nobody can act on.
    `ASR_BENCH_DEVICE` overrides this for one-off debugging."""
    import os

    torch = _torch()
    resolved = getattr(torch, dtype) if dtype else torch.float32
    choice = override or os.environ.get("ASR_BENCH_DEVICE")
    if choice:
        return choice, resolved
    if torch.backends.mps.is_available():
        return "mps", resolved
    return "cpu", resolved


def _release(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    torch = _torch()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _expected_sample_rate(processor, default: int = SAMPLE_RATE) -> int:
    """The rate a model was trained on, read off its own processor.

    Not every speech model wants 16 kHz — Kyutai STT is trained at 24 kHz
    and refuses anything else. Asking the processor rather than keeping a
    table in the registry means a new model cannot silently be fed the
    wrong rate, which degrades output without raising anything."""
    extractor = getattr(processor, "feature_extractor", processor)
    return int(getattr(extractor, "sampling_rate", None) or default)


def _resample(chunk: np.ndarray, target_sr: int) -> np.ndarray:
    """Sessions are stored at 16 kHz; resample only when a model needs
    something else."""
    if target_sr == SAMPLE_RATE:
        return chunk
    import librosa

    return librosa.resample(
        chunk.astype(np.float32), orig_sr=SAMPLE_RATE, target_sr=target_sr
    )


def run_seq2seq(
    chunks: list[np.ndarray],
    model_id: str,
    language: str = "en",
    device: str | None = None,
    dtype: str | None = None,
) -> list[str]:
    """Plain speech-to-text encoder-decoders: Moonshine, Kyutai STT.

    Same shape as Whisper — feature extractor in, token ids out — so one
    function covers both."""
    torch = _torch()
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device, dtype = _device_and_dtype(device, dtype)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()
    sample_rate = _expected_sample_rate(processor)

    texts: list[str] = []
    for chunk in chunks:
        # `audio=` by keyword, never positionally: Kyutai's processor uses
        # the generic multimodal signature whose first argument is
        # `images`, so a positional call silently passes no audio and the
        # model then fails deep inside generate.
        inputs = processor(
            audio=_resample(chunk, sample_rate),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=440)
        texts.append(
            processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        )

    _release(model, processor)
    return texts


def _load_generative_model(model_id: str, dtype):
    """Pick the AutoModel class that actually claims this architecture.

    The audio-LLM families disagree about which head they register under:
    Granite Speech is a speech-seq2seq, Qwen2-Audio and Voxtral are
    seq2seq LMs, Phi-4 is a causal LM. Trying them in order is more
    robust than a hand-maintained table, and a wrong guess raises a
    ValueError rather than loading something subtly wrong."""
    import transformers

    candidates = (
        "AutoModelForSpeechSeq2Seq",
        "AutoModelForSeq2SeqLM",
        "AutoModelForCausalLM",
    )
    last_error: Exception | None = None
    for name in candidates:
        try:
            return getattr(transformers, name).from_pretrained(model_id, dtype=dtype)
        except ValueError as error:
            last_error = error
    raise last_error  # type: ignore[misc]


def _is_granite(processor) -> bool:
    return type(processor).__name__.startswith("GraniteSpeech")


def _audio_pad_multiple(processor) -> int | None:
    """Samples the audio length must be a multiple of, or None.

    Qwen3-ASR's encoder consumes mel frames in fixed windows and rejects
    anything that does not divide evenly — the feature extractor pads to
    the longest item in the batch, which for a single chunk is its own
    unpadded length, so the caller has to supply the alignment. The
    window size is read off the model rather than hard-coded, because it
    is a config value and a future checkpoint may change it.
    """
    fe = getattr(processor, "feature_extractor", None)
    if fe is None or "qwen3asr" not in type(processor).__name__.lower():
        return None
    n_window = getattr(fe, "n_window", 50)
    hop = getattr(fe, "hop_length", 160)
    return int(n_window) * 2 * int(hop)


def _pad_to_multiple(audio: np.ndarray, multiple: int | None) -> np.ndarray:
    if not multiple or audio.size % multiple == 0:
        return audio
    pad = multiple - (audio.size % multiple)
    return np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])


def _template_owner(processor, tokenizer):
    """Return whichever object actually carries the chat template.

    Multimodal models increasingly ship the template on the processor
    rather than the tokenizer, because it has to describe audio and image
    placeholders the tokenizer knows nothing about. Qwen3-ASR has it on
    the processor only, and the tokenizer's `apply_chat_template` raises
    instead of falling back — so asking the wrong object is a hard error,
    not a silent degradation.
    """
    return processor if getattr(processor, "chat_template", None) else tokenizer


def _audio_llm_prompt(processor, tokenizer) -> str:
    """Granite's chat template expects a literal `<|audio|>` marker inside
    a plain string; Qwen2-Audio expects the structured content list. Both
    are the documented path for their own model, so both are here."""
    if _is_granite(processor):
        conversation = [
            {"role": "user", "content": f"<|audio|>{TRANSCRIBE_PROMPT}"}
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                ],
            }
        ]
    return _template_owner(processor, tokenizer).apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )


def _audio_llm_inputs(processor, prompt: str, audio: np.ndarray, sample_rate: int):
    """Granite wants a `(1, samples)` torch tensor and rejects
    `sampling_rate` (it reaches the tokenizer and raises); Qwen2-Audio
    wants numpy and requires `sampling_rate`."""
    if _is_granite(processor):
        torch = _torch()
        return processor(
            prompt,
            torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0),
            return_tensors="pt",
        )
    return processor(
        text=prompt,
        audio=[audio],
        sampling_rate=sample_rate,
        return_tensors="pt",
    )


def run_audio_llm(
    chunks: list[np.ndarray],
    model_id: str,
    language: str = "en",
    device: str | None = None,
    dtype: str | None = None,
) -> list[str]:
    """Audio-conditioned LLMs driven by a chat template: Granite Speech,
    Qwen2-Audio.

    The template is what tells the model this is a transcription task, so
    the two have to stay together — the processor knows where the audio
    placeholder goes for its own model."""
    torch = _torch()
    from transformers import AutoProcessor

    device, dtype = _device_and_dtype(device, dtype)
    processor = AutoProcessor.from_pretrained(model_id)
    model = _load_generative_model(model_id, dtype)
    model.to(device).eval()
    sample_rate = _expected_sample_rate(processor)

    tokenizer = getattr(processor, "tokenizer", processor)
    # A processor carrying `apply_transcription_request` is an ASR-native
    # multimodal processor: it expands the audio placeholder into a token
    # count derived from the waveform, so the template has to see the
    # audio and the prompt cannot be built once up front.
    #
    # The method itself is not used. On transformers 5.15.1 it raises
    # `continue_final_message is set but the final message does not
    # appear in the chat` for every language including None — the shipped
    # template and the newer template validation disagree. Building the
    # same messages by hand goes through the working path. Its presence
    # is still the honest capability marker for this family.
    audio_in_template = hasattr(processor, "apply_transcription_request")
    prompt = None if audio_in_template else _audio_llm_prompt(processor, tokenizer)
    pad_multiple = _audio_pad_multiple(processor)

    texts: list[str] = []
    for chunk in chunks:
        audio = _pad_to_multiple(_resample(chunk, sample_rate), pad_multiple)
        if audio_in_template:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text, audio=audio)
        else:
            inputs = _audio_llm_inputs(processor, prompt, audio, sample_rate)
        inputs = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=440, do_sample=False)
        # Strip the prompt: these models echo the full conversation back.
        prompt_len = inputs["input_ids"].shape[-1]
        new_ids = ids[:, prompt_len:]
        texts.append(
            tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        )

    _release(model, processor)
    return texts


def run_voxtral(
    chunks: list[np.ndarray],
    model_id: str,
    language: str = "en",
    device: str | None = None,
    dtype: str | None = None,
) -> list[str]:
    """Voxtral ships a dedicated transcription request builder, which is
    the supported path — the generic chat template puts it in
    understanding mode and it starts answering questions about the audio
    instead of transcribing it."""
    import tempfile
    from pathlib import Path

    import soundfile as sf

    torch = _torch()
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    device, dtype = _device_and_dtype(device, dtype)
    processor = AutoProcessor.from_pretrained(model_id)
    # Voxtral's decoder is a Ministral with sliding-window attention. On
    # MPS the fused SDPA path submits a Metal command buffer that never
    # completes: the run does not fail, it parks forever in
    # `waitUntilCompleted`, which reads as a very slow model rather than
    # a broken one. Eager attention is slower per token but finishes.
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager"
    )
    model.to(device).eval()

    tmpdir = Path(tempfile.mkdtemp(prefix="voxtral-"))
    texts: list[str] = []
    try:
        for index, chunk in enumerate(chunks):
            wav = tmpdir / f"chunk-{index}.wav"
            sf.write(str(wav), chunk, SAMPLE_RATE, subtype="PCM_16")
            inputs = processor.apply_transcription_request(
                language=language, audio=str(wav), model_id=model_id
            )
            inputs = inputs.to(device, dtype=dtype)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=440)
            decoded = processor.batch_decode(
                ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            texts.append(decoded[0].strip())
            wav.unlink(missing_ok=True)
    finally:
        try:
            tmpdir.rmdir()
        except OSError:
            pass
        _release(model, processor)
    return texts


def run_phi4_multimodal(
    chunks: list[np.ndarray],
    model_id: str,
    language: str = "en",
    device: str | None = None,
    dtype: str | None = None,
) -> list[str]:
    """Phi-4-multimodal uses literal `<|audio_1|>` markers in a
    hand-built prompt rather than a chat template, and needs its own
    generation config."""
    torch = _torch()
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

    _drop_remote_import("flash_attn")
    device, dtype = _device_and_dtype(device, dtype)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        _attn_implementation="eager",
        # Placed by accelerate during loading rather than moved afterwards.
        # The remote code reads tensor values while building submodules, and
        # a deferred `.to(device)` leaves those on meta, where `.item()`
        # raises instead of returning a number.
        device_map=device,
    )
    model.eval()
    generation_config = GenerationConfig.from_pretrained(model_id)

    prompt = (
        f"<|user|><|audio_1|>{TRANSCRIBE_PROMPT}<|end|><|assistant|>"
    )

    texts: list[str] = []
    for chunk in chunks:
        inputs = processor(
            text=prompt, audios=[(chunk, SAMPLE_RATE)], return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=440,
                generation_config=generation_config,
                do_sample=False,
            )
        new_ids = ids[:, inputs["input_ids"].shape[1]:]
        texts.append(
            processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        )

    _release(model, processor)
    return texts


HF_RUNNERS = {
    "seq2seq": run_seq2seq,
    "audio-llm": run_audio_llm,
    "voxtral": run_voxtral,
    "phi4": run_phi4_multimodal,
}


def run_hf(
    chunks: list[np.ndarray],
    model_id: str,
    family: str,
    language: str = "en",
    device: str | None = None,
    dtype: str | None = None,
) -> list[str]:
    try:
        runner = HF_RUNNERS[family]
    except KeyError:
        raise ValueError(f"Unknown HF family: {family}") from None
    return runner(chunks, model_id, language, device=device, dtype=dtype)
