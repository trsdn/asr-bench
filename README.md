# asr-bench

Benchmark local open-weight ASR models against each other on **multi-speaker conversations with a known ground truth**.

The trick: instead of hunting for labelled meeting audio, the harness *synthesises* the conversation from a script. One TTS voice per speaker, laid out on a timeline with realistic pauses and overlaps, then pushed through degradation profiles (telephony codec, room reverb, background noise, clipping). Because we started from the text, every session ships a word-exact reference transcript — so the benchmark reports real **WER/CER**, not just side-by-side transcripts to eyeball.

```
conversations/*.yaml  ──synth.py──▶  sessions/<name>__<profile>/  ──bench.py──▶  runs/<name>/  ──compare.py──▶  comparison.md
   the script            TTS +           audio + ground truth          ASR models        metrics           the report
                       degradation
```

Recorded audio still works: point `bench.py` at any directory with an `audio/` folder and you get speed/RAM numbers and side-by-side transcripts, just without error rates.

## Models

| Model | Runtime | Notes |
|---|---|---|
| `canary` | NeMo | `nvidia/canary-1b-flash` — multilingual (en/de/fr/es) |
| `whisper-large-v3` | faster-whisper (CTranslate2) | OpenAI Whisper Large-v3, full (not Turbo) |
| `parakeet-live` | — | Reads a recording app's own `transcript.live.jsonl`; only applies to app-produced sessions, so it is excluded from the default model set |

Voxtral can be added once you have Mistral's HuggingFace access.

## Install

```sh
uv sync
```

`uv` creates `.venv/` and resolves all deps. NeMo pulls in a large PyTorch stack; expect 3–5 minutes the first time. **ffmpeg is a hard runtime dependency** (`brew install ffmpeg`) — it does the decoding and the degradation DSP.

### Model cache location

```sh
cp .env.example .env
# edit .env — point HF_HOME / NEMO_CACHE_DIR / TORCH_HOME at wherever
# you have 10-15 GB of free space for the weight downloads.
```

`.env` is gitignored (per-machine config). Without it, caches land in `./model-cache/`. First run downloads ~10–15 GB; later runs reuse the cache.

## 1. Write a conversation

A script is a YAML file listing speakers and turns. See [`conversations/standup-de.yaml`](conversations/standup-de.yaml) (German, 3 speakers) and [`conversations/support-call-en.yaml`](conversations/support-call-en.yaml) (English, 2 speakers).

```yaml
name: standup-de
language: de
speakers:
  - id: a
    name: Anna
    rate: 185          # words per minute (optional)
  - id: b
    name: Markus
    voice: Anna        # optional; auto-assigned per language if omitted
turns:
  - speaker: a
    text: "Guten Morgen zusammen."
  - speaker: b
    text: "Kurzer Einwurf dazu."
    gap: -0.35         # seconds before this turn; negative = overlapping speech
```

Voices are auto-assigned per language and guaranteed distinct — two speakers sharing a voice would make speaker-attribution numbers meaningless. `--list-voices` shows what's installed for the script's language.

Write scripts that stress what you care about: loanwords, numbers, proper nouns, spelled-out email addresses, cross-talk. Those are where models actually differ.

## 2. Synthesise sessions

```sh
# one session per difficulty level, from a single TTS pass
uv run python synth.py --script conversations/standup-de.yaml \
    --degrade clean phone farfield
```

Each session directory contains:

```
sessions/standup-de__phone/
  session.json          # manifest: language, speakers, duration, overlap, degradation
  audio/mixed.wav       # the conversation as one stream — the realistic case
  audio/spk-a.wav       # per-speaker isolated channels (same timeline, silent elsewhere)
  reference/mixed.txt   # ground-truth transcript per channel
  reference.json        # segments with speaker, start, end, text
```

TTS renders are cached in `.tts-cache/`, so generating eight difficulty levels costs one synthesis pass, not eight.

### Degradation profiles

| Profile | What it simulates |
|---|---|
| `clean` | untouched TTS output — the ceiling |
| `noisy` / `very-noisy` | pink background noise at 10 dB / 3 dB SNR |
| `phone` | 300–3400 Hz band-limit + Opus at 12 kbps |
| `voip` | Opus at 24 kbps, light noise floor |
| `farfield` | room reverb, −12 dB level, ambient noise |
| `clipped` | +14 dB gain into a hard clipper |
| `worst-case` | large room + narrowband + Opus 8 kbps + 5 dB SNR |

Same `--seed` gives byte-identical audio, so re-runs stay comparable.

### TTS backends

`say` (macOS, default) needs nothing installed. Piper is optional and cross-platform — pass `--tts piper` and set each speaker's `voice:` to a `.onnx` model path.

## 3. Benchmark

```sh
uv run python bench.py --session sessions/standup-de__phone \
    --models canary whisper-large-v3 \
    --channels mixed
```

Defaults: every channel in the session, every real ASR model. Outputs land in `runs/<run-name>/<model>/<channel>.txt` with `*.metrics.json` (wall-clock, RTF, peak RSS, WER/CER, error), plus `summary.json` and `run.json` for the whole run.

## 4. Compare

```sh
uv run python compare.py --run-name standup-de__phone_2026-08-28_01-14-22
```

Writes `runs/<run-name>/comparison.md`:

- **accuracy ranking** — WER, CER, and substitution/deletion/insertion counts, best first
- speed and memory table (wall time, realtime factor, peak RAM)
- hallucination heuristics (Whisper ghost phrases like *"Thank you for watching"* / *"Bitte abonnieren"*, plus the most-repeated 5-gram — high counts flag loop degeneration)
- side-by-side transcript previews with the ground truth as the first column

## How scoring works

Two numbers per row, because scoring is mostly a normalisation problem:

- **`wer_raw`** — case and punctuation stripped, nothing else.
- **`wer`** — additionally spells out digits and collapses German/English number-word variants, so a model writing `2.4.1` isn't penalised against a reference saying *"zwei Punkt vier Punkt eins"*.

The gap between them is formatting, not misrecognition. How a spoken digit string *should* be written is genuinely ambiguous (`4821` vs. *"four eight two one"*), which is exactly why both are reported rather than one authoritative score.

`score.py` has no third-party dependency — it ships its own Levenshtein alignment and a German/English number speller. If you install [`jiwer`](https://pypi.org/project/jiwer/) and [`num2words`](https://pypi.org/project/num2words/) yourself, it picks them up automatically for the reference implementation and wider language coverage:

```sh
uv pip install jiwer num2words
```

## Why these models

They are open-weight, run locally, and represent genuinely different architectures — so the comparison teaches you something instead of benchmarking minor variants of the same approach.

- **Parakeet-TDT** is RNN-T-ish: frame-synchronous, alignment-forced, structurally can't hallucinate on silence. Leads English-only benchmarks at its size.
- **Canary** is attention-encoder-decoder but trained with explicit non-speech / noise tokens; claims low hallucination on silence *and* multilingual coverage (en/de/fr/es). On the Open ASR Leaderboard it frequently beats Whisper on German.
- **Whisper Large-v3** is the reference "big autoregressive decoder" — generous multilingual coverage but the well-known hallucination tendency on silence, music and cross-talk.

Canary transcribes in the session language rather than auto-translating to English: `bench.py` pins its `source_lang`/`target_lang` from `session.json`.

Both NeMo models are fed in silence-aware windows: `bench.py` cuts near the quietest frame around each window boundary rather than on a fixed grid, pads every window with 0.3 s of silence, and re-splits any window that decodes to an empty string. Canary occasionally emits EOS immediately on a perfectly clean window — one 10 s window came back empty while 8 s and 12 s of the same audio transcribed fine — and without those mitigations the empty windows scored as bulk deletions (WER 37.6 % vs 20.3 %), i.e. we were measuring our own chunking rather than the model.

## Hardware

Tested on Apple Silicon. NeMo falls back to CPU for some ops on MPS via `PYTORCH_ENABLE_MPS_FALLBACK=1` (already set in `.env.example`). faster-whisper runs CPU-only with int8 quantisation — fast enough and sidesteps Metal backend quirks.

Measured on an Apple Silicon laptop, 94 s synthetic German session (`standup-de`, `clean`, `mixed` channel):

| Model | Download | Wall clock | RTF | Peak RSS | WER | CER |
|---|---|---|---|---|---|---|
| whisper-large-v3 | ~3 GB | 70 s | 0.75 | ~4.0 GB | 7.5 % | 3.8 % |
| canary | ~4 GB | 32 s | 0.34 | ~7.0 GB | 20.3 % | 17.0 % |

Canary is the faster model here by a wide margin but mangles the English technical vocabulary sprinkled through the German dialogue (*caching layer* → *kachin leier*), which is exactly the kind of failure the scripts are written to provoke. Treat these as a smoke-test baseline, not a leaderboard.

A full matrix (2 models × 4 channels) on a 90-second session takes roughly 10 minutes. Longer sessions scale linearly — plan accordingly.

## Repo layout

| File | Role |
|---|---|
| `synth.py` | conversation script → session with audio + ground truth |
| `degrade.py` | degradation profiles (noise, codec, reverb, clipping) |
| `bench.py` | runs the model × channel matrix, scores against the reference |
| `score.py` | normalisation + WER/CER alignment |
| `compare.py` | run directory → Markdown report |
| `audio_io.py` | shared decode/encode helpers (ffmpeg + libsndfile) |
| `conversations/` | conversation scripts — the source of truth for sessions |

`sessions/`, `runs/` and `.tts-cache/` are gitignored: all three are reproducible from the scripts, so the scripts are what gets committed, not the waveforms.
