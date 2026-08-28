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

Transcription:

| Model | Runtime | Languages | Notes |
|---|---|---|---|
| `parakeet-tdt-v3` | NeMo | 25 European | `nvidia/parakeet-tdt-0.6b-v3` (2025). Transducer — frame-synchronous, structurally cannot hallucinate on silence |
| `parakeet-tdt-v2` | NeMo | English | `nvidia/parakeet-tdt-0.6b-v2` (2025). Long-time leader of the English Open ASR Leaderboard at its size |
| `canary-1b-v2` | NeMo | 25 European | `nvidia/canary-1b-v2` (2025) |
| `canary` | NeMo | en/de/fr/es | `nvidia/canary-1b-flash` |
| `canary-180m-flash` | NeMo | en/de/fr/es | Small sibling — the "how much do you lose going tiny" data point |
| `whisper-large-v3` | faster-whisper | multilingual | The reference big autoregressive decoder |
| `whisper-large-v3-turbo` | faster-whisper | multilingual | Same encoder, 4-layer decoder — much faster, slightly worse |
| `distil-whisper-large-v3` | faster-whisper | English | Distilled Whisper |

Diarisation:

| Backend | Runtime | Notes |
|---|---|---|
| `sortformer` | NeMo | `nvidia/diar_sortformer_4spk-v1` (2025), end-to-end, hard limit of 4 speakers |
| `titanet` | NeMo | TitaNet-Large embeddings + agglomerative clustering — the classic baseline |

Models declare which languages they support, and `bench.py` skips the pairings that make no sense (running an English-only model on a German session produces garbage that looks like model failure but is really operator error). `--ignore-language-support` forces them to run anyway.

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

Defaults: every channel in the session, every model that supports the session language. Outputs land in `runs/<run-name>/<model>/<channel>.txt` with `*.metrics.json` (wall-clock, RTF, peak RSS, WER/CER, error), plus `summary.json` and `run.json` for the whole run.

## 4. Diarise

```sh
uv run python diarize.py --session sessions/standup-de__clean \
    --run-name standup-de__clean_2026-08-28_01-14-22
```

Answers "who spoke when" and scores it as **DER** against the speaker-labelled turns in `reference.json`. Point `--run-name` at an existing bench run and the diarisation lands in the same directory, so `compare.py` picks it up.

The speaker count is read from `session.json` and handed to backends that accept one. `--num-speakers 0` forces auto-detection, which is the harder and more realistic setting — worth running both ways, since "knows how many people are in the room" hides a large part of the problem.

Only the `mixed` channel is a real diarisation task. The isolated per-speaker channels contain one voice each; running them is a sanity check (a backend that finds three speakers in a single-speaker track has a problem).

## 5. Compare

```sh
uv run python compare.py --run-name standup-de__phone_2026-08-28_01-14-22
```

Writes `runs/<run-name>/comparison.md`:

- **accuracy ranking** — WER, CER, and substitution/deletion/insertion counts, best first
- speed and memory table (wall time, realtime factor, peak RAM)
- **diarisation ranking** — DER split into missed speech, false alarm and speaker confusion (when `diarize.py` ran on this run)
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

### Diarisation: DER

```
DER = (missed speech + false alarm + speaker confusion) / reference speech
```

Speaker labels are arbitrary, so the scorer first picks the one-to-one mapping between the system's labels and the real speakers that explains the most audio, then counts errors on a 10 ms grid. Frames within 0.25 s of a reference boundary are excluded — the NIST convention, and a necessary one: nobody can annotate the exact moment a word begins to better than about 100 ms, so without a collar you mostly measure boundary jitter.

Reference frames can hold more than one speaker, because the conversation scripts contain deliberate overlaps, and DER counts those: a backend that reports one speaker while two people are talking takes a miss for the second. The three components are reported separately, which is what tells you *how* a backend fails — all-miss means the VAD is too conservative, all-confusion means the embeddings can't separate these voices.

## Why these models

They are open-weight, run locally, and represent genuinely different architectures — so the comparison teaches you something instead of benchmarking minor variants of the same approach.

- **Parakeet-TDT** is a transducer: frame-synchronous, alignment-forced, structurally can't hallucinate on silence. Leads English benchmarks at its size. v3 (2025) extends it to 25 European languages, v2 stays English-only — running both shows what multilingual coverage costs on English.
- **Canary** is attention-encoder-decoder but trained with explicit non-speech / noise tokens; claims low hallucination on silence *and* multilingual coverage. `canary-180m-flash` is the same idea at a fraction of the size, which is the interesting question for anyone running on a laptop: how much accuracy does the small model actually give up?
- **Whisper Large-v3** is the reference "big autoregressive decoder" — generous multilingual coverage but the well-known hallucination tendency on silence, music and cross-talk. **Turbo** keeps the encoder and cuts the decoder to 4 layers; **Distil** is a distilled English-only variant. Both trade accuracy for speed, and the point of the harness is to put a number on that trade rather than repeating the claim.

For diarisation, **Sortformer** (2025) is end-to-end: no VAD to tune, no clustering threshold to fiddle with, but a hard limit of 4 speakers. **TitaNet + clustering** is the pipeline it is trying to replace, and it is here as the baseline — if the end-to-end model can't beat a threshold-tuned classic on your audio, that is worth knowing before you adopt it.

Transcription language is pinned from `session.json` for every model that accepts a hint (Canary's `source_lang`/`target_lang`, Whisper's `language`). Left to auto-detect, a model can lose an entire session to one bad guess in the first few seconds, which measures language ID rather than transcription.

The NeMo models are fed in silence-aware windows: `bench.py` cuts near the quietest frame around each window boundary rather than on a fixed grid, pads every window with 0.3 s of silence, and re-splits any window that decodes to an empty string. Canary occasionally emits EOS immediately on a perfectly clean window — one 10 s window came back empty while 8 s and 12 s of the same audio transcribed fine — and without those mitigations the empty windows scored as bulk deletions (WER 37.6 % vs 20.3 %), i.e. we were measuring our own chunking rather than the model.

## Hardware

Tested on Apple Silicon. NeMo falls back to CPU for some ops on MPS via `PYTORCH_ENABLE_MPS_FALLBACK=1` (already set in `.env.example`). faster-whisper runs CPU-only with int8 quantisation — fast enough and sidesteps Metal backend quirks.

Measured on an Apple Silicon laptop, `mixed` channel, `clean` profile, one subprocess per model so the memory figures are attributable.

**German** — `standup-de`, 94 s, 3 speakers, 226 reference words:

| Model | Wall clock | RTF | Peak RSS | WER | CER |
|---|---:|---:|---:|---:|---:|
| parakeet-tdt-v3 | 19 s | 0.20 | 5.5 GB | **7.1 %** | 3.3 % |
| whisper-large-v3 | 94 s | 1.00 | 4.1 GB | 7.5 % | 3.8 % |
| whisper-large-v3-turbo | 43 s | 0.46 | 2.5 GB | 8.4 % | 3.1 % |
| canary-1b-v2 | 42 s | 0.45 | 9.4 GB | 10.2 % | 4.2 % |
| canary | 34 s | 0.37 | 7.1 GB | 20.3 % | 17.0 % |
| canary-180m-flash | 19 s | 0.20 | 2.6 GB | 23.4 % | 20.7 % |

**English** — `support-call-en`, 80 s, 2 speakers, 237 reference words:

| Model | Wall clock | RTF | Peak RSS | WER | CER |
|---|---:|---:|---:|---:|---:|
| canary | 26 s | 0.32 | 7.1 GB | **5.9 %** | 2.2 % |
| canary-180m-flash | 17 s | 0.22 | 2.4 GB | 6.3 % | 2.5 % |
| parakeet-tdt-v2 | 14 s | 0.17 | 5.4 GB | 8.0 % | 7.1 % |
| whisper-large-v3-turbo | 40 s | 0.50 | 2.5 GB | 8.0 % | 6.8 % |
| distil-whisper-large-v3 | 28 s | 0.35 | 2.3 GB | 8.4 % | 7.0 % |
| parakeet-tdt-v3 | 18 s | 0.23 | 5.6 GB | 9.7 % | 7.8 % |
| canary-1b-v2 | 44 s | 0.54 | 8.5 GB | 13.5 % | 8.8 % |
| whisper-large-v3 | 273 s | 3.42 | 6.2 GB | 43.9 % | 39.4 % |

Diarisation, same sessions:

| Backend | German (3 spk) | English (2 spk) |
|---|---:|---:|
| sortformer | 36.6 % DER | **0.3 % DER** |
| titanet | **27.3 % DER** | 10.1 % DER |

Three things in those tables are worth more than the ranking itself.

**No model wins both languages.** `parakeet-tdt-v3` is first on German and sixth on English; `canary` is first on English and fifth on German. Picking a model from an English leaderboard and deploying it on German is not sound, which is the whole reason this repo scores per language.

**whisper-large-v3 degenerates on the English session** — 43.9 % WER from 90 insertions, and an RTF of 3.42 rather than the 0.5 it manages elsewhere. The transcript is correct up to the last real words ("Thanks for calling.") and then invents about 78 more, ending in *"I love you too... I love you..."*. This is the well-known Whisper trailing-silence loop: the decoder is autoregressive, so once it starts a repetition it will happily continue and burn wall-clock doing it. `whisper-large-v3-turbo` on the identical audio stops cleanly. The repeated-n-gram column in the report flags it automatically — this is exactly the failure the heuristic exists for, and exactly the failure a transducer like Parakeet cannot have by construction.

**Sortformer's DER swings from 0.3 % to 36.6 %.** Near-perfect on the two-speaker English call, worst-in-class on the three-speaker German one. Two causes stack: it is trained on English, and the German session has three macOS `say` voices. Measured directly, those voices sit at cosine distance 0.71–0.92 from each other while windows *within* one voice reach 0.38 — separable, but with far less margin than real speech. Miss and false-alarm rates stay near zero in both cases, so the voice activity detection is fine; the entire error is speaker confusion. Read the German diarisation numbers as a hard floor set by synthetic audio, not as the performance you would see on a real meeting.

Canary mangles the English technical vocabulary sprinkled through the German dialogue (*caching layer* → *kachin leier*), which is exactly the kind of failure the scripts are written to provoke.

Each model runs in its own subprocess. That costs a few seconds of start-up per cell and buys honest memory numbers: `ru_maxrss` is a per-process high-water mark, so a matrix run in one process reports the largest model's peak for every model after it. Pass `--in-process` to skip the isolation when you only care about accuracy.

A full 8-model matrix on a 90-second session takes roughly 10 minutes. Longer sessions scale linearly — plan accordingly.

## Repo layout

| File | Role |
|---|---|
| `synth.py` | conversation script → session with audio + ground truth |
| `degrade.py` | degradation profiles (noise, codec, reverb, clipping) |
| `bench.py` | runs the model × channel matrix, scores against the reference |
| `diarize.py` | speaker diarisation backends, scored as DER |
| `score.py` | normalisation + WER/CER alignment + DER |
| `compare.py` | run directory → Markdown report |
| `audio_io.py` | shared decode/encode helpers (ffmpeg + libsndfile) |
| `envfile.py` | loads `.env` before torch/NeMo import, so caches land where you asked |
| `conversations/` | conversation scripts — the source of truth for sessions |

`sessions/`, `runs/` and `.tts-cache/` are gitignored: all three are reproducible from the scripts, so the scripts are what gets committed, not the waveforms.
