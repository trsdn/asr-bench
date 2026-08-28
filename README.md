# asr-bench

Benchmark local open-weight ASR models against each other on **multi-speaker conversations with a known ground truth**.

The trick: instead of hunting for labelled meeting audio, the harness *synthesises* the conversation from a script. One TTS voice per speaker, laid out on a timeline with realistic pauses and overlaps, then pushed through degradation profiles (telephony codec, room reverb, background noise, clipping). Because we started from the text, every session ships a word-exact reference transcript — so the benchmark reports real **WER/CER**, not just side-by-side transcripts to eyeball.

The catch that comes with the trick: a synthesiser has a fingerprint, and scoring against one engine measures the engine as much as the model. So every script renders through three unrelated TTS lineages (Apple `say`, Piper, Kokoro), and the engine is part of the session identity. It matters more than it sounds — the same model on the same script varies by up to 6.4 WER points across engines, and not by the same amount for every model. See [TTS backends](#tts-backends).

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
| `parakeet-ctc-1.1b` | NeMo | English | CTC head rather than a transducer — the "how much does the decoder matter" data point |
| `canary-qwen-2.5b` | NeMo (SALM) | English | Speech encoder feeding an LLM decoder |
| `kyutai-stt-2.6b` | transformers | English | `kyutai/stt-2.6b-en-trfs`. Streaming architecture; trained at 24 kHz, not 16 |
| `moonshine-base` | transformers | English | 61M params — the small-and-fast end of the range |
| `granite-speech-4.1` | transformers | English | IBM's audio LLM (2026) |
| `voxtral-mini-3b` | transformers | multilingual | Mistral's audio LLM. **Blocked:** deadlocks on MPS (see below) |
| `qwen2-audio-7b` | transformers | multilingual | Audio LLM, wired up but unverified |
| `phi-4-multimodal` | transformers | multilingual | Audio LLM, wired up but unverified |
| `qwen3-asr-1.7b` | transformers | multilingual | **Blocked:** no transformers release both knows and correctly loads this checkpoint |

Two entries are wired up but cannot currently produce a number, and are listed
as blocked rather than quietly dropped:

- **`voxtral-mini-3b`** hangs on Apple Silicon. `generate()` never returns; a
  single 5-second chunk still hangs after 300 s, while the same weights load in
  30 s, so this is a deadlock and not slowness. Sampling the process shows a
  Metal command buffer that never completes. Forcing
  `attn_implementation="eager"` — the usual remedy for SDPA on MPS — does not
  help. Re-test against a newer torch MPS backend.
- **`qwen3-asr-1.7b`** loads with 708 missing and 708 unexpected keys: the whole
  audio tower is randomly initialised, so transformers 5.15.1's implementation
  does not match the published checkpoint. The model does not exist at all in
  4.57.x, so no available version both knows it and loads it.

Neither is run on CPU as a workaround. A model that will not run on the GPU is
recorded as a failure, because a CPU number would not be comparable to any other
row in the table.

Blocked models are excluded from the default `--models` set. They stay in the
registry so the reason stays recorded, but naming one explicitly is the only way
to spend time on it.

### Time budget: slowness is a result, not a missing number

Every model here that has ever produced a transcript runs between RTF 0.14 and
1.6, worst case 3.4. The ones that fail sit above 10. There is nothing in
between, which makes the gap easy to exploit: each cell gets
`--timeout-floor` (180 s, for weight loading, which does not scale with the
audio) plus `--timeout-rtf` × the audio duration (5×). That is a little over
twice the slowest cell that has ever succeeded, so it abandons only models that
were never going to be usable locally.

Two details matter more than the numbers:

- **The budget is relative to the audio.** A flat timeout is wrong in both
  directions — too generous on a 30-second clip, too tight on a ten-minute one.
- **One timeout characterises a model.** The channels of a session are the same
  conversation seen three ways, so a model that cannot finish the mixed track
  will not finish the isolated ones. The remaining channels are skipped rather
  than re-confirming the same failure at full price. This is what turned a
  single dead model into 45 minutes of held machine before it was fixed.

This follows from what the benchmark is for. The question is whether a model is
usable for local transcription on a laptop, and at RTF > 1 the answer is already
no. Exceeding the budget is therefore a finding, not a gap in the table.

Diarisation:

| Backend | Runtime | Notes |
|---|---|---|
| `sortformer` | NeMo | `nvidia/diar_sortformer_4spk-v1` (2025), end-to-end, hard limit of 4 speakers |
| `sortformer-streaming` | NeMo | `nvidia/diar_streaming_sortformer_4spk-v2` (2025), the streaming successor — same 4-speaker limit, far cheaper, and on these sessions the most consistent of the three |
| `titanet` | NeMo | TitaNet-Large embeddings + agglomerative clustering — the classic baseline |

Models declare which languages they support, and `bench.py` skips the pairings that make no sense (running an English-only model on a German session produces garbage that looks like model failure but is really operator error). `--ignore-language-support` forces them to run anyway.

Everything runs on the GPU. There is deliberately no CPU fallback: a 3B audio
LLM on CPU is slower than realtime by a wide margin, which makes the RTF column
meaningless and ties up the machine for nothing. A model that will not run on
Metal is a result, and belongs in the table as one.

Not every failure raises, either — some models deadlock inside a Metal command
buffer that never completes, which reads as a very slow run rather than a bug.
`--timeout` (default 900 s) turns that into a normal cell result so the rest of
the matrix still finishes.

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

A script is a YAML file listing speakers and turns. See [`conversations/standup-de.yaml`](conversations/standup-de.yaml) (German, 3 speakers), [`conversations/support-call-en.yaml`](conversations/support-call-en.yaml) (English, 2 speakers), [`conversations/crosstalk-de.yaml`](conversations/crosstalk-de.yaml) (German, 4 speakers, 9.3 % overlapping speech) and [`conversations/crosstalk-en.yaml`](conversations/crosstalk-en.yaml) (English, 4 speakers, ~8.5 % overlap).

```yaml
name: standup-de
language: de
speakers:
  - id: a
    name: Anna
    rate: 185          # words per minute (optional)
  - id: b
    name: Markus
    gender: m          # optional; only used to pick a plausible voice
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

Overlap deserves its own script rather than a sprinkling of negative gaps. `standup-de` has 0.55 s of overlap in 85 s of speech — 0.6 %, effectively none, where real meetings run 5–20 %. `crosstalk-de` and `crosstalk-en` are built for it: interruptions mid-sentence, two people answering at once, back-channels, and four speakers to sit right on Sortformer's limit. They land at 9.3 % and ~8.5 % overlap. The English one exists because Kokoro has no German voices, so `crosstalk-de` alone cannot answer whether an overlap result is a property of the audio or of the engine; and it is an independent conversation rather than a translation, so the two are two samples instead of one measured twice.

A turn that begins while the previous speaker is still talking is synthesised 3 dB louder. People raise their voice to be heard over someone else, and an interruption rendered at ordinary conversational level is easier to pull apart than the real thing — which would flatter every diarisation backend on precisely the input meant to be hard.

## 2. Synthesise sessions

```sh
# one session per difficulty level, from a single TTS pass
uv run python synth.py --script conversations/standup-de.yaml \
    --degrade clean phone farfield

# same script through a second engine — this is what makes the numbers
# mean something (see "TTS backends" below)
uv run python synth.py --script conversations/standup-de.yaml --tts piper \
    --degrade clean phone farfield
```

Each session directory contains:

```
sessions/standup-de__say__phone/
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

The synthesiser is not a neutral pipe. Score a model against one engine's
output and part of what you measure is that engine, so `--tts` renders the
same script through three unrelated synthesis lineages:

| Backend | Lineage | Languages | Needs |
|---|---|---|---|
| `say` | Apple's system voices | whatever macOS has installed | macOS |
| `piper` | VITS | de, en (voice pool in `synth.py`) | `piper-tts`, voices download on first use |
| `kokoro` | StyleTTS2, 82M, runs on MPS | en only — it has no German | `kokoro` |

The engine is part of the session directory name (`standup-de__piper__clean`),
because two sessions from the same script and profile but different engines are
different test data and their numbers are not interchangeable.

Voices are assigned one per speaker and never reused — a shared voice would make
diarisation meaningless. A speaker's optional `gender:` picks a plausible voice
where one is free; it has no effect on scoring.

**This is not a small effect.** Same script, same profile, mixed channel:

| Model | `say` | `piper` | `kokoro` | Spread |
|---|---|---|---|---|
| `parakeet-tdt-v2` | **9.3 %** | **11.4 %** | **9.3 %** | 2.1 |
| `whisper-large-v3-turbo` | 11.8 % | 13.9 % | 9.7 % | 4.2 |
| `moonshine-base` | 11.4 % | 16.5 % | 10.1 % | 6.4 |

The engine moves absolute WER by up to 6.4 points, and it does not move every
model by the same amount: switching from `say` to `piper` costs Parakeet 2.1
points and Moonshine 5.1. So an absolute WER quoted without its engine is not a
meaningful number, and a small gap between two models on one engine is well
inside the range the engine alone can produce.

The ranking, however, is stable here — Parakeet wins on all three. An earlier
version of this table showed the ranking *inverting* between `say` and `piper`,
and that was wrong: it was an artefact of a bug where a speaker's `rate:` was
only honoured by `say`, so the `piper` and `kokoro` sessions were not just
differently voiced but differently paced. With tempo applied on all three
engines the inversion disappears. It is left recorded here rather than quietly
deleted, because it is a good illustration of the failure mode this axis exists
to catch: a confident cross-engine conclusion that was really measuring an
uncontrolled variable.

The honest rule is the weaker one: compare models within an engine, and treat a
result that only holds on one engine as a property of that engine until it is
shown on another.

The axis is not specific to transcription. `sortformer-streaming` scores 1.5 %
DER on `standup-de__say__clean` and 4.0 % on the same script under `piper` —
the diarisation task got harder because the voices changed, not because the
recording did. (That measurement predates the `rate` fix and should be
re-checked; see the open issues.)

A speaker's `rate:` is written in words per minute, the unit macOS `say` uses,
and is converted to each engine's own control — Piper's `length_scale`, Kokoro's
`speed` — against `say`'s default of 175 wpm. The engines do not respond
identically: over a 140→220 wpm range the resulting duration ratio is 1.36 for
`say`, 1.29 for `piper` and 1.50 for `kokoro`, against 1.57 if tempo scaled
duration perfectly. Tempo is therefore comparable across engines in direction
and roughly in magnitude, but not exactly.

Caveat when reading German cross-engine numbers: the German Piper voices are
unevenly trained (only Thorsten reaches `medium` quality), so a German `say` ↔
`piper` delta mixes model behaviour with voice quality. The English voices are
all medium/high and do not have this problem.

TTS renders are cached in `.tts-cache/` per engine, so generating eight
difficulty levels costs one synthesis pass, not eight.

## 3. Benchmark

```sh
uv run python bench.py --session sessions/standup-de__say__phone \
    --models canary whisper-large-v3 \
    --channels mixed
```

Defaults: every channel in the session, every model that supports the session language. Outputs land in `runs/<run-name>/<model>/<channel>.txt` with `*.metrics.json` (wall-clock, RTF, peak RSS, WER/CER, error), plus `summary.json` and `run.json` for the whole run.

## 4. Diarise

```sh
uv run python diarize.py --session sessions/standup-de__say__clean \
    --run-name standup-de__say__clean_2026-08-28_01-14-22
```

Answers "who spoke when" and scores it as **DER** against the speaker-labelled turns in `reference.json`. Point `--run-name` at an existing bench run and the diarisation lands in the same directory, so `compare.py` picks it up.

The speaker count is read from `session.json` and handed to backends that accept one. `--num-speakers 0` forces auto-detection, which is the harder and more realistic setting — worth running both ways, since "knows how many people are in the room" hides a large part of the problem.

Only the `mixed` channel is a real diarisation task. The isolated per-speaker channels contain one voice each; running them is a sanity check (a backend that finds three speakers in a single-speaker track has a problem).

### Tuning

Every knob in the `titanet` pipeline is a CLI flag, with the defaults shown:

| Flag | Default | What it does |
|---|---:|---|
| `--window` | 1.5 s | length of each embedded window — longer is more reliable per window, but cannot resolve short turns |
| `--hop` | 0.75 s | step between windows; the resolution limit on turn boundaries |
| `--cluster-threshold` | 0.55 | cosine distance at which clusters stop merging. **Only consulted when `--num-speakers 0`** — if the count is given, the count decides when merging stops and this value is inert |
| `--min-turn` | 0.3 s | drop turns shorter than this |
| `--vad-db` | −40 dB | below this, a window counts as silence |

`sortformer` and `sortformer-streaming` have no knobs at all — audio in, spans out. All of the above applies to `titanet` only.

`--sweep` runs a coordinate-descent search over these:

```sh
uv run python diarize.py --sweep \
    --sweep-sessions sessions/standup-de__say__clean sessions/standup-de__say__phone \
                     sessions/standup-de__say__noisy sessions/support-call-en__say__clean
```

**Sweep across several sessions or not at all.** Tuned on `standup-de__say__clean` alone, the search reports DER 27.3 % → 3.6 %; those same values then score 58.8 % on `standup-de__say__phone`, where they find 32 speakers in a 3-speaker recording. Scored honestly across four sessions the whole search is worth 12.4 % → 11.7 %, and the defaults for `--window` and `--hop` come out already optimal. That is the reason `--sweep` refuses to take a single session as the objective, and a failed session is scored as DER 1.0 so a configuration cannot win by collapsing on hard input.

The one change with cross-session evidence behind it is the VAD threshold, which is why the default is −40 dB rather than the −33 dB this repo started with.

## 5. Compare

```sh
uv run python compare.py --run-name standup-de__say__phone_2026-08-28_01-14-22
```

Writes `runs/<run-name>/comparison.md`:

- **accuracy ranking** — WER, CER, and substitution/deletion/insertion counts, best first
- speed and memory table (wall time, realtime factor, peak RAM)
- **diarisation ranking** — DER split into missed speech, false alarm and speaker confusion (when `diarize.py` ran on this run), plus a separate table restricted to overlapping speech if the session contains any
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

### Diarisation: overlapping speech

The global DER hides overlap almost completely. In a session with 9 % overlapped speech, a backend that never reports two people at once still only forfeits those 9 % — it can post a respectable overall number while being structurally blind to crosstalk. So the scorer also reports a separate block restricted to frames where two or more people speak, with its own DER, miss, confusion and a `detection_recall`: the share of overlapped time where the system reported more than one speaker at all.

The 0.25 s collar is deliberately *not* applied there. Overlap sits on speaker boundaries by definition, so a collar would mask exactly the frames being measured.

One caveat on reading `detection_recall` for `titanet`: a clustering backend assigns one label per window, so it cannot represent simultaneity. Any recall it shows comes from adjacent overlapping *windows* landing on different labels, which produces overlapping turns as a side effect of the windowing. It is an artefact, not detection.

## Why these models

They are open-weight, run locally, and represent genuinely different architectures — so the comparison teaches you something instead of benchmarking minor variants of the same approach.

- **Parakeet-TDT** is a transducer: frame-synchronous, alignment-forced, structurally can't hallucinate on silence. Leads English benchmarks at its size. v3 (2025) extends it to 25 European languages, v2 stays English-only — running both shows what multilingual coverage costs on English.
- **Canary** is attention-encoder-decoder but trained with explicit non-speech / noise tokens; claims low hallucination on silence *and* multilingual coverage. `canary-180m-flash` is the same idea at a fraction of the size, which is the interesting question for anyone running on a laptop: how much accuracy does the small model actually give up?
- **Whisper Large-v3** is the reference "big autoregressive decoder" — generous multilingual coverage but the well-known hallucination tendency on silence, music and cross-talk. **Turbo** keeps the encoder and cuts the decoder to 4 layers; **Distil** is a distilled English-only variant. Both trade accuracy for speed, and the point of the harness is to put a number on that trade rather than repeating the claim.

For diarisation, **Sortformer** (2025) is end-to-end: no VAD to tune, no clustering threshold to fiddle with, but a hard limit of 4 speakers. Both the offline v1 and the **streaming v2** successor are here because they do not rank the same way — v2 is cheaper and steadier across sessions, v1 is stronger when it works, and neither wins everywhere. **TitaNet + clustering** is the pipeline they are trying to replace, and it is here as the baseline — if the end-to-end model can't beat a threshold-tuned classic on your audio, that is worth knowing before you adopt it. The end-to-end models can also report two people talking at once, which clustering cannot do at all.

Transcription language is pinned from `session.json` for every model that accepts a hint (Canary's `source_lang`/`target_lang`, Whisper's `language`). Left to auto-detect, a model can lose an entire session to one bad guess in the first few seconds, which measures language ID rather than transcription.

The NeMo models are fed in silence-aware windows: `bench.py` cuts near the quietest frame around each window boundary rather than on a fixed grid, pads every window with 0.3 s of silence, and re-splits any window that decodes to an empty string. Canary occasionally emits EOS immediately on a perfectly clean window — one 10 s window came back empty while 8 s and 12 s of the same audio transcribed fine — and without those mitigations the empty windows scored as bulk deletions (WER 37.6 % vs 20.3 %), i.e. we were measuring our own chunking rather than the model.

## Hardware

Tested on Apple Silicon. NeMo falls back to CPU for some ops on MPS via `PYTORCH_ENABLE_MPS_FALLBACK=1` (already set in `.env.example`). faster-whisper runs CPU-only with int8 quantisation — fast enough and sidesteps Metal backend quirks.

Measured on an Apple Silicon laptop, `mixed` channel, `clean` profile, one subprocess per model so the memory figures are attributable.

> **These tables are `--tts say` only, and predate three changes: `jiwer`/`num2words`
> are now installed (so `score.py` uses its reference implementations rather than
> the fallbacks), a speaker's `rate:` is now honoured by every TTS engine rather
> than `say` alone, and the newer models below are not in them. Read them as a
> snapshot, not as current numbers — and given the cross-engine spread documented
> under "TTS backends", do not read an absolute WER without its engine.**

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

Diarisation, same sessions plus a deliberately overlap-heavy one:

| Backend | German standup (3 spk) | English call (2 spk) | German crosstalk (4 spk) |
|---|---:|---:|---:|
| sortformer | 36.6 % DER | **0.3 % DER** | **0.3 % DER** |
| sortformer-streaming | **1.5 % DER** | 1.7 % DER | 14.7 % DER |
| titanet | 26.0 % DER | 8.8 % DER | 33.7 % DER |

### Diarisation does not degrade, it collapses

Taking a single column of that table at face value is a mistake. Running all
three backends over two TTS engines × six degradation profiles — 14 cells each —
gives a distribution with a hole in the middle:

| Backend | DER < 5 % | 5–15 % | > 15 % | Sorted values |
|---|---:|---:|---:|---|
| sortformer | 11 | 0 | 3 | 0.0 … 1.0, then 27.6 / 36.6 / 59.9 |
| sortformer-streaming | 8 | 1 | 5 | 0.0 … 6.5, then 15.8 … 32.6 |
| titanet | 2 | 5 | 7 | 1.9 … 7.5, then 25.4 … 34.5 |

Sortformer v1 has nothing at all between 1.0 % and 27.6 %. It either separates
the speakers essentially perfectly or fails outright, and the mean of those two
states describes no run that ever happened. The number worth reporting for a
diarisation backend on this data is therefore how often it collapses, not its
average DER.

**Every collapse is the same failure.** In 14 of the 15 cells above 15 %, two
reference speakers are fused into one cluster of 49–84 s, and a sub-second
phantom (0.4 s, 1.0 s, 1.5 s, 1.6 s …) holds the slot that the second speaker
should have occupied. Against a reference of 33.8 / 30.3 / 21.0 s, the failures
look like `[62.2, 21.3, 0.4]` or `[83.8]`. Checking the hypothesis speakers for
one implausibly short cluster identifies the failure immediately, and is a far
better diagnostic than the DER itself.

**Audio quality does not predict it.** This was originally filed as "clean audio
diarises worse than degraded", which held on `say` — 36.6 % clean against 0.0 %
on five separate degradation profiles. It does not survive a second engine:

| Profile | say | piper |
|---|---:|---:|
| clean | 36.6 % | **1.0 %** |
| noisy | **0.0 %** | 0.7 % |
| phone | **0.0 %** | 0.9 % |
| farfield | **0.0 %** | 0.2 % |
| clipped | 59.9 % | **0.7 %** |
| worst-case | **0.3 %** | 27.6 % |

The two engines fail on disjoint profiles. Whatever tips a session into the
collapsed state is a property of the specific voices and the specific
distortion together, not of how degraded the audio is — so a diarisation result
measured on one TTS engine says nothing about the same backend on another. This
is the same lesson the transcription side produced, and it is the argument for
carrying the engine as an axis rather than a footnote.

**Overlapping speech** — `crosstalk-de`, 71 s, 4 speakers, 6.6 s (9.3 %) of genuinely simultaneous speech:

| Backend | Overlap DER | Miss | Confusion | Detection recall |
|---|---:|---:|---:|---:|
| sortformer | **10.3 %** | 6.7 % | 2.9 % | 87.5 % |
| sortformer-streaming | 19.1 % | 10.6 % | 8.5 % | 79.7 % |
| titanet | 51.8 % | 36.8 % | 15.0 % | 27.1 % |

Transcription on that same overlapping session, against the same models on the isolated single-speaker tracks:

| Model | WER on `mixed` | WER on solo tracks | Cost of overlap |
|---|---:|---:|---:|
| whisper-large-v3 | **8.3 %** | 5.6 % | +2.8 |
| whisper-large-v3-turbo | 11.3 % | 4.7 % | +6.6 |
| parakeet-tdt-v3 | 24.0 % | 9.8 % | +14.2 |
| canary-180m-flash | 40.7 % | 27.8 % | +12.9 |
| canary-1b-v2 | 40.7 % | 16.9 % | +23.8 |
| canary | 50.5 % | 63.0 % | −12.5 |

(`parakeet-tdt-v2` and `distil-whisper-large-v3` are English-only and skipped on this German session. `canary` is the one model that scores *worse* on the isolated tracks than on the mixed one — those tracks are mostly silence, which is the input it degenerates on, so its overlap cost is not measurable this way.)

Several things in those tables are worth more than the ranking itself.

**No model wins both languages.** `parakeet-tdt-v3` is first on German and sixth on English; `canary` is first on English and fifth on German. Picking a model from an English leaderboard and deploying it on German is not sound, which is the whole reason this repo scores per language.

**Overlap reorders the transcription ranking.** `parakeet-tdt-v3` wins the clean German session at 7.1 % and drops to fourth under crosstalk, giving up 14 points; `whisper-large-v3` gives up 3. A large autoregressive decoder with broad context turns out to degrade more gracefully when a second voice cuts in than a frame-synchronous transducer does — the reverse of the silence-hallucination story below. Neither architecture is simply better; they fail on different inputs, which is only visible if you test both.

**whisper-large-v3 degenerates on the English session** — 43.9 % WER from 90 insertions, and an RTF of 3.42 rather than the 0.5 it manages elsewhere. The transcript is correct up to the last real words ("Thanks for calling.") and then invents about 78 more, ending in *"I love you too... I love you..."*. This is the well-known Whisper trailing-silence loop: the decoder is autoregressive, so once it starts a repetition it will happily continue and burn wall-clock doing it. `whisper-large-v3-turbo` on the identical audio stops cleanly. The repeated-n-gram column in the report flags it automatically — this is exactly the failure the heuristic exists for, and exactly the failure a transducer like Parakeet cannot have by construction.

**A phantom speaker costs a real one.** On `standup-de` the reference is three speakers of 33.8 s, 30.3 s and 21.0 s. `sortformer` v1 returns 62.2 s, 0.4 s and 21.1 s — the 0.4 s cluster occupies a speaker slot, so the two largest speakers are merged into one. `titanet` produces the same 0.4 s fragment and the same merge. `sortformer-streaming` returns 33.7 s / 29.0 s / 21.0 s and scores 1.5 %. The failure is not that the voices are hard to tell apart, it is that a sub-second fragment is allowed to hold a slot; when it doesn't, the same audio separates almost perfectly. Checking the hypothesis speakers for one implausibly short cluster is the fastest diagnostic available, and it caught the same thing on `crosstalk-de/titanet` (1.5 s and 0.6 s fragments alongside a 38.8 s merge). Measured across 42 cells it is not an anecdote but the failure mode: see the collapse table above, where it accounts for 14 of 15 failures.

An earlier version of this README explained the German diarisation gap as macOS `say` voices sitting too close together in embedding space. That was wrong. The voices are separable — `sortformer-streaming` separates them at 1.5 % DER, and auto-detect clustering separates them at 0.5 % confusion. The measurement that seemed to support it (between-speaker cosine distance 0.71–0.92 vs. 0.14–0.38 within) was real but did not explain the errors.

**Only the end-to-end models detect overlap at all**, which follows from their design: they emit a per-speaker activity signal and can raise two at once. Clustering assigns exactly one label per window and cannot. That is the strongest architectural argument in these tables, and it is worth weighing against the 4-speaker ceiling both Sortformer variants have.

**The streaming model is not simply the newer one.** It is the most consistent across sessions (1.5 / 1.7 / 14.7) and by far the cheapest — 3.2 s against 9.9 s for offline v1 on the same audio — but offline v1 is the better model when it works. Two of three sessions favour streaming, the overlap-heavy one strongly favours offline. Three sessions is not enough to settle that.

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
| `hf_runners.py` | transcription runners for the `transformers`-based models |
| `conversations/` | conversation scripts — the source of truth for sessions |

`sessions/`, `runs/`, `.tts-cache/` and `.piper-voices/` are gitignored: all four are reproducible from the scripts, so the scripts are what gets committed, not the waveforms.
