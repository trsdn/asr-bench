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
| `voxtral-mini-3b` | transformers | multilingual | Mistral's audio LLM. **Blocked:** exceeds the time budget on MPS (see below) |
| `qwen2-audio-7b` | transformers | multilingual | Audio LLM. **Blocked:** exceeds the time budget on MPS |
| `phi-4-multimodal` | transformers | multilingual | Audio LLM. **Blocked:** exceeds the time budget on MPS |
| `qwen3-asr-1.7b` | transformers | multilingual | **Blocked:** no transformers release both knows and correctly loads this checkpoint |

Four entries are wired up but cannot currently produce a number, and are listed
as blocked rather than quietly dropped:

- **`voxtral-mini-3b`** hangs on Apple Silicon. `generate()` never returns; a
  single 5-second chunk still hangs after 300 s, while the same weights load in
  30 s, so this is a deadlock and not slowness. Sampling the process shows a
  Metal command buffer that never completes. Forcing
  `attn_implementation="eager"` — the usual remedy for SDPA on MPS — does not
  help. Re-test against a newer torch MPS backend.
- **`qwen2-audio-7b`** needs `dtype: float16` to load at all — in float32 its
  weights alone are 29.5 GiB against Metal's 30.19 GiB watermark, leaving nothing
  for the KV cache. float16 clears the allocation failure and buys a timeout
  instead.
- **`phi-4-multimodal`** now loads and generates, after a chain of dependency
  work: `torchvision` pinned to 0.24.1 (the unpinned install pulls torch 2.13 and
  breaks NeMo), plus `peft` and `backoff`, plus filtering `flash_attn` out of the
  remote-code import scan — it sits behind `if is_flash_attn_2_available()`,
  which is False here, but the scanner only understands `try/except` as optional
  and there is no Metal build to satisfy it with. All of that got the model to
  the same place as the other two: over budget.
- **`qwen3-asr-1.7b`** loads with 708 missing and 708 unexpected keys: the whole
  audio tower is randomly initialised, so transformers 5.15.1's implementation
  does not match the published checkpoint. The model does not exist at all in
  4.57.x, so no available version both knows it and loads it.

**Three of the four are the same shape of model.** Every encoder-decoder ASR
model in the table runs at RTF 0.14–1.6. Every decoder-only audio LLM tried so
far — Voxtral, Qwen2-Audio, Phi-4 — exceeds a 5× realtime budget on MPS, each
after a different amount of setup work and for reasons that look identical from
outside. That is a pattern rather than three accidents, and the honest reading is
that autoregressive audio LLMs are not currently practical on Apple Silicon,
whatever their accuracy might be. Treat the next one as blocked until shown
otherwise rather than budgeting a day for it.

None of them is run on CPU as a workaround. A model that will not run on the GPU
is recorded as a failure, because a CPU number would not be comparable to any
other row in the table.

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

**This is not a small effect.** Same script, same profile, mixed channel, all 13
models that produce a transcript. WER, best per row in bold:

| Model | `say` | `piper` | `kokoro` | Spread |
|---|---:|---:|---:|---:|
| `granite-speech-4.1` | 3.4 | 3.8 | **0.9** | 3.0 |
| `kyutai-stt-2.6b` | **2.1** | 5.1 | 3.4 | 3.0 |
| `parakeet-tdt-v2` | **3.0** | 5.1 | **3.0** | 2.1 |
| `canary-qwen-2.5b` | **3.4** | 4.7 | **3.4** | 1.3 |
| `canary` | 5.1 | 6.0 | **2.5** | 3.4 |
| `parakeet-tdt-v3` | 5.1 | 6.0 | **3.4** | 2.6 |
| `parakeet-ctc-1.1b` | 5.1 | 6.8 | **2.5** | 4.3 |
| `whisper-large-v3-turbo` | 5.5 | **4.3** | 4.7 | 1.3 |
| `canary-180m-flash` | 5.5 | 5.1 | **4.3** | 1.3 |
| `distil-whisper-large-v3` | **3.8** | 7.7 | **3.8** | 3.8 |
| `whisper-large-v3` | **3.8** | 8.1 | **3.8** | 4.3 |
| `moonshine-base` | 5.1 | 9.8 | **3.8** | 6.0 |
| `canary-1b-v2` | 8.5 | 6.4 | **6.0** | 2.5 |

**The engine decides which model wins.** `say` → `kyutai-stt-2.6b` at 2.1 %,
`piper` → `whisper-large-v3-turbo`, `kokoro` → `granite-speech-4.1` at 0.9 %.
Rank correlation between engines is weak: Spearman ρ = +0.22 (say/piper), +0.59
(say/kokoro), +0.23 (piper/kokoro). Only `say` and `kokoro` agree even loosely,
and no model wins twice.

`piper` is consistently the hardest engine and the one that disagrees with the
others. Every model scores worse on it, and the models it punishes hardest —
`moonshine-base` (+4.7 against `say`), `whisper-large-v3` (+4.3) — are not the
ones the other two engines separate.

**Read small gaps with suspicion — here that is most of the table.** The
reference is 237 words, so one word is 0.42 points. The whole 13-model field
spans 2.1 to 9.8 % across all engines, and within `say` twelve of thirteen models
fit in 3.4 points — about eight words. Most adjacent pairs are one or two words
apart, which is not a ranking. The defensible reading is that **on short clean
English these models are not distinguishable by accuracy**, and the columns worth
deciding on are RTF and peak RSS. Separating them needs harder or longer
material.

This table has been wrong twice, in opposite directions, and both corrections are
recorded rather than deleted because the failure modes are the point of the repo:

1. An early version used 3 models and showed the ranking *inverting* between
   `say` and `piper`. That was an artefact of a bug where a speaker's `rate:` was
   honoured only by `say`, so the other engines were not merely differently
   voiced but differently paced.
2. The replacement showed a clean 5–6 % / 9–11 % split with ρ ≈ +0.7 and was an
   artefact of **scoring**, not synthesis. The reference reads a case ID digit by
   digit; models that emit `4821-773` had it expanded as a cardinal and the
   spoken "dash" counted as a deletion — together about 6 WER points for
   formatting choices. Fixing it (see *How scoring works*) moved
   `parakeet-tdt-v2` from 7th to 3rd, compressed the table, and dropped the rank
   correlations from +0.65…+0.76 to +0.22…+0.59. **The "the engine broadly
   preserves the ranking" claim did not survive its own bug fix** — the
   correlation it rested on was largely the artefact.

The honest rule is the weaker one: compare models within an engine, and treat a
result that only holds on one engine as a property of that engine until it is
shown on another.

On German the effect is much stronger, and it survives the scoring fix (which
shifted German by a uniform ~0.8 points and changed no ranks).
`whisper-large-v3` is first of six on `say` and fifth on `piper`, while both
Canary models improve by 7–9 points going the other way. German is currently the
only part of this benchmark that separates models at all. See the German table in
the results section.

The axis is not specific to transcription. Diarisation shows the same thing more
violently — `sortformer` scores 36.6 % DER on `standup-de__say__clean` and 1.0 %
on the same script under `piper`. See the diarisation results below.

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

## 5. Run the whole pipeline

`bench.py` asks which model transcribes best and `diarize.py` asks which
backend separates speakers best. A meeting transcript is judged on neither:
it is judged on whether the right words end up under the right person, and a
configuration can win both halves separately and still produce unusable
minutes.

```sh
uv run python pipeline.py --session sessions/crosstalk-de__piper__clean \
    --diarizer sortformer --asr parakeet-tdt-v3 --num-speakers 4
```

Diarise first, then transcribe per speaker, then score with **cpWER**. The
ordering is deliberate: transcribe-first attribution needs word-level
timestamps from every backend, which not all of them expose and none expose
the same way, while diarise-first needs nothing from an ASR model except
that it accepts samples — so all thirteen models are usable immediately.

What the caller knows is part of the configuration, not a fixed fact. You
usually do know how many people are in the room and what language they speak,
and a backend given that information is a different algorithm from one
guessing it. `--num-speakers` and `--language` are knobs to search over.

`--attribution speaker` (the default) concatenates each speaker's audio and
makes one ASR call per speaker; `segment` calls per diarised turn, which
keeps timing and avoids splicing but costs one model load per turn. Every
runner loads its weights on each call, so that is the difference between four
calls and forty.

## 6. Search the configuration space

```sh
uv run python search.py \
  --sessions sessions/crosstalk-de__piper__clean \
             sessions/allhands-de__piper__clean \
             sessions/allhands-en__piper__clean \
             sessions/crosstalk-de__say__clean \
  --holdout engine --run-name search-v1
```

Sweeps diarizer × ASR × attribution × speaker-count hint, reports the best
configuration overall *and* the best per condition, then validates each
winner on sessions held out of the search. `--holdout engine` trains on one
synthesiser and validates on another, which is the strictest split available
here — the engine reorders the model ranking outright, so a config picked and
validated on the same synthesiser has proved nothing about speech.

`--samples N` switches from the full grid to random search. `--dry-run`
prints the plan and the evaluation count before committing the machine to an
hour of work. Diarisation is cached across ASR models within a run, which is
what makes the sweep affordable: the diarizer does not depend on what
transcribes afterwards.

`--halving` drops the worst half of the field after each session. It is off
by default and refuses to run when the sessions span more than one condition,
for a reason discovered by running it: see below.

## 7. Compare

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

The gap between them is formatting, not misrecognition.

**Digit runs are scored under both readings, and the better one wins.** How a
spoken number *should* be written is genuinely ambiguous, and the ambiguity has
two distinct branches: a quantity is read as a cardinal (*"hundertzwanzig"* →
`120`) while an identifier is read digit by digit (*"four eight two one"* →
`4821`). A model that emits `4821` has thrown that distinction away; one that
spells the words out has kept it. Expanding digits only as cardinals therefore
charges digit-emitting models for a decision the reference made rather than for
anything they misheard. So `score()` normalises both sides twice — cardinal and
digit-wise — and keeps whichever gives the lower WER.

**Spoken punctuation next to a number is dropped.** A reference saying *"four
eight two one, dash, seven seven three"* and a model writing `4821-773` agree,
but the hyphen is stripped as punctuation while the word *dash* survives, so the
word becomes a deletion nobody earned. Words like *dash*, *point*, *dot* and
*Punkt* are removed when they sit next to a number — only then, so an ordinary
*"that's a good point"* is still scored normally.

Neither rule hides a real error: `1338` against *"one two two seven"* scores 0.6,
and *"one hundred thirty"* against *"one hundred twenty"* scores 0.25.

This is not cosmetic. Before these fixes the English table showed a clean split
between a 5–6 % group and a 9–11 % group, and the split was entirely artefact:
the "good" group happened to spell numbers out like the reference, the "bad"
group wrote digits. Correcting it moved `parakeet-tdt-v2` from 7th place to 3rd,
took up to 6 WER points off nine of thirteen models, and collapsed the apparent
structure of the whole table — including the cross-engine rank correlation that
had been quoted as evidence the ranking was stable.

`test_score.py` pins all of this down; run it with `python test_score.py`. When a
rule changes, `uv run python rescore.py --dry-run` shows which stored results the
change would move — the decoded text lives in `summary.json`, so correcting the
whole result history costs seconds and no GPU.


`score.py` has no third-party dependency — it ships its own Levenshtein alignment and a German/English number speller. If you install [`jiwer`](https://pypi.org/project/jiwer/) and [`num2words`](https://pypi.org/project/num2words/) yourself, it picks them up automatically for the reference implementation and wider language coverage:

```sh
uv pip install jiwer num2words
```

### Speaker-attributed transcription: cpWER

WER and DER measure different halves of the same job and neither answers the
question a meeting transcript is judged on. A pipeline can score 5 % WER and
20 % DER and still produce minutes nobody can use, because every sentence is
correct and half of them are filed under the wrong name.

**cpWER** (concatenated minimum-permutation WER, from CHiME-6/7 DASR) closes
that gap. Each speaker's utterances are concatenated into one stream per
side, the speaker assignment that minimises total errors is found, and WER is
reported over the whole thing. Assignment is by content, so a system labelling
people `spk0`/`spk1` is not punished for it. A speaker never found costs all
their words as deletions; an invented one costs all of its own as insertions,
otherwise a system could hedge by splitting everyone in two.

The assignment is exact, not heuristic: total errors decompose as a sum over
assigned pairs and the denominator is fixed, so Hungarian on the padded error
matrix minimises cpWER itself.

How much this sees that WER cannot: three turns, every word transcribed
correctly, one turn attributed to the wrong speaker. **Flat WER scores that
0.0 %. cpWER scores it 62.5 %.** That case is in `test_score.py`.

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

**English** — `support-call-en__say__clean`, 80 s, 2 speakers, 237 reference words,
all 13 models that produce a transcript:

| Model | Wall clock | RTF | Peak RSS | WER | CER |
|---|---:|---:|---:|---:|---:|
| kyutai-stt-2.6b | 203 s | 2.54 | 7.8 GB | **2.1 %** | 0.7 % |
| parakeet-tdt-v2 | 14 s | **0.17** | 5.3 GB | 3.0 % | 1.4 % |
| canary-qwen-2.5b | 74 s | 0.93 | 10.4 GB | 3.4 % | 1.5 % |
| granite-speech-4.1 | 42 s | 0.53 | 9.0 GB | 3.4 % | 1.6 % |
| whisper-large-v3 | 150 s | 1.87 | 3.6 GB | 3.8 % | 1.8 % |
| distil-whisper-large-v3 | 27 s | 0.33 | 2.3 GB | 3.8 % | 1.7 % |
| canary | 34 s | 0.42 | 6.7 GB | 5.1 % | 1.4 % |
| parakeet-tdt-v3 | 17 s | 0.22 | 5.4 GB | 5.1 % | 2.5 % |
| parakeet-ctc-1.1b | 21 s | 0.26 | 8.3 GB | 5.1 % | 1.6 % |
| moonshine-base | 17 s | 0.21 | **0.8 GB** | 5.1 % | 2.3 % |
| canary-180m-flash | 15 s | 0.19 | 2.0 GB | 5.5 % | 1.7 % |
| whisper-large-v3-turbo | 42 s | 0.52 | 2.4 GB | 5.5 % | 3.8 % |
| canary-1b-v2 | 45 s | 0.57 | 8.0 GB | 8.5 % | 2.9 % |

**This column does not separate these models, and saying so is the result.**
237 words means one word is 0.42 points. Twelve of thirteen models sit between
2.1 % and 5.5 % — eight words end to end — and six of them are inside a single
word of each other. Ranking them would be reading noise. The task is too short
and too clean to discriminate; a benchmark that cannot tell its candidates apart
should report that rather than publish an order.

What the table does support is the choice on the other columns, where the spread
is real: `parakeet-tdt-v2` is 15× faster than `kyutai-stt-2.6b`, and
`moonshine-base` runs in 0.8 GB against `canary-qwen-2.5b`'s 10.4 GB. Those are
factors, not fractions of a word. **On clean short English, pick on RTF and
memory, because accuracy will not decide it for you.**

`kyutai-stt-2.6b` is the one model outside the pack at 2.1 %, and it is also the
only one slower than realtime by a wide margin — and its win does not survive a
change of TTS engine, so even that separation is conditional.

An earlier version of this table showed a 5–6 % group cleanly separated from a
9–11 % group, and that structure was a scoring artefact: models writing
`4821-773` for a case ID the reference reads digit by digit were charged for the
cardinal expansion *and* for the spoken "dash". The CER column had been flagging
it all along — 7.9 % CER beside 9.3 % WER is not what scattered word errors look
like — and this README said as much for several revisions without anyone
following it up. Once both were fixed, nine of thirteen rows dropped by up to 6
points and the structure disappeared. The remaining CER outlier is
`whisper-large-v3-turbo` at 3.8 %.

**German** — `standup-de__clean`, 94 s, 3 speakers, 226 reference words. Only six
models claim German. The `piper` column is the same script under the other
engine:

| Model | Wall clock | RTF | Peak RSS | WER (`say`) | CER (`say`) | WER (`piper`) |
|---|---:|---:|---:|---:|---:|---:|
| whisper-large-v3 | 155 s | 1.65 | 4.0 GB | **4.9 %** | 1.9 % | 12.5 % |
| whisper-large-v3-turbo | 104 s | 1.11 | 2.4 GB | 5.4 % | 1.4 % | **8.5 %** |
| parakeet-tdt-v3 | 28 s | 0.30 | 5.3 GB | 6.7 % | 2.7 % | 9.4 % |
| canary-1b-v2 | 70 s | 0.75 | 7.8 GB | 9.8 % | 3.8 % | 8.9 % |
| canary | 47 s | 0.51 | 7.2 GB | 20.1 % | 16.8 % | 11.6 % |
| canary-180m-flash | 28 s | 0.29 | 2.2 GB | 22.8 % | 19.8 % | 15.2 % |

**German is the only part of this benchmark that currently separates models.**
The spread is 4.9 % to 22.8 % — a factor of nearly five, against English's
2.1–5.5 % — so unlike the English table this one supports an ordering.

**It is also where the engine genuinely reorders the ranking.**
`whisper-large-v3` is first on `say` and fifth of six on `piper`, losing 7.6
points; both Canary models go the other way and *gain* 8.5 and 7.6. This is a
much larger effect than English shows, and it is the concrete reason the engine
is an axis rather than a footnote. Read it with the caveat in the TTS section:
the German Piper voices are unevenly trained, so part of this is voice quality
rather than model behaviour — the direction is trustworthy, the magnitude is not.

**What breaks Canary on German is what the script was built to break.** Its
errors are concentrated in English loanwords and spoken numbers — `Backend`
becomes *Wacken*, `Deployment` becomes *Dokumentarfilm*, and where the reference
says "ungefähr hundertzwanzig Millisekunden" Canary emits "ungefähr
Millisekunden", deleting the quantity outright. Of the 23 number-words in the
reference, `canary` keeps 15 and `canary-180m-flash` keeps 9, neither producing a
single digit; Whisper and Parakeet write 19 digit characters instead, which score
as correct because the scorer normalises `120` against *hundertzwanzig*. That
ordering — 9, 15, then digits — tracks the WER column exactly.

The CER column is the tell. A model at 20 % WER and 3 % CER is getting words
slightly wrong; one at 20 % WER and 17 % CER is dropping content. That is worth
more than the WER ranking, because a transcript missing its numbers fails
differently from one that misspells them.

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

### Pipeline results: the halves do not add up

`crosstalk-de__piper__clean`, 67 s, 4 speakers, 9.3 % overlapping speech,
scored end to end with cpWER:

| Diarizer | ASR | speakers told | cpWER | DER | speakers found | Wall clock |
|---|---|---|---:|---:|---:|---:|
| sortformer | whisper-large-v3 | 4 | **18.6 %** | 1.0 % | 4 | 94 s |
| sortformer | parakeet-tdt-v3 | 4 | 20.6 % | 0.5 % | 4 | 55 s |
| sortformer-streaming | parakeet-tdt-v3 | 4 | 22.1 % | 1.1 % | 4 | **49 s** |
| titanet | parakeet-tdt-v3 | *auto* | 32.4 % | 15.2 % | 10 | 117 s |
| titanet | parakeet-tdt-v3 | 4 | 53.9 % | 24.5 % | 4 | 65 s |

**Telling the clustering backend the right answer makes it worse — by 21
points.** TitaNet given the correct speaker count scores 53.9 %; the same
backend left to guess finds *ten* speakers, six of them wrong, and scores
32.4 %. This is the opposite of the obvious assumption, and it is the
strongest argument for treating "what the caller knows" as a parameter to be
searched rather than a fact to be passed through.

The mechanism is visible in the DER split. Forced to four clusters, TitaNet
must place every window in one of them, so overlapped and ambiguous speech is
merged into confident wrong assignments and each stream is contaminated.
Left free, it over-segments; four clusters carry most of the words cleanly
and the six spurious ones cost only their own content as insertions. Being
fragmented is recoverable, being merged is not.

That mechanism is real, but it is **not** a rule you can apply blind — see the
six-speaker results below, where the same hint on the same backend helps in one
language and hurts in the other.

**Near-perfect diarisation still costs about four points of word accuracy.**
The same ASR model on the undiarised mix scores 16.7 % WER against the
pipeline's 20.6 % cpWER, at DER 0.5 % — so the loss is not the diarizer
getting speakers wrong. It is that overlapped regions hand both speakers the
same mixed audio, boundary errors clip word edges, and concatenating
non-adjacent speech removes context. For `whisper-large-v3` the gap is wider
still: 12.2 % flat against 18.6 % attributed, +6.4 points.

That trade has no equivalent in either half's own table, and it is the number
a deployment actually pays: **attribution is not free.** On this session the
bill is roughly a quarter of your word accuracy; on others it is larger, and
on one it is negative (see below). Whether it is worth paying depends
entirely on whether anyone needs to know who said what — which is exactly the
choice this repo exists to inform.

The end-to-end models win here by a wide margin, and the ordering of the ASR
models under the pipeline matches their ordering on overlapped audio in the
flat benchmark: `whisper-large-v3` degrades most gracefully when a second
voice cuts in, and that survives being wrapped in a pipeline.

A full 8-model matrix on a 90-second session takes roughly 10 minutes. Longer sessions scale linearly — plan accordingly.

### Six speakers: where the end-to-end diarizers stop

`crosstalk-*` has four speakers, which happens to be exactly where the
Sortformer family is comfortable. `allhands-de` (123.6 s, 2.8 % overlap) and
`allhands-en` (105.5 s, 2.6 % overlap) have six, and are otherwise built the
same way. ASR is `parakeet-tdt-v3` throughout so the diarizer is the only
variable.

| Session | Diarizer | speakers told | cpWER | DER | speakers found | Wall clock |
|---|---|---|---:|---:|---:|---:|
| `allhands-de` | titanet | 6 | **21.5 %** | 6.1 % | 6 | 125 s |
| `allhands-de` | titanet | *auto* | 32.5 % | 11.1 % | 13 | 195 s |
| `allhands-de` | sortformer | 6 | 47.2 % | 20.6 % | **4** | 60 s |
| `allhands-de` | sortformer-streaming | 6 | 60.3 % | 34.2 % | **4** | 52 s |
| `allhands-en` | titanet | *auto* | **12.4 %** | 8.0 % | 9 | 123 s |
| `allhands-en` | titanet | 6 | 35.1 % | 21.9 % | 6 | 96 s |
| `allhands-en` | sortformer-streaming | 6 | 55.8 % | 25.9 % | **4** | 54 s |
| `allhands-en` | sortformer | 6 | 67.0 % | 28.3 % | **4** | 48 s |

**Both Sortformer variants report exactly four speakers, in both languages,
even when told there are six.** Four runs, four times the same answer. This is
an architectural ceiling, not a tuning problem: the end-to-end models emit a
fixed number of speaker channels, and the count they were told is ignored.
They do not *drop* the two extra people — they **merge** them into existing
channels, which is the failure mode cpWER prices most harshly, because one
merge contaminates two streams. The winner of the four-speaker table is the
loser here by 26 points.

**The best diarizer flips with speaker count.** End-to-end wins at four,
clustering wins at six. Nothing in either component's own benchmark predicts
this, and neither architecture is simply better.

**The speaker-count hint is not a reliable input.** Across three sessions the
same knob points in different directions:

| Session | told the truth | left to guess | hint helps? |
|---|---:|---:|:--:|
| `crosstalk-de` (4 spk, 9.3 % overlap) | 53.9 % | 32.4 % (found 10) | no, −21.5 |
| `allhands-de` (6 spk, 2.8 % overlap) | 21.5 % | 32.5 % (found 13) | yes, +11.0 |
| `allhands-en` (6 spk, 2.6 % overlap) | 35.1 % | 12.4 % (found 9) | no, −22.7 |

Two of three prefer the guess, but the exception is large and it is not
explained by speaker count, language, or overlap density — the two German
sessions differ in both directions at once. Overlap density is confounded with
speaker count in this design, so these runs cannot separate them. The
actionable conclusion is the conservative one: **`--num-speakers` is a
parameter to search, not a fact to pass through**, even when you genuinely
know the answer. That is precisely the case [#17](../../issues/17) exists for.

**The attribution bill is not a constant either.** Against the same model on
the undiarised mix: German 23.3 % flat vs 21.5 % attributed — the pipeline is
*cheaper* than the baseline, because splitting a six-way mix gives the ASR
model cleaner input than the overlapped whole. English 5.6 % flat vs 12.4 %
attributed, +6.8 points. So the "roughly a quarter of your word accuracy"
figure from the four-speaker table is a data point, not a law.

### The config search: what one answer costs you

18 configurations (3 diarizers × 3 ASR models × speaker-count hint on/off) on
three sessions, `say` held out against `piper`. 54 evaluations, about 75
minutes.

| | best config | cpWER |
|---|---|---:|
| **Best overall** (mean of three) | `titanet` + `parakeet-tdt-v3`, n=auto | 25.8 % |
| de, 4 speakers | `sortformer` + `whisper-large-v3` | **18.6 %** |
| de, 6 speakers | `titanet` + `parakeet-tdt-v3`, n=**known** | **21.5 %** |
| en, 6 speakers | `titanet` + `parakeet-tdt-v3`, n=auto | **12.4 %** |

**The single best configuration is the best one nowhere except English.** Run
it on the four-speaker German session and it scores 32.4 % against the 18.6 %
a condition-aware choice gets — **13.8 points paid for the convenience of one
answer.** Across the three conditions the average penalty is 8.3 points. That
is the price of "just tell me which model to use", measured rather than
asserted, and it is the argument for the pipeline being configurable at all.

**Held out, the winner holds up.** `sortformer` + `whisper-large-v3` picked on
`crosstalk-de__piper` scores 18.6 % there and 12.8 % on the same conversation
synthesised by `say` — a *negative* gap of 5.9 points. The held-out session is
easier, not the config overfitted. This is the check that would have caught it
if it had.

**The speaker-count hint is a property of the diarizer, not of the run.** For
both Sortformer variants, `n=auto` and `n=known` are bit-identical in all 18
cells — mechanical confirmation that a fixed-channel architecture cannot use
the hint. For TitaNet the hint is worth 10–29 points, and its *sign* is
constant within a session across all three ASR models:

| Session | parakeet | whisper | canary |
|---|---:|---:|---:|
| `crosstalk-de` (4 spk) | +21.6 | +15.2 | +10.8 |
| `allhands-de` (6 spk) | **−11.1** | **−9.5** | **−13.1** |
| `allhands-en` (6 spk) | +22.7 | +28.6 | +26.8 |

(positive = telling the truth makes it worse)

Nine measurements, three flips of sign, and never once a disagreement between
ASR models on the same audio. So the effect is **reproducible but not
predictable**: it belongs to the diarizer-audio pair, and nothing a caller
knows in advance — language, speaker count, overlap — tells you which way it
will go. Searching it is not laziness, it is the only sound option.

**The diarizer dominates; the ASR model barely matters.** Within one diarizer
and session the three ASR models span 2–4 points. Across diarizers on the same
session the spread is 55 points. Almost all the accuracy in a speaker-attributed
transcript is decided before a word is decoded — which is the exact opposite of
where a model comparison table directs your attention.

That has a cheap consequence: on `allhands-en`, `canary-180m-flash` scores
15.9 % in 25 s where `whisper-large-v3` scores 13.0 % in 207 s. **Eight times
faster for 2.9 points**, once a competent diarizer is doing the hard part.

### Multi-model fusion: real, and not worth it

`--fusion rover` runs every model in `--asr` on each speaker stream and votes
word by word. `--fusion escalate` runs the first two and sends only the
speakers where they *disagree* to the third — disagreement between two cheap
models is a confidence signal that needs no reference, so it works on real
audio, not just on benchmarks.

| Session | best single model | `rover` (3 models) | `escalate` |
|---|---|---|---|
| `allhands-en`, titanet | 12.4 % / 117 s | **12.1 %** / 313 s | **12.1 %** / 206 s |
| `allhands-de`, titanet n=6 | 21.5 % / 106 s | **20.6 %** / 247 s | **20.6 %** / 231 s |

**Fusion works and it barely matters.** Three models buy 0.3 points in
English and 0.9 in German, for two to three times the compute. On the same
German session, choosing TitaNet over Sortformer is worth **25.7 points**. The
diarizer is worth roughly thirty times what the ASR ensemble is worth, and
compute spent on a second and third opinion is compute not spent on the half
of the pipeline that decides the outcome.

**Escalation does exactly what it promises, when the premise holds.** In
English it escalated 5 of 9 speaker streams and reached the *identical* cpWER
as full fusion for a third less wall clock — the four skipped streams were
ones where the third opinion would not have changed the vote. In German it
escalated 6 of 6 and saved nothing: with a weaker base model the cheap pair
rarely agrees, so the policy degenerates into full fusion. **The saving is a
function of how often your models already agree**, which is worth measuring
before designing around it.

One structural note that falls out of ROVER and is easy to get wrong: with
two hypotheses there is never a majority, so voting returns the pivot
unchanged. **Fusion needs three voters to do anything at all.** Paying for a
second model buys nothing unless you also pay for a third, or use the second
only as an escalation trigger — which is what `escalate` does.

### Where the search harness was wrong about itself
The first run of `search.py` used successive halving, and it produced a
confident, wrong answer. Round one ran all 18 configs on the four-speaker
session and eliminated every TitaNet configuration — correctly, TitaNet is the
worst backend there by 20 points. TitaNet is also the **winner at six
speakers** by 26 points, and it was never measured there.

Halving assumes a configuration that loses on the first session loses
everywhere. That assumption is precisely what the rest of this README
refutes. The two ideas are incompatible: you cannot budget by early
elimination while searching for condition-dependent winners.

`--halving` is now off by default and refuses to run when the development
sessions span more than one condition, printing why. It remains useful for
sweeping many sessions of the *same* condition, which is the case it was
designed for.

The same run also compared a six-speaker development mean against a
four-speaker held-out session and reported a 39-point gap as evidence of
overfitting. It was measuring difficulty. Held-out validation is now
condition-matched.

## Repo layout

| File | Role |
|---|---|
| `synth.py` | conversation script → session with audio + ground truth |
| `degrade.py` | degradation profiles (noise, codec, reverb, clipping) |
| `bench.py` | runs the model × channel matrix, scores against the reference |
| `diarize.py` | speaker diarisation backends, scored as DER |
| `score.py` | normalisation + WER/CER alignment + DER + cpWER |
| `test_score.py` | regression tests for the normalisation rules |
| `rescore.py` | re-scores stored transcripts after a `score.py` change, no model runs |
| `pipeline.py` | diarise → transcribe per speaker → cpWER, as one configurable pipeline |
| `search.py` | sweeps the pipeline config space, best overall and best per condition, held-out validated |
| `fuse.py` | ROVER voting over several ASR hypotheses + the reference-free agreement signal |
| `compare.py` | run directory → Markdown report |
| `audio_io.py` | shared decode/encode helpers (ffmpeg + libsndfile) |
| `envfile.py` | loads `.env` before torch/NeMo import, so caches land where you asked |
| `hf_runners.py` | transcription runners for the `transformers`-based models |
| `conversations/` | conversation scripts — the source of truth for sessions |

`sessions/`, `runs/`, `.tts-cache/` and `.piper-voices/` are gitignored: all four are reproducible from the scripts, so the scripts are what gets committed, not the waveforms.
