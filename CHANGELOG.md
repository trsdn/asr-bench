# Changelog

All notable changes to asr-bench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`cleanup.py` — LLM transcript cleanup that is checked rather than trusted.** The model proposes and a reference-free guard decides. Numbers are compared as a **digit signature**, so `vier null neun strich zwei eins` and `409-21` both reduce to `40921`: rewriting the notation is the point of the step and changing the value is the thing that must never happen, and only a value-level comparison can tell those apart. Identifiers and acronyms are compared as tokens, with acronyms deliberately given the weaker test because ASR output is uncased and `api` → `API` is the desired behaviour, not an invention. A similarity floor on number-masked prose catches the model summarising instead of tidying — masked because the digit check and the floor must look at disjoint text, or approving a number rewrite looks like a divergence failure.

  Every proposal is retained even when vetoed, so one generation scores both variants. `allhands-de`: 21.5 % raw, 28.7 % guarded, **51.3 % unguarded**. `allhands-en`: 12.4 % raw, 22.4 % guarded, **32.1 % unguarded**. The guard prevented 22.6 and 9.7 points of damage and every veto was correct on inspection — it caught the model reordering a segment's digits (`409211824` → `182440921`), dropping a digit from a phone number, and replacing two segments with a summary.

  Cleanup is nonetheless net negative, by 7.2 and 10.0 points. The accepted proposals show why: the ASR heard `Loks` where the reference says `Logs`, and the cleanup promoted it to `Lokalitäten` — fluent, grammatical and further from the truth. **A cleanup model repairs fluency, not truth**, and where the ASR was already wrong it makes the error more plausible rather than less. Plausible errors cost more than visible ones, because visible garbage is a signal to go back to the audio.

  Recorded caveat: cpWER scores against a verbatim reference and a cleanup is supposed to change words, so part of the penalty is the metric refusing to credit wanted work. That does not rescue the result — dropped sentences and invented clauses are not artefacts — but it does mean the right evaluation of a cleanup is entity preservation plus readability, not WER. Entity preservation is what the guard already measures, reference-free: **4 of 6 German and 2 of 9 English proposals broke something checkable**, a number available in production where cpWER never is.

  Backend is `Qwen2.5-1.5B-Instruct`, chosen as the smallest model that could plausibly succeed so that a positive result would have been cheap. Phi-4-multimodal, already cached for the audio benchmark, could not be used: it attaches its adapters through `peft`, which calls a `transformers` method removed in 4.57.

- **12 guard cases in `test_score.py`** (36 total). Half of them assert *acceptance* — punctuation and casing, spoken numbers rewritten to digits, digits rewritten to words, fillers removed, an acronym capitalised — because a guard that vetoes everything is trivially safe and useless, so the permissive side needs pinning as hard as the restrictive one.

- **`conversations/longform-de.yaml` — a length axis, controlled.** 4.6 minutes with the same six speakers, voices and rates as `allhands-de`, whose forty-one turns appear unchanged as the final segment. The same material is therefore scored twice: standalone, and at the end of a recording more than twice as long. Nothing else differs, so anything that moves is runtime effect.

  Nothing moved. That material scores **21.5 %** flat WER inside the long session against 23.3 % standalone, whole-session cpWER is 18.7 % against 21.5 %, DER 6.5 % against 6.1 %, all six speakers found, and no repetition loops. Splitting at the content boundaries puts the worst block in the **middle** (27.9 %, the passage built dense with overlap) rather than at the end — error rate tracks what is said, not how long the recording has run.

  Two side effects worth keeping: the attribution bill nearly vanished (**+0.4** points over the undiarised baseline against the usual +6.4, because more speech per speaker helps a clustering diarizer), and diarisation cost 107 s against 62 s of ASR, so on long audio the diarizer is the larger bill in wall clock too.

  Caveat recorded in the README: this rules out degradation at five minutes, not at sixty. Loops and drift appear non-linearly.

- **`conversations/accent-en.yaml` — an accent axis, with its limits stated.** The `allhands-en` conversation unchanged, except three of the six speakers are voiced by *German* Piper models reading English text: a German VITS voice phonetises English with German grapheme-to-phoneme rules, landing the substitutions roughly where an L2 German speaker puts them. Because the reference, timing and overlap are identical and both groups share one recording and one diarisation, the comparison is **within-session** — nothing differs but the voice.

  Native speakers score **11.2 %** cpWER, German-voiced speakers **78.1 %**. Seven times worse from the voice alone; the session goes 12.4 % → 49.6 % cpWER and the undiarised baseline 5.6 % → 38.9 % WER. The failure is legibly phonetic (`through` → `truk`, `the` → `be`, `walk` → `valk`), and the stream then truncates — the model does not merely mis-hear the accent, it gives up on it.

  **Diarisation was almost untouched** (DER 11.4 % vs 8.0 %, every speaker still its own cluster). Accent damages *what was said*, not *who said it*, so the two halves of the pipeline fail on different things.

  The proxy overshoots and is documented as doing so: a German TTS model has never seen English, so it applies German phonology unconditionally where a real L2 speaker applies it partially, and it carries none of the prosody or disfluency. The number is an **upper bound on accent difficulty**, not a model of a mild accent. Measuring that needs recorded L2 speech.

- **`fuse.py` — multi-model voting and selective escalation, measured and found marginal.** `--fusion rover` runs several ASR models per speaker stream and votes word by word; `--fusion escalate` runs two and sends only the speakers where they *disagree* to a third, using inter-model disagreement as a confidence signal that needs no reference and therefore works on real audio.

  Both work. Neither is worth much. Three models buy **0.3 points in English and 0.9 in German for two to three times the compute**, while on the same German session choosing TitaNet over Sortformer is worth 25.7 points. The diarizer is worth about thirty times what the ASR ensemble is worth.

  Escalation behaves exactly as designed when its premise holds: in English it escalated 5 of 9 streams and reached the identical cpWER as full fusion for a third less wall clock. In German it escalated 6 of 6 and saved nothing, because a weaker base model makes the cheap pair disagree everywhere. The saving is a function of how often the models already agree, which is worth measuring before designing around it.

  One structural property, pinned in the tests: with two hypotheses there is never a majority, so voting returns the pivot unchanged. **Fusion needs three voters to do anything at all** — a second model buys nothing unless a third follows, or unless it is used purely as an escalation trigger.

- **`search.py` — the pipeline config space, searched rather than assumed.** Sweeps diarizer × ASR × attribution × speaker-count hint and reports two different answers: the best configuration overall, and the best one *per condition* (language × speaker count). The gap between them is the finding. On three sessions the single best config — `titanet` + `parakeet-tdt-v3` with the speaker count left to the backend, 25.8 % mean cpWER — is the best choice **nowhere except English**. Applied to the four-speaker German session it scores 32.4 % where a condition-aware choice gets 18.6 %: **13.8 points paid for the convenience of one answer**, 8.3 points averaged across conditions. That is the price of "just tell me which model to use", measured instead of asserted.

  Held-out validation is built in and on by default, because the TTS engine reorders model rankings outright: `--holdout engine` picks on one synthesiser and validates on another, condition-matched. The first winner survived it (18.6 % on `piper`, 12.8 % on `say`).

  Two structural findings came out of the sweep. **The speaker-count hint belongs to the diarizer, not to the run:** for both Sortformer variants `n=auto` and `n=known` are bit-identical in all 18 cells, mechanical confirmation that a fixed-channel architecture cannot use it, while for TitaNet it is worth 10–29 points and its *sign* flips between sessions — yet is identical across all three ASR models on the same audio. Reproducible, and not predictable from anything a caller knows in advance. **And the diarizer dominates:** swapping the ASR model moves cpWER by 2–4 points, swapping the diarizer by up to 55. Almost all the accuracy in a speaker-attributed transcript is decided before a word is decoded. Consequence: `canary-180m-flash` scores 15.9 % in 25 s where `whisper-large-v3` scores 13.0 % in 207 s.

  The harness also caught itself being wrong. Its first run used successive halving, eliminated every TitaNet config on the four-speaker session — correct there, it loses by 20 points — and so never measured TitaNet at six speakers, where it *wins* by 26. Halving assumes a config that loses first loses everywhere, which is exactly what this repo's results refute; the two ideas are incompatible. `--halving` is now off by default and refuses to run across mixed conditions. Held-out validation likewise no longer compares a six-speaker development mean to a four-speaker held-out session, which had reported a 39-point "overfitting" gap that was really a difficulty gap.

- **Diarisation caching in `pipeline.py`** (`cache_diarization=True`, used by the search). Diarisation does not depend on which ASR model runs after it, so a sweep across models otherwise repeats identical work dozens of times. Off by default so a one-shot run reports honest wall clock.

- **`pipeline.py` and cpWER — the benchmark now scores a pipeline, not a component.** `bench.py` measures transcription and `diarize.py` measures speaker separation, and neither answers what a meeting transcript is judged on: whether the right words end up under the right person. cpWER (concatenated minimum-permutation WER, CHiME-6/7 DASR) does. The gap is not academic — three turns transcribed word-perfectly with one filed under the wrong speaker score **0.0 % on flat WER and 62.5 % on cpWER**.

  Two findings fell out of the first run, both invisible to either half alone. **Telling TitaNet the correct speaker count made it 21 points worse** (53.9 % vs 32.4 % when left to guess and finding ten speakers): forced to four clusters it merges ambiguous speech into confident wrong assignments, while over-segmenting keeps clusters pure and costs only the spurious ones. And **near-perfect diarisation still costs about four points of word accuracy** — 16.7 % flat WER against 20.6 % cpWER at DER 0.5 %, rising to +6.4 for `whisper-large-v3`. Attribution is not free, and neither component table shows the bill.

- **Six-speaker sessions (`conversations/allhands-de.yaml`, `allhands-en.yaml`), which is where the end-to-end diarizers stop.** `crosstalk-*` has four speakers, exactly the count the Sortformer family is built for. Given six, **both Sortformer variants report exactly four, in both languages, even when told there are six** — four runs, the same answer every time. It is an architectural ceiling, and they *merge* the extra speakers rather than dropping them, which is the failure cpWER prices most harshly. The winner of the four-speaker table (20.6 %) is the loser here by 26 points (47.2 %). **The best diarizer flips with speaker count:** end-to-end at four, clustering (TitaNet, 21.5 %) at six.

  This also retires the "cluster purity beats cluster count" reading of the four-speaker result. Across three sessions the speaker-count hint helps once and hurts twice, and the direction is not explained by speaker count, language, or overlap density — the two German sessions disagree. `--num-speakers` is therefore **a parameter to search, not a fact to pass through**, even when the caller genuinely knows it. The attribution bill is not constant either: on `allhands-de` the pipeline is 1.8 points *cheaper* than the undiarised baseline, because splitting a six-way mix hands the ASR model cleaner input than the overlapped whole.

- **`test_score.py`** — regression tests for the normalisation rules, runnable without pytest. Two published rankings turned out to be scoring artefacts, so the behaviour that fixed them is pinned rather than left to be re-broken.
- **`rescore.py`** — re-scores every stored transcript against its reference without re-running a single model. `summary.json` keeps the decoded text, so a normalisation change can be applied to the whole result history in seconds; `--dry-run` shows which cells would move, which is the evidence that a scoring change did what was intended.
- **English and German re-measured across TTS engines** (13 models × 3 engines on English, 6 × 2 on German). On English no model wins on more than one engine and rank correlation is weak (Spearman ρ +0.22 to +0.59). On German the engine reorders outright: `whisper-large-v3` goes from first of six to fifth, while both Canary models gain 7–9 points in the other direction.
- **Diarisation measured across engines and degradation profiles** (3 backends × 2 TTS engines × 6 profiles). The result is bimodal: Sortformer v1 has no cell at all between 1.0 % and 27.6 % DER. It separates the speakers almost perfectly or collapses, so its mean DER describes no run that ever happened, and the statistic worth reporting is how often a backend collapses. In 14 of the 15 failing cells the mechanism is identical — two reference speakers fused into one 49–84 s cluster with a sub-second phantom holding the freed slot.
- **Lombard effect on interrupting speech.** A turn that starts while the previous speaker is still talking is raised 3 dB. People raise their voice to be heard over someone else, and synthesising an interruption at conversational level produces overlap that is easier to separate than the real thing — which would flatter every diarisation backend on exactly the input they are meant to find hard.
- `conversations/crosstalk-en.yaml`: four speakers, ~8.5 % overlapping speech. Deliberately not a translation of `crosstalk-de` but an independent conversation, so the two are separate samples rather than the same content measured twice. This makes overlap × engine × language available for the first time: `crosstalk-de` cannot run on Kokoro, which has no German voices.
- `interpreter` field in the model registry, so a model needing a conflicting dependency set can run from a second virtualenv while the rest of the matrix stays on the main one. The path resolves against both the script directory and the active venv's parent, so it works from a git worktree as well as from the main checkout.
- **Multiple TTS engines as a first-class benchmark axis.** `--tts` now takes `say`, `piper` or `kokoro` — three unrelated synthesis lineages (Apple system voices, VITS, StyleTTS2) — and the engine is part of the session directory name (`standup-de__piper__clean`). This is not cosmetic: on `support-call-en__clean` the same three models score 9.3 / 11.8 / 11.4 % WER under `say` and 11.4 / 13.9 / 16.5 % under `piper`. The engine moves absolute WER by up to 6.4 points, and moves models unequally, so an absolute WER quoted without its engine means very little.
- Piper and Kokoro voices are auto-assigned per language from a curated pool, one distinct voice per speaker, downloading on first use. Piper previously required an explicit `.onnx` path per speaker, which made it unusable as a comparison axis.
- Optional `gender:` on a speaker, used only to pick a plausible voice from an engine's catalogue. Distinctness still wins over matching, and it has no effect on scoring.
- Eight models: `parakeet-ctc-1.1b`, `canary-qwen-2.5b` (SALM, LLM decoder), `moonshine-base`, `kyutai-stt-2.6b`, `granite-speech-4.1`, `voxtral-mini-3b`, `qwen2-audio-7b` and `phi-4-multimodal`. Measured on `support-call-en__say__clean`: kyutai 2.5 %, parakeet-ctc 5.5 %, canary-qwen 9.7 %, granite 9.7 %, moonshine 11.4 % WER.
- `hf_runners.py`: the `transformers`-based models, one runner per family. The HuggingFace side is not uniform the way NeMo and faster-whisper are — an audio LLM wants a chat template, an encoder-decoder wants a feature tensor, Voxtral wants its own request builder.
- `--timeout` on `bench.py` (default 900 s). Not every failure raises: some models deadlock inside a Metal command buffer that never completes, which presents as a very slow run and holds the machine indefinitely. A timeout records the cell as failed and lets the matrix finish.
- `sortformer-streaming` diarisation backend (nvidia/diar_streaming_sortformer_4spk-v2, 2025) — the streaming successor to Sortformer v1. Cheaper (3.2 s vs 9.9 s on a 71 s session) and more consistent across sessions, though v1 remains stronger on overlapping speech.
- Overlap-specific diarisation metrics: `diarization_error_rate` now reports a separate `overlap` block — seconds, share of speech, DER/miss/false-alarm/confusion restricted to frames with two or more simultaneous reference speakers, plus `detection_recall`. The 0.25 s collar is deliberately not applied there, since overlap sits on the boundaries a collar would mask. Surfaced by `compare.py` as its own table.
- `conversations/crosstalk-de.yaml`: four speakers, 9.3 % overlapping speech, built to exercise crosstalk. The existing scripts have 0.6 % overlap, which is no test at all.
- Tunable `titanet` parameters exposed as CLI flags — `--window`, `--hop`, `--cluster-threshold`, `--min-turn`, `--vad-db` — and `--sweep` / `--sweep-sessions` for a coordinate-descent search over them. The sweep scores every configuration across several sessions and optimises mean DER, counting a failed session as DER 1.0; single-session tuning overfits badly (27.3 % → 3.6 % on the tuned session, 58.8 % on another).
- `diarize.py`: speaker diarisation scored as DER against the session's speaker-labelled ground truth. Two backends — `sortformer` (nvidia/diar_sortformer_4spk-v1, end-to-end, max 4 speakers) and `titanet` (energy VAD → TitaNet embeddings → agglomerative clustering, no speaker limit). The clustering is implemented in-repo, so diarisation needs no extra dependency.
- `score.py` gained `diarization_error_rate`: optimal speaker-label mapping, 0.25 s collar, overlap-aware, reported split into missed speech / false alarm / speaker confusion.
- More models, including current ones: `parakeet-tdt-v3` and `parakeet-tdt-v2` (2025), `canary-1b-v2` (2025), `canary-180m-flash`, `whisper-large-v3-turbo` and `distil-whisper-large-v3`.
- Models declare the languages they support; `bench.py` skips pairings that cannot work (English-only model on a German session) unless `--ignore-language-support` is passed.
- `compare.py` reports the diarisation ranking when `diarize.py` wrote results into the same run directory.
- `envfile.py`: the `.env` loading that has to happen before torch/NeMo import, shared by every entry point instead of copied into each.
- `synth.py`: conversation scripts (YAML) are rendered to session audio via TTS — one voice per speaker, laid out on a timeline with configurable gaps and genuine overlaps, written as per-speaker isolated tracks plus a mixed channel. Because the text is known up front, every session ships a word-exact reference transcript.
- `degrade.py`: reproducible degradation profiles (`clean`, `noisy`, `very-noisy`, `phone`, `voip`, `farfield`, `clipped`, `worst-case`) built from noise, gain, clipping, reverb and codec round-trip steps, seeded so a given seed reproduces byte-identical audio.
- `score.py`: dependency-free WER/CER with Levenshtein alignment and German/English number normalisation, so `3712` and `dreitausendsiebenhundertzwölf` count as a match. Uses `jiwer` / `num2words` when installed.
- `audio_io.py`: shared decode/encode helpers, so synthesis and benchmarking cannot drift on sample rate or channel layout.
- Example conversation scripts: `conversations/standup-de.yaml` and `conversations/support-call-en.yaml`.
- `bench.py` scores every transcript against the reference and ranks results by WER; runs write `run.json` alongside `summary.json`.
- `compare.py` leads with an accuracy ranking and shows the ground truth beside each model's transcript.

### Changed

- **The per-cell time budget is relative to the audio, and one timeout retires a model.** A flat 900 s was wrong in both directions: too generous on a short clip, too tight on a long session, and it let a single dead model burn 45 minutes by timing out identically on all three channels. Each cell now gets `--timeout-floor` (180 s of load time, which does not scale with the audio) plus `--timeout-rtf` × duration (5×), and the remaining channels are skipped once one has gone over budget. The gap this exploits is real: working models run at RTF 0.14-1.6, worst case 3.4, while failing ones sit above 10, with nothing in between. Since the question is whether a model is usable for local transcription — where RTF > 1 already means no — exceeding the budget is a result rather than a missing number.
- Blocked models are excluded from the default `--models` set. They stay in the registry so the reason stays recorded, but naming one explicitly is now the only way to spend time on one.

- **Session directories now carry the TTS engine**: `standup-de__clean` becomes `standup-de__say__clean`. Two sessions from the same script and profile but different engines are different test data, and their numbers are not interchangeable — burying that in a manifest field invites comparing what cannot be compared.
- No CPU fallback for the `transformers` models. A 3B audio LLM on CPU is slower than realtime by a wide margin, which makes the RTF column meaningless and ties up the machine for nothing; a model that will not run on Metal is a result in itself. `ASR_BENCH_DEVICE` overrides this for debugging.
- Voxtral loads with eager attention, and it does not help. Its Ministral decoder parks forever on MPS: a single 5-second chunk still hangs after 300 s while the same weights load in 30 s, so this is a deadlock in a Metal command buffer, not slowness. Eager attention is the usual remedy for SDPA on MPS and was worth trying; the model is now labelled blocked rather than left looking merely untested.
- Renamed the project to `asr-bench`; it is no longer tied to any particular recording app.
- Sessions are now described by a `session.json` manifest with arbitrarily named channels, replacing the fixed mic/system layout.
- Transcription language is pinned from the session manifest for every model that accepts a hint, Whisper included — auto-detection can lose a whole session to one bad guess, which measures language ID rather than transcription.
- Default `titanet` VAD threshold from −33 dB to −40 dB. The only sweep result that held up across sessions; the window and hop defaults were already optimal, and the honest cross-session gain from the whole search is 12.4 % → 11.7 % DER.

### Removed

- The `parakeet-live` entry, which read a recording app's own `transcript.live.jsonl`. Sessions are synthetic now, so there is nothing for it to read.

### Fixed

- **`pipeline.py` results no longer overwrite each other across sessions.** The result filename was built from the config alone, so benchmarking two sessions under one `--run-name` silently discarded the first. The session name is now part of the filename.

- **Numeral and punctuation formatting was scored as transcription error, and it invented the English ranking.** Two separate bugs, same class. (a) A digit run carries no information about how it was spoken — a quantity is a cardinal (*hundertzwanzig* → `120`), an identifier is read digit by digit (*four eight two one* → `4821`) — but scoring only expanded cardinals, so a model writing `4821` for a spelled-out case ID was charged the whole expansion. (b) The reference says *"dash"* where a model writes `-`; the hyphen was stripped as punctuation while the word survived, making it a deletion nobody earned. `score()` now normalises both sides under both digit readings and keeps the better, and drops spoken punctuation words that sit next to a number.

  The effect on the results was not marginal. The apparent 5–6 % / 9–11 % split in the English table — which had looked like a real difference between model families — disappeared entirely; it was separating models that spell numbers out from models that write digits. `parakeet-tdt-v2` moved from 7th place to 3rd. Cross-engine rank correlation fell from +0.65…+0.76 to +0.22…+0.59, which retires the "the engine preserves the broad ranking" claim: that claim rested on the artefact. German shifted by a uniform ~0.8 points and kept its ranking, so its findings stand.

  The CER column had been signalling this for several revisions — 7.9 % CER beside 9.3 % WER is not what scattered word errors look like — and the README said so without following it up.

- **The English task no longer distinguishes the models, and that is now stated as the result.** Twelve of thirteen sit between 2.1 % and 5.5 % on a 237-word reference — eight words end to end, with six inside a single word of each other. The README now directs the choice to RTF and peak RSS, where the spread is a factor of 15 and 13 rather than a fraction of a word.
- **`phi-4-multimodal` now loads and generates.** Pinned `torchvision==0.24.1` (the unpinned install pulls torch 2.13 and breaks NeMo), added `peft` and `backoff`, and filtered `flash_attn` out of the remote-code import scan — it is guarded by `if is_flash_attn_2_available()`, but the scanner only treats `try/except` as optional and there is no Metal build to satisfy it with. The model then exceeds the time budget, so it is marked blocked, joining Voxtral and Qwen2-Audio.
- **All three decoder-only audio LLMs behave the same way on MPS.** Every encoder-decoder ASR model runs at RTF 0.14–1.6; Voxtral, Qwen2-Audio and Phi-4 all exceed a 5× realtime budget, each after a different amount of setup work. Documented as a pattern rather than three accidents, so the next one is assumed blocked until shown otherwise.
- **A speaker's `rate:` was only applied by `say`.** Piper ignored it and Kokoro had `speed=1.0` hard-coded, so the per-speaker tempo declared by all nine speakers across the three scripts silently vanished on two engines out of three. The value is words per minute — `say`'s unit, default 175 — and is now converted to Piper's `length_scale` and Kokoro's `speed` against that baseline. The engines do not respond identically (a 140→220 wpm range yields a duration ratio of 1.36 / 1.29 / 1.50 against an ideal 1.57), so tempo is comparable in direction and roughly in magnitude, not exactly.

  This mattered more than a missing knob suggests: it meant a `say` ↔ `piper` comparison varied voice *and* tempo together. **The ranking inversion previously documented as this repo's headline finding does not survive the fix** — with tempo controlled, `parakeet-tdt-v2` wins on all three engines (9.3 / 11.4 / 9.3 % WER). What remains is a large but order-preserving effect: up to 6.4 points of spread, distributed unequally across models (`say`→`piper` costs Parakeet 2.1 points and Moonshine 5.1). The claim is corrected in the README rather than deleted, since it is a clean example of the failure mode the TTS axis exists to detect.
- TTS cache keys carry a version, so a change that alters the waveform behind an existing key invalidates it instead of serving stale audio.
- Each model now runs in its own subprocess, so `Peak RSS` is attributable per model. `ru_maxrss` is a monotonic per-process high-water mark, so the previous single-process matrix reported the largest model seen so far for every model after the first — with all six German models in one process every one of them read 7559 MB, against the 2.5–9.4 GB spread the isolated runs show. `--in-process` restores the old behaviour.
- NeMo windows are cut near the quietest frame around each boundary instead of on a fixed 30 s grid, padded with silence, and re-split when a window decodes to an empty string. Canary was losing whole utterances to abrupt window edges — WER 37.6 % before, 20.3 % after.
- Sortformer speaker labels are normalised from NeMo's `speaker_0` to the same `spk0` form the clustering backend emits, so the two backends' output can be compared directly.

## [0.1.0] - 2026-04-30

### Added

- Side-by-side ASR benchmark harness for real recorded session audio with mic and system channels.
- Initial model matrix for Parakeet, Canary, and Whisper Large-v3.
- Run output directories with per-model transcripts, metrics, and machine-readable summary data.
- Comparison report generation with performance tables, transcript previews, and hallucination heuristics.
- GitHub Actions secret scan workflow and local guardrail compatibility.

### Changed

- Routed Canary through German language hints and generalized NeMo transcription keyword arguments.
- Swapped NeMo Parakeet execution to live transcript handling and chunked NeMo inference.

### Fixed

- Added ffmpeg fallback for CAF audio files that libsndfile cannot parse.
- Skipped private hosted secret scan jobs by default unless explicitly enabled.

[Unreleased]: https://github.com/trsdn/asr-bench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/trsdn/asr-bench/releases/tag/v0.1.0