# CTC Alignment Notes: Punctuation Handling and Block Construction

This document describes the text-speech alignment stage in `ctc_tts`:
how punctuation is treated, and how each aligned block is built.

## 1. Scope

- Core implementation: `ctc_model/ctc_align.py`
- Drivers: `local/VoiceAssistant/align_data.py` (VoiceAssistant-400K), `local/data_prep_multi_proc.py` (LibriSpeech)
- Per-sample inputs: a WAV file + its text. For VoiceAssistant the alignment text is
  `item["units_label"]` (the phoneme/unit string); the tokenizer then encodes it.
- Per-sample outputs (written into `aligned_{split}.json`):
  `units` (space-joined phoneme symbols with punctuation), `seqs` (flattened block token ids),
  and optionally a `.TextGrid` when `--write_textgrid` is enabled.

The entry point is `align_text_logits_and_speech_tokens(...)`, which chains:

```
text ──encode_text_with_separators──> TextTokenizationResult
      ──forced_align_ctc_logits────> raw CTC path
      ──postprocess_alignment_path─> token_path / token_lengths
      ──align_text_tokens_to_speech_tokens──> per-token speech spans
      ──build_interleaved_sequence / build_unit_speech_blocks──> sequences
```

## 2. Config values used by alignment

Taken from `config/monophone/single-speaker/o21_single_speaker_ctc_tts_l.json`:

| Key | Value | Meaning |
| --- | --- | --- |
| `encoder.tokens_per_second` | `25` | CTC encoder frame rate (40 ms per frame) |
| `nac.tokens_per_second` | `75` | WavTokenizer speech-token rate (13.33 ms per token) |
| `nac.codebook_size` | `4096` | Speech codebook size; also the text-token offset |
| `data.separators` | `{" ":4196, ",":4197, ".":4198, "!":4199, "?":4200}` | Separator chars -> token ids |
| `data.unit_token_start` / `_end` | `4096` / `4195` | Text-unit (phoneme) token id range |
| `data.separator_token_start` / `_end` | `4196` / `4200` | Separator token id range |
| `data.eob` | `4201` | End-of-block token |
| `data.pad` | `4202` | Padding token |
| `data.eos` | `4203` | End-of-sentence token |
| `data.block_type` | `"biword"` | Block granularity: current word (+next word lookahead) |

Derived values:

- Text offset `_get_text_offset` = `nac.codebook_size` = **4096** (fallback `data.unit_token_start`).
  A tokenizer id `t` becomes the "shifted" id `t + 4096` so text, speech and separator tokens
  share one id space.
- Timeline `ratio` `_get_ratio` = `nac.tokens_per_second / encoder.tokens_per_second`
  = `75 / 25` = **3.0** (one encoder frame = three speech tokens).

## 3. How punctuation is handled

### 3.1 Text is split on separators

`split_text_by_config_separators(text, config)` splits the raw text using the separator set.
Because `" "` (space) is itself a separator, each resulting **segment is a word**.
It returns:

- `segments`: word strings, separators removed, inner whitespace normalized, empty parts dropped.
- `separators`: the separator character that follows each segment (only for segments that are
  actually followed by one). If the last segment has no trailing separator, `separators` is
  one entry shorter than `segments`.

Example: `"ab c."` -> `segments = ["ab", "c"]`, `separators = [" ", "."]`.

### 3.2 Encoding keeps three parallel sequences

`encode_text_with_separators` encodes each segment with `tokenizer.encode(segment)` and builds:

| Field | Content | Contains punctuation? |
| --- | --- | --- |
| `ctc_token_ids` | concat of encoded segments (original tokenizer ids) | **No** |
| `shifted_token_ids` | `ctc_token_ids` + offset 4096 | **No** |
| `shifted_token_ids_with_separators` | offset text tokens with separator token ids inserted after each segment | **Yes** |
| `segment_lengths` | token count per segment | - |
| `segments` | segment text per word | - |
| `separators` / `separator_token_ids` | separator char / id per segment | - |

### 3.3 Consequences for alignment

1. **CTC uses only `ctc_token_ids`.** `forced_align_ctc_logits` aligns logits against the
   separator-free target. Therefore punctuation occupies **no encoder frames** and consumes
   **no speech tokens**.
2. **No time span for punctuation.** `align_text_tokens_to_speech_tokens` partitions the speech
   tokens across text tokens (phonemes) only. Words are therefore temporally contiguous: the end
   of one word equals the start of the next.
3. **Punctuation still exists in the sequences.** It appears as separator tokens in
   `shifted_token_ids_with_separators`, in the interleaved sequence, and inside block sequences,
   always sitting between words (never bound to speech).
4. **Sentence-ending punctuation is structural.** `.`, `!` and `?` are used as decision points
   when building blocks (Section 6). `,` and space are "continuing" separators.
5. **In the `units` output** (`format_units_text`) the phoneme symbols of each segment are
   emitted in order and the segment's separator is appended right after them, then joined with
   spaces - i.e. punctuation is preserved in the text view even though it has zero duration.

## 4. CTC path post-processing

- `forced_align_ctc_logits` runs `torchaudio.functional.forced_align`, returning the raw
  per-frame path (length = encoder frames `T`) and per-frame scores, `blank_id = 0`.
- `postprocess_alignment_path` produces:
  - `raw_path`: the raw path (blanks kept).
  - `filled_path`: every blank replaced by the first following non-blank; trailing blanks
    replaced by the last non-blank.
  - `token_path` + `token_lengths`: collapse at the text-element level. Blanks before an element
    are counted into it; trailing blanks are added to the last element; **two identical labels
    separated by a blank remain two separate elements**.
- Consistency check: `token_path` must equal `ctc_token_ids`, otherwise a `ValueError` is raised.

## 5. Mapping text tokens to speech-token spans

`align_text_tokens_to_speech_tokens` walks the text tokens in order:

```
start = previous_end
cumulative_length += token_length[i]
end   = len(speech_tokens)              if i is the last token      # absorb the whole tail
      = floor(ratio * cumulative_length) otherwise
start, end = clamp to [0, len(speech_tokens)]
```

- `ratio = 3.0` for this config, so each encoder frame of duration is worth three speech tokens.
- The last text token absorbs **all remaining tail speech tokens**, guaranteeing full coverage.
- The function asserts that spans are contiguous, start at `0`, and end at `len(speech_tokens)`
  (no gaps, no overlaps, no leftover speech).

Since the span boundaries live on the speech-token grid (75 tokens/s = 13.33 ms), phoneme
boundaries are effectively quantized to 13.33 ms, even though the CTC alignment itself is
computed on the 40 ms encoder grid.

## 6. Block construction (`build_unit_speech_blocks`)

Block granularity = the **current word** (`data.block_type = "biword"`).

Each block has the form:

```
current word units + its separator + [optional next word units] + current word speech tokens + eob
```

Algorithm (per text segment / word):

1. Start the block with the current word's unit (phoneme) tokens.
2. Append the word's separator token if it has one.
3. Append the "future-text" part:
   - **last word of the whole utterance** -> append `eos`;
   - else if the separator is sentence-ending (`. ! ?`) -> append **nothing**
     (no next word, no `eos`);
   - else (space or `,`) -> append the **next word's units** (one-word lookahead).
4. Append the current word's speech tokens (aggregated from its phoneme tokens).
5. Append `eob`.

Notes:

- Punctuation is part of the block text prefix but never brings speech tokens with it.
- A non-final word without a separator raises
  `"Missing mandatory separator between adjacent text segments"`.
- Because a normal block contains the current word plus the next word, the mode is named
  **"biword"**.

Flattening: `flatten_unit_speech_blocks` concatenates all blocks into the 1-D
`seqs_sequence`, which `align_data.py` stores as the `seqs` field
(`" ".join(map(str, result.seqs_sequence))`).

Both the alignment-side builder and the inference-side `generate_unit_list`
(`src/utils.py`) implement the same "current word + separator + optional next word" layout;
`extract_unit_from_seqs` (`src/utils.py`) recovers the `current word units + separator` prefix
from each block at inference time.

## 7. Interleaved sequence (per-token view)

`build_interleaved_sequence` produces, for each segment, every text token's shifted id followed
by its aligned speech tokens; after each segment the separator token is inserted. This is a
per-token view used for inspection and is distinct from the block/`seqs` training sequence.
Punctuation is inserted here but still binds no speech.

## 8. TextGrid output

`build_textgrid` renders Praat TextGrid (long format) with two `IntervalTier`s:

- `phones`: one interval per text token. `xmin = speech_start / 75`, `xmax = speech_end / 75`,
  text = `vocab[ctc_token_id]` (the phoneme symbol).
- `words`: one interval per segment. Start = first token's `speech_start`, end = last token's
  `speech_end`, text = the segment string.

Timeline: `seconds = speech_token_index / config["nac"]["tokens_per_second"]` (75).

`xmax` = `max(audio_duration, aligned_duration)` where `aligned_duration = last.speech_end / 75`;
when the real audio is longer, a single empty trailing interval is appended to **both** tiers so
they keep covering `[xmin, xmax]`.

Punctuation never forms its own interval. Because separators carry no speech tokens, adjacent
word intervals are directly contiguous, and the `words` tier has no gap where `.`/`,`/space occur.

## 9. Worked example

Text `"ab c."` encoded as `"a"->4097, "b"->4098, "c"->4099` (offset 4096),
separators `" "=4196`, `"."=4198`, `eob=4201`, `eos=4203`:

- `segments = ["ab", "c"]`, `separators = [" ", "."]`, `segment_lengths = [2, 1]`.
- Block 0 (word `"ab"`, separator space, next word `"c"` exists and is not sentence-final):
  `[4097, 4098, 4196, 4099, <speech of "ab">, 4201]`
- Block 1 (word `"c"`, separator `"."`, last word of utterance):
  `[4099, 4198, 4203, <speech of "c">, 4201]`

`seqs` = block 0 followed by block 1 (flattened).
In the TextGrid, `words` has exactly two intervals (`ab`, `c`) that touch each other, and
`phones` has one interval per phoneme; neither tier contains an interval for `.`.

## 10. Notes and caveats

- `separators` may be shorter than `segments` (last word without trailing punctuation) or longer
  (repeated / adjacent separators such as two spaces). Only segment indices `< len(separators)`
  receive a separator token, and the mandatory-separator error only triggers for a non-final word
  that lacks one.
- The alignment-side block builder appends `eos` to the final word's block, while the
  inference-side `generate_unit_list` (`src/utils.py`) currently has that append commented out
  (no `eos` on the last word). Confirm whether this asymmetry is intended before relying on it.
- Punctuation only affects block boundaries and the text prefix; it never consumes speech tokens
  or time.

## 11. Function map

| Function | Role |
| --- | --- |
| `_get_separator_mapping` | Read `data.separators` |
| `_get_text_offset` | Text token offset (= `nac.codebook_size`) |
| `_get_ratio` | `nac.tps / encoder.tps` |
| `split_text_by_config_separators` | Split text into segments + separators |
| `encode_text_with_separators` | Build the three token sequences |
| `forced_align_ctc_logits` | CTC forced alignment |
| `replace_blanks_in_path` / `collapse_path_to_tokens` / `postprocess_alignment_path` | Path post-processing |
| `align_text_tokens_to_speech_tokens` | Text token -> speech span mapping |
| `build_interleaved_sequence` | Per-token interleaved sequence |
| `build_unit_speech_blocks` / `flatten_unit_speech_blocks` | Block construction and flattening |
| `build_textgrid` | Praat TextGrid rendering |
| `align_text_logits_and_speech_tokens` | Top-level entry point |
