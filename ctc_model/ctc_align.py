import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchaudio


@dataclass
class TextTokenizationResult:
    """
    Text tokenization result.

    ctc_token_ids:
        Target sequence fed directly into CTC forced alignment, without offset or separators.
    shifted_token_ids:
        Text token sequence after adding the multimodal offset, without separators.
    shifted_token_ids_with_separators:
        Sequence after adding the multimodal offset and inserting separator tokens at segment boundaries.
    segment_lengths:
        Number of tokens produced for each text segment.
        If spaces are also treated as separators, one segment usually corresponds to one word-level unit sequence.
    separators:
        Separator after each text segment. If the final segment has no trailing separator, it does not appear in this list.
    separator_token_ids:
        Separator token IDs aligned one-to-one with `separators`.
    """

    ctc_token_ids: List[int]
    shifted_token_ids: List[int]
    shifted_token_ids_with_separators: List[int]
    segment_lengths: List[int]
    separators: List[str]
    separator_token_ids: List[int]


@dataclass
class AlignmentPathResult:
    """
    Post-processed CTC path result.

    raw_path:
        Raw path returned by `torchaudio.forced_align`, keeping blanks.
    filled_path:
        Path after replacing each blank with the first following non-blank, and trailing blanks with the last non-blank.
    token_path:
        Path collapsed at the text-element level. Identical adjacent labels separated by blanks are still treated as separate elements.
    token_lengths:
        Duration of each element in `token_path`.
    """

    raw_path: List[int]
    filled_path: List[int]
    token_path: List[int]
    token_lengths: List[int]


@dataclass
class TextSpeechTokenAlignment:
    """
    Alignment result between one text token and discrete speech tokens.

    text_index:
        Linear index of the current text token in the full text-token sequence.
    ctc_token_id:
        Original tokenizer ID of the current text token, used to match the CTC alignment path.
    shifted_token_id:
        ID of the current text token after adding the multimodal offset, used to build training/inference sequences.
    ctc_length:
        Number of frames occupied by the current text token in the CTC alignment path.
    speech_start:
        Inclusive start index of the speech-token interval aligned to the current text token.
    speech_end:
        Exclusive end index of the speech-token interval aligned to the current text token.
    speech_tokens:
        Speech-token subsequence sliced from `speech_token_ids[speech_start:speech_end]`.
    """

    text_index: int
    ctc_token_id: int
    shifted_token_id: int
    ctc_length: int
    speech_start: int
    speech_end: int
    speech_tokens: List[int]


@dataclass
class CTCTextSpeechAlignmentResult:
    """
    Full output of the generic CTC text-speech alignment interface.

    tokenization:
        Text segmentation and tokenization result.
    path:
        Raw CTC forced-alignment path and its post-processed result.
    ratio:
        Scale ratio from the encoder-token timeline to the NAC speech-token timeline.
    token_alignments:
        Per-text-token text-speech alignment details.
    interleaved_sequence:
        Text/speech interleaved sequence per text token, excluding `eob`.
    block_sequences:
        Training sequence per block. The current block granularity is the current word-level unit sequence.
        Each block is structured as:
        `current word units + required separator + optional next-word units (sentence-final words do not append future words, and only the final word of the full segment appends `eos`) + current-word speech + eob`
    seqs_sequence:
        Flattened 1-D sequence of `block_sequences`, directly usable in training/inference pipelines.
    forced_alignment_scores:
        Frame-level alignment confidence returned by forced alignment, converted back from log probability to probability space.
    """

    tokenization: TextTokenizationResult
    path: AlignmentPathResult
    ratio: float
    token_alignments: List[TextSpeechTokenAlignment]
    interleaved_sequence: List[int]
    block_sequences: List[List[int]]
    seqs_sequence: List[int]
    forced_alignment_scores: List[float]


def _get_separator_mapping(config: Dict) -> Dict[str, int]:
    """
    Read the mapping from text separators to token IDs from the config.

    Args:
        config:
            Global config dictionary. `config["data"]["separators"]` must exist,
            for example `{" ": 4196, ",": 4197, ".": 4198, "!": 4199, "?": 4200}`.

    Returns:
        Dict[str, int]:
            Mapping from separator characters to separator token IDs.
    """

    separators = config.get("data", {}).get("separators")
    if not separators:
        raise KeyError("config['data']['separators'] is required.")
    return separators


def _get_text_offset(config: Dict) -> int:
    """
    Get the offset of text tokens in the multimodal sequence.

    Args:
        config:
            Global config dictionary. Prefer `config["nac"]["codebook_size"]`,
            and fall back to `config["data"]["unit_token_start"]` if it is missing.

    Returns:
        int:
            Offset applied to text tokens. After adding this value to the original tokenizer ID,
            text tokens can share one ID space with speech tokens and separator tokens.
    """

    if "nac" in config and "codebook_size" in config["nac"]:
        return int(config["nac"]["codebook_size"])
    if "data" in config and "unit_token_start" in config["data"]:
        return int(config["data"]["unit_token_start"])
    raise KeyError("Either config['nac']['codebook_size'] or config['data']['unit_token_start'] is required.")


def _get_ratio(config: Dict) -> float:
    """
    Compute the ratio from the encoder timeline to the NAC speech-token timeline.

    Args:
        config:
            Global config dictionary. It must contain both
            `config["encoder"]["tokens_per_second"]`
            and `config["nac"]["tokens_per_second"]`.

    Returns:
        float:
            Ratio `nac_tokens_per_second / encoder_tokens_per_second`.
            This ratio is later used to map CTC alignment lengths to speech-token spans.
    """

    encoder_tps = config.get("encoder", {}).get("tokens_per_second")
    nac_tps = config.get("nac", {}).get("tokens_per_second")
    if encoder_tps is None or nac_tps is None:
        raise KeyError("config['encoder']['tokens_per_second'] and config['nac']['tokens_per_second'] are required.")
    if encoder_tps <= 0 or nac_tps <= 0:
        raise ValueError("tokens_per_second must be positive.")
    return float(nac_tps) / float(encoder_tps)


def split_text_by_config_separators(text: str, config: Dict) -> Tuple[List[str], List[str]]:
    """
    Split text by the separators defined in the config.

    Args:
        text:
            Raw text string, which may include spaces, commas, periods, question marks, exclamation marks, and other separators.
        config:
            Global config dictionary. `config["data"]["separators"]` must define the valid separator set.

    Returns:
        Tuple[List[str], List[str]]:
            The first return value, `segments`, is the list of text segments with separators removed; empty segments are skipped.
            The second return value, `separators`, is the list of separators that follow each segment.
            If the last segment has no trailing separator, `separators` is shorter than `segments` by 1.
    """

    separator_mapping = _get_separator_mapping(config)
    separator_chars = list(separator_mapping.keys())
    if not separator_chars:
        return [text.strip()] if text.strip() else [], []

    pattern = "(" + "|".join(re.escape(char) for char in separator_chars) + ")"
    parts = re.split(pattern, text)

    segments: List[str] = []
    separators: List[str] = []

    for part in parts:
        if part == "" or part is None:
            continue
        if part in separator_mapping:
            if segments:
                separators.append(part)
            continue
        normalized = " ".join(part.split()).strip()
        if normalized:
            segments.append(normalized)

    return segments, separators


def encode_text_with_separators(text: str, tokenizer, config: Dict) -> TextTokenizationResult:
    """
    Split by separators first, then encode each segment independently.

    Two token sequences are kept here at the same time:
    - ctc_token_ids: original tokenizer IDs aligned directly with CTC logits
    - shifted_token_ids / shifted_token_ids_with_separators: offset token IDs used for later multimodal interleaving

    Args:
        text:
            Raw text string.
        tokenizer:
            Text tokenizer. It must provide an `encode(str) -> List[int]` interface.
            `encode` is called once for each segment.
        config:
            Global config dictionary used to read separators and the text offset.

    Returns:
        TextTokenizationResult:
            Contains the CTC alignment target sequence, the offset text sequence,
            the separator-aware text sequence, segment lengths, and separator metadata.
    """

    segments, separators = split_text_by_config_separators(text, config)
    if not segments:
        raise ValueError("Text becomes empty after separator split.")

    separator_mapping = _get_separator_mapping(config)
    offset = _get_text_offset(config)

    ctc_token_ids: List[int] = []
    shifted_token_ids: List[int] = []
    shifted_token_ids_with_separators: List[int] = []
    segment_lengths: List[int] = []
    separator_token_ids: List[int] = []

    for seg_idx, segment in enumerate(segments):
        encoded = tokenizer.encode(segment)
        if not isinstance(encoded, list):
            encoded = list(encoded)
        encoded = [int(token_id) for token_id in encoded]

        segment_lengths.append(len(encoded))
        ctc_token_ids.extend(encoded)

        shifted_segment = [token_id + offset for token_id in encoded]
        shifted_token_ids.extend(shifted_segment)
        shifted_token_ids_with_separators.extend(shifted_segment)

        if seg_idx < len(separators):
            separator = separators[seg_idx]
            separator_token_id = int(separator_mapping[separator])
            separator_token_ids.append(separator_token_id)
            shifted_token_ids_with_separators.append(separator_token_id)

    return TextTokenizationResult(
        ctc_token_ids=ctc_token_ids,
        shifted_token_ids=shifted_token_ids,
        shifted_token_ids_with_separators=shifted_token_ids_with_separators,
        segment_lengths=segment_lengths,
        separators=separators,
        separator_token_ids=separator_token_ids,
    )


def forced_align_ctc_logits(
    logits: torch.Tensor,
    target_tokens: Sequence[int],
    blank_id: int = 0,
) -> Tuple[List[int], List[float], torch.Tensor]:
    """
    Run forced alignment on CTC logits.

    `logits` supports two shapes:
    - [T, V]
    - [B, T, V], where this interface currently uses only the first sample in the batch

    Args:
        logits:
            CTC model output logits with shape `[T, V]` or `[1, T, V]`.
        target_tokens:
            Target text-token sequence without blank.
        blank_id:
            CTC blank class ID, default is 0.

    Returns:
        Tuple[List[int], List[float], torch.Tensor]:
            The first return value is the optimal forced-alignment path, whose length equals the number of time steps `T`.
            The second return value is the per-time-step alignment score, converted back from log probability to probability space.
            The third return value is the `log_softmax` log-probability tensor with the same shape convention as the input logits.
    """

    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if logits.dim() != 3:
        raise ValueError(f"logits must be 2-D or 3-D, but got shape {tuple(logits.shape)}")
    if logits.size(0) != 1:
        raise ValueError("This interface currently expects a single sample. Please pass logits with batch size 1.")

    if len(target_tokens) == 0:
        raise ValueError("target_tokens must not be empty.")

    targets = torch.tensor(target_tokens, dtype=torch.long, device=logits.device).unsqueeze(0)
    log_probs = torch.log_softmax(logits, dim=-1)
    alignments, scores = torchaudio.functional.forced_align(log_probs, targets, blank=blank_id)
    return (
        alignments[0].detach().cpu().tolist(),
        scores[0].exp().detach().cpu().tolist(),
        log_probs,
    )


def replace_blanks_in_path(path: Sequence[int], blank_id: int = 0) -> List[int]:
    """
    Replace each blank with the first following non-blank token.

    If the path ends with blanks, replace trailing blanks with the last non-blank token.

    Args:
        path:
            Raw CTC path sequence, usually the output of forced alignment.
        blank_id:
            Blank class ID, default is 0.

    Returns:
        List[int]:
            Path sequence after filling all blanks.
            Returns an empty list if the input is empty or the entire path is blank.
    """

    path_list = [int(token_id) for token_id in path]
    if not path_list:
        return []

    last_non_blank: Optional[int] = None
    for token_id in reversed(path_list):
        if token_id != blank_id:
            last_non_blank = token_id
            break

    if last_non_blank is None:
        return []

    filled_path = list(path_list)
    n = len(filled_path)
    i = 0
    while i < n:
        if filled_path[i] != blank_id:
            i += 1
            continue

        j = i
        while j < n and filled_path[j] == blank_id:
            j += 1

        replacement = filled_path[j] if j < n else last_non_blank
        for k in range(i, j):
            filled_path[k] = replacement
        i = j

    return filled_path


def collapse_path_to_tokens(path: Sequence[int], blank_id: int = 0) -> Tuple[List[int], List[int]]:
    """
    Collapse the raw path to the text-element level and compute the length of each element.

    Rules:
    - Intermediate blanks are counted toward the first following non-blank element
    - Trailing blanks are counted toward the last non-blank element
    - If two identical labels are separated by blanks, they are still treated as two separate elements

    Args:
        path:
            Raw CTC path sequence, usually with length equal to the encoder time steps.
        blank_id:
            Blank class ID, default is 0.

    Returns:
        Tuple[List[int], List[int]]:
            The first return value, `token_path`, is the collapsed token sequence.
            The second return value, `token_lengths`, is the time-axis duration of each token.
    """

    path_list = [int(token_id) for token_id in path]
    if not path_list:
        return [], []

    token_path: List[int] = []
    token_lengths: List[int] = []

    trailing_blank_count = 0
    idx = len(path_list) - 1
    while idx >= 0 and path_list[idx] == blank_id:
        trailing_blank_count += 1
        idx -= 1

    i = 0
    while i < len(path_list):
        blank_count = 0
        while i < len(path_list) and path_list[i] == blank_id:
            blank_count += 1
            i += 1

        if i >= len(path_list):
            break

        current_token = path_list[i]
        repeat_count = 0
        while i < len(path_list) and path_list[i] == current_token:
            repeat_count += 1
            i += 1

        token_path.append(current_token)
        token_lengths.append(blank_count + repeat_count)

    if token_lengths:
        token_lengths[-1] += trailing_blank_count

    return token_path, token_lengths


def postprocess_alignment_path(path: Sequence[int], blank_id: int = 0) -> AlignmentPathResult:
    """
    Combine blank replacement and path collapsing.

    Args:
        path:
            Raw CTC alignment path.
        blank_id:
            Blank class ID, default is 0.

    Returns:
        AlignmentPathResult:
            Contains the raw path, the blank-filled path,
            and the collapsed token path with its lengths.
    """

    raw_path = [int(token_id) for token_id in path]
    filled_path = replace_blanks_in_path(raw_path, blank_id=blank_id)
    token_path, token_lengths = collapse_path_to_tokens(raw_path, blank_id=blank_id)

    return AlignmentPathResult(
        raw_path=raw_path,
        filled_path=filled_path,
        token_path=token_path,
        token_lengths=token_lengths,
    )


def align_text_tokens_to_speech_tokens(
    shifted_token_ids: Sequence[int],
    ctc_token_ids: Sequence[int],
    token_lengths: Sequence[int],
    speech_token_ids: Sequence[int],
    ratio: float,
) -> List[TextSpeechTokenAlignment]:
    """
    Map each text token to a contiguous span of discrete speech tokens according to `ratio`.

    Args:
        shifted_token_ids:
            Text-token sequence after adding the offset, with length equal to the number of text tokens.
        ctc_token_ids:
            Original tokenizer ID sequence aligned one-to-one with `shifted_token_ids`.
        token_lengths:
            Duration of each text token on the CTC timeline.
        speech_token_ids:
            Discrete token sequence of the target speech.
        ratio:
            Scale ratio from the encoder timeline to the speech-token timeline.

    Returns:
        List[TextSpeechTokenAlignment]:
            List of per-text-token alignment results.
            Each element contains the text token itself, its duration,
            and the corresponding start/end positions and slice in the speech-token sequence.
    """

    if len(shifted_token_ids) != len(ctc_token_ids):
        raise ValueError("shifted_token_ids and ctc_token_ids must have the same length.")
    if len(shifted_token_ids) != len(token_lengths):
        raise ValueError("The number of token lengths must match the number of text tokens.")
    if ratio <= 0:
        raise ValueError("ratio must be positive.")

    speech_token_ids = [int(token_id) for token_id in speech_token_ids]
    alignments: List[TextSpeechTokenAlignment] = []
    cumulative_length = 0
    previous_end = 0

    for index, (shifted_token_id, ctc_token_id, token_length) in enumerate(
        zip(shifted_token_ids, ctc_token_ids, token_lengths)
    ):
        if token_length < 0:
            raise ValueError("token_length must be non-negative.")

        start = previous_end
        cumulative_length += int(token_length)
        if index == len(shifted_token_ids) - 1:
            # Merge all remaining tail speech tokens into the last text token.
            end = len(speech_token_ids)
        else:
            end = int(math.floor(ratio * cumulative_length))

        start = max(0, min(start, len(speech_token_ids)))
        end = max(start, min(end, len(speech_token_ids)))

        alignment = TextSpeechTokenAlignment(
            text_index=index,
            ctc_token_id=int(ctc_token_id),
            shifted_token_id=int(shifted_token_id),
            ctc_length=int(token_length),
            speech_start=start,
            speech_end=end,
            speech_tokens=speech_token_ids[start:end],
        )
        alignments.append(alignment)
        previous_end = end
        
    # After alignment, explicitly verify that the speech spans of all text tokens cover the entire `speech_token_ids` sequence continuously
    if alignments:
        if alignments[0].speech_start != 0:
            raise ValueError(
                "Speech-token alignment does not start from the beginning of the speech sequence. "
                f"speech_start={alignments[0].speech_start}"
            )
        if alignments[-1].speech_end != len(speech_token_ids):
            raise ValueError(
                "Speech-token alignment does not fully cover the tail of the speech sequence. "
                f"speech_end={alignments[-1].speech_end}, total_speech_tokens={len(speech_token_ids)}"
            )
        for previous_alignment, current_alignment in zip(alignments[:-1], alignments[1:]):
            if previous_alignment.speech_end != current_alignment.speech_start:
                raise ValueError(
                    "Speech-token alignment contains gaps or overlaps between adjacent text tokens. "
                    f"previous_end={previous_alignment.speech_end}, current_start={current_alignment.speech_start}"
                )

    return alignments


def build_interleaved_sequence(
    token_alignments: Sequence[TextSpeechTokenAlignment],
    tokenization: TextTokenizationResult,
) -> List[int]:
    """
    Build the interleaved sequence of text tokens and speech tokens.

    Separators are inserted after the corresponding text segment but do not bind extra speech tokens.

    Args:
        token_alignments:
            Per-text-token text-speech alignment result.
        tokenization:
            Text segmentation and tokenization result.

    Returns:
        List[int]:
            Interleaved 1-D sequence.
            Its structure is `text token + one or more corresponding speech tokens`,
            with the corresponding separator token inserted after each text segment.
    """

    interleaved: List[int] = []
    alignment_index = 0

    for segment_index, segment_length in enumerate(tokenization.segment_lengths):
        for _ in range(segment_length):
            alignment = token_alignments[alignment_index]
            interleaved.append(alignment.shifted_token_id)
            interleaved.extend(alignment.speech_tokens)
            alignment_index += 1

        if segment_index < len(tokenization.separator_token_ids):
            interleaved.append(tokenization.separator_token_ids[segment_index])

    return interleaved


def _build_segment_text_units(
    tokenization: TextTokenizationResult,
) -> List[List[int]]:
    """
    Reconstruct linear text tokens into segment-level unit sequences according to `segment_lengths`.

    With space separators enabled, one segment usually corresponds to one word.

    Args:
        tokenization:
            Text segmentation and tokenization result.

    Returns:
        List[List[int]]:
            List of segment-level text-unit sequences.
            Each element contains all shifted tokens for one text segment.
    """

    segments: List[List[int]] = []
    cursor = 0
    total_token_count = len(tokenization.shifted_token_ids)

    for segment_length in tokenization.segment_lengths:
        segment_length = int(segment_length)
        if segment_length < 0:
            raise ValueError("segment_length must be non-negative.")
        next_cursor = cursor + segment_length
        if next_cursor > total_token_count:
            raise ValueError("segment_lengths exceed the number of shifted_token_ids.")
        segments.append([int(token_id) for token_id in tokenization.shifted_token_ids[cursor:next_cursor]])
        cursor = next_cursor

    if cursor != total_token_count:
        raise ValueError("segment_lengths do not cover all shifted_token_ids.")

    return segments


def _build_segment_speech_units(
    token_alignments: Sequence[TextSpeechTokenAlignment],
    tokenization: TextTokenizationResult,
) -> List[List[int]]:
    """
    Aggregate speech tokens by text segment.

    All speech tokens aligned to text tokens within the same segment are concatenated in order.

    Args:
        token_alignments:
            Per-text-token text-speech alignment result.
        tokenization:
            Text segmentation and tokenization result, used to determine how many text tokens belong to each segment.

    Returns:
        List[List[int]]:
            List of segment-level speech-token sequences.
            Each element contains all aggregated speech tokens for one text segment.
    """

    segments: List[List[int]] = []
    cursor = 0
    total_alignment_count = len(token_alignments)

    for segment_length in tokenization.segment_lengths:
        segment_length = int(segment_length)
        if segment_length < 0:
            raise ValueError("segment_length must be non-negative.")
        next_cursor = cursor + segment_length
        if next_cursor > total_alignment_count:
            raise ValueError("segment_lengths exceed the number of token_alignments.")

        speech_tokens: List[int] = []
        for alignment in token_alignments[cursor:next_cursor]:
            speech_tokens.extend(int(token_id) for token_id in alignment.speech_tokens)
        segments.append(speech_tokens)
        cursor = next_cursor

    if cursor != total_alignment_count:
        raise ValueError("segment_lengths do not cover all token_alignments.")

    return segments


def _build_segment_separator_metadata(
    tokenization: TextTokenizationResult,
) -> Tuple[List[Optional[str]], List[Optional[int]]]:
    """
    Build segment-level separator metadata.

    Args:
        tokenization:
            Text segmentation and tokenization result.

    Returns:
        Tuple[List[Optional[str]], List[Optional[int]]]:
            The first return value is the separator character after each segment;
            the second return value is the corresponding separator token ID.
            If a segment has no trailing separator, the corresponding position is `None`.
    """

    segment_count = len(tokenization.segment_lengths)
    separator_texts: List[Optional[str]] = [None] * segment_count
    separator_token_ids: List[Optional[int]] = [None] * segment_count

    for segment_index, separator in enumerate(tokenization.separators):
        if segment_index >= segment_count:
            break
        separator_texts[segment_index] = separator
        separator_token_ids[segment_index] = int(tokenization.separator_token_ids[segment_index])

    return separator_texts, separator_token_ids


def build_unit_speech_blocks(
    token_alignments: Sequence[TextSpeechTokenAlignment],
    tokenization: TextTokenizationResult,
    config: Dict,
) -> List[List[int]]:
    """
    Build block sequences organized by text segment.

    The current block granularity is the unit sequence of the current word or text segment, with the following output structure:
    `current segment units + separator + optional next-segment units (sentence-final words do not append future words, and only the last word of the full segment appends eos) + current-segment speech + eob`

    Rule summary:
    - If there is a following segment and the separator is a space or comma, append the next segment’s unit sequence;
    - If there is a following segment and the separator is a sentence-ending punctuation mark `. ! ?`, do not append future segments and do not append `eos`;
    - Only the final segment of the whole utterance appends `eos` as the text-end marker.

    Args:
        token_alignments:
            Per-text-token text-speech alignment result.
        tokenization:
            Text segmentation and tokenization result.
        config:
            Global config dictionary. It must provide `data.eos`, `data.eob`, and `data.separators`.

    Returns:
        List[List[int]]:
            List of segment-level blocks.
            Each block corresponds to one current segment and can be further flattened into a 1-D sequence for training/inference.
    """

    if len(token_alignments) != len(tokenization.shifted_token_ids):
        raise ValueError("The number of token alignments must match the number of shifted token ids.")

    eos_token = int(config["data"]["eos"])
    eob_token = int(config["data"]["eob"])
    sentence_end_separators = {".", "!", "?"}
    segment_units = _build_segment_text_units(tokenization)
    segment_speech_units = _build_segment_speech_units(token_alignments, tokenization)
    separator_texts, separator_token_ids = _build_segment_separator_metadata(tokenization)

    blocks: List[List[int]] = []
    for index, current_units in enumerate(segment_units):
        if not current_units:
            continue

        separator_text = separator_texts[index]
        separator_token_id = separator_token_ids[index]
        has_next_segment = index + 1 < len(segment_units)

        if has_next_segment and separator_token_id is None:
            raise ValueError(
                "Missing mandatory separator between adjacent text segments. "
                f"segment_index={index}, total_segments={len(segment_units)}"
            )

        block = list(current_units)
        if separator_token_id is not None:
            block.append(int(separator_token_id))

        if not has_next_segment:
            block.append(eos_token)
        elif separator_text in sentence_end_separators:
            pass
        else:
            block.extend(segment_units[index + 1])

        block.extend(segment_speech_units[index])
        block.append(eob_token)
        blocks.append(block)

    return blocks


def flatten_unit_speech_blocks(blocks: Sequence[Sequence[int]]) -> List[int]:
    """
    Flatten a list of blocks into a 1-D sequence.

    Args:
        blocks:
            2-D block sequence, typically returned by `build_unit_speech_blocks`.

    Returns:
        List[int]:
            1-D token sequence obtained by concatenating all blocks in order.
    """

    flattened: List[int] = []
    for block in blocks:
        flattened.extend(int(token_id) for token_id in block)
    return flattened


def align_text_logits_and_speech_tokens(
    text: str,
    logits: torch.Tensor,
    tokenizer,
    speech_token_ids: Sequence[int],
    config: Dict,
    blank_id: int = 0,
) -> CTCTextSpeechAlignmentResult:
    """
    Generic CTC text-speech alignment interface.

    Args:
        text:
            Raw text string.
        logits:
            Logits output by the CTC model.
        tokenizer:
            Must provide an `encode(str) -> List[int]` interface.
        speech_token_ids:
            Discrete speech-token sequence corresponding to the text.
        config:
            Dictionary-style config.
        blank_id:
            CTC blank ID, default is 0.

    Returns:
        CTCTextSpeechAlignmentResult:
            Complete text-speech alignment result, including:
            text tokenization result, post-processed CTC path result, per-token alignment result,
            the interleaved sequence, segment-organized block sequences, flattened `seqs_sequence`,
            and frame-level forced-alignment scores.
    """

    tokenization = encode_text_with_separators(text, tokenizer, config)
    aligned_path, scores, _ = forced_align_ctc_logits(
        logits=logits,
        target_tokens=tokenization.ctc_token_ids,
        blank_id=blank_id,
    )
    path_result = postprocess_alignment_path(aligned_path, blank_id=blank_id)

    if path_result.token_path != tokenization.ctc_token_ids:
        raise ValueError(
            "Postprocessed alignment path does not match tokenizer output. "
            f"path={path_result.token_path}, target={tokenization.ctc_token_ids}"
        )

    ratio = _get_ratio(config)
    token_alignments = align_text_tokens_to_speech_tokens(
        shifted_token_ids=tokenization.shifted_token_ids,
        ctc_token_ids=tokenization.ctc_token_ids,
        token_lengths=path_result.token_lengths,
        speech_token_ids=speech_token_ids,
        ratio=ratio,
    )
    interleaved_sequence = build_interleaved_sequence(token_alignments, tokenization)
    block_sequences = build_unit_speech_blocks(token_alignments, tokenization, config)
    seqs_sequence = flatten_unit_speech_blocks(block_sequences)

    return CTCTextSpeechAlignmentResult(
        tokenization=tokenization,
        path=path_result,
        ratio=ratio,
        token_alignments=token_alignments,
        interleaved_sequence=interleaved_sequence,
        block_sequences=block_sequences,
        seqs_sequence=seqs_sequence,
        forced_alignment_scores=scores,
    )


class _DummyTokenizer:
    """
    Minimal tokenizer used by module-local self-tests.
    """

    def __init__(self, mapping: Dict[str, int]):
        self.mapping = mapping

    def encode(self, text: str) -> List[int]:
        """
        Map test text to discrete IDs character by character.

        Args:
            text:
                Input text string. Spaces are removed first, then each character is looked up one by one.

        Returns:
            List[int]:
                Corresponding sequence of character IDs.
        """

        normalized = text.replace(" ", "")
        return [self.mapping[ch] for ch in normalized]


def _test_split_and_encode_text():
    """
    Minimal self-test for text splitting and encoding logic.

    Returns:
        None:
            Validated only through `assert` statements and returns no value.
    """

    config = {
        "nac": {"codebook_size": 4096},
        "data": {"separators": {" ": 4196, ",": 4197, ".": 4198, "!": 4199, "?": 4200}},
    }
    tokenizer = _DummyTokenizer({"a": 1, "b": 2, "c": 3})
    result = encode_text_with_separators("ab c.", tokenizer, config)

    assert result.ctc_token_ids == [1, 2, 3]
    assert result.shifted_token_ids == [4097, 4098, 4099]
    assert result.shifted_token_ids_with_separators == [4097, 4098, 4196, 4099, 4198]
    assert result.segment_lengths == [2, 1]
    assert result.separator_token_ids == [4196, 4198]


def _test_postprocess_alignment_path():
    """
    Minimal self-test for post-processing alignment paths.

    Returns:
        None:
            Validated only through `assert` statements and returns no value.
    """

    path = [0, 1, 1, 0, 2, 0, 2, 2, 0]
    result = postprocess_alignment_path(path, blank_id=0)

    assert result.filled_path == [1, 1, 1, 2, 2, 2, 2, 2, 2]
    assert result.token_path == [1, 2, 2]
    assert result.token_lengths == [3, 2, 4]


def _test_align_text_tokens_to_speech_tokens():
    """
    Minimal self-test for mapping text tokens to speech tokens.

    Returns:
        None:
            Validated only through `assert` statements and returns no value.
    """

    alignments = align_text_tokens_to_speech_tokens(
        shifted_token_ids=[4097, 4098, 4099],
        ctc_token_ids=[1, 2, 3],
        token_lengths=[2, 2, 2],
        speech_token_ids=list(range(12)),
        ratio=2.0,
    )

    assert [item.speech_tokens for item in alignments] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]


def _test_end_to_end_alignment():
    """
    Minimal end-to-end self-test from text encoding to block construction.

    Returns:
        None:
            Validated only through `assert` statements and returns no value.
    """

    config = {
        "encoder": {"tokens_per_second": 2},
        "nac": {"codebook_size": 4096, "tokens_per_second": 4},
        "data": {
            "separators": {" ": 4196, ",": 4197, ".": 4198, "!": 4199, "?": 4200},
            "eob": 4201,
            "eos": 4202,
        },
    }
    tokenizer = _DummyTokenizer({"a": 1, "b": 2, "c": 3})

    logits = torch.full((1, 6, 4), -8.0)
    logits[0, 0, 1] = 8.0
    logits[0, 1, 1] = 8.0
    logits[0, 2, 0] = 8.0
    logits[0, 3, 2] = 8.0
    logits[0, 4, 3] = 8.0
    logits[0, 5, 0] = 8.0

    result = align_text_logits_and_speech_tokens(
        text="ab c.",
        logits=logits,
        tokenizer=tokenizer,
        speech_token_ids=list(range(10)),
        config=config,
        blank_id=0,
    )

    assert result.tokenization.ctc_token_ids == [1, 2, 3]
    assert result.path.token_path == [1, 2, 3]
    assert sum(result.path.token_lengths) == logits.size(1)
    assert result.tokenization.shifted_token_ids_with_separators == [4097, 4098, 4196, 4099, 4198]
    assert [item.shifted_token_id for item in result.token_alignments] == [4097, 4098, 4099]
    assert result.interleaved_sequence[0] == 4097
    assert result.interleaved_sequence.count(4196) == 1
    assert 4196 in result.interleaved_sequence
    assert result.block_sequences == [
        [4097, 4098, 4196, 4099, 0, 1, 2, 3, 4, 5, 4201],
        [4099, 4198, 4202, 6, 7, 8, 9, 4201],
    ]
    assert result.seqs_sequence == [
        4097, 4098, 4196, 4099, 0, 1, 2, 3, 4, 5, 4201,
        4099, 4198, 4202, 6, 7, 8, 9, 4201,
    ]

    previous_end = 0
    for alignment in result.token_alignments:
        assert alignment.speech_start == previous_end
        assert alignment.speech_end >= alignment.speech_start
        previous_end = alignment.speech_end


def _test_mid_sentence_sentence_end_separator_has_no_future_or_eos():
    """
    Verify that when `. ! ?` appears mid-sentence, the current block keeps only the current segment and separator,
    does not append future segments or `eos`, and only the final segment of the full utterance appends `eos`.

    Returns:
        None:
            Validated only with `assert` statements and returns no value.
    """

    config = {
        "encoder": {"tokens_per_second": 2},
        "nac": {"codebook_size": 4096, "tokens_per_second": 4},
        "data": {
            "separators": {" ": 4196, ",": 4197, ".": 4198, "!": 4199, "?": 4200},
            "eob": 4201,
            "eos": 4202,
        },
    }

    tokenization = TextTokenizationResult(
        ctc_token_ids=[1, 2, 3, 4],
        shifted_token_ids=[4097, 4098, 4099, 4100],
        shifted_token_ids_with_separators=[4097, 4098, 4198, 4099, 4100, 4198],
        segment_lengths=[2, 2],
        separators=[".", "."],
        separator_token_ids=[4198, 4198],
    )
    token_alignments = [
        TextSpeechTokenAlignment(1, 4097, 2, 0, 2, [0, 1]),
        TextSpeechTokenAlignment(2, 4098, 2, 2, 4, [2, 3]),
        TextSpeechTokenAlignment(3, 4099, 2, 4, 6, [4, 5]),
        TextSpeechTokenAlignment(4, 4100, 2, 6, 8, [6, 7]),
    ]

    blocks = build_unit_speech_blocks(token_alignments, tokenization, config)
    assert blocks == [
        [4097, 4098, 4198, 0, 1, 2, 3, 4201],
        [4099, 4100, 4198, 4202, 4, 5, 6, 7, 4201],
    ]


def run_module_tests():
    """
    Run all built-in minimal self-tests for this module.

    Returns:
        None:
            Execute all `_test_*` functions in order and return no value.
        """

    _test_split_and_encode_text()
    _test_postprocess_alignment_path()
    _test_align_text_tokens_to_speech_tokens()
    _test_end_to_end_alignment()
    _test_mid_sentence_sentence_end_separator_has_no_future_or_eos()
    print("ctc_align.py tests passed")


if __name__ == "__main__":
    run_module_tests()
