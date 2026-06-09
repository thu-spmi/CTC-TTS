import torch
from torch import nn
import json
import re
from typing import *

import torchaudio

from .extract_fbank_feature_with_kaldi import extract_fbank_with_kaldi
from .phontisaurus_g2p import g2p_phonetize, g2p_phonetize_batch
from ctc_model.encoder_loader import load_pretrained_encoder

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.syllabify import syllabify_tokens

# old alignment implementation. 

class LexiconTokenizer:
    def __init__(self, phoneme2token=None):
        """Initialize LexiconTokenizer. By default it uses the phoneme set from whistle1.0."""
        if phoneme2token is None: # Use the phoneme inventory from whistle1.0 by default
            self.token2phoneme = {0: '<s>', 1: '<unk>', 2: 'j', 3: 's', 4: 'ɡ', 5: 'c', 6: 'm', 7: 'ɥ', 
                8: 'ʏ', 9: 'ɒ', 10: 'ɽ', 11: 'ʌ', 12: 'β', 13: 'v', 14: 'h', 15: 'ɜ', 16: 'ɪ', 
                17: 'ɧ', 18: 'ɲ', 19: 'k', 20: 'θ', 21: 'ʐ', 22: 'd', 23: 'a', 24: 'ə', 25: 'n', 
                26: 'e', 27: 'ʉ', 28: 'd͡z', 29: 'ɑ', 30: 'p', 31: 'z', 32: 'ɣ', 33: 'u', 34: 'œ', 
                35: 'ʋ', 36: 'ɫ', 37: 'ɟ', 38: 'ɨ', 39: 't͡ʃ', 40: 'i', 41: 'f', 42: 'y', 43: 'ʁ', 
                44: 'ɔ', 45: 'o', 46: 'ʎ', 47: 'æ', 48: 'χ', 49: 'd͡ʒ', 50: 'ʂ', 51: 'r', 52: 'ʔ',
                53: 'x', 54: 'ɯ', 55: 't͡s', 56: 'ɛ', 57: 'b', 58: 'ð', 59: 'ø', 60: 't', 61: 'ɕ', 
                62: 'ʑ', 63: 'ɹ', 64: 'ʊ', 65: 'w', 66: 'ŋ', 67: 'ɱ', 68: 'ʃ', 69: 'ɤ', 70: 'l', 
                71: 'ɾ', 72: 'ʒ', 73: 'ɵ', 74: 'q'}
            self.phoneme2token = {v: k for k, v in self.token2phoneme.items()}
        else:
            self.phoneme2token = phoneme2token
            self.token2phoneme = {v: k for k, v in self.phoneme2token.items()}
        
        self._units = {v: k for k, v in self.token2phoneme.items()} # phoneme2token
    def decode(self, indices: Union[Iterable[int], Iterable[Iterable[int]]]):
        str_phone = ""
        is_first = True
        for j in range(len(indices)):
            ind = indices[j]
            phone = list(self._units.keys())[list(self._units.values()).index(ind)]
            if is_first:
                str_phone = str_phone + phone
            else:
                str_phone = str_phone + " " + phone
            is_first = False
        return str_phone

class WhistleEncoder(nn.Module):
    def __init__(self, args, word_phoneme_mapping=None):
        super().__init__()
        if args["encoder"]["phoneme2token"] is not None:
            self.tokenizer = LexiconTokenizer(args["encoder"]["phoneme2token"])
        else:
            self.tokenizer = LexiconTokenizer()
        self.model = self.from_pretrained_model(args)
        self.codebook_size = args["nac"]["codebook_size"]
        self.word_phoneme_mapping = word_phoneme_mapping
        for param in self.model.parameters():
            param.requires_grad = args["encoder"]["finetune_encoder"]

        # Punctuation token mapping
        self.separator_tokens = {
            ',': 75,
            '.': 76,
            '!': 77,
            '?': 78,
            ' ': 79
        }
        
    def from_pretrained_model(self, args):
        return load_pretrained_encoder(args)
    
    def align(self, fbank, lens, target):

        device = next(self.model.parameters()).device

        # Ensure the inputs are placed on the correct device
        fbank = fbank.to(device)
        lens = lens.to(device)
        logits, olens = self.model(fbank, lens)
    
        targets = torch.tensor(target, device=device).unsqueeze(0)  # Add the batch dimension
        log_probs = torch.log_softmax(logits, dim=-1).to(device)
        alignments, scores = torchaudio.functional.forced_align(log_probs, targets, blank=0)
        alignments, scores = alignments[0], scores[0] # remove batch dimension for simplicity
        scores = scores.exp() # convert back to probability
        return alignments, scores, log_probs
        
    def split_text_by_separators(self, text):
        """
        Split text by separators and return both the word sequence and the corresponding separators.
        The supported separators are `,.!?` and spaces. Consecutive separators are normalized first. If punctuation is adjacent to spaces, keep only the punctuation separator.
        """
        
        # 1. Text preprocessing: append `.` if the text does not end with `.`, `!`, or `?`
        text = text.strip()
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        # 2. Normalize consecutive separators
        # Use a regular expression to match consecutive separators (punctuation and spaces)
        
        # Replace multiple consecutive spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Handle punctuation-space combinations and prefer punctuation over spaces
        # Match: space+punctuation, punctuation+space, or repeated punctuation
        def process_separators(match):
            sep_group = match.group(0)
            # Find the first non-space punctuation mark
            for char in sep_group:
                if char in ',.!?':
                    return char
            # If the group contains only spaces, return a single space
            return ' '
        
        # Normalize consecutive separators with a regular expression
        text = re.sub(r'[\s,.!?]+', process_separators, text)
        
        # 3. Split the text into words by separators and record the separators
        # Split with a regular expression while keeping separators
        pattern = r'([,.!?\s])'
        parts = re.split(pattern, text)
        
        words = []
        separators = []
        
        for i in range(len(parts)):
            part = parts[i]
            if part == '':
                continue
            if part and not re.match(r'^[,.!?\s]$', part):  # This part is not a separator
                # Further split words by spaces
                word = part.strip()
                words.append(word)
            elif part in ',.!?':
                # This is a punctuation separator
                separators.append(part)
            elif part == ' ':
                # This is a space separator
                separators.append(' ')
            else:
                print(f"Unknown separator: ---{part}---")

        # Because the text always ends with punctuation, `separators` should have the same length as `words`
        if len(separators) != len(words):
            print(f"Warning: Length mismatch - words: {len(words)}, separators: {len(separators)}")
            print(f"Words: {words}")
            print(f"Separators: {separators}")
        return words, separators

    def get_word_phonemes(self, word):
        """
        Get the phoneme sequence for a word
        Look up the word in `word_phoneme_mapping` first; if it is not found, fall back to `g2p_phonetize_batch`
        
        Args:
            word (str): Input word
            
        Returns:
            str: Phoneme string, and whether the word was converted into an empty phoneme sequence
        """
        word = word.strip(",.?!'")

        no_phoneme = False # Whether phonetisaurus generated an empty phoneme sequence. This can happen for a small number of words.

        # Try the lexicon first
        if self.word_phoneme_mapping and word in self.word_phoneme_mapping:
            phonemes = self.word_phoneme_mapping[word]
            if phonemes == '' or phonemes.strip() == '':
                no_phoneme = True
            return phonemes, no_phoneme
        
        # If the word is not in the lexicon, use G2P as a fallback
        phonemes = g2p_phonetize_batch([word])
        print(f"Warning: '{word}' not found in word_phoneme_mapping, using phonetisaurus as fallback. {word} {phonemes}")
        # Add the generated phonemes back into `word_phoneme_mapping`
        if self.word_phoneme_mapping is not None:
            self.word_phoneme_mapping[word] = phonemes
            if phonemes == '' or phonemes.strip() == '':
                no_phoneme = True

        return phonemes, no_phoneme

    def generate_phonemes(self, words, separators, tknz):
        # Assert if `words` and `separators` do not have the same length
        assert len(words) == len(separators), f"words and separators must have the same length.\nwords={words}\nseparators={separators}"
        target_phonemes=[]
        target_phonemes_with_separators=[]
        exist_null_phoneme = False
        if self.word_phoneme_mapping == None:
            for i in range(len(words)):
                phoneme_block = g2p_phonetize(words[i])
                if phoneme_block == '':
                    exist_null_phoneme = True
                # print(f"phoneme_block=\n{phoneme_block}")
                phone_tokens = []
                # print(f"phoneme_block={phoneme_block}")

                for j in phoneme_block:
                    # print(f"current_phoneme={j}")
                    phone_tokens.append(tknz._units[j])
                target_phonemes.extend(phone_tokens)
                target_phonemes_with_separators.extend(phone_tokens)
                target_phonemes_with_separators.append(self.separator_tokens[separators[i]])
        else:
            for i in range(len(words)):
                # Prefer lexicon lookup
                word = words[i]
                phoneme_block, no_phoneme = self.get_word_phonemes(word)
                if no_phoneme:
                    exist_null_phoneme = True
                phone_tokens = []
                # print(f"phoneme_block=\n{phoneme_block}")
                # print(f"phoneme_block.split()=\n{phoneme_block.split()}")
                for j in phoneme_block.split():
                    # print(f"current_phoneme={j}")
                    phone_tokens.append(tknz._units[j])
                target_phonemes.extend(phone_tokens)
                target_phonemes_with_separators.extend(phone_tokens)
                target_phonemes_with_separators.append(self.separator_tokens[separators[i]])

        return target_phonemes, target_phonemes_with_separators, exist_null_phoneme       

    def calculate_ctc_lengths(self, path, target_phonemes_with_separators, blank_token=0):
        """
        Compute the duration of each label after collapsing the CTC path.
        Length = the number of repeated labels + all blank labels between this phoneme label and the previous phoneme label
        
        Args:
            path: List[int], CTC model output path, including blanks and repeated labels.
            target_phonemes_with_separators: List[int], Target phoneme path including separators.
            blank_token: int, Blank label, default is 0
        
        Returns:
            path_uni: List[int], Collapsed path where identical neighboring labels separated by blanks remain separate
            lengths: List[int], Length of each label in `path_uni`
        """
        if not path:
            return [], []
        
        # Handle trailing zeros in `path`: if the final element is 0, replace trailing zeros with the last non-zero element
        if path and path[-1] == blank_token:
            # Find the last non-zero element
            last_non_zero = None
            for token in reversed(path):
                if token != blank_token:
                    last_non_zero = token
                    break
            
            # If a non-zero element is found, replace trailing zeros with slicing
            if last_non_zero is not None:
                # Find the start of the trailing zero span
                i = len(path) - 1
                while i >= 0 and path[i] == blank_token:
                    i -= 1
                # Replace by slicing: `path[i+1:]` contains only trailing zeros
                path[i+1:] = [last_non_zero] * (len(path) - i - 1)
        # print(f"Processed path={path}")
        path_uni = []    # Collapsed path
        lengths = []     # Length for each label
        prev_consumed_blanks = 0  # Track how many trailing blanks were already absorbed by the previous label
        
        i = 0
        while i < len(path):
            # Skip leading blank labels
            while i < len(path) and path[i] == blank_token:
                i += 1

            if i >= len(path):
                break

            # Find the next non-blank label
            current_label = path[i]
            path_uni.append(current_label)

            # Count repetitions of this label
            repeat_count = 0
            start_pos = i
            while i < len(path) and path[i] == current_label:
                repeat_count += 1
                i += 1

            # Count consecutive blanks after the current repeated label and find the next non-blank label
            trailing_blank_count = 0
            j2 = i
            while j2 < len(path) and path[j2] == blank_token:
                trailing_blank_count += 1
                j2 += 1
            next_label = path[j2] if j2 < len(path) else None

            # Count blanks between this label and the previous label
            blanks_between = 0
            if len(path_uni) == 1:
                # For the first label, count the leading blanks and subtract blanks already absorbed by the previous label (usually 0)
                blanks_between = start_pos
                blanks_between = max(blanks_between - prev_consumed_blanks, 0)
            else:
                # For non-first labels, locate the end of the previous label
                # Count blanks backward from the current label start
                j = start_pos - 1
                while j >= 0 and path[j] == blank_token:
                    blanks_between += 1
                    j -= 1
                # Subtract trailing blanks already absorbed by the previous label to avoid double counting
                blanks_between = max(blanks_between - prev_consumed_blanks, 0)

            # Compute total length using the original rule
            total_length = repeat_count + blanks_between

            # If the current phoneme in `target_phonemes_with_separators` is followed by a punctuation separator (not a space), and the repeated label in the path is followed by blanks
            # count those consecutive blanks toward the current label length and mark them as absorbed; `i` can also be advanced to `j2` to skip them
            sep_tokens = getattr(self, 'separator_tokens', {})
            sep_id_set = set(sep_tokens.values()) if isinstance(sep_tokens, dict) else set()
            punct_ids = set(v for k, v in sep_tokens.items() if k != ' ') if isinstance(sep_tokens, dict) else set()

            # Current non-separator phoneme index (0-based), aligned with the length of `path_uni`
            cur_nonsep_index = len(path_uni) - 1

            # Find the corresponding position index in `target_phonemes_with_separators`
            tps_pos = -1
            seen_nonsep = -1
            for idx, tok in enumerate(target_phonemes_with_separators):
                if tok not in sep_id_set:  # Non-separator phoneme
                    seen_nonsep += 1
                if seen_nonsep == cur_nonsep_index:
                    tps_pos = idx
                    break

            # Check whether the next position after this phoneme is a punctuation separator (not a space)
            is_punct = False
            if tps_pos != -1 and tps_pos + 1 < len(target_phonemes_with_separators):
                next_in_tps = target_phonemes_with_separators[tps_pos + 1]
                if next_in_tps in punct_ids:
                    is_punct = True

            if is_punct and trailing_blank_count > 0:
                total_length += trailing_blank_count
                prev_consumed_blanks = trailing_blank_count
                i = j2  # Consume these blanks directly, equivalent to `i += trailing_blank_count`
            else:
                prev_consumed_blanks = 0

            lengths.append(total_length)
        
        return path_uni, lengths
    
    def align_phonemes_speech(self, phonemes, phonemes_lengths, phonemes_with_separators,speech_tokens, silence_token, future_phoneme_num=1, eob=4176, eos=4178):
        """
        Align phonemes with speech tokens. Each phoneme position corresponds to three speech tokens.

        phonemes: List of phonemes
        phonemes_lengths: Length of each phoneme
        phonemes_with_separators: List of phonemes with separators. Each word-ending phoneme is followed by a separator, and separators do not appear consecutively
        speech_tokens: List of speech tokens
        silence_token: Silence token
        future_phoneme_num: Number of future phonemes appended after each phoneme
        eob: End-of-block token
        eos: End-of-sentence token
        """
        assert len(phonemes) == len(phonemes_lengths)

        # Compute the required total number of `speech_tokens`
        required_tokens = sum(3 * length for length in phonemes_lengths)

        # Adjust the length of `speech_tokens`.
        # Check whether the ratio between `speech_tokens` and phonemes is 3. Truncate when it is larger than 3, otherwise pad with `silence_token`
        if len(speech_tokens) > required_tokens:
            speech_tokens = speech_tokens[:required_tokens]
        elif len(speech_tokens) < required_tokens:
            speech_tokens += [silence_token] * (required_tokens - len(speech_tokens))
        
        assert len(speech_tokens) == required_tokens

        # Append `future_phoneme_num` future phonemes after one phoneme, then insert `length * 3` speech tokens and finally an `eob`
        # If there are not enough future phonemes, append `<eos>` as padding
        # Separators are not treated as standalone phonemes; they only mark the end of a word-level phoneme span
        p_s_path = []
        speech_idx = 0  # Current index into `speech_tokens`
        separator_count = 0 # How many separators have been seen so far
        for i, phoneme in enumerate(phonemes_with_separators):
            # Skip separators
            if phoneme > 74: 
                separator_count += 1
                continue
            # Append the current phoneme
            p_s_path.append(phoneme + self.codebook_size) # Offset by the discrete speech codebook size

            # Append `future_phoneme_num` future phonemes
            for j in range(future_phoneme_num):
                future_idx = i + 1 + j
                if future_idx < len(phonemes_with_separators):
                    # Enough future phonemes are available
                    p_s_path.append(phonemes_with_separators[future_idx] + self.codebook_size)
                else:
                    # Not enough future phonemes are available, append <eos>
                    p_s_path.append(eos)

            tokens_for_this_phoneme = 3 * phonemes_lengths[i - separator_count]
            for k in range(tokens_for_this_phoneme):
                if speech_idx < len(speech_tokens):
                    p_s_path.append(speech_tokens[speech_idx])
                    speech_idx += 1
                else:
                    # Pad with `silence_token` when out of range
                    p_s_path.append(silence_token)
                    print(f"Warning: speech_tokens is out of range, padding with silence_token")
            # Insert the end-of-block token
            p_s_path.append(eob)
        return p_s_path

    def align_words_speech(self, phonemes, phonemes_lengths, phonemes_with_separators,speech_tokens, silence_token, future_word_num=1, eob=4176, eos=4178):
        """
        Align phonemes with speech tokens. Each phoneme position corresponds to three speech tokens.

        phonemes: List of phonemes
        phonemes_lengths: Length of each phoneme
        phonemes_with_separators: List of phonemes with separators. Each word-ending phoneme is followed by a separator, and separators do not appear consecutively
        speech_tokens: List of speech tokens
        silence_token: Silence token
        future_word_num: Number of future words appended after each word
        eob: End-of-block token
        eos: End-of-sentence token
        """
        assert len(phonemes) == len(phonemes_lengths)

        # 1) Split by separators into words and compute the length of each word as the sum of its phoneme lengths
        words = []            # List[List[int]]: Phoneme ID list for each word, excluding separators
        word_lengths = []     # List[int]: Length of each word (sum of phoneme lengths)
        word_separators = []  # Separator for each word; `None` if the word has no trailing separator
        curr_word_phonemes = []
        curr_word_len_sum = 0
        p_idx = 0  # Index into `phonemes` / `phonemes_lengths` excluding separators

        for tok in phonemes_with_separators:
            if tok <= 74:
                # Non-separator: phoneme
                curr_word_phonemes.append(tok)
                curr_word_len_sum += phonemes_lengths[p_idx]
                p_idx += 1
            else:
                # Separator: terminates a word
                if curr_word_phonemes:
                    words.append(curr_word_phonemes)
                    word_lengths.append(curr_word_len_sum)
                    word_separators.append(tok + self.codebook_size)  # Record the separator for this word and add the offset
                    curr_word_phonemes = []
                    curr_word_len_sum = 0
        # Finalize the last word even if it does not end with a separator
        if curr_word_phonemes:
            words.append(curr_word_phonemes)
            word_lengths.append(curr_word_len_sum)
            word_separators.append(None)  # No trailing separator

        # Consistency check: the number of non-separator phonemes must match
        assert p_idx == len(phonemes), f"Number of non-separator phonemes does not match：p_idx={p_idx}, len(phonemes)={len(phonemes)}"

        # 2) Compute the required total number of speech tokens aligned by word length
        required_tokens = sum(3 * wl for wl in word_lengths)

        # 3) Adjust the length of `speech_tokens` by truncating or padding silence
        if len(speech_tokens) > required_tokens:
            speech_tokens = speech_tokens[:required_tokens]
        elif len(speech_tokens) < required_tokens:
            speech_tokens += [silence_token] * (required_tokens - len(speech_tokens))

        assert len(speech_tokens) == required_tokens

        # 4) Build the alignment path at the word level
        p_s_path = []
        speech_idx = 0  # Current index into speech_tokens

        for w_i, word_phs in enumerate(words):
            # All phonemes of the current word, offset by `codebook_size`
            for ph in word_phs:
                p_s_path.append(ph + self.codebook_size)

            # Keep the separator of the current word if it exists
            sep = word_separators[w_i]
            if sep is not None:
                p_s_path.append(sep)

            # Future-word context containing only phonemes, without separators
            added_words = 0
            next_w = w_i + 1
            while added_words < future_word_num:
                if next_w < len(words):
                    # Append all phonemes of the next word, offset by `codebook_size`
                    for ph in words[next_w]:
                        p_s_path.append(ph + self.codebook_size)
                    next_w += 1
                    added_words += 1
                else:
                    # If future words are insufficient, use one `eos` token as a placeholder for each missing word
                    p_s_path.append(eos)
                    added_words += 1

            # The `3 × word_length` speech tokens corresponding to this word
            tokens_for_this_word = 3 * word_lengths[w_i]
            for _ in range(tokens_for_this_word):
                if speech_idx < len(speech_tokens):
                    p_s_path.append(speech_tokens[speech_idx])
                    speech_idx += 1
                else:
                    p_s_path.append(silence_token)
                    print(f"Warning: speech_tokens is out of range, padding with silence_token")

            # Insert the end-of-block token
            p_s_path.append(eob)

        return p_s_path
    
    def align_syllables_speech(self, phonemes, phonemes_lengths, phonemes_with_separators, speech_tokens, silence_token, future_syllable_num=1, eob=4176, eos=4178):
        """
        Align syllables with speech tokens:
        - All phonemes of the current syllable (+codebook_size)
        - If this syllable is the last syllable of its word, append the word separator (+codebook_size)
        - Then append all phonemes from `future_syllable_num` future syllables (pad with `eos` if needed)
        - Append `3 × current_syllable_length` speech tokens
        - Append `eob` at the end
        """
        assert len(phonemes) == len(phonemes_lengths)

        # 1) First split into words by separators and record each word’s phoneme IDs and lengths, excluding separators
        words_ids: List[List[int]] = []
        words_lenlists: List[List[int]] = []
        word_separators: List[Optional[int]] = []

        cur_ids: List[int] = []
        cur_lens: List[int] = []
        p_idx = 0  # Index into the non-separator phoneme length list

        for tok in phonemes_with_separators:
            if tok <= 74:
                cur_ids.append(tok)
                cur_lens.append(phonemes_lengths[p_idx])
                p_idx += 1
            else:
                if cur_ids:
                    words_ids.append(cur_ids)
                    words_lenlists.append(cur_lens)
                    word_separators.append(tok + self.codebook_size)  # Add offset to the separator
                    cur_ids, cur_lens = [], []
        if cur_ids:
            words_ids.append(cur_ids)
            words_lenlists.append(cur_lens)
            word_separators.append(None)

        assert p_idx == len(phonemes), f"Number of non-separator phonemes does not match：p_idx={p_idx}, len(phonemes)={len(phonemes)}"

        # 2) Syllabify each word and expand the result into a global syllable sequence
        syllables_ids: List[List[int]] = []
        syllables_lengths: List[int] = []
        syllables_separators: List[Optional[int]] = []

        for w_idx, word_ids in enumerate(words_ids):
            word_len_list = words_lenlists[w_idx]
            # Convert phoneme IDs to strings for syllabification
            word_ph_strs = [self.tokenizer.token2phoneme[t] for t in word_ids]
            syllables_str_tokens: List[List[str]] = syllabify_tokens(word_ph_strs)

            # Slice by syllables in strict order and compute each syllable length as the sum of its phoneme lengths
            ptr = 0
            for s_idx, s_tokens in enumerate(syllables_str_tokens):
                L = len(s_tokens)
                syll_ids = word_ids[ptr:ptr+L]
                syll_len_sum = sum(word_len_list[ptr:ptr+L])
                ptr += L

                syllables_ids.append(syll_ids)
                syllables_lengths.append(syll_len_sum)
                # The last syllable of the word carries the separator; other syllables do not
                if s_idx == len(syllables_str_tokens) - 1:
                    syllables_separators.append(word_separators[w_idx])
                else:
                    syllables_separators.append(None)

            # Assert that all phonemes have been consumed
            assert ptr == len(word_ids), f"Not all phonemes of the word were consumed after syllabification：ptr={ptr}, len(word_ids)={len(word_ids)}"

        # 3) Adjust the length of `speech_tokens` by truncating or padding silence
        required_tokens = sum(3 * L for L in syllables_lengths)
        if len(speech_tokens) > required_tokens:
            speech_tokens = speech_tokens[:required_tokens]
        elif len(speech_tokens) < required_tokens:
            speech_tokens += [silence_token] * (required_tokens - len(speech_tokens))
        assert len(speech_tokens) == required_tokens

        # 4) Build the syllable-speech alignment path
        p_s_path: List[int] = []
        speech_idx = 0
        for s_i, syll_ids in enumerate(syllables_ids):
            # Phonemes of the current syllable (+codebook_size)
            for ph in syll_ids:
                p_s_path.append(ph + self.codebook_size)

            # If this is the last syllable of the word, append the separator (+codebook_size)
            sep_tok = syllables_separators[s_i]
            if sep_tok is not None:
                p_s_path.append(sep_tok)

            # Append future-syllable context (pad with `eos` if needed)
            added = 0
            next_s = s_i + 1
            while added < future_syllable_num:
                if next_s < len(syllables_ids):
                    for ph in syllables_ids[next_s]:
                        p_s_path.append(ph + self.codebook_size)
                    next_s += 1
                    added += 1
                else:
                    p_s_path.append(eos)
                    added += 1

            # Append `3 × current_syllable_length` speech tokens
            tokens_for_this_syllable = 3 * syllables_lengths[s_i]
            for _ in range(tokens_for_this_syllable):
                if speech_idx < len(speech_tokens):
                    p_s_path.append(speech_tokens[speech_idx])
                    speech_idx += 1
                else:
                    p_s_path.append(silence_token)
                    print("Warning: speech_tokens is out of range, padding with silence_token")

            # End-of-block token
            p_s_path.append(eob)

        return p_s_path
            
    
    def generate_phonemes_speech_alignment(self, audio_path, text, fbank_config, speech_tokens, silence_token, unit='',future_word_num=1, eob=4176, eos=4178):

        """
        Generate the phoneme-speech alignment path. The `unit` argument controls whether alignment is performed at the phoneme, word, or syllable level.
        audio_path: Audio path
        text: Text corresponding to the audio
        fbank_config: Fbank configuration used by whistle
        speech_tokens: Discrete speech token sequence generated by WavTokenizer
        silence_token: Silence token used by the WavTokenizer model
        unit: Alignment unit. `word` aligns by word-level phoneme groups, and `phoneme` aligns by individual phonemes.
        future_word_num: Number of future word/phoneme context units, default is 1.
        eob: End-of-block token, default is 4176.
        eos: Padding token, default is 4178.
        """

        device = next(self.model.parameters()).device

        fbank_features = extract_fbank_with_kaldi(audio_path, fbank_config)
        fbank = fbank_features.unsqueeze(0).to(device) 
        # print(f"FBank feature shape: {fbank.shape}")  # (B, T, 80)
        fbank_length = torch.full((fbank.shape[0],), fbank.shape[1], device=device) 

        # Generate the word and separator sequences
        words, separators = self.split_text_by_separators(text)

        # Generate phoneme sequences and separator-aware phoneme sequences from text chunks
        target_phonemes, target_phonemes_with_separators, exist_null_phoneme = self.generate_phonemes(words, separators, self.tokenizer)

        # Generate alignment information. The path includes blanks and repeated labels
        aligned_tokens, _, _ = self.align(fbank=fbank, lens=fbank_length, target=target_phonemes)
        aligned_tokens = aligned_tokens.tolist()

        # Compute the collapsed aligned_tokens path and the length of each label. If aligned_tokens ends with consecutive blanks, replace those trailing blanks with the last phoneme token first
        path_uni, lengths = self.calculate_ctc_lengths(aligned_tokens, target_phonemes_with_separators) 
        assert len(path_uni) == len(lengths), f"Length mismatch: path_uni has length {len(path_uni)}, while lengths has length {len(lengths)}"
        assert path_uni == target_phonemes, f"Collapsed path does not match the target phoneme sequence. Collapsed path: {path_uni}, target phoneme sequence: {target_phonemes}"
        assert sum(lengths) == len(aligned_tokens), f"Collapsed-path length does not match the original alignment-path length. Collapsed length: {sum(lengths)}, original alignment-path length: {len(aligned_tokens)}"

        if unit == 'word':
            # Align phonemes with discrete speech tokens. Each phoneme position corresponds to three speech tokens because the frame-rate ratio between WavTokenizer and whistle output labels is 3
            w_s_path = self.align_words_speech(path_uni, lengths, target_phonemes_with_separators, speech_tokens, silence_token, future_word_num, eob=eob, eos=eos)
            
            # Return the word-speech alignment path, the phoneme path, the length of each label, the separator-aware phoneme path, and whether any word produced an empty phoneme sequence
            return w_s_path, path_uni, lengths, target_phonemes_with_separators, exist_null_phoneme
        elif unit == 'phoneme':
            # Align phonemes with discrete speech tokens. Each phoneme position corresponds to three speech tokens because the frame-rate ratio between WavTokenizer and whistle output labels is 3
            p_s_path = self.align_phonemes_speech(path_uni, lengths, target_phonemes_with_separators, speech_tokens, silence_token, future_word_num, eob=eob, eos=eos)
            
            # Return the phoneme-speech alignment path, the phoneme path, the length of each label, the separator-aware phoneme path, and whether any word produced an empty phoneme sequence
            return p_s_path, path_uni, lengths, target_phonemes_with_separators, exist_null_phoneme
        elif unit == 'syllable':
            s_s_path = self.align_syllables_speech(path_uni, lengths, target_phonemes_with_separators, speech_tokens, silence_token, future_syllable_num=future_word_num, eob=eob, eos=eos)
            return s_s_path, path_uni, lengths, target_phonemes_with_separators, exist_null_phoneme
        else:
            raise ValueError(f"unit must be `word`, `phoneme`, or `syllable`, but got {unit}")


if __name__ == "__main__":
    config = json.load(open("/mnt/workspace/liuhw/streaming_tts/config/config_llmvox.json"))

    from src.utils import  load_lexicon
    print(f"Loading word_phoneme_mapping from: {config['data']['lexicon_path']}")
    word_phoneme_mapping = load_lexicon(config["data"]["lexicon_path"])
    print(f"Load word_phoneme_mapping successfully! Length={len(word_phoneme_mapping)}")

    encoder = WhistleEncoder(config, word_phoneme_mapping)
    # encoder = WhistleEncoder(config)
    # print(encoder)
    print(encoder.tokenizer._units)

    from WavTokenizer.encoder.utils import convert_audio
    from WavTokenizer.decoder.pretrained import WavTokenizer

    wavtokenizer = WavTokenizer.from_pretrained0802(
        config["nac"]["nac_config"], 
        config["nac"]["nac_model"]
    )
    wavtokenizer = wavtokenizer.to('cuda')
    wavtokenizer.eval() 
    bandwidth_id = torch.tensor([0]).to('cuda')

    audio_file = "/mnt/workspace/liuhw/whistle/121_123859_000017_000001.wav"
    text_file = "/mnt/workspace/liuhw/whistle/121_123859_000017_000001.normalized.txt"
    config_path = "/mnt/workspace/liuhw/whistle/ckpt/fbank.conf"

    fbank_features = extract_fbank_with_kaldi(audio_file, config_path)
    fbank = fbank_features.unsqueeze(0)
    # print(f"FBank feature shape: {fbank.shape}")  # (B, T, 80)
    speech_length = torch.full((fbank.shape[0],), fbank.shape[1])
    
    with open(text_file, 'r') as file:
        target_text = file.read().strip().lower()
    
    # Generate the word and separator sequences
    words, separators = encoder.split_text_by_separators(target_text)
    print(f"words={words}\nlen_words={len(words)}\nseparators{separators}\nlen_separators={len(separators)}")

    # Generate phoneme sequences from text chunks while recording where punctuation should appear
    target_phonemes, target_phonemes_with_separators, exist_null_phoneme = encoder.generate_phonemes(words, separators, encoder.tokenizer)
    print(f"target_phonemes=\n{target_phonemes}\nlen_target_phonemes={len(target_phonemes)}")
    print(f"target_phonemes_with_separators={target_phonemes_with_separators}\nlen_target_phonemes_with_separators={len(target_phonemes_with_separators)}")

    # Generate alignment information
    aligned_tokens, _, _ = encoder.align(fbank=fbank, lens=speech_length, target=target_phonemes)
    aligned_tokens = aligned_tokens.tolist()
    print(f"Original alignment path:\n{aligned_tokens}")

    # Compute the collapsed aligned_tokens path and the length of each label
    path_uni, lengths = encoder.calculate_ctc_lengths(aligned_tokens, target_phonemes_with_separators) 
    assert len(path_uni) == len(lengths)
    print(f"Collapsed path and the length of each phoneme:\n{path_uni}\nlen={len(path_uni)}\n{lengths}")
    assert path_uni == target_phonemes, f"Collapsed path does not match the target phoneme sequence. Collapsed path: {path_uni}, target phoneme sequence: {target_phonemes}"
    assert sum(lengths) == len(aligned_tokens), f"Collapsed-path length does not match the original alignment-path length. Collapsed length: {sum(lengths)}, original alignment-path length: {len(aligned_tokens)}"

    # Get WavTokenizer discrete speech tokens and align them with phonemes
    wav, sr = torchaudio.load(audio_file)
    wav = convert_audio(wav, sr, 24000, 1)
    wav = wav.to('cuda')

    with torch.no_grad():
        _, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    speech_tokens = discrete_code.tolist()[0][0]
    print(f"WavTokenizer discrete speech tokens:\n{speech_tokens}")

    # p_s_path= encoder.align_phonemes_speech(
    #     path_uni, lengths,target_phonemes_with_separators, speech_tokens, silence_token=config['nac']['SILENCE_TOKEN_ID'], 
    #     future_phoneme_num=config['data']['future_num'], eob=config['data']['eob'], eos=config['data']['eos']
    # )
    # print(f"Phoneme-speech alignment path:\n{p_s_path}")

    s_s_path = encoder.align_syllables_speech(
        path_uni, lengths,target_phonemes_with_separators, speech_tokens, silence_token=config['nac']['SILENCE_TOKEN_ID'], 
        future_syllable_num=config['data']['future_num'], eob=config['data']['eob'], eos=config['data']['eos']
    )
    print(f"Syllable-speech alignment path:\n{s_s_path}")

    # w_s_path = encoder.align_words_speech(
    #     path_uni, lengths,target_phonemes_with_separators, speech_tokens, silence_token=config['nac']['SILENCE_TOKEN_ID'], 
    #     future_word_num=config['data']['future_num'], eob=config['data']['eob'], eos=config['data']['eos']
    # )
    # print(f"Word-speech alignment path:\n{w_s_path}")
