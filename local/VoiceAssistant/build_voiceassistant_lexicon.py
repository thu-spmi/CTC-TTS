"""
Generate pronunciation lexicon and tokenizer for the VoiceAssistant dataset.
"""
import os
import sys
import re
import json
import argparse
import tempfile
from collections import OrderedDict
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ctc_model.tokenizer import LexiconTokenizer, load_tknz, save
from ctc_model.whistle.phontisaurus_g2p import g2p_phonetize

WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")

DEFAULT_INPUT_JSON = "/mnt/workspace/liuhw/VoiceAssistant/voiceassistant.json"
DEFAULT_G2P_MODEL = "/mnt/workspace/liuhw/ctc_tts/config/model.fst"
DEFAULT_BASE_TKNZ = "/mnt/workspace/liuhw/VoiceAssistant/dict_librispeech_own_replace/tokenizer.tknz"
DEFAULT_OUTPUT_DIR = "/mnt/workspace/liuhw/datasets/voiceassistant"
DEFAULT_BATCH_SIZE = 1024

# Phoneme replacement rules: original IPA -> replaced IPA
# Note: this replaces individual phonemes, not strings, to avoid the t in tʃ being wrongly replaced with tʰ
PHONEME_REPLACEMENTS = {
    "k": "kʰ",
    "p": "pʰ",
    "t": "tʰ",
    "ɹ": "ɻ",
    "ɛ": "e",
    "ɔ": "o",
    "ɪ": "i",
}


def apply_phoneme_replacements(phonemes: list) -> list:
    """Apply the replacement rules to each phoneme in the list."""
    return [PHONEME_REPLACEMENTS.get(p, p) for p in phonemes]


def extract_vocab_from_json(json_path: str) -> set:
    """Extract all unique words (lowercase) from the cleaned JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = set()
    for item in tqdm(data, desc="Extracting vocabulary", unit="utt"):
        for field in ["text", "units_label"]:
            text = item.get(field, "")
            if not isinstance(text, str) or not text:
                continue
            for word in WORD_PATTERN.findall(text.lower()):
                vocab.add(word)
    return vocab


def batch_g2p(words_lower: list, model_path: str, batch_size: int = 1024) -> dict:
    """Batch G2P, reusing phontisaurus_g2p.g2p_phonetize, returns {lower_word: [phonemes...]}."""
    import subprocess
    mapping = {}
    total = len(words_lower)
    words_upper = sorted([w.upper() for w in words_lower])
    upper_to_lower = {w.upper(): w for w in words_lower}

    for i in tqdm(range(0, total, batch_size), desc="Batch G2P prediction", unit="batch"):
        batch_upper = words_upper[i:i + batch_size]
        try:
            cmd = ["phonetisaurus", "predict", "--model", model_path] + batch_upper
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                if "#" in line:
                    parts = line.split("#")
                    word_token = parts[0].strip().split()[0] if parts[0].strip() else ""
                    phonemes_str = parts[1].strip() if len(parts) >= 2 else ""
                else:
                    tokens = line.split()
                    word_token = tokens[0] if tokens else ""
                    phonemes_str = " ".join(tokens[1:]) if len(tokens) >= 2 else ""

                if word_token and phonemes_str:
                    phonemes = [p for p in phonemes_str.split() if p]
                    phonemes = apply_phoneme_replacements(phonemes)
                    lower_word = upper_to_lower.get(word_token, word_token.lower())
                    mapping[lower_word] = phonemes
        except subprocess.CalledProcessError as e:
            # Fall back to per-word calls if the batch fails
            tqdm.write(f"  Batch G2P failed, falling back to per-word processing: {e}")
            for w in batch_upper:
                phonemes = g2p_phonetize(w, model_path=model_path, apply_processing=False)
                if phonemes:
                    phonemes = apply_phoneme_replacements(phonemes)
                    lower_word = upper_to_lower.get(w, w.lower())
                    mapping[lower_word] = phonemes
    return mapping


def save_lexicon(mapping: dict, output_path: str):
    """Save the lexicon file, format: word\tphoneme1 phoneme2 ... (words lowercase)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sorted_words = sorted(mapping.keys())
    with open(output_path, "w", encoding="utf-8") as f:
        for word in sorted_words:
            phonemes = mapping[word]
            f.write(f"{word}\t{' '.join(phonemes)}\n")
    print(f"Lexicon saved: {output_path} ({len(sorted_words)} words)")


def save_phone_list(new_mapping: dict, base_tokenizer_units: OrderedDict, output_path: str):
    """Save the full phone list (merging existing tokenizer phones and newly generated phones)."""
    all_phones = OrderedDict()
    for unit, idx in base_tokenizer_units.items():
        if unit not in ("<s>", "<unk>"):
            all_phones[unit] = True
    for phonemes in new_mapping.values():
        for p in phonemes:
            all_phones[p] = True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for phone in all_phones:
            f.write(phone + "\n")
    print(f"Phone list saved: {output_path} ({len(all_phones)} phones)")


def main():
    parser = argparse.ArgumentParser(description="Build lexicon and tokenizer for VoiceAssistant")
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--g2p_model", type=str, default=DEFAULT_G2P_MODEL)
    parser.add_argument("--base_tknz", type=str, default=DEFAULT_BASE_TKNZ)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    lexicon_path = os.path.join(args.output_dir, "lexicon.txt")
    phone_list_path = os.path.join(args.output_dir, "phone_list.txt")
    output_tknz_path = os.path.join(args.output_dir, "tokenizer.tknz")

    # 1. Extract vocabulary (lowercase)
    print(f"Reading data: {args.input_json}")
    vocab = extract_vocab_from_json(args.input_json)
    print(f"Extracted {len(vocab)} unique words (lowercase)")

    # 2. Load the base tokenizer, convert existing words to lowercase, find new words
    print(f"\nLoading base tokenizer: {args.base_tknz}")
    base_tokenizer = load_tknz(args.base_tknz)
    if not isinstance(base_tokenizer, LexiconTokenizer):
        raise TypeError(f"Base tokenizer is not a LexiconTokenizer: {type(base_tokenizer).__name__}")

    lower_w2pid = OrderedDict()
    for word, token_ids in base_tokenizer._w2pid.items():
        if word in ("<s>", "<unk>"):
            lower_w2pid[word] = token_ids
        else:
            lower_w2pid[word.lower()] = token_ids
    base_tokenizer._w2pid = lower_w2pid

    existing_words = set(w for w in base_tokenizer._w2pid.keys() if w not in ("<s>", "<unk>"))
    new_words = vocab - existing_words
    print(f"Base tokenizer already has {len(existing_words)} words (lowercase); new words needing G2P: {len(new_words)}")

    # 3. Generate pronunciations for new words via G2P
    if new_words:
        print(f"\nStarting G2P prediction for new words...")
        new_mapping = batch_g2p(sorted(new_words), args.g2p_model, args.batch_size)
        failed = len(new_words) - len(new_mapping)
        print(f"G2P done: {len(new_mapping)} words succeeded, {failed} words failed")
    else:
        new_mapping = {}

    # 4. Merge existing pronunciations and newly generated pronunciations (all lowercase)
    combined_mapping = OrderedDict()
    for word, token_ids in base_tokenizer._w2pid.items():
        if word in ("<s>", "<unk>"):
            continue
        phonemes = [base_tokenizer._id2unit[tid] for tid in token_ids]
        combined_mapping[word] = phonemes
    for word, phonemes in new_mapping.items():
        combined_mapping[word] = phonemes

    # 5. Save lexicon
    save_lexicon(combined_mapping, lexicon_path)

    # 6. Save phone list
    save_phone_list(new_mapping, base_tokenizer._units, phone_list_path)

    # 7. Extend the tokenizer and save
    print(f"\nExtending tokenizer...")
    tmp_lex = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    for word, phonemes in new_mapping.items():
        tmp_lex.write(f"{word}\t{' '.join(phonemes)}\n")
    tmp_lex.close()

    stats = base_tokenizer.update_from_lexicon(tmp_lex.name, overwrite_existing_words=False)
    os.unlink(tmp_lex.name)

    os.makedirs(os.path.dirname(output_tknz_path), exist_ok=True)
    save(base_tokenizer, output_tknz_path)
    print(f"Tokenizer saved: {output_tknz_path}")
    print(f"Update stats: added_units={stats['added_units']}, added_words={stats['added_words']}, "
          f"total_units={stats['total_units']}, total_words={stats['total_words']}")

    # 8. Verify phoneme ID consistency
    print(f"\n===== Verifying phoneme ID consistency =====")
    orig_tokenizer = load_tknz(args.base_tknz)
    new_tokenizer = load_tknz(output_tknz_path)
    id_match = True
    for unit, idx in orig_tokenizer._units.items():
        if new_tokenizer._units.get(unit) != idx:
            print(f"  Mismatch: '{unit}' original id={idx}, new id={new_tokenizer._units.get(unit)}")
            id_match = False
    if id_match:
        print(f"  All phoneme IDs exactly match the original tokenizer ✓")
    print(f"  Original vocab_size: {orig_tokenizer.vocab_size}, new vocab_size: {new_tokenizer.vocab_size}")

    test_text = "hello world"
    ids = new_tokenizer.encode(test_text)
    dec = new_tokenizer.decode(ids)
    print(f"  encode('{test_text}'): {ids}")
    print(f"  decode: {dec}")


if __name__ == "__main__":
    main()
