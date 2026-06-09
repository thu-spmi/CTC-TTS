import os
import sys
import json
import argparse
import re
from tqdm import tqdm
import subprocess


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ctc_model.whistle.phontisaurus_g2p import process_phonemes 
from ctc_model.tokenizer import LexiconTokenizer, save

WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")

def build_unique_vocab_from_json(json_path: str) -> set:
    """
    Extract text fields from cleaned JSON entries and return the unique word set.

    Supported field priority:
    - text / units_label
    - prompt_text / target_text
    - prompt_unit / target_unit
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = set()
    for item in data:
        candidate_fields = [
            item.get("text"),
            item.get("units_label"),
            item.get("prompt_text"),
            item.get("target_text"),
            item.get("prompt_unit"),
            item.get("target_unit"),
        ]
        for text in candidate_fields:
            if not isinstance(text, str) or not text:
                continue
            for word in WORD_PATTERN.findall(text.lower()):
                # vocab.add(word.upper())
                vocab.add(word.upper())
    return vocab

def build_unique_vocab_from_json_files(json_paths: list) -> set:
    """
    Extract vocabularies from multiple cleaned JSON files and merge them into a unique word set.
    """
    vocab = set()
    for json_path in tqdm(json_paths, desc="Reading JSON files", unit="file"):
        try:
            v = build_unique_vocab_from_json(json_path)
            vocab.update(v)
        except Exception as e:
            print(f"Failed to read {json_path}: {e}")
    return vocab

def batch_g2p_predict(words_batch, model_path, apply_processing=True):
    """
    Use phonetisaurus to predict a batch of words at once and return a {WORD: [PHONEMES,...]} mapping
    Support two output formats (different `.fst` models may emit different formats):
      1) "WORD PHONEME PHONEME ..."
      2) "WORD # PHONEME PHONEME #"  Support for this format is still incomplete
    """
    if not words_batch:
        return {}
    cmd = ['phonetisaurus', 'predict', '--model', model_path] + list(words_batch)
    mapping = {}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if '#' in line:
                parts = line.split('#')
                left = parts[0].strip()
                word_token = left.split()[0] if left else ""
                phonemes_str = parts[1].strip() if len(parts) >= 2 else ""
            else:
                tokens = line.split()
                word_token = tokens[0] if tokens else ""
                phonemes_str = ' '.join(tokens[1:]) if len(tokens) >= 2 else ""
            if apply_processing and phonemes_str:
                phonemes_str = process_phonemes(phonemes_str)
            phonemes = [p for p in phonemes_str.split() if p]
            if word_token:
                mapping[word_token] = phonemes
    except subprocess.CalledProcessError as e:
        print(f"Error running phonetisaurus: {e}")
    except Exception as e:
        print(f"Unexpected error during batch G2P: {e}")
    return mapping

def generate_lexicon(vocab: set, output_path: str, model_path: str, batch_size: int = 256) -> None:
    """
    Generate a lexicon from the unique word set and write it to `output_path`
    Support batch prediction to reduce external process launches and improve speed
    Format: WORD\tPHONEME1 PHONEME2 ...
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sorted_vocab = sorted(vocab)
    total_words = len(sorted_vocab)
    success = 0
    failed = 0

    # Show progress batch by batch
    num_batches = (total_words + batch_size - 1) // batch_size
    with open(output_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, total_words, batch_size), desc="Generating lexicon (G2P batches)", unit="batch", total=num_batches):
            batch = sorted_vocab[i:i + batch_size]
            mapping = batch_g2p_predict(batch, model_path, apply_processing=True)
            for word in batch:
                phonemes = mapping.get(word, [])
                if phonemes:
                    fout.write(f"{word}\t{' '.join(phonemes)}\n")
                    success += 1
                else:
                    failed += 1

    print(f"Lexicon written to: {output_path}")
    print(f"Total unique words: {total_words}, success: {success}, failed: {failed}")


def build_and_save_lexicon_tokenizer(lexicon_path: str, tokenizer_path: str) -> None:
    """
    Build a LexiconTokenizer from the generated pronunciation lexicon and save it as a `.tknz` file.
    """
    output_dir = os.path.dirname(tokenizer_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tokenizer = LexiconTokenizer(lexicon=lexicon_path, add_special_token=True)
    save(tokenizer, tokenizer_path)
    print(f"Tokenizer written to: {tokenizer_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate lexicon.txt from cleaned dataset json files")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config containing data.raw_train_path/raw_dev_path/raw_test_path/lexicon_path and encoder.phonetisaurus_model_path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for G2P prediction",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    input_jsons = []
    input_jsons.append(cfg['data']['raw_train_path'])
    input_jsons.append(cfg['data']['raw_dev_path'])
    input_jsons.append(cfg['data']['raw_test_path'])
    cross_data_path = cfg.get("inference", {}).get("cross_data")
    if cross_data_path:
        input_jsons.append(cross_data_path)

    output_lexicon = cfg['data']['lexicon_path']
    output_tokenizer = cfg['tokenizer']['file']
    model_path = cfg['encoder']['phonetisaurus_model_path']

    vocab = build_unique_vocab_from_json_files(input_jsons)
    if not vocab:
        print("No words extracted from input JSON. Please check the file and its 'text' or 'units_label' fields.")
        sys.exit(1)

    generate_lexicon(vocab, output_lexicon, model_path, args.batch_size)
    build_and_save_lexicon_tokenizer(output_lexicon, output_tokenizer)

if __name__ == "__main__":
    main()
