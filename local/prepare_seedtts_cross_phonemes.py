import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ctc_model.whistle.phontisaurus_g2p import g2p_phonetize, process_phonemes


DEFAULT_INPUT_JSON = "/mnt/workspace/liuhw/streaming_tts/local/seedtts/seedtts_cross.json"
DEFAULT_OUTPUT_JSON = "/mnt/workspace/liuhw/datasets/ctc_tts/phone_bpe_100/seedtts_cross_phoneme.json"
DEFAULT_MODEL_PATH = "/mnt/workspace/liuhw/sftp_dir/lexicon/model.fst"

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
IPA_REPLACEMENTS = {
    "k": "kʰ",
    "p": "pʰ",
    "t": "tʰ",
    "ɹ": "ɻ",
    "ɛ": "e",
    "ɔ": "o",
    "ɪ": "i",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SeedTTS cross text into phoneme units.")
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--chunk_size", type=int, default=512)
    return parser.parse_args()


def tokenize_text(text: str) -> List[str]:
    return WORD_PATTERN.findall(text)


def is_word_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", token))


def apply_ipa_replacements(phonemes: Sequence[str]) -> List[str]:
    return [IPA_REPLACEMENTS.get(phoneme, phoneme) for phoneme in phonemes]


def _parse_phonetisaurus_line(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []

    if "#" in line:
        parts = line.split("#")
        phonemes_str = parts[1].strip() if len(parts) >= 2 else ""
    else:
        tokens = line.split()
        phonemes_str = " ".join(tokens[1:]) if len(tokens) >= 2 else ""

    phonemes_str = process_phonemes(phonemes_str) if phonemes_str else ""
    return [phoneme for phoneme in phonemes_str.split() if phoneme]


def batch_g2p_words(words: Sequence[str], model_path: str, chunk_size: int = 512) -> Dict[str, List[str]]:
    unique_words = sorted({word.upper() for word in words if word})
    phoneme_map: Dict[str, List[str]] = {}

    if not unique_words:
        return phoneme_map

    total_chunks = math.ceil(len(unique_words) / chunk_size)
    progress = tqdm(total=len(unique_words), desc="G2P words", unit="word")

    for chunk_index in range(total_chunks):
        start = chunk_index * chunk_size
        end = min(len(unique_words), start + chunk_size)
        chunk_words = unique_words[start:end]
        cmd = ["phonetisaurus", "predict", "--model", model_path] + chunk_words

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = [line for line in result.stdout.splitlines() if line.strip()]
        except subprocess.CalledProcessError:
            lines = []

        for word, line in zip(chunk_words, lines):
            phoneme_map[word] = _parse_phonetisaurus_line(line)

        missing_words = [word for word in chunk_words if word not in phoneme_map]
        for word in missing_words:
            phoneme_map[word] = g2p_phonetize(word, model_path=model_path, apply_processing=True)

        progress.update(len(chunk_words))

    progress.close()
    return phoneme_map


def phonemize_text(text: str, phoneme_map: Dict[str, List[str]]) -> str:
    tokens = tokenize_text(text)
    rendered_tokens: List[str] = []

    for token in tokens:
        if is_word_token(token):
            phonemes = apply_ipa_replacements(phoneme_map.get(token.upper(), []))
            phoneme_word = "".join(phonemes)
            if phoneme_word:
                rendered_tokens.append(phoneme_word)
            else:
                rendered_tokens.append(token.lower())
            continue

        if rendered_tokens:
            rendered_tokens[-1] = rendered_tokens[-1] + token
        else:
            rendered_tokens.append(token)

    return " ".join(rendered_tokens)


def collect_unique_words(items: Sequence[dict]) -> List[str]:
    words: List[str] = []
    for item in items:
        for field in ("prompt_text", "target_text"):
            text = item.get(field, "")
            for token in tokenize_text(text):
                if is_word_token(token):
                    words.append(token)
    return words


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    unique_words = collect_unique_words(items)
    phoneme_map = batch_g2p_words(unique_words, model_path=args.model_path, chunk_size=args.chunk_size)

    converted_items = []
    for item in tqdm(items, desc="Convert samples", unit="item"):
        converted_item = dict(item)
        converted_item["prompt_unit"] = phonemize_text(item.get("prompt_text", ""), phoneme_map)
        converted_item["target_unit"] = phonemize_text(item.get("target_text", ""), phoneme_map)
        converted_items.append(converted_item)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted_items, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(converted_items)} items to {output_path}")


if __name__ == "__main__":
    main()
