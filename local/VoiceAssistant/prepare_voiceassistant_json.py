"""
Process the Whisper-transcribed text in the VoiceAssistant400k dataset and save it in the json format required for ctc_tts training after cleaning.

Features:
  1. Read the whisper_text field from metadata.json
  2. Replace numbered labels such as "1. " "2. " ... "9. " in the text with English words "one. " "two. " ... "nine. "
  3. Use clean_text_content in process_text.py for text normalization and cleaning
  4. Output a json file in the same format as prepare_librispeech_json.py
     (containing speech_folder, speech_file, id, text, units_label, duration)

Usage:
    python prepare_voiceassistant_json.py
    python prepare_voiceassistant_json.py --data_dir /mnt/nas4_workspace1/HanwenLiu/VoiceAssistant400k
    python prepare_voiceassistant_json.py --output /path/to/output.json
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

from tqdm import tqdm

# Add ctc_tts/local to sys.path to import process_text
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from process_text import clean_text_content

DEFAULT_DATA_DIR = "/mnt/nas4_workspace1/HanwenLiu/VoiceAssistant400k"
DEFAULT_OUTPUT = "/mnt/workspace/liuhw/VoiceAssistant/voiceassistant.json"

# Map numbers 1-9 to English words
NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def replace_numbered_lists(text: str) -> str:
    """
    Replace numbered labels such as "1. " "2. " ... "9. " in the text with English words.
    Match pattern: a digit 1-9 immediately followed by a period, then a space or end of string.
    For example: "1. Eat" -> "one. Eat", "2." -> "two."
    """
    def _replace_match(match):
        num = int(match.group(1))
        if num in NUMBER_WORDS:
            return NUMBER_WORDS[num] + "."
        return match.group(0)

    # Match "digit." followed by a space or end of line/string
    text = re.sub(r"\b([1-9])\.(?=\s|$)", _replace_match, text)
    return text


VALID_CHAR_PATTERN = re.compile(r"^[a-z.,?!' ]+$")


def find_invalid_chars(text: str) -> set:
    """Return the set of characters in the text that are not in the allowed character set [a-z.,?!' ]."""
    return set(text) - set("abcdefghijklmnopqrstuvwxyz.,?!' ")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare cleaned VoiceAssistant json from Whisper transcripts."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="VoiceAssistant400k data directory (containing metadata.json and wavs/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output json file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    meta_path = os.path.join(args.data_dir, "metadata.json")
    print(f"Reading metadata: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Total entries: {len(metadata)}")

    processed_items = []
    empty_text = 0
    none_duration = 0
    missing_audio = 0
    error_count = 0
    invalid_char_count = 0
    invalid_char_counter = Counter()

    for item in tqdm(metadata, desc="Processing VoiceAssistant", unit="utt"):
        try:
            item_id = item["id"]
            audio_rel = item["audio_path"]
            audio_abs = os.path.join(args.data_dir, audio_rel)
            duration_val = item.get("duration", None)
            if duration_val is None:
                none_duration += 1
                continue
            duration = float(duration_val)

            # Use whisper_text as the source text
            source_text = item.get("whisper_text", "")
            if not source_text or not source_text.strip():
                empty_text += 1
                continue

            if not os.path.isfile(audio_abs):
                missing_audio += 1
                continue

            # 1. Replace numbered labels with English words
            text = replace_numbered_lists(source_text)

            # 2. Clean the text with clean_text_content
            cleaned_text = clean_text_content(text)
            if not cleaned_text:
                empty_text += 1
                continue

            # 3. Check whether it contains invalid characters
            invalid_chars = find_invalid_chars(cleaned_text)
            if invalid_chars:
                invalid_char_count += 1
                invalid_char_counter.update(invalid_chars)
                continue

            speech_folder, speech_file = os.path.split(audio_abs)
            processed_items.append({
                "speech_folder": speech_folder,
                "speech_file": speech_file,
                "id": item_id,
                "text": cleaned_text,
                "units_label": cleaned_text,
                "duration": duration,
            })
        except Exception as e:
            error_count += 1
            if error_count <= 10:
                tqdm.write(f"  [Error] {item.get('id', 'unknown')}: {e}")

    total_duration = sum(item["duration"] for item in processed_items)
    total_hours = total_duration / 3600

    print(f"\nProcessing complete:")
    print(f"  Kept entries: {len(processed_items)}")
    print(f"  Total duration: {total_duration:.1f}s ({total_hours:.2f}h)")
    print(f"  Empty text/empty after cleaning: {empty_text}")
    print(f"  duration is None: {none_duration}")
    print(f"  Missing audio files: {missing_audio}")
    print(f"  Containing invalid characters: {invalid_char_count}")
    print(f"  Processing errors: {error_count}")

    if invalid_char_counter:
        print(f"\nInvalid character statistics ({len(invalid_char_counter)} kinds in total):")
        for char, cnt in invalid_char_counter.most_common():
            if char == '\n':
                display = '\\n'
            elif char == '\t':
                display = '\\t'
            elif char == '\r':
                display = '\\r'
            else:
                display = char
            print(f"  '{display}': {cnt}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed_items, f, ensure_ascii=False, indent=2)
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
