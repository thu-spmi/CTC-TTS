import argparse
import json
import os
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from process_text import clean_text_content

# LibriSpeech path
DEFAULT_LIBRISPEECH_ROOT = "/mnt/workspace/liuhw/datasets/librispeech/LibriSpeech"
# Path to punctuated LibriSpeech-PC transcription data
DEFAULT_LIBRISPEECH_PC_ROOT = "/mnt/workspace/liuhw/datasets/librispeech/librispeech_pc"
# SeedTTS data path
DEFAULT_SEEDTTS_EN_ROOT = "/mnt/workspace/liuhw/datasets/seedtts/seedtts_testset/en"
DEFAULT_SEEDTTS_META = os.path.join(DEFAULT_SEEDTTS_EN_ROOT, "meta.lst")
DEFAULT_SEEDTTS_OUTPUT = (
    "/mnt/workspace/liuhw/datasets/ctc_tts/mono_phone_librispeech/seedtts_cross_text.json"
)

DEFAULT_OUTPUT_DIR = "/mnt/workspace/liuhw/datasets/ctc_tts/mono_phone_librispeech"


SPLIT_TO_PC_FILES = {
    "train": [
        "train-clean-100.json",
        "train-clean-360.json",
        "train-other-500.json",
    ],
    "dev": [
        "dev-clean.json",
        "dev-other.json",
    ],
    "test": [
        "test-clean.json",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cleaned LibriSpeech json files from librispeech_pc punctuated transcripts."
    )
    parser.add_argument("--librispeech_root", type=str, default=DEFAULT_LIBRISPEECH_ROOT)
    parser.add_argument("--librispeech_pc_root", type=str, default=DEFAULT_LIBRISPEECH_PC_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seedtts_en_root", type=str, default=DEFAULT_SEEDTTS_EN_ROOT)
    parser.add_argument("--seedtts_meta", type=str, default=DEFAULT_SEEDTTS_META)
    parser.add_argument("--seedtts_output", type=str, default=DEFAULT_SEEDTTS_OUTPUT)
    parser.add_argument(
        "--prefer_text_raw",
        action="store_true",
        default=True,
        help="Prefer text_raw over text before cleaning. Enabled by default.",
    )
    parser.add_argument(
        "--disable_prefer_text_raw",
        action="store_false",
        dest="prefer_text_raw",
        help="Use text instead of text_raw when both are available.",
    )
    return parser.parse_args()


def read_jsonl(file_path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid json line in {file_path}:{line_num}") from exc
    return items


def choose_source_text(item: Dict, prefer_text_raw: bool) -> str:
    if prefer_text_raw and item.get("text_raw"):
        return item["text_raw"]
    if item.get("text"):
        return item["text"]
    if item.get("text_raw"):
        return item["text_raw"]
    return ""


def build_sample(item: Dict, librispeech_root: str, prefer_text_raw: bool) -> Optional[Dict]:
    rel_audio_path = item["audio_filepath"]
    abs_audio_path = os.path.join(librispeech_root, rel_audio_path)
    if not os.path.isfile(abs_audio_path):
        return None

    source_text = choose_source_text(item, prefer_text_raw)
    cleaned_text = clean_text_content(source_text)
    if not cleaned_text:
        return None

    speech_folder, speech_file = os.path.split(abs_audio_path)
    sample_id = os.path.splitext(speech_file)[0]
    return {
        "speech_folder": speech_folder,
        "speech_file": speech_file,
        "id": sample_id,
        "text": cleaned_text,
        "units_label": cleaned_text,
        "duration": float(item["duration"]),
    }


def iter_split_items(split_name: str, librispeech_pc_root: str) -> Iterable[Dict]:
    for file_name in SPLIT_TO_PC_FILES[split_name]:
        file_path = os.path.join(librispeech_pc_root, file_name)
        for item in read_jsonl(file_path):
            yield item


def process_split(
    split_name: str,
    librispeech_root: str,
    librispeech_pc_root: str,
    prefer_text_raw: bool,
) -> List[Dict]:
    input_file_paths = [
        os.path.join(librispeech_pc_root, file_name) for file_name in SPLIT_TO_PC_FILES[split_name]
    ]
    expected_count = sum(len(read_jsonl(file_path)) for file_path in input_file_paths)
    processed_items: List[Dict] = []
    missing_audio = 0
    empty_text = 0

    for item in tqdm(
        iter_split_items(split_name, librispeech_pc_root),
        total=expected_count,
        desc=f"Processing {split_name}",
        unit="utt",
    ):
        sample = build_sample(item, librispeech_root, prefer_text_raw)
        if sample is None:
            rel_audio_path = item["audio_filepath"]
            abs_audio_path = os.path.join(librispeech_root, rel_audio_path)
            if not os.path.isfile(abs_audio_path):
                missing_audio += 1
            else:
                empty_text += 1
            continue
        processed_items.append(sample)

    print(
        f"{split_name}: kept={len(processed_items)}, "
        f"missing_audio={missing_audio}, empty_text={empty_text}"
    )
    return processed_items


def save_split(items: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def process_seedtts_cross(seedtts_meta: str, seedtts_en_root: str) -> List[Dict]:
    processed_items: List[Dict] = []

    with open(seedtts_meta, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines, desc="Processing seedtts_en", unit="utt"):
        parts = line.split("|")
        if len(parts) != 4:
            raise ValueError(f"Invalid meta line: {line}")

        sample_id, prompt_text, prompt_speech_relpath, target_text = parts
        prompt_speech_path = (
            prompt_speech_relpath
            if os.path.isabs(prompt_speech_relpath)
            else os.path.join(seedtts_en_root, prompt_speech_relpath)
        )

        prompt_unit = clean_text_content(prompt_text)
        target_unit = clean_text_content(target_text)
        if not prompt_unit or not target_unit:
            continue

        processed_items.append(
            {
                "id": sample_id,
                "prompt_speech_path": prompt_speech_path,
                "prompt_text": prompt_text,
                "target_text": target_text,
                "prompt_unit": prompt_unit,
                "target_unit": target_unit,
            }
        )

    print(f"seedtts_en: kept={len(processed_items)}")
    return processed_items


def main() -> None:
    args = parse_args()

    outputs = {
        "train": os.path.join(args.output_dir, "train.json"),
        "dev": os.path.join(args.output_dir, "dev.json"),
        "test": os.path.join(args.output_dir, "test_clean.json"),
    }

    for split_name, output_path in outputs.items():
        items = process_split(
            split_name=split_name,
            librispeech_root=args.librispeech_root,
            librispeech_pc_root=args.librispeech_pc_root,
            prefer_text_raw=args.prefer_text_raw,
        )
        save_split(items, output_path)
        print(f"Saved {split_name} json to {output_path}")

    seedtts_items = process_seedtts_cross(
        seedtts_meta=args.seedtts_meta,
        seedtts_en_root=args.seedtts_en_root,
    )
    save_split(seedtts_items, args.seedtts_output)
    print(f"Saved seedtts cross json to {args.seedtts_output}")


if __name__ == "__main__":
    main()
