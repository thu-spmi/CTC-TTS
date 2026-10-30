"""
Split voiceassistant.json into three datasets: train/dev/test.

Split rules:
  - dev: randomly sample data totaling about 50 hours
  - test: randomly sample about 5 hours of data from the remainder
  - train: all remaining data as the training set

Usage:
    python split_data.py
    python split_data.py --input /path/to/voiceassistant.json --output_dir /path/to/output
"""
import json
import os
import sys
import random
import argparse
from collections import OrderedDict

DEFAULT_INPUT = "/mnt/workspace/liuhw/VoiceAssistant/voiceassistant.json"
DEFAULT_OUTPUT_DIR = "/mnt/workspace/liuhw/datasets/voiceassistant/raw_data"
DEFAULT_DEV_HOURS = 50.0
DEFAULT_TEST_HOURS = 5.0
DEFAULT_SEED = 42


def calc_duration_hours(data: list) -> float:
    return sum(item.get("duration", 0) for item in data) / 3600.0


def split_data(input_json: str, output_dir: str, dev_hours: float, test_hours: float, seed: int):
    print(f"Reading data: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    total_hours = calc_duration_hours(data)
    print(f"Total entries: {total}, total duration: {total_hours:.2f}h")

    # Shuffle randomly
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    # 1. Sample dev: accumulate about dev_hours hours
    dev_target_sec = dev_hours * 3600
    dev_indices = []
    dev_dur = 0.0
    remaining_after_dev = []
    for idx in indices:
        item_dur = data[idx].get("duration", 0)
        if dev_dur < dev_target_sec:
            dev_indices.append(idx)
            dev_dur += item_dur
        else:
            remaining_after_dev.append(idx)

    # 2. Sample test from the remainder: accumulate about test_hours hours
    test_target_sec = test_hours * 3600
    test_indices = []
    test_dur = 0.0
    train_indices = []
    for idx in remaining_after_dev:
        item_dur = data[idx].get("duration", 0)
        if test_dur < test_target_sec:
            test_indices.append(idx)
            test_dur += item_dur
        else:
            train_indices.append(idx)

    # Build datasets
    dev_data = [data[i] for i in dev_indices]
    test_data = [data[i] for i in test_indices]
    train_data = [data[i] for i in train_indices]

    dev_h = calc_duration_hours(dev_data)
    test_h = calc_duration_hours(test_data)
    train_h = calc_duration_hours(train_data)

    print(f"\nSplit result (seed={seed}):")
    print(f"  dev:   {len(dev_data):>7} entries, {dev_h:.2f}h")
    print(f"  test:  {len(test_data):>7} entries, {test_h:.2f}h")
    print(f"  train: {len(train_data):>7} entries, {train_h:.2f}h")
    print(f"  Total: {len(dev_data)+len(test_data)+len(train_data):>7} entries, {dev_h+test_h+train_h:.2f}h")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    splits = OrderedDict([
        ("train", train_data),
        ("dev", dev_data),
        ("test", test_data),
    ])
    for name, split_data in splits.items():
        out_path = os.path.join(output_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Split VoiceAssistant data into train/dev/test")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dev_hours", type=float, default=DEFAULT_DEV_HOURS)
    parser.add_argument("--test_hours", type=float, default=DEFAULT_TEST_HOURS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    split_data(args.input, args.output_dir, args.dev_hours, args.test_hours, args.seed)


if __name__ == "__main__":
    main()
