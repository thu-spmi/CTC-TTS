import json
import torch
import os
import sys
from tqdm import tqdm
import time
import multiprocessing as mp
import math
import argparse
import soundfile as sf
from typing import List

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.nac import build_nac
from ctc_model.ctc_align import align_text_logits_and_speech_tokens, build_textgrid
from ctc_model.encoder_loader import infer_ctc_logits, load_pretrained_encoder
from ctc_model.tokenizer import load_tknz

PROCESSES_PER_GPU = 4  # Processes per GPU

# When --write_textgrid is enabled, the TextGrid path is derived from the audio path by a direct
# string replacement: replace the --audio_root prefix with --textgrid_dir and change the extension to .TextGrid
DEFAULT_AUDIO_ROOT = "/mnt/workspace/liuhw/datasets/librispeech/LibriSpeech"
DEFAULT_TEXTGRID_DIR = "/mnt/workspace/liuhw/datasets/librispeech/LibriSpeech/textgrid"


def build_textgrid_file_path(textgrid_dir, audio_path, audio_root=DEFAULT_AUDIO_ROOT):
    """Convert an audio path into a TextGrid path by string replacement: swap the root prefix and change the extension."""
    replaced = audio_path.replace(audio_root, textgrid_dir, 1)
    return os.path.splitext(replaced)[0] + ".TextGrid"


def parse_args():
    parser = argparse.ArgumentParser(description="Multiprocess data preparation")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device to run on (e.g., 'cuda', 'cpu', or 'cuda:0')"
    )
    parser.add_argument(
        "--write_textgrid",
        action="store_true",
        help="Additionally write a .TextGrid for each audio (words + phones tiers)"
    )
    parser.add_argument(
        "--textgrid_dir",
        type=str,
        default=DEFAULT_TEXTGRID_DIR,
        help="Root directory for TextGrid output; the relative path is appended after replacing the --audio_root prefix"
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default=DEFAULT_AUDIO_ROOT,
        help="Root prefix of the original audio; replaced with --textgrid_dir when writing TextGrid"
    )
    return parser.parse_args()

def resolve_worker_device(proc_id, requested_device):
    if requested_device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if requested_device.startswith("cuda:"):
        return requested_device
    gpu_count = torch.cuda.device_count()
    return f"cuda:{proc_id % gpu_count}"


def get_total_processes(requested_device):
    if requested_device == "cpu" or not torch.cuda.is_available():
        return 1
    if requested_device.startswith("cuda:"):
        return PROCESSES_PER_GPU
    return max(1, torch.cuda.device_count() * PROCESSES_PER_GPU)


def load_tokenizer_from_config(config):
    return load_tknz(config["tokenizer"]["file"])


def format_units_text(tokenizer, tokenization) -> str:
    vocab = tokenizer.dump_vocab()
    units_parts: List[str] = []
    cursor = 0
    for segment_index, segment_length in enumerate(tokenization.segment_lengths):
        segment_token_ids = tokenization.ctc_token_ids[cursor:cursor + segment_length]
        cursor += segment_length
        if segment_token_ids:
            units_parts.extend(vocab[int(token_id)] for token_id in segment_token_ids)
        if segment_index < len(tokenization.separators):
            units_parts.append(tokenization.separators[segment_index])
    return " ".join(units_parts)


def process_worker(proc_id, data_subset, config, output_path, dataset_name, progress_counter, requested_device,
                   write_textgrid=False, textgrid_dir=DEFAULT_TEXTGRID_DIR, audio_root=DEFAULT_AUDIO_ROOT):
    """Multiprocessing worker function"""
    device = resolve_worker_device(proc_id, requested_device)
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":")[1]))
    
    try:
        print(f"Process {proc_id} loading models on {device}...")

        nac = build_nac(config, device=device)
        ctc_encoder = load_pretrained_encoder(config, device=device)
        tokenizer = load_tokenizer_from_config(config)

        print(f"Process {proc_id} models loaded successfully")

        processed_data = []
        error_count = 0
        textgrid_error_count = 0
        created_textgrid_dirs = set()
        
        for item in data_subset:
            audio_path = os.path.join(item['speech_folder'], item['speech_file'])
            try:
                audio_id = item['id']
                text = item['units_label']

                if not os.path.exists(audio_path):
                    error_count += 1
                    progress_counter.value += 1
                    continue

                speech_tokens = nac.encode(audio_path).tolist()[0][0]
                logits = infer_ctc_logits(audio_path, ctc_encoder, cfg=config, device=device)
                alignment_result = align_text_logits_and_speech_tokens(
                    text=text,
                    logits=logits,
                    tokenizer=tokenizer,
                    speech_token_ids=speech_tokens,
                    config=config,
                )

                seqs_str = ' '.join(map(str, alignment_result.seqs_sequence))
                units_str = format_units_text(tokenizer, alignment_result.tokenization)

                if write_textgrid:
                    try:
                        duration = sf.info(audio_path).duration
                        textgrid = build_textgrid(alignment_result, tokenizer, config, audio_duration=duration)
                        textgrid_path = build_textgrid_file_path(textgrid_dir, audio_path, audio_root)
                        textgrid_subdir = os.path.dirname(textgrid_path)
                        if textgrid_subdir not in created_textgrid_dirs:
                            os.makedirs(textgrid_subdir, exist_ok=True)
                            created_textgrid_dirs.add(textgrid_subdir)
                        with open(textgrid_path, 'w', encoding='utf-8') as f:
                            f.write(textgrid)
                    except Exception as e:
                        textgrid_error_count += 1
                        if textgrid_error_count <= 5:
                            print(f"Process {proc_id} failed to write TextGrid {audio_id}: {str(e)}")

                data_item = {
                    "id": audio_id,
                    "audio_path": audio_path,
                    "text": text,
                    "units": units_str,
                    "seqs": seqs_str
                }

                processed_data.append(data_item)
                progress_counter.value += 1
                
            except Exception as e:
                print(f"Process {proc_id} error processing {audio_path}: {str(e)}")
                error_count += 1
                progress_counter.value += 1
        
        # Save processed results to a temporary file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Process {proc_id} completed: {len(processed_data)} processed, {error_count} errors, {textgrid_error_count} textgrid errors")
        
    except Exception as e:
        print(f"Process {proc_id} fatal error: {e}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

def process_dataset_multiprocess(data_list, config, output_file, dataset_name, device,
                                 write_textgrid=False, textgrid_dir=DEFAULT_TEXTGRID_DIR,
                                 audio_root=DEFAULT_AUDIO_ROOT):
    """Process the dataset with multiple processes"""
    if not data_list:
        print(f"No data to process for {dataset_name}")
        return
    
    total_processes = min(len(data_list), get_total_processes(device))
    print(f"Processing {len(data_list)} files for {dataset_name} using {total_processes} processes")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    progress_counter = manager.Value('i', 0)
    
    chunk_size = math.ceil(len(data_list) / total_processes)
    processes = []
    
    for i in range(total_processes):
        chunk = data_list[i * chunk_size:(i + 1) * chunk_size]
        if not chunk:
            continue
        
        part_output_path = output_file.replace(".json", f"_part{i}.json")
        p = mp.Process(
            target=process_worker,
            args=(i, chunk, config, part_output_path, dataset_name, progress_counter, device,
                  write_textgrid, textgrid_dir, audio_root),
        )
        processes.append((p, part_output_path))
        p.start()
    
    # Show progress
    with tqdm(total=len(data_list), desc=f"Processing {dataset_name}") as pbar:
        last_val = 0
        while any(p[0].is_alive() for p in processes):
            time.sleep(1)
            cur_val = progress_counter.value
            pbar.update(cur_val - last_val)
            last_val = cur_val
        
        for p, _ in processes:
            p.join()
        
        final_val = progress_counter.value
        if final_val > last_val:
            pbar.update(final_val - last_val)
    
    print(f"Merging results for {dataset_name}...")
    all_results = []
    for p, part_path in processes:
        if os.path.exists(part_path):
            try:
                with open(part_path, 'r', encoding='utf-8') as f:
                    part_results = json.load(f)
                    all_results.extend(part_results)
                os.remove(part_path)
            except Exception as e:
                print(f"Error reading {part_path}: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"{dataset_name} processing completed: {len(all_results)} items saved to {output_file}")

def main(config_path, device="cuda" if torch.cuda.is_available() else "cpu",
         write_textgrid=False, textgrid_dir=DEFAULT_TEXTGRID_DIR, audio_root=DEFAULT_AUDIO_ROOT):
    config = json.load(open(config_path))

    train_data = json.load(open(config['data']['raw_train_path']))
    dev_data = json.load(open(config['data']['raw_dev_path']))
    test_data = json.load(open(config['data']['raw_test_path']))

    print(f"Raw train data length: {len(train_data)} items")
    print(f"Raw dev data length: {len(dev_data)} items")
    print(f"Raw test data length: {len(test_data)} items")
    if write_textgrid:
        print(f"TextGrid: enabled -> {textgrid_dir} (derived by replacing the {audio_root} prefix in audio paths)")
    
    dev_output_file = config['data']['valid_data']
    test_output_file = config['data']['test_data']
    train_output_file = config['data']['train_data']

    for ofile in [dev_output_file, test_output_file, train_output_file]:
        os.makedirs(os.path.dirname(ofile), exist_ok=True)
    
    print("\n=== Processing Development Set ===")
    process_dataset_multiprocess(dev_data, config, dev_output_file, "dev", device,
                                 write_textgrid, textgrid_dir, audio_root)

    print("\n=== Processing Test Set ===")
    process_dataset_multiprocess(test_data, config, test_output_file, "test", device,
                                 write_textgrid, textgrid_dir, audio_root)

    print("\n=== Processing Training Set ===")
    process_dataset_multiprocess(train_data, config, train_output_file, "train", device,
                                 write_textgrid, textgrid_dir, audio_root)
    
    print("\nAll processing completed!")
    print(f"Training set saved to: {train_output_file}")
    print(f"Development set saved to: {dev_output_file}")
    print(f"Test set saved to: {test_output_file}")

if __name__ == "__main__":

    args = parse_args()
    main(args.config, device=args.device, write_textgrid=args.write_textgrid,
         textgrid_dir=args.textgrid_dir, audio_root=args.audio_root)
