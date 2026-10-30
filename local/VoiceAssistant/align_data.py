"""
VoiceAssistant dataset CTC alignment script (multi-GPU, multi-process).
Only fixes the encoder checkpoint's s2p_model. prefix and the n_classes->num_classes issue;
all other logic reuses the existing code from ctc_model and data_prep_multi_proc.py.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python align_data.py
    CUDA_VISIBLE_DEVICES=4,5,6,7 python align_data.py --procs_per_gpu 2
    python align_data.py --gpus 4,5,6,7
    python align_data.py --write_textgrid   # additionally write Praat TextGrid

    After enabling --write_textgrid, each audio's TextGrid path is obtained by direct string replacement of the wav path:
    replace the --audio_root prefix with --textgrid_dir, and change .wav to .TextGrid, e.g.
        /mnt/nas4_workspace1/HanwenLiu/VoiceAssistant400k/wavs/rlhf/rlhf_99826.wav
     -> /mnt/workspace/liuhw/datasets/voiceassistant/textgrid/rlhf/rlhf_99826.TextGrid
    Each TextGrid contains two IntervalTier layers: words (word-level) and phones (phoneme/unit-level).
"""
import json
import torch
import os
import sys
import math
import time
import argparse
import soundfile as sf
from collections import OrderedDict
from typing import List
from tqdm import tqdm
import multiprocessing as mp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.nac import build_nac
from ctc_model.ctc_align import align_text_logits_and_speech_tokens, build_textgrid
from ctc_model import encoder as encoder_module
from ctc_model import encoder_loader
from ctc_model.encoder_loader import load_encoder_input_for_ctc, _load_checkpoint
from ctc_model.tokenizer import load_tknz

DEFAULT_CONFIG = "/mnt/workspace/liuhw/ctc_tts/config/monophone/single-speaker/o21_single_speaker_ctc_tts_l.json"
DEFAULT_PROCS_PER_GPU = 1
DEFAULT_TEXTGRID_DIR = "/mnt/nas4_workspace1/HanwenLiu/VoiceAssistant400k/textgrid"
# Root directory of the original wav; when writing TextGrid, this prefix is replaced with --textgrid_dir
DEFAULT_AUDIO_ROOT = "/mnt/nas4_workspace1/HanwenLiu/VoiceAssistant400k/wavs"


def build_textgrid_file_path(textgrid_dir, audio_path, audio_root=DEFAULT_AUDIO_ROOT):
    """Convert a wav path to a TextGrid path by string replacement: replace the root prefix + change extension."""
    replaced = audio_path.replace(audio_root, textgrid_dir, 1)
    return os.path.splitext(replaced)[0] + ".TextGrid"


def patch_encoder_loader():
    """Monkey-patch encoder_loader to fix the s2p_model. prefix issue."""
    def _extract_encoder_state_dict_fixed(path_or_checkpoint):
        checkpoint = _load_checkpoint(path_or_checkpoint)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        normalized = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]
            if key.startswith("s2p_model."):
                key = key[len("s2p_model."):]
            key = key.replace("decoder.", "predictor.", 1).replace("joint.", "joiner.", 1)
            key = key.replace("am.", "encoder.", 1)
            key = key.replace(".mhsencoder.", ".mhsam.")
            if key.startswith("encoder."):
                key = key[len("encoder."):]
            normalized[key] = value
        return normalized
    encoder_loader.extract_encoder_state_dict = _extract_encoder_state_dict_fixed


def load_encoder(cfg, device):
    """Load the encoder, handling n_classes->num_classes."""
    encoder_cfg = cfg["encoder"]
    encoder_cls = getattr(encoder_module, encoder_cfg["type"])
    kwargs = dict(encoder_cfg["kwargs"])
    if "n_classes" in kwargs and "num_classes" not in kwargs:
        kwargs["num_classes"] = kwargs.pop("n_classes")
    model = encoder_cls(**kwargs)
    state_dict = encoder_loader.extract_encoder_state_dict(encoder_cfg["model_path"])
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def format_units_text(tokenizer, tokenization) -> str:
    vocab = tokenizer.dump_vocab()
    parts: List[str] = []
    cursor = 0
    for seg_idx, seg_len in enumerate(tokenization.segment_lengths):
        seg_ids = tokenization.ctc_token_ids[cursor:cursor + seg_len]
        cursor += seg_len
        if seg_ids:
            parts.extend(vocab[int(tid)] for tid in seg_ids)
        if seg_idx < len(tokenization.separators):
            parts.append(tokenization.separators[seg_idx])
    return " ".join(parts)


def process_worker(proc_id, data_subset, config, output_path, local_gpu_id, progress_counter,
                   write_textgrid=False, textgrid_dir=DEFAULT_TEXTGRID_DIR, audio_root=DEFAULT_AUDIO_ROOT):
    """Child process function; local_gpu_id is the local CUDA index (0,1,2...)."""
    try:
        patch_encoder_loader()
        torch.cuda.set_device(local_gpu_id)
        device_str = f"cuda:{local_gpu_id}"

        print(f"[Process {proc_id}] Using cuda:{local_gpu_id}")
        nac = build_nac(config, device=device_str)
        ctc_encoder = load_encoder(config, device=device_str)
        tokenizer = load_tknz(config["tokenizer"]["file"])
        print(f"[Process {proc_id}] Model loaded")

        processed = []
        errors = 0
        textgrid_errors = 0
        created_textgrid_dirs = set()
        for item in data_subset:
            audio_path = os.path.join(item["speech_folder"], item["speech_file"])
            try:
                if not os.path.exists(audio_path):
                    errors += 1
                    progress_counter.value += 1
                    continue

                speech_tokens = nac.encode(audio_path).tolist()[0][0]
                inputs, lengths = load_encoder_input_for_ctc(audio_path, config, device=device_str)
                with torch.no_grad():
                    logits, output_lengths = ctc_encoder(inputs, lengths)
                valid_len = int(output_lengths[0].item())
                logits_valid = logits[0, :valid_len]

                result = align_text_logits_and_speech_tokens(
                    text=item["units_label"],
                    logits=logits_valid,
                    tokenizer=tokenizer,
                    speech_token_ids=speech_tokens,
                    config=config,
                )
                seqs_str = " ".join(map(str, result.seqs_sequence))
                units_str = format_units_text(tokenizer, result.tokenization)

                if write_textgrid:
                    try:
                        duration = sf.info(audio_path).duration
                        textgrid = build_textgrid(result, tokenizer, config, audio_duration=duration)
                        textgrid_path = build_textgrid_file_path(textgrid_dir, audio_path, audio_root)
                        textgrid_subdir = os.path.dirname(textgrid_path)
                        if textgrid_subdir not in created_textgrid_dirs:
                            os.makedirs(textgrid_subdir, exist_ok=True)
                            created_textgrid_dirs.add(textgrid_subdir)
                        with open(textgrid_path, "w", encoding="utf-8") as f:
                            f.write(textgrid)
                    except Exception as e:
                        textgrid_errors += 1
                        if textgrid_errors <= 5:
                            print(f"[Process {proc_id}] TextGrid write failed {item.get('id','?')}: {e}")

                processed.append({
                    "id": item["id"],
                    "audio_path": audio_path,
                    "text": item["units_label"],
                    "units": units_str,
                    "seqs": seqs_str,
                })
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"[Process {proc_id}] Error {item.get('id','?')}: {e}")
            finally:
                progress_counter.value += 1

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        print(f"[Process {proc_id}] Done: {len(processed)} succeeded, {errors} errors, {textgrid_errors} TextGrid failures")
    except Exception as e:
        print(f"[Process {proc_id}] Fatal error: {e}")
        import traceback; traceback.print_exc()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)


def process_dataset(data_list, config, output_file, name, num_gpus, procs_per_gpu,
                    write_textgrid=False, textgrid_dir=DEFAULT_TEXTGRID_DIR, audio_root=DEFAULT_AUDIO_ROOT):
    if not data_list:
        print(f"  {name}: no data"); return
    # Local GPU index: 0,1,...,num_gpus-1; each GPU gets procs_per_gpu processes
    gpu_assign = []
    for local_id in range(num_gpus):
        for _ in range(procs_per_gpu):
            gpu_assign.append(local_id)
    total_procs = min(len(data_list), len(gpu_assign))
    gpu_assign = gpu_assign[:total_procs]

    print(f"  {name}: {len(data_list)} items, {total_procs} processes, {num_gpus} GPUs, {procs_per_gpu} processes per GPU")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    counter = manager.Value("i", 0)

    chunk_size = math.ceil(len(data_list) / total_procs)
    processes = []
    for i in range(total_procs):
        chunk = data_list[i * chunk_size:(i + 1) * chunk_size]
        if not chunk: continue
        part_path = output_file.replace(".json", f"_part{i}.json")
        p = mp.Process(target=process_worker,
                       args=(i, chunk, config, part_path, gpu_assign[i], counter,
                             write_textgrid, textgrid_dir, audio_root))
        processes.append((p, part_path))
        p.start()

    with tqdm(total=len(data_list), desc=f"  {name}") as pbar:
        last = 0
        while any(p[0].is_alive() for p in processes):
            time.sleep(2)
            cur = counter.value
            pbar.update(cur - last); last = cur
        for p, _ in processes: p.join()
        cur = counter.value
        if cur > last: pbar.update(cur - last)

    print(f"  Merging {name}...")
    all_results = []
    for p, pp in processes:
        if os.path.exists(pp):
            try:
                with open(pp, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))
                os.remove(pp)
            except Exception as e:
                print(f"  Failed to read {pp}: {e}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  {name} done: {len(all_results)} items -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="CTC alignment for VoiceAssistant")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated physical GPU ids. If not specified, uses CUDA_VISIBLE_DEVICES or all GPUs")
    parser.add_argument("--procs_per_gpu", type=int, default=DEFAULT_PROCS_PER_GPU)
    parser.add_argument("--write_textgrid", action="store_true",
                        help="Additionally write a .TextGrid (words + phones layers) for each audio")
    parser.add_argument("--textgrid_dir", type=str, default=DEFAULT_TEXTGRID_DIR,
                        help="Root directory to save TextGrid; replaces the --audio_root prefix and then appends the relative path")
    parser.add_argument("--audio_root", type=str, default=DEFAULT_AUDIO_ROOT,
                        help="Root directory prefix of the original wav; replaced with --textgrid_dir when writing TextGrid")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.gpus:
        # User explicitly specified GPUs; set CUDA_VISIBLE_DEVICES
        gpu_list = [int(x.strip()) for x in args.gpus.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
        num_gpus = len(gpu_list)
        print(f"Using GPUs: {gpu_list} (physical ids)")
    else:
        # Use CUDA_VISIBLE_DEVICES or auto-detect
        num_gpus = torch.cuda.device_count()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        print(f"Using GPUs: {visible} ({num_gpus} devices)")

    if num_gpus == 0:
        print("Error: no available GPU"); sys.exit(1)

    print(f"===== VoiceAssistant CTC Alignment =====")
    print(f"Num GPUs: {num_gpus}, procs per GPU: {args.procs_per_gpu}")
    if args.write_textgrid:
        print(f"TextGrid: enabled -> {args.textgrid_dir} (obtained by replacing the {args.audio_root} prefix in wav paths)")

    train_data = json.load(open(config["data"]["raw_train_path"]))
    dev_data = json.load(open(config["data"]["raw_dev_path"]))
    test_data = json.load(open(config["data"]["raw_test_path"]))
    print(f"  train: {len(train_data)}, dev: {len(dev_data)}, test: {len(test_data)}")

    for p in [config["data"]["valid_data"], config["data"]["test_data"], config["data"]["train_data"]]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    print(f"\n=== test ===")
    process_dataset(test_data, config, config["data"]["test_data"], "test", num_gpus, args.procs_per_gpu,
                    args.write_textgrid, args.textgrid_dir, args.audio_root)
    print(f"\n=== dev ===")
    process_dataset(dev_data, config, config["data"]["valid_data"], "dev", num_gpus, args.procs_per_gpu,
                    args.write_textgrid, args.textgrid_dir, args.audio_root)
    print(f"\n=== train ===")
    process_dataset(train_data, config, config["data"]["train_data"], "train", num_gpus, args.procs_per_gpu,
                    args.write_textgrid, args.textgrid_dir, args.audio_root)

    print(f"\n===== All Done =====")


if __name__ == "__main__":
    main()
