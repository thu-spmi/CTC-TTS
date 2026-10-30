"""
Decode the answer_snac codes in VoiceAssistant-400K.csv into speech files (WAV).
Uses the SNAC 24kHz neural audio codec.

Usage:
    # Single GPU, single process:
    python decode_snac.py --gpus 0

    # Multiple GPUs (GPU 4,5,6), 1 process per GPU:
    python decode_snac.py --gpus 4,5,6

    # Multiple GPUs, 2 processes per GPU (6 processes total):
    python decode_snac.py --gpus 4,5,6 --procs_per_gpu 2

    # Test mode (only process the first 100 entries):
    python decode_snac.py --gpus 4,5,6 --num_samples 100

    # Merge metadata (run after all processes finish):
    python decode_snac.py --merge_metadata

    # Resume from checkpoint: just re-run the same command after an interruption to continue automatically
"""
import os
import sys
import glob
import json
import time
import argparse
import subprocess
import torch
import pandas as pd
import soundfile as sf
from snac import SNAC

# ======== Configuration ========
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VoiceAssistant-400K.csv")
OUTPUT_DIR = "/mnt/nas4_workspace1/HanwenLiu/VoiceAssistant400k"
AUDIO_SUBDIR = "wavs"
META_FILE = "metadata.json"
META_SHARD_PATTERN = "metadata_shard_*.json"
LOG_DIR = "logs"
SAMPLE_RATE = 24000
SNAC_MODEL_REPO = "hubertsiuzdak/snac_24khz"
_DEFAULT_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--hubertsiuzdak--snac_24khz/snapshots/d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"
)
SNAC_MODEL_PATH = _DEFAULT_CACHE if os.path.isdir(_DEFAULT_CACHE) else None
SAVE_EVERY = 100


def parse_snac_tokens(snac_str):
    """
    Parse an answer_snac string into a SNAC 3-layer codes tensor.
    Each group separated by # has 7 tokens, arranged as (see Mini-Omni2 reconstruct_tensors):
        [t1_0, t2_0, t3_0, t3_1, t2_1, t3_2, t3_3]
    This corresponds to the 24kHz model with vq_strides=[4,2,1] and length ratio 1:2:4.
    """
    groups = snac_str.split('#')
    groups = [g.strip() for g in groups if g.strip()]
    codes0, codes1, codes2 = [], [], []
    for g in groups:
        tokens = g.split()
        if len(tokens) != 7:
            raise ValueError(f"Expected 7 tokens per group, got {len(tokens)}: {g[:50]}...")
        t = [int(x) for x in tokens]
        codes0.append(t[0])
        codes1.append(t[1])
        codes1.append(t[4])
        codes2.append(t[2])
        codes2.append(t[3])
        codes2.append(t[5])
        codes2.append(t[6])
    codes = [
        torch.tensor(codes0, dtype=torch.long).unsqueeze(0),
        torch.tensor(codes1, dtype=torch.long).unsqueeze(0),
        torch.tensor(codes2, dtype=torch.long).unsqueeze(0),
    ]
    return codes


def shard_progress_file(output_dir, rank):
    return os.path.join(output_dir, f"decode_progress_shard{rank}.txt")


def shard_meta_file(output_dir, rank):
    return os.path.join(output_dir, f"metadata_shard_{rank}.json")


def get_shard_progress(output_dir, rank):
    p = shard_progress_file(output_dir, rank)
    if os.path.exists(p):
        with open(p, 'r') as f:
            return int(f.read().strip())
    return -1


def save_shard_progress(idx, output_dir, rank):
    p = shard_progress_file(output_dir, rank)
    with open(p, 'w') as f:
        f.write(str(idx))


def load_shard_metadata(output_dir, rank):
    p = shard_meta_file(output_dir, rank)
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_shard_metadata(metadata, output_dir, rank):
    p = shard_meta_file(output_dir, rank)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def merge_metadata(output_dir):
    """Merge all shard metadata into the final metadata.json."""
    print("Merging metadata shards...")
    all_meta = []
    shard_files = sorted(glob.glob(os.path.join(output_dir, META_SHARD_PATTERN)))
    for sf_path in shard_files:
        with open(sf_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_meta.extend(data)
        print(f"  {os.path.basename(sf_path)}: {len(data)} entries")
    all_meta.sort(key=lambda x: x['id'])
    meta_path = os.path.join(output_dir, META_FILE)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print(f"Merge complete: {len(all_meta)} entries -> {META_FILE}")
    return all_meta


def run_worker(rank, world_size, gpu_id, csv_path, output_dir, num_samples, model_path):
    """Main function for a single worker process."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = "cuda:0"

    log_path = os.path.join(LOG_DIR, f"decode_gpu{gpu_id}_rank{rank}.log")
    log_f = open(log_path, 'a', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f"=== SNAC speech decoding (shard {rank}/{world_size}, GPU {gpu_id}) ===")

    # Load model
    if model_path and os.path.isdir(model_path):
        model = SNAC.from_pretrained(model_path)
    else:
        if 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        model = SNAC.from_pretrained(SNAC_MODEL_REPO)
    model = model.to(device).eval()
    log(f"Model loaded, device: cuda:0 (physical GPU {gpu_id})")

    # Read CSV
    df = pd.read_csv(csv_path, sep='|')
    total = len(df)

    indices = list(range(total))
    if num_samples > 0:
        indices = [i for i in indices if i < num_samples]
    shard_indices = [i for i in indices if i % world_size == rank]
    shard_total = len(shard_indices)
    log(f"Total {total} entries, current shard {rank} assigned {shard_total} entries")

    last_shard_pos = get_shard_progress(output_dir, rank)
    start_shard_pos = last_shard_pos + 1
    metadata = load_shard_metadata(output_dir, rank) if start_shard_pos > 0 else []
    if start_shard_pos > 0:
        log(f"Resuming from entry {start_shard_pos} in shard ({start_shard_pos} entries already done)")

    audio_dir = os.path.join(output_dir, AUDIO_SUBDIR)
    os.makedirs(audio_dir, exist_ok=True)

    t_start = time.time()
    errors = 0

    for shard_pos in range(start_shard_pos, shard_total):
        i = shard_indices[shard_pos]
        row = df.iloc[i]
        split_name = str(row['split_name'])
        index_val = str(row['index'])
        answer = str(row['answer'])
        snac_str = str(row['answer_snac'])

        item_id = f"{split_name}_{index_val}"
        split_dir = os.path.join(audio_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        wav_rel_path = os.path.join(AUDIO_SUBDIR, split_name, f"{item_id}.wav")
        wav_abs_path = os.path.join(output_dir, wav_rel_path)

        if os.path.exists(wav_abs_path):
            metadata.append({
                "id": item_id, "split": split_name, "index": index_val,
                "text": answer, "audio_path": wav_rel_path, "duration": None,
            })
            if (shard_pos + 1) % SAVE_EVERY == 0:
                save_shard_metadata(metadata, output_dir, rank)
                save_shard_progress(shard_pos, output_dir, rank)
            continue

        try:
            codes = parse_snac_tokens(snac_str)
            codes = [c.to(device) for c in codes]
            with torch.inference_mode():
                audio = model.decode(codes)
            audio_np = audio.squeeze().cpu().numpy()
            sf.write(wav_abs_path, audio_np, SAMPLE_RATE)
            metadata.append({
                "id": item_id, "split": split_name, "index": index_val,
                "text": answer, "audio_path": wav_rel_path,
                "duration": round(len(audio_np) / SAMPLE_RATE, 3),
            })
        except Exception as e:
            errors += 1
            log(f"  [ERROR] entry {i} (id={item_id}): {e}")
            if errors <= 10:
                import traceback
                log(traceback.format_exc())
            continue

        done = shard_pos - start_shard_pos + 1
        if done % 50 == 0 or shard_pos == shard_total - 1:
            elapsed = time.time() - t_start
            speed = done / elapsed if elapsed > 0 else 0
            eta = (shard_total - shard_pos - 1) / speed if speed > 0 else 0
            log(f"[shard{rank}|gpu{gpu_id}] [{shard_pos+1}/{shard_total}] "
                f"processed {done} entries, speed {speed:.1f} entries/s, "
                f"ETA {eta/3600:.1f}h, errors {errors}")

        if (shard_pos + 1) % SAVE_EVERY == 0:
            save_shard_metadata(metadata, output_dir, rank)
            save_shard_progress(shard_pos, output_dir, rank)

    save_shard_metadata(metadata, output_dir, rank)
    save_shard_progress(shard_total - 1, output_dir, rank)
    total_time = time.time() - t_start
    log(f"[shard{rank}|gpu{gpu_id}] === Done! === processed {len(metadata)} entries, "
        f"{errors} errors, took {total_time/60:.1f} minutes")
    log_f.close()


def launch_distributed(gpu_list, procs_per_gpu, csv_path, output_dir, num_samples, model_path):
    """Launch multiple worker subprocesses using subprocess."""
    world_size = len(gpu_list) * procs_per_gpu
    script_path = os.path.abspath(__file__)
    python_exe = sys.executable

    print(f"=== SNAC multi-GPU decoding launcher ===")
    print(f"GPU list: {gpu_list}")
    print(f"Processes per GPU: {procs_per_gpu}")
    print(f"Total processes: {world_size}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {LOG_DIR}")
    print()

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    processes = []
    rank = 0
    for gpu_id in gpu_list:
        for proc_idx in range(procs_per_gpu):
            cmd = [
                python_exe, script_path,
                "--worker",
                "--rank", str(rank),
                "--world_size", str(world_size),
                "--gpu", str(gpu_id),
                "--csv_path", csv_path,
                "--output_dir", output_dir,
                "--num_samples", str(num_samples),
            ]
            if model_path:
                cmd.extend(["--model_path", model_path])

            log_path = os.path.join(LOG_DIR, f"decode_gpu{gpu_id}_rank{rank}.log")
            log_f = open(log_path, 'w', buffering=1)
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                 cwd=os.path.dirname(script_path))
            processes.append((rank, gpu_id, p, log_f))
            print(f"Started shard {rank} (GPU {gpu_id}, proc {proc_idx}), PID={p.pid}, log={log_path}")
            rank += 1

    print(f"\nAll {len(processes)} processes started.")
    print(f"To monitor progress: tail -f {LOG_DIR}/decode_gpu*_rank*.log")
    print()

    try:
        for rank_id, gpu_id, p, log_f in processes:
            p.wait()
            if p.returncode == 0:
                print(f"[shard{rank_id}|gpu{gpu_id}] finished normally")
            else:
                print(f"[shard{rank_id}|gpu{gpu_id}] exited abnormally, code={p.returncode}")
            log_f.close()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, terminating all processes...")
        for _, _, p, log_f in processes:
            p.terminate()
        for _, _, p, log_f in processes:
            p.wait()
            log_f.close()
        print("All processes terminated. You can re-run the command to resume from checkpoint.")
        return

    print("\nAll processes finished! Run the following command to merge metadata:")
    print(f"  python {script_path} --merge_metadata")


def main():
    parser = argparse.ArgumentParser(description="Decode SNAC tokens to WAV files")
    parser.add_argument("--gpus", type=str, default=None,
                        help="GPU indices to use, comma-separated, e.g. '4,5,6'")
    parser.add_argument("--procs_per_gpu", type=int, default=1,
                        help="Number of processes to launch per GPU (default 1)")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Only process the first N samples (0=all, for testing)")
    parser.add_argument("--csv_path", type=str, default=CSV_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model_path", type=str, default=SNAC_MODEL_PATH)
    parser.add_argument("--merge_metadata", action="store_true",
                        help="Merge all shard metadata into the final metadata.json")
    # Internal worker parameters (users do not need to specify these manually)
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--world_size", type=int, default=1,
                        help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=0,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Merge mode
    if args.merge_metadata:
        merge_metadata(args.output_dir)
        return

    # Worker mode (invoked by the launcher)
    if args.worker:
        run_worker(
            rank=args.rank,
            world_size=args.world_size,
            gpu_id=args.gpu,
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            model_path=args.model_path,
        )
        return

    # Launcher mode: --gpus must be specified
    if args.gpus is None:
        parser.print_help()
        print("\nError: please specify GPUs with --gpus, e.g. --gpus 4,5,6")
        sys.exit(1)

    gpu_list = [int(x.strip()) for x in args.gpus.split(',')]
    launch_distributed(
        gpu_list=gpu_list,
        procs_per_gpu=args.procs_per_gpu,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()

    # python decode_snac.py --gpus 4,5,6 --procs_per_gpu 2
    # After all processes finish, run the following command to merge metadata.json
    # python decode_snac.py --merge_metadata
