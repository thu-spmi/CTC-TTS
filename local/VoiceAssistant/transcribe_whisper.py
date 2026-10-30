"""
Re-transcribe the audio files in VoiceAssistant400k using Whisper large-v3.
Since the original answer text and the speech decoded from answer_snac may not fully match semantically
(the text is usually longer than the speech content), use Whisper transcription to get text consistent with the actual speech.

The transcription results will be updated in metadata.json, with new fields added:
  - whisper_text: Whisper transcription text
  - text: keep the original answer text unchanged

Usage:
    # Use the faster-whisper backend (CTranslate2, faster, int8 quantization):
    python transcribe_whisper.py --gpus 4,5,6 --backend faster

    # Use the openai-whisper backend (original PyTorch, aligned with paper results):
    python transcribe_whisper.py --gpus 4,5,6 --backend openai

    # 2 processes per GPU:
    python transcribe_whisper.py --gpus 4,5,6 --procs_per_gpu 2

    # Test mode (only transcribe the first 100 items):
    python transcribe_whisper.py --gpus 4,5,6 --num_samples 100

    # Merge results (run after all processes finish):
    python transcribe_whisper.py --merge

    # Resume from checkpoint: re-run the same command after an interruption to continue automatically
"""
import os
import sys
import glob
import json
import time
import argparse
import subprocess
import re

# Ensure CUDA library paths such as cublas/cudnn are correct (required by faster-whisper/ctranslate2)
def _setup_cuda_libs():
    """Add the nvidia library directories from the conda environment to LD_LIBRARY_PATH."""
    conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
    nvidia_base = os.path.join(conda_prefix, "lib", "python3.10", "site-packages", "nvidia")
    lib_dirs = []
    if os.path.isdir(nvidia_base):
        for pkg in os.listdir(nvidia_base):
            lib_dir = os.path.join(nvidia_base, pkg, "lib")
            if os.path.isdir(lib_dir):
                lib_dirs.append(lib_dir)
    if lib_dirs:
        existing = os.environ.get('LD_LIBRARY_PATH', '')
        new_path = ':'.join(lib_dirs)
        if existing:
            new_path += ':' + existing
        os.environ['LD_LIBRARY_PATH'] = new_path

_setup_cuda_libs()

# ======== Config ========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = "/mnt/nas4_workspace1/HanwenLiu/VoiceAssistant400k"
META_FILE = "metadata.json"
META_SHARD_PATTERN = "whisper_shard_*.json"
LOG_DIR = os.path.join(SCRIPT_DIR, "logs_whisper")
WHISPER_MODEL = "large-v3"
BEAM_SIZE = 5
LANGUAGE = "en"
SAVE_EVERY = 200


def shard_progress_file(data_dir, rank):
    return os.path.join(data_dir, f"whisper_progress_shard{rank}.txt")


def shard_meta_file(data_dir, rank):
    return os.path.join(data_dir, f"whisper_shard_{rank}.json")


def get_shard_progress(data_dir, rank):
    p = shard_progress_file(data_dir, rank)
    if os.path.exists(p):
        with open(p, 'r') as f:
            return int(f.read().strip())
    return -1


def save_shard_progress(idx, data_dir, rank):
    p = shard_progress_file(data_dir, rank)
    with open(p, 'w') as f:
        f.write(str(idx))


def load_shard_results(data_dir, rank):
    p = shard_meta_file(data_dir, rank)
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_shard_results(results, data_dir, rank):
    p = shard_meta_file(data_dir, rank)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)


def merge_results(data_dir):
    """Merge the transcription results of all shards and update metadata.json."""
    print("Merging Whisper transcription results...")

    all_transcripts = {}
    shard_files = sorted(glob.glob(os.path.join(data_dir, META_SHARD_PATTERN)))
    for sf in shard_files:
        with open(sf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_transcripts.update(data)
        print(f"  {os.path.basename(sf)}: {len(data)} items")
    print(f"  Total transcribed: {len(all_transcripts)} items")

    meta_path = os.path.join(data_dir, META_FILE)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"  Total metadata entries: {len(metadata)}")

    updated = 0
    missing = 0
    for item in metadata:
        if item['id'] in all_transcripts:
            item['whisper_text'] = all_transcripts[item['id']]
            updated += 1
        else:
            missing += 1

    backup_path = meta_path + ".bak"
    if not os.path.exists(backup_path):
        os.rename(meta_path, backup_path)
        print(f"  Original file backed up as: {os.path.basename(backup_path)}")

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Update complete: {updated} items had whisper_text added, {missing} items missing transcripts")
    print(f"Output file: {meta_path}")


def transcribe_faster_whisper(model, audio_path):
    """Transcribe using faster-whisper."""
    segments, info = model.transcribe(
        audio_path, language=LANGUAGE, beam_size=BEAM_SIZE,
        vad_filter=False, condition_on_previous_text=True,
        initial_prompt="The following is a clear and properly punctuated English sentence from a voice assistant.",
    )
    return ' '.join([seg.text.strip() for seg in segments]).strip()


def transcribe_openai_whisper(model, audio_path):
    """Transcribe using openai-whisper (original PyTorch), aligned with the paper standard."""
    result = model.transcribe(
        audio_path, language=LANGUAGE, beam_size=BEAM_SIZE,
        condition_on_previous_text=True,
        initial_prompt="The following is a clear and properly punctuated English sentence from a voice assistant.",
    )
    return result['text'].strip()


def run_worker(rank, world_size, gpu_id, data_dir, num_samples, model_size, backend, compute_type):
    """A single worker process: load the Whisper model and transcribe the assigned audio."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    log_path = os.path.join(LOG_DIR, f"whisper_{backend}_gpu{gpu_id}_rank{rank}.log")
    log_f = open(log_path, 'a', buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f"=== Whisper transcription (shard {rank}/{world_size}, GPU {gpu_id}, backend={backend}) ===")

    meta_path = os.path.join(data_dir, META_FILE)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    all_items = metadata
    if num_samples > 0:
        all_items = all_items[:num_samples]
    shard_items = [item for idx, item in enumerate(all_items) if idx % world_size == rank]
    shard_total = len(shard_items)
    log(f"Total {len(all_items)} items, current shard {rank} assigned {shard_total} items")

    results = load_shard_results(data_dir, rank)
    last_pos = get_shard_progress(data_dir, rank)
    start_pos = last_pos + 1
    if start_pos > 0:
        log(f"Continuing from item {start_pos} in the shard ({len(results)} items already done)")

    # Load the model
    t0 = time.time()
    if backend == 'faster':
        from faster_whisper import WhisperModel
        log(f"Loading faster-whisper {model_size} ({compute_type})...")
        model = WhisperModel(model_size, device='cuda', device_index=0, compute_type=compute_type)
        transcribe_fn = transcribe_faster_whisper
    elif backend == 'openai':
        import whisper
        log(f"Loading openai-whisper {model_size}...")
        model = whisper.load_model(model_size, device='cuda')
        transcribe_fn = transcribe_openai_whisper
    else:
        raise ValueError(f"Unknown backend: {backend}")
    log(f"Model loaded, took {time.time()-t0:.1f}s")

    t_start = time.time()
    errors = 0
    processed = 0

    for shard_pos in range(start_pos, shard_total):
        item = shard_items[shard_pos]
        item_id = item['id']
        audio_rel = item['audio_path']
        audio_abs = os.path.join(data_dir, audio_rel)

        if item_id in results:
            if (shard_pos + 1) % SAVE_EVERY == 0:
                save_shard_results(results, data_dir, rank)
                save_shard_progress(shard_pos, data_dir, rank)
            continue

        if not os.path.exists(audio_abs):
            log(f"  [skip] audio file does not exist: {audio_abs}")
            results[item_id] = ""
            errors += 1
            continue

        try:
            text = transcribe_fn(model, audio_abs)
            results[item_id] = text
        except Exception as e:
            errors += 1
            log(f"  [error] {item_id}: {e}")
            results[item_id] = ""
            if errors <= 10:
                import traceback
                log(traceback.format_exc())

        processed += 1

        if processed % 50 == 0 or shard_pos == shard_total - 1:
            elapsed = time.time() - t_start
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (shard_total - shard_pos - 1) / speed if speed > 0 else 0
            log(f"[shard{rank}|gpu{gpu_id}] [{shard_pos+1}/{shard_total}] "
                f"processed {processed} items, speed {speed:.1f} items/s, "
                f"ETA {eta/3600:.1f}h, errors {errors}")

        if (shard_pos + 1) % SAVE_EVERY == 0:
            save_shard_results(results, data_dir, rank)
            save_shard_progress(shard_pos, data_dir, rank)

    save_shard_results(results, data_dir, rank)
    save_shard_progress(shard_total - 1, data_dir, rank)
    total_time = time.time() - t_start
    log(f"[shard{rank}|gpu{gpu_id}] === done! === transcribed {len(results)} items, "
        f"errors {errors} items, took {total_time/60:.1f} minutes")
    log_f.close()


def launch_distributed(gpu_list, procs_per_gpu, data_dir, num_samples, model_size, backend, compute_type):
    """Launch multiple worker subprocesses using subprocess."""
    world_size = len(gpu_list) * procs_per_gpu
    script_path = os.path.abspath(__file__)
    python_exe = sys.executable

    print(f"=== Whisper multi-GPU transcription launcher ===")
    print(f"Backend: {backend}")
    print(f"GPU list: {gpu_list}")
    print(f"Processes per GPU: {procs_per_gpu}")
    print(f"Total processes: {world_size}")
    print(f"Model: {model_size}" + (f" ({compute_type})" if backend == 'faster' else ""))
    print(f"Data directory: {data_dir}")
    print(f"Log directory: {LOG_DIR}")
    print()

    os.makedirs(LOG_DIR, exist_ok=True)

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
                "--data_dir", data_dir,
                "--num_samples", str(num_samples),
                "--model_size", model_size,
                "--backend", backend,
                "--compute_type", compute_type,
            ]

            log_path = os.path.join(LOG_DIR, f"whisper_{backend}_gpu{gpu_id}_rank{rank}.log")
            log_f = open(log_path, 'w', buffering=1)
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                 cwd=SCRIPT_DIR)
            processes.append((rank, gpu_id, p, log_f))
            print(f"Starting shard {rank} (GPU {gpu_id}, proc {proc_idx}), PID={p.pid}, log={log_path}")
            rank += 1

    print(f"\nAll {len(processes)} processes have been started.")
    print(f"Check progress: tail -f {LOG_DIR}/whisper_{backend}_gpu*_rank*.log")
    print()

    try:
        for rank_id, gpu_id, p, log_f in processes:
            p.wait()
            if p.returncode == 0:
                print(f"[shard{rank_id}|gpu{gpu_id}] completed normally")
            else:
                print(f"[shard{rank_id}|gpu{gpu_id}] exited abnormally, code={p.returncode}")
            log_f.close()
    except KeyboardInterrupt:
        print("\nInterrupt signal received, terminating all processes...")
        for _, _, p, log_f in processes:
            p.terminate()
        for _, _, p, log_f in processes:
            p.wait()
            log_f.close()
        print("All processes terminated. Re-run the command to resume from checkpoint.")
        return

    print("\nAll processes finished! Run the following command to merge the results into metadata.json:")
    print(f"  python {script_path} --merge")


def main():
    parser = argparse.ArgumentParser(description="Whisper large-v3 transcription for VoiceAssistant400k")
    parser.add_argument("--gpus", type=str, default=None,
                        help="GPU indices to use, comma-separated, e.g. '4,5,6'")
    parser.add_argument("--procs_per_gpu", type=int, default=1,
                        help="Number of processes to launch per GPU (default 1)")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Only transcribe the first N items (0=all, for testing)")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_size", type=str, default=WHISPER_MODEL,
                        help="Whisper model size (default large-v3)")
    parser.add_argument("--backend", type=str, default="faster",
                        choices=["faster", "openai"],
                        help="Inference backend: faster (CTranslate2, fast) or openai (original PyTorch, aligned with paper)")
    parser.add_argument("--compute_type", type=str, default="int8",
                        help="faster-whisper compute precision (default int8, options float16/float32/int8)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all shard transcription results into metadata.json")
    # worker internal parameters
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--world_size", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.merge:
        merge_results(args.data_dir)
        return

    if args.worker:
        run_worker(
            rank=args.rank, world_size=args.world_size, gpu_id=args.gpu,
            data_dir=args.data_dir, num_samples=args.num_samples,
            model_size=args.model_size, backend=args.backend,
            compute_type=args.compute_type,
        )
        return

    if args.gpus is None:
        parser.print_help()
        print("\nError: please use --gpus to specify GPUs, e.g. --gpus 4,5,6")
        sys.exit(1)

    gpu_list = [int(x.strip()) for x in args.gpus.split(',')]
    launch_distributed(
        gpu_list=gpu_list, procs_per_gpu=args.procs_per_gpu,
        data_dir=args.data_dir, num_samples=args.num_samples,
        model_size=args.model_size, backend=args.backend,
        compute_type=args.compute_type,
    )


if __name__ == "__main__":
    main()
