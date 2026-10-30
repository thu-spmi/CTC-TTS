"""
Download the VoiceAssistant-400K dataset from hf-mirror.com and export it as CSV.
The original datasets-server.huggingface.co is inaccessible in China (DNS pollution / connection timeout),
so we switch to the hf-mirror.com mirror to download the parquet files directly.
"""
import os
import sys
import time
import requests
import pyarrow.parquet as pq
import pandas as pd

# ======== Config ========
MIRROR_BASE = "https://hf-mirror.com/datasets/gpt-omni/VoiceAssistant-400K/resolve/main/data"
TOTAL_FILES = 325          # total number of parquet shards
OUTPUT_FILE = "VoiceAssistant-400K.csv"
PROGRESS_FILE = "last_file.txt"   # records the index of the last completed file (0-based)
PARQUET_DIR = "parquet_cache"     # parquet cache directory (can be deleted after download)
# If you don't need to keep the parquet files, set to True to save disk space (~160GB)
DELETE_PARQUET_AFTER_CONVERT = True
# CSV output columns
OUTPUT_COLUMNS = ["split_name", "index", "answer", "answer_snac"]
# Request timeout (seconds)
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300


def get_last_file():
    """Read the index of the last downloaded file (0-based); returns -1 to start from the beginning."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return -1
    return -1


def save_last_file(idx):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(idx))


def download_parquet(file_idx, dest_path, max_retries=5):
    """Download a single parquet file, supporting resume and retry."""
    url = f"{MIRROR_BASE}/train-{file_idx:05d}-of-{TOTAL_FILES:05d}.parquet"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    # Check the already-downloaded part (resume)
    downloaded = 0
    if os.path.exists(dest_path):
        downloaded = os.path.getsize(dest_path)
        if downloaded > 0:
            headers['Range'] = f'bytes={downloaded}-'

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url, headers=headers, stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            # On resume, the server should return 206; a full download returns 200
            if downloaded > 0 and resp.status_code == 200:
                # Server does not support range; re-download from scratch
                downloaded = 0
                headers.pop('Range', None)
                resp = requests.get(
                    url, headers=headers, stream=True,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
                )
            resp.raise_for_status()

            mode = 'ab' if downloaded > 0 and resp.status_code == 206 else 'wb'
            total_size = downloaded + int(resp.headers.get('content-length', 0))

            with open(dest_path, mode) as f:
                downloaded_now = 0
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded_now += len(chunk)
                        if total_size > 0:
                            pct = (downloaded + downloaded_now) / total_size * 100
                            print(f"\r  Downloading: {downloaded + downloaded_now}/{total_size} bytes ({pct:.1f}%)", end='', flush=True)
            print()  # newline
            return True
        except Exception as e:
            print(f"\n  Download failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Waiting {wait} seconds before retry...")
                time.sleep(wait)
                # Update the downloaded size for resume
                if os.path.exists(dest_path):
                    downloaded = os.path.getsize(dest_path)
                    headers['Range'] = f'bytes={downloaded}-'
            else:
                print("  Maximum retries reached, skipping this file")
                return False


def parquet_to_csv(parquet_path, csv_path, write_header, total_written):
    """Read a parquet file and append it to the CSV. Returns the number of rows written."""
    pf = pq.ParquetFile(parquet_path)
    rows_written = 0
    for row_group_idx in range(pf.num_row_groups):
        table = pf.read_row_group(row_group_idx, columns=OUTPUT_COLUMNS)
        df = table.to_pandas()
        # Clean up newlines and separators
        for col in ['answer', 'answer_snac']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[\r\n|]', ' ', regex=True)
        df.to_csv(csv_path, sep='|', index=False, header=write_header and rows_written == 0,
                  mode='a', encoding='utf-8')
        rows_written += len(df)
        write_header = False  # only write the header the first time
    return rows_written


def main():
    os.makedirs(PARQUET_DIR, exist_ok=True)

    last_file = get_last_file()
    start_file = last_file + 1
    total_written = 0

    # If starting from the beginning, write the header first
    write_header = (start_file == 0)
    if write_header and os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    print(f"=== VoiceAssistant-400K Downloader (hf-mirror) ===")
    print(f"Starting from file {start_file} (out of {TOTAL_FILES})")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    for i in range(start_file, TOTAL_FILES):
        parquet_path = os.path.join(PARQUET_DIR, f"train-{i:05d}-of-{TOTAL_FILES:05d}.parquet")
        print(f"[{i+1}/{TOTAL_FILES}] Downloading train-{i:05d}-of-{TOTAL_FILES:05d}.parquet ...")

        t0 = time.time()
        ok = download_parquet(i, parquet_path)
        if not ok:
            print(f"  File {i} download failed, skipping")
            continue

        dl_time = time.time() - t0
        file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"  Download complete: {file_size_mb:.1f} MB, took {dl_time:.1f}s")

        # Convert and append to CSV
        print(f"  Converting to CSV ...")
        t1 = time.time()
        rows = parquet_to_csv(parquet_path, OUTPUT_FILE, write_header, total_written)
        cv_time = time.time() - t1
        total_written += rows
        write_header = False
        print(f"  Wrote {rows} rows, took {cv_time:.1f}s | total {total_written} rows")

        # Record progress
        save_last_file(i)

        # Delete parquet to save space
        if DELETE_PARQUET_AFTER_CONVERT:
            os.remove(parquet_path)

    print(f"\n=== All Done! Total {total_written} rows written to {OUTPUT_FILE} ===")
    # Clean up the progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


if __name__ == "__main__":
    main()
