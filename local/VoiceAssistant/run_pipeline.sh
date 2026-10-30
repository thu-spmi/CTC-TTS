#!/usr/bin/env bash
# VoiceAssistant data preparation pipeline.
#
# Stage 0: Download the VoiceAssistant-400K dataset (keep only answer text and answer_snac speech tokens) -> CSV
# Stage 1: SNAC decoding (https://github.com/hubertsiuzdak/snac) -> 24kHz wavs + metadata.json
# Stage 2: Whisper large-v3 transcription -> add whisper_text to metadata.json
# Stage 3: Text cleaning -> voiceassistant.json
# Stage 4: G2P pronunciation lexicon generation + tokenizer expansion -> tokenizer/
# Stage 5: Split into train/dev/test -> raw_data/
# Stage 6: CTC alignment (align phoneme tokens with WavTokenizer speech tokens) -> aligned_data/
#
# Usage:
#   bash run_pipeline.sh --start_stage 3 --stop_stage 6
#   bash run_pipeline.sh --gpus 4,5,6,7 --procs_per_gpu 2
#   PYTHON=/path/to/python GPU_LIST=0,1 bash run_pipeline.sh --start_stage 6
#
# Notes:
#   - Stage 1/2 are multi-GPU, multi-process launchers; the script runs merge automatically after all workers finish.
#   - All stages support resuming: just rerun the same command after an interruption.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ======== Configurable options ========
# Python interpreter used to run each stage script (defaults to the current conda env's python;
# torch / snac / openai-whisper etc. must already be installed)
PYTHON="${PYTHON:-python}"
# Physical GPU ids to use, comma-separated; Stage 1/2/6 use it to distribute multi-GPU, multi-process work
GPU_LIST="${GPU_LIST:-4,5,6,7}"
# Number of processes to start per GPU (Stage 1/2/6)
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"

# Master config file for training/alignment; Stage 6 reads encoder, tokenizer, nac and data paths from it
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/monophone/single-speaker/o21_single_speaker_ctc_tts_l.json}"
# VoiceAssistant-400K CSV used by Stage 0 download and Stage 1 decoding
CSV_PATH="${CSV_PATH:-${SCRIPT_DIR}/VoiceAssistant-400K.csv}"
# Output directory for Stage 1/2/3: stores wavs/ and metadata.json
DATA_DIR="${DATA_DIR:-/mnt/nas4_workspace1/HanwenLiu/VoiceAssistant400k}"
# Stage 3 output and Stage 4/5 input: the cleaned text json
CLEANED_JSON="${CLEANED_JSON:-/mnt/workspace/liuhw/VoiceAssistant/voiceassistant.json}"
# Stage 5 output directory: split train/dev/test.json (must match config's raw_*_path)
RAW_DIR="${RAW_DIR:-/mnt/workspace/liuhw/datasets/voiceassistant/raw_data}"
# Stage 4 output directory: lexicon.txt / phone_list.txt / tokenizer.tknz (must match config's tokenizer path)
TOKENIZER_DIR="${TOKENIZER_DIR:-/mnt/workspace/liuhw/datasets/voiceassistant/tokenizer}"
# phonetisaurus G2P model used by Stage 4
G2P_MODEL="${G2P_MODEL:-${PROJECT_ROOT}/config/model.fst}"
# Base tokenizer used for Stage 4 expansion (the tknz provided by huggingface can be used)
BASE_TKNZ="${BASE_TKNZ:-/mnt/workspace/liuhw/VoiceAssistant/dict_librispeech_own_replace/tokenizer.tknz}"

# Stage 2 transcription backend: openai (original PyTorch) or faster (CTranslate2)
WHISPER_BACKEND="${WHISPER_BACKEND:-openai}"
# Whisper model size used by Stage 2
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"

# Stage 5 target duration for the dev set (hours)
DEV_HOURS="${DEV_HOURS:-50.0}"
# Stage 5 target duration for the test set (hours)
TEST_HOURS="${TEST_HOURS:-5.0}"
# Random seed for the Stage 5 random split
SPLIT_SEED="${SPLIT_SEED:-42}"


START_STAGE=1
STOP_STAGE=6

# ======== Argument parsing ========
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start_stage) START_STAGE="$2"; shift 2 ;;
    --stop_stage)  STOP_STAGE="$2"; shift 2 ;;
    --gpus)        GPU_LIST="$2"; shift 2 ;;
    --procs_per_gpu) PROCS_PER_GPU="$2"; shift 2 ;;
    --config)      CONFIG="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,20p' "$0"
      exit 0 ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

run_stage() {
  local stage="$1"
  if (( START_STAGE <= stage && STOP_STAGE >= stage )); then
    return 0
  fi
  return 1
}

echo "=================================================="
echo "VoiceAssistant data preparation pipeline"
echo "  Python        : ${PYTHON}"
echo "  GPU list      : ${GPU_LIST} (${PROCS_PER_GPU} processes per GPU)"
echo "  Config file   : ${CONFIG}"
echo "  Data dir      : ${DATA_DIR}"
echo "  Stage range   : ${START_STAGE} ~ ${STOP_STAGE}"
echo "=================================================="

cd "${SCRIPT_DIR}"

# ---------- Stage 0: Download dataset ----------
if run_stage 0; then
  echo ""
  echo ">>> [Stage 0] Downloading the VoiceAssistant-400K dataset (keep only answer text and answer_snac speech tokens) ..."
  "${PYTHON}" download_data.py
fi

# ---------- Stage 1: SNAC decoding ----------
if run_stage 1; then
  echo ""
  echo ">>> [Stage 1] Decoding SNAC to wav ..."
  "${PYTHON}" decode_snac.py \
    --gpus "${GPU_LIST}" \
    --procs_per_gpu "${PROCS_PER_GPU}" \
    --csv_path "${CSV_PATH}" \
    --output_dir "${DATA_DIR}"
  echo ">>> [Stage 1] Merging metadata shards ..."
  "${PYTHON}" decode_snac.py --merge_metadata --output_dir "${DATA_DIR}"
fi

# ---------- Stage 2: Whisper transcription ----------
if run_stage 2; then
  echo ""
  echo ">>> [Stage 2] Whisper transcription ..."
  "${PYTHON}" transcribe_whisper.py \
    --gpus "${GPU_LIST}" \
    --procs_per_gpu "${PROCS_PER_GPU}" \
    --data_dir "${DATA_DIR}" \
    --backend "${WHISPER_BACKEND}" \
    --model_size "${WHISPER_MODEL}"
  echo ">>> [Stage 2] Merging transcription results into metadata.json ..."
  "${PYTHON}" transcribe_whisper.py --merge --data_dir "${DATA_DIR}"
fi

# ---------- Stage 3: Text cleaning ----------
if run_stage 3; then
  echo ""
  echo ">>> [Stage 3] Cleaning whisper text -> voiceassistant.json ..."
  "${PYTHON}" prepare_voiceassistant_json.py \
    --data_dir "${DATA_DIR}" \
    --output "${CLEANED_JSON}"
fi

# ---------- Stage 4: Pronunciation lexicon + tokenizer ----------
if run_stage 4; then
  echo ""
  echo ">>> [Stage 4] Generating the pronunciation lexicon and tokenizer ..."
  "${PYTHON}" build_voiceassistant_lexicon.py \
    --input_json "${CLEANED_JSON}" \
    --g2p_model "${G2P_MODEL}" \
    --base_tknz "${BASE_TKNZ}" \
    --output_dir "${TOKENIZER_DIR}"
fi

# ---------- Stage 5: Split dataset ----------
if run_stage 5; then
  echo ""
  echo ">>> [Stage 5] Splitting train/dev/test ..."
  "${PYTHON}" split_data.py \
    --input "${CLEANED_JSON}" \
    --output_dir "${RAW_DIR}" \
    --dev_hours "${DEV_HOURS}" \
    --test_hours "${TEST_HOURS}" \
    --seed "${SPLIT_SEED}"
fi

# ---------- Stage 6: CTC alignment ----------
if run_stage 6; then
  echo ""
  echo ">>> [Stage 6] CTC alignment ..."
  "${PYTHON}" align_data.py \
    --config "${CONFIG}" \
    --gpus "${GPU_LIST}" \
    --procs_per_gpu "${PROCS_PER_GPU}"
fi

echo ""
echo "=================================================="
echo "pipeline finished (stage ${START_STAGE} ~ ${STOP_STAGE})"
echo "=================================================="
