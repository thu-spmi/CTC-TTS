export CUDA_VISIBLE_DEVICES="4,5,6,7"
gpu_nums=4
# Remember to update the experiment directory path in the config file
config_path=/mnt/workspace/liuhw/ctc_tts/config/monophone/single-speaker/o21_single_speaker_ctc_tts_l.json
START_STAGE=-1
STOP_STAGE=-1

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start_stage)
      START_STAGE="$2"; shift 2 ;;
    --stop_stage)
      STOP_STAGE="$2"; shift 2 ;;
    --)
      shift; break ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Stage 0: Text normalization
if (( START_STAGE == 0 && STOP_STAGE >= 0 )); then
  echo "Stage 0: Text normalization..."
  # download librispeech from https://www.openslr.org/12/
  # download librispeech-pc from https://www.openslr.org/145/
  # download seed-tts test set from https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit
  cd local
  python prepare_librispeech_json.py
  cd ..

fi

# Stage 1: Generate pronunciation lexicon
if (( START_STAGE <= 1 && STOP_STAGE >= 1 )); then
  echo "Stage 1: Generate pronunciation lexicon..."
  cd local
  # Generate the pronunciation lexicon and the corresponding tokenizer. Skip this step if they are already prepared
  python generate_lexicon.py --config $config_path 
  cd ..
fi

# Stage 2: Phoneme-speech alignment
if (( START_STAGE <= 2 && STOP_STAGE >= 2 )); then
  echo "Stage 2: Phoneme-speech alignment..."
  cd local
  # Generate training sequences for phoneme-speech alignment
  python data_prep_multi_proc.py --config $config_path 
  cd ..
fi

# Stage 3: Training
if (( START_STAGE <= 3 && STOP_STAGE >= 3 )); then
  echo "Stage 3: Training..."
  torchrun --standalone --nproc_per_node=$gpu_nums train_gpt2.py \
      --config $config_path
fi

# Stage 4: Testing
if (( START_STAGE <= 4 && STOP_STAGE >= 4 )); then
  echo "Stage 4: Testing..."
  # Multi-speaker cross and continuation tests
  python test/test_multi_speaker.py --config $config_path

  # Streaming cross and single-speaker tests
  # python test/test_streaming_inferencer.py --config $config_path

fi
