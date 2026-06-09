# CTC-TTS: LLM-Based Dual-Streaming Text-to-Speech with CTC Alignment

Official implementation of the Interspeech 2026 paper:
`CTC-TTS: LLM-Based Dual-Streaming Text-to-Speech with CTC Alignment`

- Paper: [arXiv PDF](https://arxiv.org/pdf/2602.19574)
- Demo samples: [Project page](https://ctctts.github.io/)

## Overview

This repository contains the code, training pipeline, and inference workflow for the Interspeech 2026 paper `CTC-TTS: LLM-Based Dual-Streaming Text-to-Speech with CTC Alignment`.

CTC-TTS builds joint text-speech sequences from CTC alignment results and trains a GPT-style speech generation model with a dual-streaming formulation. This repository includes data preparation, pronunciation lexicon construction, phoneme-speech alignment, model training, and evaluation scripts used by the paper.

## Paper Information

If you use this repository, please refer to the paper and demo page above for the full method description, experimental setup, and synthesized speech examples.

The main workflow currently corresponds to the five stages in `run.sh`:

- `stage 0`: Text normalization that generates `*.json` files for later processing
- `stage 1`: Generate the pronunciation lexicon and tokenizer
- `stage 2`: Phoneme-speech alignment that generates the aligned sequences required for training
- `stage 3`: Model training
- `stage 4`: Testing

## Environment Setup

Download [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) to ctc_tts/

Download [SpeechMOS](https://github.com/SynthAether/SpeechMOS_UTMOS) to ctc_tts/metrics

A Python 3.9 conda environment is recommended.

You can create it like this:

```bash
conda create -n ctc_tts python=3.9 -y
conda activate ctc_tts
pip install -r /mnt/workspace2/liuhw/ctc_tts/requirements.txt
```

Notes:

- install [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
- `requirements.txt` already contains most dependencies required for training and inference in this project. It is recommended to install `flash-attn` first, and then install the remaining dependencies.
- Versions of `torch`, `torchaudio`, and `flash-attn` should stay aligned with the current project environment. `torch` should be `>= 2.5`, since that version supports `torchaudio`'s `forced_align` function. See the `torchaudio` documentation for details.
- `flash-attn` is sensitive to the versions of `torch`, CUDA, and the build toolchain. See the official releases page: https://github.com/Dao-AILab/flash-attention/releases
- If `pip install -r requirements.txt` fails, first check whether the current `torch`/CUDA versions are compatible with `flash-attn`.
- After switching machines, changing the CUDA version, or recreating the environment, `flash-attn` may need to be reinstalled.



## Config Preparation

Before running, check that the paths in the config file are correct, especially:

- CTC model path
- NAC/WavTokenizer model path
- Tokenizer path
- Data path
- Experiment output path

The default config used by `run.sh` is:



## About Stage 1

The purpose of `stage 1` is to generate the pronunciation lexicon and the corresponding tokenizer.

However, this step is not always required for the current project.

This is because:

- An existing `tknz` file can be specified directly in `config`
- That `tknz` file must match the CTC model being used
- As long as the tokenizer file in the config is already usable, there is no need to regenerate the lexicon or tokenizer

In other words:

- If you already have a `tknz` file that matches the current CTC model, you can fill it in directly in the config
- In that case, after completing `stage 0`, you can skip `stage 1` and go directly to `stage 2`

## Workflow

### 1. Stage 0: Text Normalization

Run:

```bash
cd ctc_tts
bash run.sh --start_stage 0 --stop_stage 0
```

This stage calls:

```bash
cd local
python prepare_librispeech_json.py
```

The generated `*.json` files are mainly used for later lexicon generation or alignment preparation.

### 2. Stage 1: Generate the Lexicon and Tokenizer

If you do not already have a `tknz` file that matches the current CTC model, you can run:

```bash
cd ctc_tts
bash run.sh --start_stage 1 --stop_stage 1
```

If the config file already specifies a usable `tknz` file, this stage can be skipped.

### 3. Stage 2: Phoneme-Speech Alignment

If `stage 0` is already complete and the tokenizer is ready, you can run this stage directly:

```bash
cd ctc_tts
bash run.sh --start_stage 2 --stop_stage 2
```

This stage outputs aligned-sequence JSON files required for later model training.

## Training

Training corresponds to `stage 3`:

```bash
cd /mnt/workspace2/liuhw/ctc_tts
bash run.sh --start_stage 3 --stop_stage 3
```

The actual command executed in `run.sh` is:

```bash
torchrun --standalone --nproc_per_node=$gpu_nums train_gpt2.py --config $config_path
```

Before training, it is recommended to confirm:

- `CUDA_VISIBLE_DEVICES` in `run.sh`
- `gpu_nums`
- `config_path`
- `out.out_dir` in the config file

## Testing

Testing corresponds to `stage 4`:

```bash
cd /mnt/workspace2/liuhw/ctc_tts
bash run.sh --start_stage 4 --stop_stage 4
```

The current default test script is:

```bash
python test/test_multi_speaker.py --config $config_path
```

`run.sh` also keeps the entry for the streaming cross test; enable it as needed:

```bash
python test/test_streaming_inferencer.py --config $config_path
```


## Notes

- Whether `stage 1` is needed mainly depends on whether you already have a `tknz` file that matches the current CTC model
- If you switch to a different CTC model, you should usually verify again that the tokenizer still matches
- Before training starts, make sure all paths in the config file have been switched to locations accessible on the current machine
