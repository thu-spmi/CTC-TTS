import os
import sys
import json
import time
import threading
import asyncio
from queue import Queue
from typing import List, Dict, Tuple, Optional
import argparse

import numpy as np
import torch
import whisper
from torchmetrics import WordErrorRate, CharErrorRate

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from metrics.cal import clean_english_text, transcribe_audio, load_asr_model
from metrics.calculate_utmos import calculate_utmos_batch, calculate_utmos
from metrics.speaker_similarity import create_default_speaker_extractor, compute_speaker_similarity

from multi_speaker_inferencer import MultiSpeakerInferencer

import warnings
warnings.filterwarnings(
    "ignore",
    message="Support for mismatched key_padding_mask and attn_mask is deprecated",
)

# Compute and return the objective metrics
def calculate_objective_metrics(
    text: str,
    speech_path1: str,
    speech_path2: str,
    speaker_extractor,
    asr_model,
    wer_metric,
    cer_metric,
    asr_model_type,
) -> Dict[str, float]:
    """
    Compute and return the objective metrics
    
    Args:
        text: Reference text corresponding to `speech_path1`
        speech_path1: Audio path 1, used to compute speaker similarity
        speech_path2: Audio path 2, used to compute speaker similarity, WER, and CER
        speaker_extractor: Speaker extractor
        asr_model: Whisper model
        wer_metric: WordErrorRate metric
        cer_metric: CharErrorRate metric
        asr_model_type: ASR model type: `whisper` or `conformer-transducer`
    
    Returns:
        utmos: UTMOS score
        speaker_similarity: Speaker similarity
        wer_v: WER word error rate
        cer_v: CER character error rate
        hyp: Transcribed text corresponding to `speech_path2`
    """
    # Compute UTMOS
    utmos = calculate_utmos(speech_path2)
    
    # Compute speaker similarity
    speaker_similarity = compute_speaker_similarity(
        speech_path1,
        speech_path2,
        speaker_extractor,
    )
    
    # Compute Word Error Rate (WER) and Character Error Rate (CER)
    hyp = transcribe_audio(speech_path2, asr_model, asr_model_type) or ""
    ref = clean_english_text(text)
    hyp_c = clean_english_text(hyp)
    
    wer_v = wer_metric(hyp_c.lower(), ref.lower()).item()
    cer_v = cer_metric(hyp_c.lower().replace(" ", ""), ref.lower().replace(" ", "")).item()

    return utmos, speaker_similarity, wer_v, cer_v, hyp

def main(config_path: str):
    # Load the config file
    with open(config_path, "r") as f:
        config = json.load(f)
    out_dir = os.path.join(config["inference"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    asr_model_type = config["inference"].get("asr_model_type", "whisper")
    task_type = config["inference"].get("task_type", "all").lower()
    if task_type not in {"cross", "continuation", "all"}:
        raise ValueError(f"Unsupported task_type: {task_type}")

    # Load inference and evaluation models
    inferencer = MultiSpeakerInferencer(config_path)
    speaker_extractor = create_default_speaker_extractor(config["inference"]["wavlm_path"])
    asr_model = load_asr_model(config["inference"]["asr_model_path"], asr_model_type)

    continuation_data = []
    cross_data = []
    if task_type in {"continuation", "all"}:
        continuation_data_path = config["inference"]["continuation_data"]
        with open(continuation_data_path, "r") as f:
            continuation_data = json.load(f)
    if task_type in {"cross", "all"}:
        cross_data_path = config["inference"]["cross_data"]
        with open(cross_data_path, "r") as f:
            cross_data = json.load(f)
    
    continuation_result = []
    avg_continuation_data = {"avg_utmos": 0.0, "avg_speaker_similarity": 0.0, "avg_wer_v": 0.0, "avg_cer_v": 0.0}
    count_continuation = 0
    cross_result = []
    avg_cross_data = {"avg_utmos": 0.0, "avg_speaker_similarity": 0.0, "avg_wer_v": 0.0, "avg_cer_v": 0.0}
    count_cross = 0
    
    wer_metric = WordErrorRate()
    cer_metric = CharErrorRate()

    if task_type in {"continuation", "all"}:
        continuation_dir = os.path.join(out_dir, "continuation")
        os.makedirs(continuation_dir, exist_ok=True)
        wav_dir = os.path.join(continuation_dir, "wav")
        os.makedirs(wav_dir, exist_ok=True)

        for item in continuation_data:
            if item["duration"] > 10.0 or item["duration"] < 4.0:
                continue

            speech_id = item["id"]
            speech_folder = item["speech_folder"]
            speech_file = item["speech_file"]
            text = item["text"]
            units_label = item["units_label"]
            speech_path = os.path.join(speech_folder, speech_file)

            try:
                output_path = inferencer.infer_continuation(
                    wav_path=speech_path,
                    target_text=units_label,
                    output_path=os.path.join(wav_dir, f"{speech_id}.wav"),
                )
            except Exception as e:
                print(f"Error during multi-speaker evaluation: {e}")
                print(f"[Synthsizing error] error speech_id:{speech_id}")
                continue
            try:
                utmos, speaker_similarity, wer_v, cer_v, trans_text = calculate_objective_metrics(
                    text,
                    speech_path,
                    output_path,
                    speaker_extractor,
                    asr_model,
                    wer_metric,
                    cer_metric,
                    asr_model_type
                )
            except Exception as e:
                print(f"Error during multi-speaker evaluation: {e}")
                print(f"[Calculating error] error speech_id:{speech_id}, utmos={utmos}, spk_sim={speaker_similarity}, wer={wer_v}, cer={cer_v}, trans_text={trans_text}")
                continue
            count_continuation += 1
            print(f"[Continuation] {speech_id}: utmos={utmos:.4f}, speaker_similarity={speaker_similarity:.4f}, wer_v={wer_v:.4f}, cer_v={cer_v:.4f}")
            print(f"gt_text: {text}\n trans_text: {trans_text}")
            continuation_result.append({
                "id": speech_id,
                "gt_speech_path": speech_path,
                "gt_text": text,
                "gt_units_label": units_label,
                "trans_text": trans_text,
                "ctc_tts_speech_path": output_path,
                "utmos": utmos,
                "speaker_similarity": speaker_similarity,
                "wer_v": wer_v,
                "cer_v": cer_v,
            })
            avg_continuation_data["avg_utmos"] += utmos
            avg_continuation_data["avg_speaker_similarity"] += speaker_similarity
            avg_continuation_data["avg_wer_v"] += wer_v
            avg_continuation_data["avg_cer_v"] += cer_v
            print(f"[Continuation] avg_utmos={avg_continuation_data['avg_utmos'] / count_continuation:.4f}\navg_speaker_similarity={avg_continuation_data['avg_speaker_similarity'] / count_continuation:.4f}\navg_wer_v={avg_continuation_data['avg_wer_v'] / count_continuation:.4f}\navg_cer_v={avg_continuation_data['avg_cer_v'] / count_continuation:.4f}")

        with open(os.path.join(continuation_dir, "continuation_result.json"), "w") as f:
            json.dump(continuation_result, f, indent=4, ensure_ascii=False)

        if count_continuation > 0:
            avg_continuation_data["avg_utmos"] /= count_continuation
            avg_continuation_data["avg_speaker_similarity"] /= count_continuation
            avg_continuation_data["avg_wer_v"] /= count_continuation
            avg_continuation_data["avg_cer_v"] /= count_continuation
        print(f"[Continuation] completed, synthesized a total of {count_continuation} samples，avg_utmos={avg_continuation_data['avg_utmos']:.4f}\navg_speaker_similarity={avg_continuation_data['avg_speaker_similarity']:.4f}\navg_wer_v={avg_continuation_data['avg_wer_v']:.4f}\navg_cer_v={avg_continuation_data['avg_cer_v']:.4f}")

    if task_type in {"cross", "all"}:
        cross_dir = os.path.join(out_dir, "cross")
        os.makedirs(cross_dir, exist_ok=True)
        wav_dir = os.path.join(cross_dir, "wav")
        os.makedirs(wav_dir, exist_ok=True)

        for item in cross_data:
            speech_id = item["id"]
            prompt_speech_path = item["prompt_speech_path"]
            prompt_text = item["prompt_text"]
            target_text = item["target_text"]
            prompt_unit = item["prompt_unit"]
            target_unit = item["target_unit"]

            try:
                output_path = inferencer.infer_cross(
                    wav_path=prompt_speech_path,
                    prompt_text=prompt_unit,
                    target_text=target_unit,
                    output_path=os.path.join(wav_dir, f"{speech_id}.wav"),
                )
                utmos, speaker_similarity, wer_v, cer_v, trans_text = calculate_objective_metrics(
                    target_text,
                    prompt_speech_path,
                    output_path,
                    speaker_extractor,
                    asr_model,
                    wer_metric,
                    cer_metric,
                    asr_model_type
                )
                count_cross += 1
            except Exception as e:
                print(f"Error during multi-speaker evaluation: {e}")
                print(f"[Cross] error speech_id:{speech_id}, ")
                continue
            print(f"[Cross] {speech_id}: utmos={utmos:.4f}, speaker_similarity={speaker_similarity:.4f}, wer_v={wer_v:.4f}, cer_v={cer_v:.4f}")
            print(f"gt_text: {target_text}\n trans_text: {trans_text} target_unit: {target_unit}")
            cross_result.append({
                "id": speech_id,
                "prompt_speech_path": prompt_speech_path,
                "prompt_text": prompt_text,
                "prompt_unit": prompt_unit,
                "target_unit": target_unit,
                "gt_text": target_text,
                "trans_text": trans_text,
                "ctc_tts_speech_path": output_path,
                "utmos": utmos,
                "speaker_similarity": speaker_similarity,
                "wer_v": wer_v,
                "cer_v": cer_v,
            })
            avg_cross_data["avg_utmos"] += utmos
            avg_cross_data["avg_speaker_similarity"] += speaker_similarity
            avg_cross_data["avg_wer_v"] += wer_v
            avg_cross_data["avg_cer_v"] += cer_v
            print(f"[Cross] avg_utmos={avg_cross_data['avg_utmos'] / count_cross:.4f}\navg_speaker_similarity={avg_cross_data['avg_speaker_similarity'] / count_cross:.4f}\navg_wer_v={avg_cross_data['avg_wer_v'] / count_cross:.4f}\navg_cer_v={avg_cross_data['avg_cer_v'] / count_cross:.4f}")

        with open(os.path.join(cross_dir, "cross_result.json"), "w") as f:
            json.dump(cross_result, f, indent=4, ensure_ascii=False)

        if count_cross > 0:
            avg_cross_data["avg_utmos"] /= count_cross
            avg_cross_data["avg_speaker_similarity"] /= count_cross
            avg_cross_data["avg_wer_v"] /= count_cross
            avg_cross_data["avg_cer_v"] /= count_cross
        print(f"[Cross] completed, synthesized a total of {count_cross} samples，avg_utmos={avg_cross_data['avg_utmos']:.4f}\navg_speaker_similarity={avg_cross_data['avg_speaker_similarity']:.4f}\navg_wer_v={avg_cross_data['avg_wer_v']:.4f}\navg_cer_v={avg_cross_data['avg_cer_v']:.4f}")

    if task_type in {"continuation", "all"}:
        print(f"All tasks completed! [Continuation] synthesized a total of {count_continuation} samples，avg_utmos={avg_continuation_data['avg_utmos']:.4f}\navg_speaker_similarity={avg_continuation_data['avg_speaker_similarity']:.4f}\navg_wer_v={avg_continuation_data['avg_wer_v']:.4f}\navg_cer_v={avg_continuation_data['avg_cer_v']:.4f}")
    if task_type in {"cross", "all"}:
        print(f"All tasks completed! [Cross] synthesized a total of {count_cross} samples，avg_utmos={avg_cross_data['avg_utmos']:.4f}\navg_speaker_similarity={avg_cross_data['avg_speaker_similarity']:.4f}\navg_wer_v={avg_cross_data['avg_wer_v']:.4f}\navg_cer_v={avg_cross_data['avg_cer_v']:.4f}")

if __name__ == "__main__":
    # Get the config path from command-line arguments
    
    parser = argparse.ArgumentParser(description="Test MultiSpeakerInferencer")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    config_path = args.config_path

    main(config_path)

