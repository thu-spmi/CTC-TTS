import argparse
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torchaudio
from torchmetrics import CharErrorRate, WordErrorRate

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import streaming_inferencer as si


def safe_mean(values: Sequence[float]) -> float:
    valid = [float(v) for v in values if not np.isnan(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def is_cross_history_metric_enabled(task_type: str, cfg: Dict) -> bool:
    return task_type == "cross" and not bool(cfg.get("model", {}).get("length_aligned_input", False))



def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    if "inference" not in cfg:
        cfg["inference"] = {}
    return cfg


def init_runtime(cfg: Dict):
    si.config = cfg
    si.text_streamer = si.FixedTextStreamer()
    si.init_models()


def get_audio_duration_seconds(audio_path: str) -> float:
    if not os.path.exists(audio_path):
        return float("nan")
    info = torchaudio.info(audio_path)
    if info.sample_rate <= 0:
        return float("nan")
    return float(info.num_frames) / float(info.sample_rate)


def ensure_task_output_dir(cfg: Dict, task_type: str) -> Tuple[str, str]:
    out_dir = cfg["inference"]["output_dir"]
    task_dir = os.path.join(out_dir, task_type.replace("-", "_"))
    wav_dir = os.path.join(task_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    return task_dir, wav_dir


def load_test_data_for_task(cfg: Dict, task_type: str) -> List[Dict]:
    inf_cfg = cfg.get("inference", {})
    if task_type == "single-speaker":
        data_path = inf_cfg.get("single_speaker_data")
    elif task_type == "continuation":
        data_path = inf_cfg.get("continuation_data")
    elif task_type == "cross":
        data_path = inf_cfg.get("cross_data")
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    if not data_path:
        raise ValueError(f"Missing data path for task_type='{task_type}'.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_requested_tasks(cfg: Dict) -> List[str]:
    requested = cfg["inference"].get("task_type", "single-speaker").lower()
    if requested != "all":
        return [requested]

    tasks: List[str] = []
    single_speaker_data = cfg["inference"].get("single_speaker_data")
    if single_speaker_data and os.path.exists(single_speaker_data):
        tasks.append("single-speaker")
    if cfg["inference"].get("continuation_data"):
        tasks.append("continuation")
    if cfg["inference"].get("cross_data"):
        tasks.append("cross")
    if not tasks:
        raise ValueError("task_type='all' but no available test data paths were found.")
    return tasks


def resolve_prompt_speech_path(item: Dict) -> str:
    speech_folder = item.get("speech_folder")
    speech_file = item.get("speech_file")
    if speech_folder and speech_file:
        return os.path.join(speech_folder, speech_file)
    return str(
        item.get("speech_prompt_path")
        or item.get("prompt_speech_path")
        or item.get("audio_path")
        or ""
    ).strip()


def build_sample(task_type: str, item: Dict, index: int) -> Dict:
    sample_id = str(item.get("id", index)).strip() or str(index)
    if task_type == "single-speaker":
        synth_text = item.get("units_label")
        reference_text = item.get("text") or synth_text
        return {
            "id": sample_id,
            "synth_text": synth_text,
            "reference_text": reference_text,
            "speech_prompt": None,
            "text_prompt": None,
            "prompt_speech_path": "",
        }

    if task_type == "continuation":
        prompt_speech_path = resolve_prompt_speech_path(item)
        full_units_text = item.get("units_label")
        text_prompt = full_units_text
        synth_text = full_units_text
        reference_text = item.get("text")
        return {
            "id": sample_id,
            "synth_text": synth_text,
            "reference_text": reference_text,
            "speech_prompt": prompt_speech_path,
            "text_prompt": text_prompt,
            "prompt_speech_path": prompt_speech_path,
        }

    if task_type == "cross":
        prompt_speech_path = resolve_prompt_speech_path(item)
        text_prompt = item.get("prompt_unit")
        synth_text = item.get("target_unit")
        reference_text = item.get("target_text")
        return {
            "id": sample_id,
            "synth_text": synth_text,
            "reference_text": reference_text,
            "speech_prompt": prompt_speech_path,
            "text_prompt": text_prompt,
            "prompt_speech_path": prompt_speech_path,
        }

    raise ValueError(f"Unsupported task_type: {task_type}")


async def consume_one_async(
    text: str,
    wav_path: str,
    enable_chunked_decode: bool,
    enable_smoothing: bool,
    speech_prompt: Optional[str] = None,
    text_prompt: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    Consume the streaming generator and return:
    - First-chunk latency (ms)
    - Total generation time (ms)
    """
    t0 = time.time()
    first_audio_time: Optional[float] = None
    end_time: Optional[float] = None

    gen = await si.tts(
        si.config,
        text,
        wav_path=wav_path,
        enable_chunked_decode=enable_chunked_decode,
        enable_smoothing=enable_smoothing,
        speech_prompt=speech_prompt,
        text_prompt=text_prompt,
    )
    try:
        while True:
            item = await gen.__anext__()
            if item is None:
                end_time = time.time()
                await gen.aclose()
                break
            if first_audio_time is None and isinstance(item, (bytes, bytearray)) and len(item) > 0:
                first_audio_time = time.time()
    except StopAsyncIteration:
        end_time = time.time()

    latency_ms = getattr(si, "first_packet_latency_ms", None)
    generation_ms = getattr(si, "total_generation_time_ms", None)

    if latency_ms is None:
        if first_audio_time is None:
            latency_ms = float("nan")
        else:
            latency_ms = round((first_audio_time - t0) * 1000.0, 4)
    else:
        latency_ms = float(latency_ms)

    if generation_ms is None:
        if end_time is None:
            generation_ms = float("nan")
        else:
            generation_ms = round((end_time - t0) * 1000.0, 4)
    else:
        generation_ms = float(generation_ms)

    cross_first_chunk_history_text_token_count = getattr(
        si,
        "cross_first_chunk_history_text_token_count",
        None,
    )
    cross_first_chunk_history_text_update_time_ms = getattr(
        si,
        "cross_first_chunk_history_text_update_time_ms",
        None,
    )

    return {
        "latency_ms": latency_ms,
        "generation_ms": generation_ms,
        "cross_first_chunk_history_text_token_count": cross_first_chunk_history_text_token_count,
        "cross_first_chunk_history_text_update_time_ms": cross_first_chunk_history_text_update_time_ms,
    }


def synthesize_one_sample(sample: Dict, wav_dir: str, cfg: Dict, task_type: str) -> Dict[str, Optional[float]]:
    wav_path = os.path.join(wav_dir, f"ctctts_{sample['id']}.wav")
    inf_cfg = cfg.get("inference", {})
    enable_chunked_decode = bool(inf_cfg.get("enable_chunked_decode", False))
    enable_smoothing = bool(inf_cfg.get("enable_smoothing", True))

    consume_result = asyncio.run(
        consume_one_async(
            text=sample["synth_text"],
            wav_path=wav_path,
            enable_chunked_decode=enable_chunked_decode,
            enable_smoothing=enable_smoothing,
            speech_prompt=sample.get("speech_prompt"),
            text_prompt=sample.get("text_prompt"),
        )
    )
    latency_ms = consume_result["latency_ms"]
    generation_ms = consume_result["generation_ms"]

    audio_duration_s = get_audio_duration_seconds(wav_path)
    if np.isnan(audio_duration_s) or audio_duration_s <= 0 or np.isnan(generation_ms):
        rtf = float("nan")
    else:
        rtf = float(generation_ms) / (audio_duration_s * 1000.0)

    cross_first_chunk_history_text_token_count = None
    cross_first_chunk_history_text_update_time_ms = None
    if is_cross_history_metric_enabled(task_type, cfg):
        token_count = consume_result["cross_first_chunk_history_text_token_count"]
        update_time_ms = consume_result["cross_first_chunk_history_text_update_time_ms"]
        if token_count is not None:
            cross_first_chunk_history_text_token_count = int(token_count)
        if update_time_ms is not None:
            cross_first_chunk_history_text_update_time_ms = float(update_time_ms)

    return {
        "wav_path": wav_path,
        "latency_ms": latency_ms,
        "generation_ms": generation_ms,
        "audio_duration_s": audio_duration_s,
        "rtf": rtf,
        "cross_first_chunk_history_text_token_count": cross_first_chunk_history_text_token_count,
        "cross_first_chunk_history_text_update_time_ms": cross_first_chunk_history_text_update_time_ms,
    }


def calculate_objective_metrics(
    text: str,
    speech_path1: str,
    speech_path2: str,
    speaker_extractor,
    asr_model,
    wer_metric,
    cer_metric,
    asr_model_type: str,
) -> Tuple[float, float, float, float, str]:
    from metrics.cal import clean_english_text, transcribe_audio
    from metrics.calculate_utmos import calculate_utmos
    from metrics.speaker_similarity import compute_speaker_similarity

    utmos = calculate_utmos(speech_path2)

    speaker_similarity = float("nan")
    if speaker_extractor is not None and speech_path1 and os.path.exists(speech_path1):
        speaker_similarity = compute_speaker_similarity(
            speech_path1,
            speech_path2,
            speaker_extractor,
        )

    hyp = transcribe_audio(speech_path2, asr_model, asr_model_type) or ""
    ref = clean_english_text(text)
    hyp_c = clean_english_text(hyp)
    wer_v = wer_metric(hyp_c.lower(), ref.lower()).item()
    cer_v = cer_metric(hyp_c.lower().replace(" ", ""), ref.lower().replace(" ", "")).item()
    print(f"Reference text: {ref}\nRecognized text: {hyp}\nwer: {wer_v:.4f}\tcer: {cer_v:.4f}")
    return utmos, speaker_similarity, wer_v, cer_v, hyp


def create_result_item(item: Dict, task_type: str, sample: Dict) -> Dict:
    result = dict(item)
    result["task_type"] = task_type
    result["synthesis_text"] = sample["synth_text"]
    result["reference_text"] = sample["reference_text"]
    result["prompt_text"] = sample.get("text_prompt") or ""
    result["prompt_speech_path"] = sample.get("prompt_speech_path") or ""
    result["ctctts_speech_path"] = ""
    result["ctctts_latency"] = float("nan")
    result["ctctts_first_packet_latency_ms"] = float("nan")
    result["ctctts_generation_time_ms"] = float("nan")
    result["ctctts_audio_duration_s"] = float("nan")
    result["ctctts_rtf"] = float("nan")
    result["ctctts_speech_trans"] = ""
    result["ctctts_wer"] = float("nan")
    result["ctctts_cer"] = float("nan")
    result["ctctts_utmos"] = float("nan")
    result["ctctts_speaker_similarity"] = float("nan")
    result["ctctts_cross_first_chunk_history_text_token_count"] = None
    result["ctctts_cross_first_chunk_history_text_update_time_ms"] = None
    return result


def should_skip_item(task_type: str, item: Dict) -> bool:
    if task_type != "continuation":
        return False
    duration = item.get("duration")
    if duration is None:
        return False
    return duration > 10.0 or duration < 4.0


def run_task(
    task_type: str,
    cfg: Dict,
    asr_model,
    asr_model_type: str,
    speaker_extractor,
):
    si.config["inference"]["task_type"] = task_type
    task_data = load_test_data_for_task(cfg, task_type)
    task_dir, wav_dir = ensure_task_output_dir(cfg, task_type)

    wer_metric = WordErrorRate()
    cer_metric = CharErrorRate()
    results: List[Dict] = []
    latencies: List[float] = []
    generation_times: List[float] = []
    rtfs: List[float] = []
    utmos_values: List[float] = []
    speaker_sims: List[float] = []
    wers: List[float] = []
    cers: List[float] = []
    cross_history_token_counts: List[float] = []
    cross_history_update_times: List[float] = []
    track_cross_history_metrics = is_cross_history_metric_enabled(task_type, cfg)

    for index, item in enumerate(task_data, start=1):
        if should_skip_item(task_type, item):
            print(f"[{task_type}] Skip idx={index} due to duration filter")
            continue
        sample = build_sample(task_type, item, index)
        result = create_result_item(item, task_type, sample)
        if not sample["synth_text"]:
            print(f"[{task_type}] Skip idx={index} due to empty synthesis text")
            results.append(result)
            continue

        try:
            print(f"[{task_type}] Synthesize idx={index}, id={sample['id']}")
            synth_result = synthesize_one_sample(sample, wav_dir, cfg, task_type)
            result["ctctts_speech_path"] = synth_result["wav_path"]
            result["ctctts_latency"] = synth_result["latency_ms"]
            result["ctctts_first_packet_latency_ms"] = synth_result["latency_ms"]
            result["ctctts_generation_time_ms"] = synth_result["generation_ms"]
            result["ctctts_audio_duration_s"] = synth_result["audio_duration_s"]
            result["ctctts_rtf"] = synth_result["rtf"]
            result["ctctts_cross_first_chunk_history_text_token_count"] = synth_result[
                "cross_first_chunk_history_text_token_count"
            ]
            result["ctctts_cross_first_chunk_history_text_update_time_ms"] = synth_result[
                "cross_first_chunk_history_text_update_time_ms"
            ]
        except Exception as exc:
            print(f"[{task_type}] Synthesize error idx={index}, id={sample['id']}: {exc}")
            results.append(result)
            continue

        if not os.path.exists(result["ctctts_speech_path"]):
            print(f"[{task_type}] Missing synthesized wav for id={sample['id']}")
            results.append(result)
            continue

        try:
            utmos, speaker_similarity, wer_v, cer_v, trans_text = calculate_objective_metrics(
                sample["reference_text"],
                sample["prompt_speech_path"],
                result["ctctts_speech_path"],
                speaker_extractor,
                asr_model,
                wer_metric,
                cer_metric,
                asr_model_type,
            )
            result["ctctts_speech_trans"] = trans_text.strip()
            result["ctctts_wer"] = float(wer_v)
            result["ctctts_cer"] = float(cer_v)
            result["ctctts_utmos"] = float(utmos)
            result["ctctts_speaker_similarity"] = float(speaker_similarity)
        except Exception as exc:
            print(f"[{task_type}] Metric error idx={index}, id={sample['id']}: {exc}")

        if not np.isnan(result["ctctts_first_packet_latency_ms"]):
            latencies.append(result["ctctts_first_packet_latency_ms"])
        if not np.isnan(result["ctctts_generation_time_ms"]):
            generation_times.append(result["ctctts_generation_time_ms"])
        if not np.isnan(result["ctctts_rtf"]):
            rtfs.append(result["ctctts_rtf"])
        if not np.isnan(result["ctctts_utmos"]):
            utmos_values.append(result["ctctts_utmos"])
        if not np.isnan(result["ctctts_speaker_similarity"]):
            speaker_sims.append(result["ctctts_speaker_similarity"])
        if not np.isnan(result["ctctts_wer"]):
            wers.append(result["ctctts_wer"])
        if not np.isnan(result["ctctts_cer"]):
            cers.append(result["ctctts_cer"])
        if result["ctctts_cross_first_chunk_history_text_token_count"] is not None:
            cross_history_token_counts.append(result["ctctts_cross_first_chunk_history_text_token_count"])
        if result["ctctts_cross_first_chunk_history_text_update_time_ms"] is not None:
            cross_history_update_times.append(result["ctctts_cross_first_chunk_history_text_update_time_ms"])

        progress_message = (
            f"[{task_type}] Progress idx={index}, "
            f"latency={result['ctctts_first_packet_latency_ms']:.4f} ms, "
            f"rtf={result['ctctts_rtf']:.6f}, "
            f"avg_latency={safe_mean(latencies):.4f} ms, "
            f"avg_rtf={safe_mean(rtfs):.6f}, "
            f"avg_wer={safe_mean(wers):.6f}, "
            f"avg_cer={safe_mean(cers):.6f}, "
            f"avg_utmos={safe_mean(utmos_values):.4f}"
        )
        if track_cross_history_metrics:
            progress_message += (
                f", avg_cross_history_text_tokens={safe_mean(cross_history_token_counts):.4f}, "
                f"avg_cross_history_update_time_ms={safe_mean(cross_history_update_times):.4f}"
            )
        print(progress_message)
        results.append(result)

    result_json = os.path.join(task_dir, f"{task_type.replace('-', '_')}_result.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "task_type": task_type,
        "result_json": result_json,
        "sample_count": len(results),
        "avg_latency_ms": safe_mean(latencies),
        "avg_generation_time_ms": safe_mean(generation_times),
        "avg_rtf": safe_mean(rtfs),
        "avg_utmos": safe_mean(utmos_values),
        "avg_speaker_similarity": safe_mean(speaker_sims),
        "avg_wer": safe_mean(wers),
        "avg_cer": safe_mean(cers),
        "avg_cross_first_chunk_history_text_token_count": safe_mean(cross_history_token_counts),
        "avg_cross_first_chunk_history_text_update_time_ms": safe_mean(cross_history_update_times),
    }
    summary_message = (
        f"[{task_type}] completed, samples={summary['sample_count']}, "
        f"avg_latency={summary['avg_latency_ms']:.4f} ms, "
        f"avg_generation_time={summary['avg_generation_time_ms']:.4f} ms, "
        f"avg_rtf={summary['avg_rtf']:.6f}, "
        f"avg_utmos={summary['avg_utmos']:.4f}, "
        f"avg_speaker_similarity={summary['avg_speaker_similarity']:.4f}, "
        f"avg_wer={summary['avg_wer']:.6f}, "
        f"avg_cer={summary['avg_cer']:.6f}"
    )
    if track_cross_history_metrics:
        summary_message += (
            f", avg_cross_history_text_tokens={summary['avg_cross_first_chunk_history_text_token_count']:.4f}, "
            f"avg_cross_history_update_time_ms={summary['avg_cross_first_chunk_history_text_update_time_ms']:.4f}"
        )
    print(summary_message)
    return summary



def main(config_path: str):
    from metrics.cal import load_asr_model
    from metrics.speaker_similarity import create_default_speaker_extractor

    cfg = load_config(config_path)
    init_runtime(cfg)

    inf_cfg = cfg.get("inference", {})
    asr_model_type = inf_cfg.get("asr_model_type", "whisper")
    asr_model_path = inf_cfg.get("asr_model_path")
    if not asr_model_path:
        raise ValueError("Missing `inference.asr_model_path` in config.")

    print(f"[ASR] Loading {asr_model_type} model from: {asr_model_path}")
    asr_model = load_asr_model(asr_model_path, asr_model_type)

    requested_tasks = get_requested_tasks(cfg)
    use_speaker_similarity = any(task in {"continuation", "cross"} for task in requested_tasks)
    speaker_extractor = None
    if use_speaker_similarity:
        wavlm_path = inf_cfg.get("wavlm_path")
        if wavlm_path:
            print(f"[Speaker] Loading speaker extractor from: {wavlm_path}")
            speaker_extractor = create_default_speaker_extractor(wavlm_path)

    summaries = []
    for task_type in requested_tasks:
        summaries.append(
            run_task(
                task_type=task_type,
                cfg=cfg,
                asr_model=asr_model,
                asr_model_type=asr_model_type,
                speaker_extractor=speaker_extractor,
            )
        )

    print("[Final] All tasks completed")
    for summary in summaries:
        final_message = (
            f"[Final] {summary['task_type']}"
            f"avg_latency={summary['avg_latency_ms']:.4f} ms, "
            f"avg_generation_time={summary['avg_generation_time_ms']:.4f} ms, "
            f"avg_rtf={summary['avg_rtf']:.6f}, "
            f"avg_utmos={summary['avg_utmos']:.4f}, "
            f"avg_speaker_similarity={summary['avg_speaker_similarity']:.4f}, "
            f"avg_wer={summary['avg_wer']:.6f}, "
            f"avg_cer={summary['avg_cer']:.6f}, "
            f"result_json={summary['result_json']}"
        )
        if is_cross_history_metric_enabled(summary["task_type"], cfg):
            final_message += (
                f", avg_cross_history_text_tokens={summary['avg_cross_first_chunk_history_text_token_count']:.4f}, "
                f"avg_cross_history_update_time_ms={summary['avg_cross_first_chunk_history_text_update_time_ms']:.4f}"
            )
        print(final_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test streaming inferencer")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    main(args.config_path)
