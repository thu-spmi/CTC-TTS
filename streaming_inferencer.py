"""
Multiprocess streaming TTS inference utilities.
"""

import asyncio
import json
import multiprocessing as mp
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from queue import Empty
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io.wavfile import write as wav_write

from inference.FixedTextStreaming import FixedTextStreamer
from inference.model_handler import ModelHandler
from src.utils import generate_unit_list, get_overlap_weights, init_condition, sample_token_from_probs


config = {}
text_streamer: Optional[FixedTextStreamer] = None
model_handler_1 = None
model_handler_2 = None
generation_start = None
first_packet_latency_ms = None
total_generation_time_ms = None
cross_first_chunk_history_text_token_count = None
cross_first_chunk_history_text_update_time_ms = None
mp_ctx = None
worker_request_queue_1 = None
worker_request_queue_2 = None
worker_output_queue_1 = None
worker_output_queue_2 = None
worker_control_queue = None
worker_process_1 = None
worker_process_2 = None
worker_config_signature = None


@dataclass
class StreamingDecoderState:
    index: int
    model_handler: ModelHandler
    kvcache: Optional[object] = None
    speech_decoder_output: Optional[torch.Tensor] = None
    speech_decoder_input: list[int] = field(default_factory=list)
    text_seqs: list[int] = field(default_factory=list)
    speech_seqs: list[int] = field(default_factory=list)
    token_batch: list[int] = field(default_factory=list)
    first_block: bool = True
    dump_size: int = 0
    track_cross_history_metrics: bool = False
    history_text_token_count: int = 0
    history_text_update_time_ms: float = 0.0
    first_chunk_metrics_sent: bool = False


def _get_sentence_end_tokens(cfg: dict) -> set[int]:
    separators = cfg.get("data", {}).get("separators", {})
    sentence_end_tokens = {
        int(token_id)
        for separator, token_id in separators.items()
        if separator in {".", "!", "?"}
    }
    return sentence_end_tokens


def _get_initial_dump_size(cfg: dict, index: int) -> int:
    if index == 0:
        return int(cfg["inference"]["initial_dump_size_1"])
    return int(cfg["inference"]["initial_dump_size_2"])


def _get_task_type(cfg: dict) -> str:
    return cfg.get("inference", {}).get("task_type", "single-speaker")


def _validate_prompt_inputs(cfg: dict, speech_prompt: Optional[str], text_prompt: Optional[str]):
    task_type = _get_task_type(cfg)
    if task_type == "single-speaker":
        return
    if not speech_prompt:
        raise ValueError(f"`speech_prompt` is required for task_type='{task_type}'.")
    if not text_prompt:
        raise ValueError(f"`text_prompt` is required for task_type='{task_type}'.")


def _run_length_aligned_step(
    model_handler: ModelHandler,
    text_tokens: list[int],
    speech_tokens: list[int],
    kvcache=None,
):
    text_tensor = torch.tensor([text_tokens], dtype=torch.long, device=model_handler.device)
    speech_tensor = torch.tensor([speech_tokens], dtype=torch.long, device=model_handler.device)
    return model_handler.model(tokens={"text_seqs": text_tensor, "speech_seqs": speech_tensor}, kvcache=kvcache)


def _run_sequential_step(model_handler: ModelHandler, seq_tokens: list[int], kvcache=None):
    input_seq = torch.tensor([seq_tokens], dtype=torch.long, device=model_handler.device)
    return model_handler.model(input_seq, kvcache=kvcache)


def _clone_kvcache(kvcache):
    if kvcache is None:
        return None
    cloned_cache = []
    for layer_cache in kvcache:
        if layer_cache is None:
            cloned_cache.append(None)
            continue
        key_cache, value_cache = layer_cache
        cloned_cache.append([key_cache.detach().clone(), value_cache.detach().clone()])
    return cloned_cache


def _clone_cached_init_result(cached_result, length_aligned_input: bool):
    if length_aligned_input:
        text_seqs, speech_seqs, kvcache = cached_result
        return list(text_seqs), list(speech_seqs), _clone_kvcache(kvcache)
    speech_decoder_input, kvcache = cached_result
    return list(speech_decoder_input), _clone_kvcache(kvcache)


def _get_prompt_cache_key(
    index: int,
    model_handler: ModelHandler,
    cfg: dict,
    speech_prompt: Optional[str],
    text_prompt: Optional[str],
):
    return (
        index,
        _get_task_type(cfg),
        bool(model_handler.model.config.length_aligned_input),
        str(model_handler.device),
        speech_prompt or "",
        text_prompt or "",
    )


def _get_initialized_condition(
    index: int,
    model_handler: ModelHandler,
    cfg: dict,
    prompt_init_cache: dict,
    speech_prompt: Optional[str],
    text_prompt: Optional[str],
):
    if _get_task_type(cfg) == "single-speaker":
        return init_condition(model_handler, cfg, speech_prompt, text_prompt)

    cache_key = _get_prompt_cache_key(index, model_handler, cfg, speech_prompt, text_prompt)
    cached_result = prompt_init_cache.get(cache_key)
    if cached_result is None:
        cached_result = init_condition(model_handler, cfg, speech_prompt, text_prompt)
        prompt_init_cache[cache_key] = _clone_cached_init_result(
            cached_result,
            model_handler.model.config.length_aligned_input,
        )
    return _clone_cached_init_result(
        prompt_init_cache[cache_key],
        model_handler.model.config.length_aligned_input,
    )


def _build_decoder_state(
    index: int,
    model_handler: ModelHandler,
    cfg: dict,
    prompt_init_cache: dict,
    speech_prompt: Optional[str] = None,
    text_prompt: Optional[str] = None,
) -> StreamingDecoderState:
    state = StreamingDecoderState(
        index=index,
        model_handler=model_handler,
        dump_size=_get_initial_dump_size(cfg, index),
        track_cross_history_metrics=(
            _get_task_type(cfg) == "cross" and not bool(model_handler.model.config.length_aligned_input)
        ),
    )
    if model_handler.model.config.length_aligned_input:
        state.text_seqs, state.speech_seqs, state.kvcache = _get_initialized_condition(
            index,
            model_handler,
            cfg,
            prompt_init_cache,
            speech_prompt,
            text_prompt,
        )
    else:
        state.speech_decoder_input, state.kvcache = _get_initialized_condition(
            index,
            model_handler,
            cfg,
            prompt_init_cache,
            speech_prompt,
            text_prompt,
        )
        state.first_block = False
    return state


def _prepare_text_tokens_for_model(cfg: dict, text_tokens: list[int]) -> list[int]:
    codebook_size = int(cfg["nac"]["codebook_size"])
    unit_vocab_size = int(cfg["data"]["unit_vocab_size"])
    return [int(token) + codebook_size if int(token) < unit_vocab_size else int(token) for token in text_tokens]


def _prime_decoder_with_text(state: StreamingDecoderState, text_tokens: list[int]):
    model_handler = state.model_handler
    if model_handler.model.config.length_aligned_input:
        if state.first_block:
            state.text_seqs.append(text_tokens[0])
            state.speech_seqs.append(-1)
            state.speech_decoder_output, _, state.kvcache = _run_length_aligned_step(
                model_handler,
                state.text_seqs,
                state.speech_seqs,
                kvcache=state.kvcache,
            )
            state.first_block = False
            return state.speech_decoder_output, 1
        if state.speech_decoder_output is None:
            raise RuntimeError("Length-aligned decoder state is not initialized.")
        return state.speech_decoder_output, 0

    speech_decoder_output = None
    update_start_time = time.time() if state.track_cross_history_metrics and not state.first_chunk_metrics_sent else None
    for token in text_tokens:
        state.speech_decoder_input.append(token)
        state.speech_decoder_output, _, state.kvcache = _run_sequential_step(
            model_handler,
            state.speech_decoder_input,
            kvcache=state.kvcache,
        )
    if update_start_time is not None:
        state.history_text_token_count += len(text_tokens)
        state.history_text_update_time_ms += (time.time() - update_start_time) * 1000.0
    return state.speech_decoder_output, 0


def _advance_decoder_state(
    state: StreamingDecoderState,
    cfg: dict,
    current_token: int,
    text_tokens: list[int],
    text_point: int,
):
    model_handler = state.model_handler
    eob_token = int(cfg["data"]["eob"])
    if model_handler.model.config.length_aligned_input:
        if current_token == eob_token:
            state.text_seqs.append(eob_token)
            state.speech_seqs.append(-1)
        else:
            pad_token = int(cfg["data"]["pad"])
            next_text_token = text_tokens[text_point] if text_point < len(text_tokens) else pad_token
            state.text_seqs.append(next_text_token)
            state.speech_seqs.append(current_token)
            text_point += 1
        state.speech_decoder_output, _, state.kvcache = _run_length_aligned_step(
            model_handler,
            state.text_seqs,
            state.speech_seqs,
            kvcache=state.kvcache,
        )
        return state.speech_decoder_output, text_point

    state.speech_decoder_input.append(current_token)
    state.speech_decoder_output, _, state.kvcache = _run_sequential_step(
        model_handler,
        state.speech_decoder_input,
        kvcache=state.kvcache,
    )
    return state.speech_decoder_output, text_point


def _decode_speech_tokens(model_handler: ModelHandler, speech_tokens: list[int]) -> np.ndarray:
    predicted_tokens = torch.tensor([speech_tokens], dtype=torch.long, device=model_handler.device)
    audio_out = model_handler.nac.decode(predicted_tokens).squeeze(0)
    return audio_out.detach().cpu().numpy().astype(np.float32)


def _update_generation_metrics_from_decode_time(decode_time: float):
    global generation_start, first_packet_latency_ms, total_generation_time_ms
    if generation_start is None:
        return
    elapsed_ms = round((decode_time - generation_start) * 1000.0, 4)
    total_generation_time_ms = elapsed_ms
    if first_packet_latency_ms is None:
        first_packet_latency_ms = elapsed_ms


def _enqueue_audio_chunk(
    request_id: str,
    state: StreamingDecoderState,
    output_queue,
    speech_tokens: list[int],
):
    if not speech_tokens:
        return
    audio_np = _decode_speech_tokens(state.model_handler, speech_tokens)
    if state.track_cross_history_metrics and not state.first_chunk_metrics_sent:
        output_queue.put(
            (
                request_id,
                "cross_first_chunk_history",
                {
                    "text_token_count": int(state.history_text_token_count),
                    "text_update_time_ms": round(float(state.history_text_update_time_ms), 4),
                },
            )
        )
        state.first_chunk_metrics_sent = True
    output_queue.put((request_id, "audio", (audio_np, time.time())))


def _flush_non_chunked_batch(
    request_id: str,
    state: StreamingDecoderState,
    cfg: dict,
    output_queue,
):
    if not state.token_batch:
        return
    overlap_token_count = int(cfg["inference"]["overlap_token_count"])
    min_dump_size = int(cfg["inference"]["min_dump_size"])
    if len(state.token_batch) < min_dump_size:
        if overlap_token_count > 0:
            state.token_batch = state.token_batch[:-overlap_token_count]
        return
    _enqueue_audio_chunk(
        request_id,
        state,
        output_queue,
        state.token_batch,
    )
    state.token_batch = []


def _flush_chunked_batch(
    request_id: str,
    state: StreamingDecoderState,
    cfg: dict,
    output_queue,
    force: bool = False,
):
    max_dump_size = int(cfg["inference"]["max_dump_size"])
    if not state.token_batch:
        return
    if not force and len(state.token_batch) < state.dump_size:
        return
    _enqueue_audio_chunk(
        request_id,
        state,
        output_queue,
        state.token_batch,
    )
    state.token_batch = []
    state.dump_size = min(state.dump_size * 3, max_dump_size)


def _build_text_unit_list(model_handler: ModelHandler, cfg: dict, text: str, eos_token: int) -> list[list[int]]:
    local_streamer = FixedTextStreamer()
    _, tokenization = model_handler.tokenize_text(text)
    unit_list = generate_unit_list(cfg, tokenization.shifted_token_ids_with_separators)
    unit_list[-1].append(eos_token)
    return list(local_streamer.generate_text_stream(unit_list))


def _run_decoder_request(
    index: int,
    model_handler: ModelHandler,
    cfg: dict,
    prompt_init_cache: dict,
    request_id: str,
    text: str,
    audio_output_queue,
    control_queue,
    enable_chunked_decode: bool = False,
    speech_prompt: Optional[str] = None,
    text_prompt: Optional[str] = None,
):
    try:
        sentence_end_tokens = _get_sentence_end_tokens(cfg)
        eob_token = int(cfg["data"]["eob"])
        eos_token = int(cfg["data"]["eos"])
        codebook_size = int(cfg["nac"]["codebook_size"])
        state = _build_decoder_state(index, model_handler, cfg, prompt_init_cache, speech_prompt, text_prompt)
        received_first_text = False
        active_index = 0
        unit_stream = _build_text_unit_list(model_handler, cfg, text, eos_token)

        with torch.inference_mode():
            for text_tokens in unit_stream:
                if not text_tokens:
                    continue

                end_of_sentence = any(int(token) in sentence_end_tokens for token in text_tokens)
                belongs_to_worker = active_index == index

                if belongs_to_worker:
                    if not received_first_text:
                        audio_output_queue.put((request_id, "metric_start", time.time()))
                        received_first_text = True

                    end_of_speech = end_of_sentence
                    end_generation = int(text_tokens[-1]) == eos_token
                    prepared_text_tokens = _prepare_text_tokens_for_model(cfg, text_tokens)
                    speech_decoder_output, text_point = _prime_decoder_with_text(state, prepared_text_tokens)

                    while True:
                        logits = speech_decoder_output[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        current_token = sample_token_from_probs(probs, cfg)

                        if current_token == eob_token:
                            if not enable_chunked_decode:
                                _flush_non_chunked_batch(
                                    request_id,
                                    state,
                                    cfg,
                                    audio_output_queue,
                                )
                            speech_decoder_output, text_point = _advance_decoder_state(
                                state,
                                cfg,
                                current_token,
                                prepared_text_tokens,
                                text_point,
                            )
                            break

                        if current_token >= codebook_size:
                            raise ValueError(f"Generated invalid speech token {current_token}")

                        state.token_batch.append(current_token)
                        if enable_chunked_decode:
                            _flush_chunked_batch(
                                request_id,
                                state,
                                cfg,
                                audio_output_queue,
                            )

                        speech_decoder_output, text_point = _advance_decoder_state(
                            state,
                            cfg,
                            current_token,
                            prepared_text_tokens,
                            text_point,
                        )

                    if (end_of_speech or end_generation) and enable_chunked_decode:
                        _flush_chunked_batch(
                            request_id,
                            state,
                            cfg,
                            audio_output_queue,
                            force=True,
                        )

                    if end_generation:
                        audio_output_queue.put((request_id, "end", None))
                    elif end_of_speech:
                        audio_output_queue.put((request_id, "switch", 1 if index == 0 else 0))

                    if end_of_speech or end_generation:
                        state = _build_decoder_state(
                            index,
                            model_handler,
                            cfg,
                            prompt_init_cache,
                            speech_prompt,
                            text_prompt,
                        )

                if end_of_sentence:
                    active_index = 1 - active_index

        control_queue.put((request_id, "done", index))
    except Exception as exc:
        control_queue.put((request_id, "error", f"worker {index}: {exc}\n{traceback.format_exc()}"))
        control_queue.put((request_id, "done", index))


def _decoder_worker_process(
    index: int,
    cfg: dict,
    device_id: Optional[int],
    request_queue,
    audio_output_queue,
    control_queue,
):
    prompt_init_cache = {}
    model_handler = ModelHandler(cfg, device_id=device_id)
    while True:
        request = request_queue.get()
        if request is None:
            break
        command = request.get("command")
        if command == "shutdown":
            break
        if command != "synthesize":
            continue
        _run_decoder_request(
            index=index,
            model_handler=model_handler,
            cfg=cfg,
            prompt_init_cache=prompt_init_cache,
            request_id=request["request_id"],
            text=request["text"],
            audio_output_queue=audio_output_queue,
            control_queue=control_queue,
            enable_chunked_decode=bool(request.get("enable_chunked_decode", False)),
            speech_prompt=request.get("speech_prompt"),
            text_prompt=request.get("text_prompt"),
        )


async def audio_generator_async(
    audio_byte_queue_1,
    audio_byte_queue_2,
    control_queue,
    request_id: str,
    wav_path: str,
    processes,
    enable_smoothing: bool = True,
):
    queue_list = [audio_byte_queue_1, audio_byte_queue_2]
    current_queue = queue_list[0]
    loop = asyncio.get_event_loop()

    sample_rate = int(config["inference"]["sample_rate"])
    smoothing_ms = float(config["inference"]["smoothing_ms"])
    smoothing_samples = max(1, int(sample_rate * smoothing_ms / 1000))
    tail = None
    wav_buf = []
    seen_done = set()
    wrote_wav = False

    def finalize_wav():
        nonlocal tail, wav_buf, wrote_wav
        if wrote_wav:
            return
        final_segments = list(wav_buf)
        if enable_smoothing and isinstance(tail, np.ndarray) and len(tail) > 0:
            final_segments.append(tail.astype(np.float32))
            tail = None
        if not final_segments:
            return
        audio_full = np.concatenate(final_segments).astype(np.float32)
        wav_dir = os.path.dirname(wav_path)
        if wav_dir:
            os.makedirs(wav_dir, exist_ok=True)
        wav_write(wav_path, sample_rate, audio_full)
        wav_buf = []
        wrote_wav = True

    try:
        while True:
            try:
                event_request_id, message_type, payload = control_queue.get_nowait()
            except Empty:
                event_request_id, message_type, payload = None, None, None

            if event_request_id == request_id:
                if message_type == "error":
                    print(f"Error in multiprocess streaming worker: {payload}")
                    continue

                if message_type == "done":
                    seen_done.add(int(payload))
                    continue

            try:
                event_request_id, message_type, payload = current_queue.get_nowait()
            except Empty:
                if len(seen_done) == len(processes):
                    finalize_wav()
                    yield None
                    break
                if any(process.is_alive() for process in processes):
                    await asyncio.sleep(0.01)
                    continue
                finalize_wav()
                yield None
                break

            if event_request_id != request_id:
                continue

            if message_type == "metric_start":
                global generation_start
                start_time = float(payload)
                if generation_start is None or start_time < generation_start:
                    generation_start = start_time
                continue

            if message_type == "cross_first_chunk_history":
                global cross_first_chunk_history_text_token_count
                global cross_first_chunk_history_text_update_time_ms
                metrics_payload = payload or {}
                cross_first_chunk_history_text_token_count = int(
                    metrics_payload.get("text_token_count", 0)
                )
                cross_first_chunk_history_text_update_time_ms = float(
                    metrics_payload.get("text_update_time_ms", 0.0)
                )
                continue

            if message_type == "switch":
                current_queue = queue_list[int(payload)]
                if enable_smoothing:
                    tail = None
                continue

            if message_type == "end":
                finalize_wav()
                yield None
                break

            if message_type != "audio":
                continue

            audio_item, decode_time = payload
            if isinstance(audio_item, np.ndarray):
                audio_np = audio_item.astype(np.float32)
            else:
                audio_np = np.asarray(audio_item, dtype=np.float32)
            _update_generation_metrics_from_decode_time(float(decode_time))

            if enable_smoothing:
                overlap_len = min(
                    smoothing_samples,
                    len(audio_np),
                    len(tail) if isinstance(tail, np.ndarray) else 0,
                )
                if overlap_len > 0 and isinstance(tail, np.ndarray):
                    w_prev, w_new = get_overlap_weights(overlap_len)
                    audio_np[:overlap_len] = audio_np[:overlap_len] * w_new + tail[-overlap_len:] * w_prev

            emit_len = max(0, len(audio_np) - smoothing_samples) if enable_smoothing else len(audio_np)
            if emit_len > 0:
                to_emit = audio_np[:emit_len].astype(np.float32)
                wav_buf.append(to_emit)
                yield to_emit.tobytes()

            if enable_smoothing:
                tail = audio_np[-smoothing_samples:] if len(audio_np) >= smoothing_samples else audio_np.copy()
            else:
                tail = None
    finally:
        finalize_wav()


def _config_signature(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, ensure_ascii=True)


def _shutdown_worker_processes():
    global worker_process_1, worker_process_2
    global worker_request_queue_1, worker_request_queue_2
    global worker_output_queue_1, worker_output_queue_2, worker_control_queue
    global worker_config_signature

    for request_queue in (worker_request_queue_1, worker_request_queue_2):
        if request_queue is not None:
            try:
                request_queue.put({"command": "shutdown"})
            except Exception:
                pass

    for process in (worker_process_1, worker_process_2):
        if process is None:
            continue
        if process.is_alive():
            process.join(timeout=2)
        if process.is_alive():
            process.terminate()

    worker_process_1 = None
    worker_process_2 = None
    worker_request_queue_1 = None
    worker_request_queue_2 = None
    worker_output_queue_1 = None
    worker_output_queue_2 = None
    worker_control_queue = None
    worker_config_signature = None


def init_models(force_restart: bool = False):
    """Start decoder worker processes."""
    global text_streamer, model_handler_1, model_handler_2
    global mp_ctx, worker_request_queue_1, worker_request_queue_2
    global worker_output_queue_1, worker_output_queue_2, worker_control_queue
    global worker_process_1, worker_process_2, worker_config_signature

    if text_streamer is None:
        text_streamer = FixedTextStreamer()

    current_signature = _config_signature(config)
    if (
        not force_restart
        and
        worker_process_1 is not None
        and worker_process_2 is not None
        and worker_process_1.is_alive()
        and worker_process_2.is_alive()
        and worker_config_signature == current_signature
    ):
        print("Multiprocess streaming workers are already running.")
        return

    _shutdown_worker_processes()

    mp_ctx = mp.get_context("spawn")
    worker_request_queue_1 = mp_ctx.Queue()
    worker_request_queue_2 = mp_ctx.Queue()
    worker_output_queue_1 = mp_ctx.Queue()
    worker_output_queue_2 = mp_ctx.Queue()
    worker_control_queue = mp_ctx.Queue()

    device_id_1 = config["inference"]["tts_device_1"]
    device_id_2 = config["inference"]["tts_device_2"]

    worker_process_1 = mp_ctx.Process(
        target=_decoder_worker_process,
        args=(0, config, device_id_1, worker_request_queue_1, worker_output_queue_1, worker_control_queue),
        daemon=True,
    )
    worker_process_2 = mp_ctx.Process(
        target=_decoder_worker_process,
        args=(1, config, device_id_2, worker_request_queue_2, worker_output_queue_2, worker_control_queue),
        daemon=True,
    )
    worker_process_1.start()
    worker_process_2.start()

    worker_config_signature = current_signature
    model_handler_1 = "multiprocess-worker-0"
    model_handler_2 = "multiprocess-worker-1"
    print("Multiprocess streaming workers started successfully.")


async def tts(
    config,
    text,
    wav_path: str,
    enable_chunked_decode: bool = False,
    enable_smoothing: bool = True,
    speech_prompt=None,
    text_prompt=None,
):
    """Generate speech from text with streaming output using two decoder processes."""
    globals()["config"] = config
    _validate_prompt_inputs(config, speech_prompt, text_prompt)
    globals()["generation_start"] = None
    globals()["first_packet_latency_ms"] = None
    globals()["total_generation_time_ms"] = None
    globals()["cross_first_chunk_history_text_token_count"] = None
    globals()["cross_first_chunk_history_text_update_time_ms"] = None

    init_models(force_restart=True)

    request_id = uuid.uuid4().hex
    request_payload = {
        "command": "synthesize",
        "request_id": request_id,
        "text": text,
        "enable_chunked_decode": enable_chunked_decode,
        "speech_prompt": speech_prompt,
        "text_prompt": text_prompt,
    }
    worker_request_queue_1.put(request_payload)
    worker_request_queue_2.put(request_payload)
    processes = [worker_process_1, worker_process_2]

    async def _gen():
        try:
            async for item in audio_generator_async(
                worker_output_queue_1,
                worker_output_queue_2,
                worker_control_queue,
                request_id,
                wav_path,
                processes,
                enable_smoothing=enable_smoothing,
            ):
                yield item
        finally:
            _shutdown_worker_processes()

    return _gen()


if __name__ == "__main__":
    config_path = ""

    with open(config_path, "r") as f:
        config = json.load(f)

    init_models()

    text = "Recognizing that the current economic climate, characterized by fluctuating interest rates and unpredictable supply chain disruptions, presents unprecedented challenges for small businesses, the federal government has unveiled a comprehensive stimulus package that includes grants, low interest loans, and tax incentives designed to provide a crucial lifeline and foster long term resilience."
    wav_path = "long1_equal_multiproc.wav"

    async def _run():
        gen = await tts(config, text, wav_path, enable_chunked_decode=True, enable_smoothing=True)
        async for _ in gen:
            pass

    asyncio.run(_run())
