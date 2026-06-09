import argparse
import json
from typing import Optional, Union

import torch
import torchaudio

from ctc_model.encoder import WavLMEncoder
from ctc_model.encoder_loader import build_encoder, load_pretrained_encoder
from ctc_model.tokenizer import load_tknz


def build_wavlm_encoder(cfg: dict) -> WavLMEncoder:
    model = build_encoder(cfg)
    if not isinstance(model, WavLMEncoder):
        raise TypeError(f"Expected WavLMEncoder, but got {type(model).__name__}.")
    return model


def load_pretrained_wavlm_encoder(
    cfg: dict,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> WavLMEncoder:
    model = load_pretrained_encoder(cfg, device=device, strict=strict)
    if not isinstance(model, WavLMEncoder):
        raise TypeError(f"Expected WavLMEncoder, but got {type(model).__name__}.")
    return model


def load_tokenizer_from_config(cfg: dict):
    tokenizer_path = cfg["tokenizer"]["file"]
    return load_tknz(tokenizer_path)


def ctc_greedy_decode_token_ids(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    blank_id: int = 0,
):
    pred_ids = logits.argmax(dim=-1).detach().cpu()
    lengths = lengths.detach().cpu().tolist()

    batch_token_ids = []
    for frame_ids, valid_length in zip(pred_ids, lengths):
        collapsed_ids = []
        prev_id = None
        for token_id in frame_ids[:valid_length].tolist():
            if token_id == blank_id:
                prev_id = token_id
                continue
            if token_id != prev_id:
                collapsed_ids.append(token_id)
            prev_id = token_id
        batch_token_ids.append(collapsed_ids)
    return batch_token_ids


def greedy_decode_to_text_list(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    tokenizer,
    blank_id: int = 0,
):
    batch_token_ids = ctc_greedy_decode_token_ids(logits, lengths, blank_id=blank_id)
    decoded_texts = tokenizer.decode(batch_token_ids)
    if isinstance(decoded_texts, str):
        return [decoded_texts], batch_token_ids
    return decoded_texts, batch_token_ids


def test_wavlm_encoder(
    config_path: str,
    audio_path: str,
    device: Optional[Union[str, torch.device]] = None,
    sample_rate: int = 16000,
    strict: bool = True,
):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = load_pretrained_wavlm_encoder(cfg, device=device, strict=strict)
    tokenizer = load_tokenizer_from_config(cfg)
    print(tokenizer._vocab_to_dict())
    print(tokenizer.encode("i t i tʰ wɑz ðis ɑbzəɻveiʃən ðætʰ dɻu fɻəm dəɡləs nɑtʰ imidiətʰli bətʰ leitʰəɻ in ðə ivniŋ ə ɻipʰlai ðætʰ hæd ðə intʰɻəstʰiŋ kʰɑnsəkʰwəns tʰoʊ hwitʃ ai kʰol ətʰenʃən səm wən els tʰoʊld ə stʰoɻi nɑtʰ pʰəɻtʰikʰjələɻli ifekʰtʰiv hwitʃ ai so hi wɑz nɑtʰ fɑloʊiŋ"))
    print(tokenizer.encode("itʰ"))
    print(tokenizer.encode("wɑz"))
    print(tokenizer.encode("ðis"))
    print(tokenizer.encode("ɑbzəɻveiʃən"))
    print(tokenizer.decode([12,13,74]))
    device = next(model.parameters()).device

    waveform, source_sample_rate = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform[:1]
    if source_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sample_rate, sample_rate)

    waveform = waveform.to(device)
    lengths = torch.tensor([waveform.size(-1)], device=device)

    with torch.no_grad():
        outputs, output_lengths = model(waveform, lengths)
    decoded_texts, decoded_token_ids = greedy_decode_to_text_list(
        outputs,
        output_lengths,
        tokenizer=tokenizer,
        blank_id=0,
    )

    print(f"input_shape={tuple(waveform.shape)}")
    print(f"output_shape={tuple(outputs.shape)}")
    print(f"output_lengths_shape={tuple(output_lengths.shape)}")
    print(f"output_lengths={output_lengths.tolist()}")
    print(f"decoded_token_ids={decoded_token_ids}")
    print(f"decoded_texts={decoded_texts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/mnt/workspace/liuhw/experiments/ctc_tts/mono_phone/m2_wavlmbase_ctc_monophone/monophone.json")
    parser.add_argument("--audio", default="/mnt/workspace/liuhw/streaming_tts/121_127105_000007_000002.wav")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    test_wavlm_encoder(
        config_path=args.config,
        audio_path=args.audio,
        device=args.device,
        sample_rate=args.sample_rate,
    )
