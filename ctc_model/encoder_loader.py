from collections import OrderedDict
from typing import Mapping, Optional, Tuple, Type, Union

import torch
import torchaudio

from . import encoder as encoder_module
from .encoder import AbsEncoder
from .whistle.extract_fbank_feature_with_kaldi import extract_fbank_with_kaldi


CheckpointLike = Union[str, Mapping[str, object], OrderedDict]


def translate_prev_checkpoint(state_dict: OrderedDict) -> OrderedDict:
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace("decoder.", "predictor.", 1).replace("joint.", "joiner.", 1)
        key = key.replace("am.", "encoder.", 1)
        key = key.replace(".mhsencoder.", ".mhsam.")
        new_state_dict[key] = value
    return new_state_dict


def get_encoder_class(cfg: dict) -> Type[AbsEncoder]:
    if "encoder" not in cfg:
        raise KeyError("Missing 'encoder' field in config.")

    encoder_type = cfg["encoder"].get("type")
    if not encoder_type:
        raise KeyError("Missing 'type' in encoder config.")

    encoder_cls = getattr(encoder_module, encoder_type, None)
    if not isinstance(encoder_cls, type) or not issubclass(encoder_cls, AbsEncoder):
        available_encoders = sorted(
            name
            for name, value in vars(encoder_module).items()
            if isinstance(value, type) and issubclass(value, AbsEncoder) and value is not AbsEncoder
        )
        raise NameError(
            f"'{encoder_type}' is not a valid AbsEncoder subclass. "
            f"Available encoders: {', '.join(available_encoders)}."
        )

    return encoder_cls


def build_encoder(cfg: dict) -> AbsEncoder:
    encoder_cfg = cfg["encoder"]
    encoder_cls = get_encoder_class(cfg)
    return encoder_cls(**encoder_cfg["kwargs"])


def _load_checkpoint(path_or_checkpoint: CheckpointLike) -> Mapping[str, object]:
    if isinstance(path_or_checkpoint, str):
        return torch.load(path_or_checkpoint, map_location="cpu", weights_only=False)
    return path_or_checkpoint


def _normalize_state_dict(path_or_checkpoint: CheckpointLike) -> OrderedDict:
    checkpoint = _load_checkpoint(path_or_checkpoint)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    normalized_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        normalized_state_dict[key] = value
    return translate_prev_checkpoint(normalized_state_dict)


def extract_encoder_state_dict(path_or_checkpoint: CheckpointLike) -> OrderedDict:
    state_dict = _normalize_state_dict(path_or_checkpoint)

    # Encoder weights from an external s2p pretrained checkpoint carry an "s2p_model." prefix
    # (the same file may also contain p2g_model. / projector.);
    # keep only the s2p_model.* entries and strip the prefix.
    if any(key.startswith("s2p_model.") for key in state_dict):
        return OrderedDict(
            (key[len("s2p_model.") :], value)
            for key, value in state_dict.items()
            if key.startswith("s2p_model.")
        )

    if any(key.startswith("encoder.") for key in state_dict):
        return OrderedDict(
            (key[len("encoder.") :], value)
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        )

    return state_dict


def load_pretrained_encoder(
    cfg: dict,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> AbsEncoder:
    model = build_encoder(cfg)
    model_path = cfg["encoder"].get("model_path")
    if model_path:
        model.load_state_dict(
            extract_encoder_state_dict(model_path),
            strict=strict,
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()
    return model


def load_encoder_input_for_ctc(
    audio_path: str,
    cfg: dict,
    device: Union[str, torch.device],
    sample_rate: int = 16000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    encoder_cfg = cfg["encoder"]

    if encoder_cfg.get("kaldi_fbank_in", False):
        fbank_config = encoder_cfg.get("fbank_config")
        fbank_features = extract_fbank_with_kaldi(audio_path, fbank_config)
        if isinstance(fbank_features, bool) and not fbank_features:
            raise RuntimeError(f"Failed to extract kaldi fbank features from {audio_path}")
        inputs = fbank_features.unsqueeze(0).to(device)
        lengths = torch.tensor([inputs.size(1)], device=device)
        return inputs, lengths

    waveform, source_sample_rate = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform[:1]
    if source_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sample_rate, sample_rate)
    inputs = waveform.to(device)
    lengths = torch.tensor([inputs.size(-1)], device=device)
    return inputs, lengths


def infer_ctc_logits(
    audio_path: str,
    encoder: AbsEncoder,
    cfg: dict,
    device: Union[str, torch.device],
    sample_rate: int = 16000,
) -> torch.Tensor:
    encoder_inputs, lengths = load_encoder_input_for_ctc(
        audio_path,
        cfg,
        device,
        sample_rate=sample_rate,
    )
    with torch.no_grad():
        logits, output_lengths = encoder(encoder_inputs, lengths)
    valid_length = int(output_lengths[0].item())
    return logits[0, :valid_length]
