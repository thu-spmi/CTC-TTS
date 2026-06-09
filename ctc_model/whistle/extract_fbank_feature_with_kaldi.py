import os

import numpy as np
import torch


TARGET_SAMPLE_RATE = 16000.0
NUM_MEL_BINS = 80


def _build_fbank_options(config_file):
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    _ = config_file
    opts.frame_opts.samp_freq = TARGET_SAMPLE_RATE
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = NUM_MEL_BINS
    opts.mel_opts.debug_mel = False
    return opts


def _load_audio(audio_file):
    import torchaudio

    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"audio file not found: {audio_file}")

    waveform, sample_rate = torchaudio.load(audio_file)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() != 2:
        raise ValueError(f"unexpected waveform shape: {tuple(waveform.shape)}")
    waveform = waveform.squeeze(0).contiguous().cpu().numpy().astype(np.float32, copy=False)
    # torchaudio.load() returns normalized floats in [-1, 1], while Kaldi's
    # compute-fbank-feats consumes PCM-scale samples. Scale back so the native
    # extractor matches the legacy Kaldi command output.
    waveform = waveform * 32768.0
    return waveform, sample_rate


def _maybe_resample(waveform, sample_rate, target_sample_rate):
    import torchaudio.functional as F

    if sample_rate == target_sample_rate:
        return waveform, sample_rate

    if sample_rate < target_sample_rate:
        raise ValueError(
            f"sample rate {sample_rate} is lower than target {target_sample_rate}, "
            "upsample is not supported"
        )

    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
    waveform_tensor = F.resample(waveform_tensor, sample_rate, target_sample_rate)
    waveform = waveform_tensor.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    return waveform, target_sample_rate


def _extract_fbank_np(audio_file, config_file=None):
    import kaldi_native_fbank as knf

    opts = _build_fbank_options(config_file)
    waveform, sample_rate = _load_audio(audio_file)

    target_sample_rate = int(round(opts.frame_opts.samp_freq))
    waveform, sample_rate = _maybe_resample(waveform, sample_rate, target_sample_rate)

    extractor = knf.OnlineFbank(opts)
    extractor.accept_waveform(sample_rate, waveform.tolist())
    extractor.input_finished()

    num_frames = int(extractor.num_frames_ready)
    feat_dim = int(opts.mel_opts.num_bins)
    if num_frames == 0:
        return np.empty((0, feat_dim), dtype=np.float32)

    features = np.stack([extractor.get_frame(i) for i in range(num_frames)], axis=0)
    return np.asarray(features, dtype=np.float32)


def extract_fbank_with_kaldi(audio_file, config_file=None, return_tensor=True, tensor_type='torch'):

    if return_tensor and tensor_type not in {"torch", "numpy"}:
        raise ValueError("tensor_type must be 'torch' or 'numpy'")

    try:
        features = _extract_fbank_np(audio_file, config_file)
    except Exception as e:
        print(f"Error extracting fbank features from {audio_file}: {e}")
        return False

    if return_tensor:
        if tensor_type == 'torch':
            return torch.from_numpy(features.copy()).float()
        return features

    return True
