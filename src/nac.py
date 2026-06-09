from abc import ABC, abstractmethod
import torch
import torchaudio
from WavTokenizer.decoder.pretrained import WavTokenizer
from WavTokenizer.encoder.utils import convert_audio


class NACBase(ABC):
    @abstractmethod
    def encode(self, audio_path: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, features_input: torch.Tensor, **kwargs):
        raise NotImplementedError


class WavTokenizerNAC(NACBase):
    def __init__(self, config_path, model_path, device=None, target_sr=24000, target_channels=1):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.model = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, audio_path: str, bandwidth_id: int = 0, **kwargs):
        """
        Inputs:
            audio_path (str): Audio file path; after loading, a waveform tensor with shape `[C, T]` is expected
            bandwidth_id (int): Bandwidth-selection ID for WavTokenizer; only `0` is supported
        Outputs:
            features (torch.Tensor): Speech features, typically with shape `[B, C, L]`
            discrete_codes (torch.Tensor): Discrete codes, typically with shape `[B, Nq, L]`
        """
        wav, sr = torchaudio.load(audio_path)
        if sr != self.target_sr or wav.shape[0] != self.target_channels:
            wav = convert_audio(wav, sr, self.target_sr, self.target_channels)
        wav = wav.to(self.device)
        bandwidth_id = torch.tensor([bandwidth_id], device=self.device)
        with torch.no_grad():
            features, discrete_codes = self.model.encode_infer(wav, bandwidth_id=bandwidth_id, **kwargs)
        return discrete_codes

    def encode_infer(self, audio_path: str, bandwidth_id: int = 0, **kwargs):
        """
        Inputs:
            audio_path (str): Audio file path; after loading, a waveform tensor with shape `[C, T]` is expected
            bandwidth_id (int): Bandwidth-selection ID for WavTokenizer; only `0` is supported
        Outputs:
            features (torch.Tensor): Speech features, typically with shape `[B, C, L]`
            discrete_codes (torch.Tensor): Discrete codes, typically with shape `[B, Nq, L]`
        """
        return self.encode(audio_path, bandwidth_id=bandwidth_id, **kwargs)

    def decode(self, discrete_codes: torch.Tensor, **kwargs):
        """
        Inputs:
            discrete_codes (torch.Tensor or list): Discrete codes, typically with shape `[B, Nq, L]` / [Nq, L]
        Outputs:
            audio_output (torch.Tensor): Reconstructed audio, typically with shape `[B, T]`
        """

        if isinstance(discrete_codes, list):
            discrete_codes = torch.stack(discrete_codes, dim=0)
        
        if discrete_codes.dim() == 1:
            discrete_codes = discrete_codes.unsqueeze(0)

        discrete_codes = discrete_codes.to(self.device)
        with torch.no_grad():
            features = self.model.codes_to_features(discrete_codes)
            bandwidth_id = torch.tensor([0], device=self.device)
            audio_out = self.model.decode(features, bandwidth_id=bandwidth_id)
        return audio_out


def build_nac(config, device=None):
    nac_cfg = config.get("nac", {})
    nac_type = nac_cfg.get("type", "").lower()
    if nac_type == "wavtokenizer":
        config_path = nac_cfg.get("config_path", nac_cfg.get("nac_config"))
        model_path = nac_cfg.get("model_path", nac_cfg.get("nac_model"))
        target_sr = nac_cfg.get("sample_rate", 24000)
        target_channels = nac_cfg.get("channels", 1)
        return WavTokenizerNAC(
            config_path=config_path,
            model_path=model_path,
            device=device,
            target_sr=target_sr,
            target_channels=target_channels,
        )
    raise ValueError(f"Unknown NAC type: {nac_type}")
