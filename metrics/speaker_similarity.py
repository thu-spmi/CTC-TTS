import torch
import torchaudio
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor


class SimpleSpeakerExtractor:
    def __init__(self, model_path: str = "microsoft/wavlm-base-plus-sv", device: str = None):
        """
        Simple speaker embedding extractor based on WavLMForXVector.

        Args:
            model_path: Hugging Face Hugging Face model name or local directory path
            device: Device string such as `cuda` or `cpu`; automatically selected by default
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        if model_path.startswith("/") or model_path.startswith("./"):
            self.model = WavLMForXVector.from_pretrained(model_path).to(device)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        else:
            self.model = WavLMForXVector.from_pretrained(model_path).to(device)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

        self.model.eval()

    def extract_speaker_vector(self, audio_path_or_tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract a speaker embedding vector from a single audio sample.

        Args:
            audio_path_or_tensor: Audio file path, or a waveform tensor with shape [1, T]
            sample_rate: Target sample rate

        Returns:
            Speaker vector with shape [D] after squeezing the batch dimension
        """
        with torch.no_grad():
            if isinstance(audio_path_or_tensor, str):
                waveform, sr = torchaudio.load(audio_path_or_tensor)
                if sr != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            else:
                waveform = audio_path_or_tensor

            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            speaker_vector = outputs.embeddings

            return speaker_vector.squeeze(0)


def create_default_speaker_extractor(
    model_path: str = "microsoft/wavlm-base-plus-sv",
    device: str = None,
) -> SimpleSpeakerExtractor:
    """
    Instantiate a default SimpleSpeakerExtractor.

    Args:
        model_path: Hugging Face model name or local path
        device: Device string; automatically selected by default

    Returns:
        Initialized SimpleSpeakerExtractor instance
    """
    return SimpleSpeakerExtractor(model_path=model_path, device=device)


def compute_speaker_similarity(
    audio_path_1: str,
    audio_path_2: str,
    extractor: SimpleSpeakerExtractor,
    sample_rate: int = 16000,
) -> float:
    """
    Compute speaker similarity between two audio clips using cosine similarity.

    Args:
        audio_path_1: Path to the first audio file
        audio_path_2: Path to the second audio file
        extractor: Speaker embedding extractor instance
        sample_rate: Target sample rate used during extraction

    Returns:
        Cosine similarity scalar, approximately in [-1, 1]; values closer to 1 mean more similar speakers
    """
    with torch.no_grad():
        v1 = extractor.extract_speaker_vector(audio_path_1, sample_rate=sample_rate)
        v2 = extractor.extract_speaker_vector(audio_path_2, sample_rate=sample_rate)

        v1 = v1.float()
        v2 = v2.float()

        v1_norm = v1 / (v1.norm(p=2) + 1e-12)
        v2_norm = v2 / (v2.norm(p=2) + 1e-12)

        similarity = torch.dot(v1_norm, v2_norm).item()
        return float(similarity)

if __name__ == "__main__":
    extractor = create_default_speaker_extractor("/mnt/workspace/liuhw/LLMVoX-main/CHECKPOINTS/wavlm", "cuda")
    similarity = compute_speaker_similarity(
        "/mnt/workspace/liuhw/streaming_tts/test_8s.wav",
        "/mnt/workspace/liuhw/streaming_tts/continuous.wav",
        extractor,
    )
    print(similarity)