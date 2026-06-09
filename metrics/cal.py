import os
import torch
import re
from torchmetrics import WordErrorRate, CharErrorRate

def clean_english_text(text):
    # Replace non-English characters with a space
    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Condense consecutive spaces into a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.lower()

def transcribe_audio(audio_path, model, model_type="whisper"):
    """
    Transcribe audio with an OpenAI Whisper model or a Conformer Transducer model
    
    Args:
        audio_path: Audio file path
        model: Whisper model or Conformer Transducer model
        model_type: Model type: `whisper` or `conformer-transducer`
    
    Returns:
        Transcribed text
    """
    try:
        if model_type == "whisper":
            result = model.transcribe(audio_path)
            return result["text"].strip()
        elif model_type == "conformer-transducer":
            result = model.transcribe([audio_path])
            return result[0][0].strip()
        else:
            print(f"Unsupported model type: {model_type}")
            return None
    except Exception as e:
        print(f"Error while transcribing audio: {e}")
        return None

def load_conformer_tansducer(model_path):

    """
    model_path: Local `.nemo` file path, for example
        /mnt/workspace/liuhw/sftp_dir/ckpts/conformer_transducer/stt_en_conformer_transducer_large.nemo
    """
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    return asr_model

def load_whisper_model(model_path):
    """
    Load a Whisper model
    
    Args:
        model_path: Model file path
    
    Returns:
        Whisper model
    """
    try:
        import whisper

        model = whisper.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error while loading the Whisper model: {e}")
        return None

def load_asr_model(model_path, model_type="whisper"):
    """
    Load an ASR model
    
    Args:
        model_path: Model file path
        model_type: Model type: `whisper` or `conformer-transducer`
    
    Returns:
        ASR model
    """
    if model_type == "whisper":
        return load_whisper_model(model_path)
    elif model_type == "conformer-transducer":
        return load_conformer_tansducer(model_path)
    else:
        print(f"Unsupported model type: {model_type}")
        return None

if __name__ == "__main__":
    audio_path = "/mnt/workspace/liuhw/streaming_tts/test_wav/streaming/streaming_test_smooth_seg_3.wav"
    text = "Recognizing that the current economic climate, characterized by fluctuating interest rates and unpredictable supply chain disruptions, presents unprecedented challenges for small businesses, the federal government has unveiled a comprehensive stimulus package that includes grants, low interest loans, and tax incentives designed to provide a crucial lifeline and foster long term resilience."
    
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the local model, or download it automatically if missing
    local_model_path = "/mnt/workspace/liuhw/sftp_dir/ckpts/whisper/large-v3/large-v3.pt"
    
    if os.path.exists(local_model_path):
        print(f"Loading local model from: {local_model_path}")
        model = whisper.load_model(local_model_path, device=device)
    else:
        print("Local model not found, downloading large-v3 model...")
        # Set the cache directory
        os.environ['WHISPER_CACHE_DIR'] = '/mnt/workspace/liuhw/sftp_dir/ckpts/whisper'
        model = whisper.load_model("large-v3", device=device)
    
    wer = WordErrorRate()
    cer = CharErrorRate()
    
    # Transcribe audio
    transcription = transcribe_audio(audio_path, model)
    
    if transcription:
        # Clean the text
        text = clean_english_text(text)
        transcription = clean_english_text(transcription)
        
        print(f"text={text.lower()}\ntrans={transcription.lower()}")
        
        # Compute WER and CER
        wer_value = wer(transcription.lower(), text.lower()).detach().cpu().numpy().tolist()
        cer_value = cer(transcription.lower().replace(" ", ""), text.lower().replace(" ", "")).detach().cpu().numpy().tolist()
        print(f"wer: {wer_value}, cer: {cer_value}")
    else:
        print("Transcription failed")
