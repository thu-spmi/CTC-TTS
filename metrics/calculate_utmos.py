import torch

import os
import librosa
from tqdm import tqdm

import sys
import json
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/mnt/workspace2/liuhw/ckpts/utmos/utmos22_strong_step7459_v1.pt"
predictor = None


def get_predictor():
    global predictor
    if predictor is not None:
        return predictor

    from .SpeechMOS.speechmos.utmos22.strong.model import UTMOS22Strong

    predictor = UTMOS22Strong()
    state_dict = torch.load(model_path)
    predictor.load_state_dict(state_dict)
    predictor = predictor.eval().to(device)
    return predictor

def calculate_utmos(speech_path):
    predictor_model = get_predictor()
    wave, sr = librosa.load(speech_path, sr=None, mono=True)
    # assert sr == 24000
    with torch.no_grad():
        x = torch.from_numpy(wave).unsqueeze(0).to(device=device, dtype=torch.float32)
        score = predictor_model(x, sr)
    return score.item()

def calculate_utmos_from_dir(speech_dir):
    predictor_model = get_predictor()
    utmos_values = []
    # Show progress with tqdm
    for root, dirs, files in os.walk(speech_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.endswith(".wav") or file.endswith(".flac") or file.endswith(".mp3"):
                speech_path = os.path.join(root, file)
                wave, sr = librosa.load(speech_path, sr=None, mono=True)
                # assert sr == 24000
                with torch.no_grad():
                    x = torch.from_numpy(wave).unsqueeze(0).to(device=device, dtype=torch.float32)
                    score = predictor_model(x, sr)
                utmos_values.append(score.item())

    return sum(utmos_values) / len(utmos_values), utmos_values

def calculate_utmos_batch(json_file, key="ctctts_speech_path"):
    predictor_model = get_predictor()
    mos_value = []
    with open(json_file, 'r') as f:
        data = json.load(f)
        for item in tqdm(data):
            speech_path = item[key]
            if not os.path.exists(speech_path):
                print(f"Speech path not found: {speech_path}")
                continue
            wave, sr = librosa.load(speech_path, sr=None, mono=True)
            # assert sr == 24000
            with torch.no_grad():
                x = torch.from_numpy(wave).unsqueeze(0).to(device=device, dtype=torch.float32)
                score = predictor_model(x, sr)
            mos_value.append(score.item())
    return mos_value

if __name__ == '__main__':

    speech_dir = "/mnt/workspace/liuhw/datasets/seedtts/seedtts_testset/en/prompt-wavs"
    utmos_mean, utmos_values = calculate_utmos_from_dir(speech_dir)
    print(f"utmos: {utmos_mean}")
