import torch
import torch.nn.functional as F
import json
import torchaudio

from inference.model_handler import ModelHandler

from src.utils import generate_unit_list
from src.utils import extract_unit_from_seqs, split_sequence_by_speech_tokens, build_length_aligned_text_speech_blockwise, sample_token_from_probs


class MultiSpeakerInferencer:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        device_id_1 = self.config["inference"]["tts_device_1"]
        self.model_handler = ModelHandler(self.config, device_id=device_id_1)

    def _encode_speech_tokens(self, wav_path):
        return self.model_handler.nac.encode(wav_path).tolist()[0][0]

    def _save_audio(self, speech_tokens, output_path):
        speech_token_tensor = torch.tensor([speech_tokens], dtype=torch.long)
        audio_out = self.model_handler.nac.decode(speech_token_tensor)
        torchaudio.save(
            output_path,
            audio_out.to("cpu"),
            sample_rate=self.config["inference"].get("sample_rate", 24000),
            encoding="PCM_S",
            bits_per_sample=16,
        )
        return output_path

    def _run_length_aligned_step(self, text_tokens, speech_tokens, kvcache=None):
        text_tensor = torch.tensor([text_tokens], dtype=torch.long, device=self.model_handler.device)
        speech_tensor = torch.tensor([speech_tokens], dtype=torch.long, device=self.model_handler.device)
        return self.model_handler.model(tokens={"text_seqs": text_tensor, "speech_seqs": speech_tensor}, kvcache=kvcache)

    def _run_sequential_step(self, seq_tokens, kvcache=None):
        input_seq = torch.tensor([seq_tokens], dtype=torch.long, device=self.model_handler.device)
        return self.model_handler.model(input_seq, kvcache=kvcache)

    def _prime_decoder(self, seq):
        if self.model_handler.model.config.length_aligned_input:
            text_seq, speech_seq = build_length_aligned_text_speech_blockwise(
                seq=seq,
                eob_token=self.config["data"]["eob"],
                pad_token=self.config["data"]["pad"],
                speech_upper=self.config["nac"]["codebook_size"] - 1,
            )
            text_pre = []
            speech_pre = []
            kvcache = None
            speech_decoder_output = None
            for text_token, speech_token in zip(text_seq, speech_seq):
                text_pre.append(text_token)
                speech_pre.append(speech_token)
                speech_decoder_output, _, kvcache = self._run_length_aligned_step(text_pre, speech_pre, kvcache=kvcache)
            return {"text_pre": text_pre, "speech_pre": speech_pre, "kvcache": kvcache, "output": speech_decoder_output}

        seq_pre = []
        kvcache = None
        speech_decoder_output = None
        for token in seq:
            seq_pre.append(token)
            speech_decoder_output, _, kvcache = self._run_sequential_step(seq_pre, kvcache=kvcache)
        return {"seq_pre": seq_pre, "kvcache": kvcache, "output": speech_decoder_output}

    def _generate_from_units(self, initial_seq, units):
        speech_tokens = []
        codebook_size = self.config["nac"]["codebook_size"]
        eob_token = self.config["data"]["eob"]
        pad_token = self.config["data"]["pad"]

        with torch.inference_mode():
            state = self._prime_decoder(initial_seq)
            if self.model_handler.model.config.length_aligned_input:
                text_pre = state["text_pre"]
                speech_pre = state["speech_pre"]
                kvcache = state["kvcache"]
                speech_decoder_output = state["output"]

                for unit in units:
                    text_point = 0
                    while True:
                        logits = speech_decoder_output[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        current_token = sample_token_from_probs(probs, self.config)
                        if current_token == eob_token:
                            text_pre.append(eob_token)
                            speech_pre.append(-1)
                            speech_decoder_output, _, kvcache = self._run_length_aligned_step(text_pre, speech_pre, kvcache=kvcache)
                            break
                        if current_token >= codebook_size:
                            raise ValueError(f"Generated an invalid token {current_token}")
                        speech_tokens.append(current_token)
                        if text_point < len(unit):
                            text_pre.append(unit[text_point])
                            text_point += 1
                        else:
                            text_pre.append(pad_token)
                        speech_pre.append(current_token)
                        speech_decoder_output, _, kvcache = self._run_length_aligned_step(text_pre, speech_pre, kvcache=kvcache)
                return speech_tokens

            seq_pre = state["seq_pre"]
            kvcache = state["kvcache"]
            speech_decoder_output = state["output"]
            for unit in units:
                for token in unit:
                    seq_pre.append(token)
                    speech_decoder_output, _, kvcache = self._run_sequential_step(seq_pre, kvcache=kvcache)
                while True:
                    logits = speech_decoder_output[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    current_token = sample_token_from_probs(probs, self.config)
                    seq_pre.append(current_token)
                    if current_token == eob_token:
                        speech_decoder_output, _, kvcache = self._run_sequential_step(seq_pre, kvcache=kvcache)
                        break
                    if current_token >= codebook_size:
                        raise ValueError(f"Generated an invalid token {current_token}")
                    speech_tokens.append(current_token)
                    speech_decoder_output, _, kvcache = self._run_sequential_step(seq_pre, kvcache=kvcache)
            return speech_tokens

    def _extract_cross_prefix(self, first_unit):
        unit_token_start = self.config["data"]["unit_token_start"]
        unit_token_end = self.config["data"]["unit_token_end"]
        prefix = []
        for token in first_unit:
            if unit_token_start <= token <= unit_token_end:
                prefix.append(token)
            else:
                break
        if not prefix:
            raise ValueError("Failed to extract the initial text token for cross-sentence inference from the target text.")
        return prefix

    def infer_cross(self, wav_path, prompt_text, target_text, output_path="cross.wav"):
        self.config["inference"]["task_type"] = "cross"
        eos_token = self.config["data"]["eos"]
        speech_tokens = self._encode_speech_tokens(wav_path)
        _, prompt_alignment = self.model_handler.align_text_and_speech(wav_path, prompt_text, speech_tokens)

        print(f"prompt_alignment.seqs_sequence: {prompt_alignment.seqs_sequence}")
        _, target_tokenization = self.model_handler.tokenize_text(target_text)
        print(f"target_tokenization.shifted_token_ids_with_separators: {target_tokenization.shifted_token_ids_with_separators}")
        units = generate_unit_list(self.config, target_tokenization.shifted_token_ids_with_separators)
        if not units:
            raise ValueError("The target text did not produce any text units.")
        units[-1].append(eos_token)
        print(f"units: {units}")
        prompt_sequence = list(prompt_alignment.seqs_sequence)
        replace_cross_eos = self.config["inference"].get("replace_cross_eos_with_target_prefix", False)
        
        if replace_cross_eos: # Legacy training sequence: use the first word of the next sentence as the future word for the last word of the current sentence
            if eos_token not in prompt_sequence:
                raise ValueError("The prompt sequence does not contain `eos`, so cross inference cannot be performed.")
            eos_index = len(prompt_sequence) - 1 - prompt_sequence[::-1].index(eos_token)
            prompt_sequence = (
                prompt_sequence[:eos_index]
                + self._extract_cross_prefix(units[0])
                + prompt_sequence[eos_index + 1:]
            )
        else: # Remove `eos` from `prompt_sequence`
            prompt_sequence = [token for token in prompt_sequence if token != eos_token]

        speech_token = self._generate_from_units(prompt_sequence, units)
        return self._save_audio(speech_token, output_path)

    def infer_continuation(self, wav_path, target_text, output_path="continuous.wav"):
        self.config["inference"]["task_type"] = "continuation"
        speech_tokens = self._encode_speech_tokens(wav_path)
        _, alignment_result = self.model_handler.align_text_and_speech(wav_path, target_text, speech_tokens)

        codebook_size = self.config["nac"]["codebook_size"]
        eob_token = self.config["data"]["eob"]
        eos_token = self.config["data"]["eos"]
        s_prev, s_next, _ = split_sequence_by_speech_tokens(
            alignment_result.seqs_sequence,
            self.config["nac"]["tokens_per_second"] * 3,
            eob_token,
            codebook_size=codebook_size,
        )
        speech_token = [token for token in s_prev if token < codebook_size]
        # self._save_audio(speech_token, "continuous_pre.wav")

        text_tokens = extract_unit_from_seqs(
            s_next,
            block_type=self.config['data']['block_type'],
            eob_token=eob_token,
            codebook_size=codebook_size,
            unit_token_start=self.config["data"]["unit_token_start"],
            unit_token_end=self.config["data"]["unit_token_end"],
            separator_token_start=self.config["data"]["separator_token_start"],
            separator_token_end=self.config["data"]["separator_token_end"],
        )
        text_tokens = [token for unit in text_tokens for token in unit]
        units = generate_unit_list(self.config, text_tokens)
        if not units:
            raise ValueError("The continuation segment did not generate any text units.")
        units[-1].append(eos_token)
        print(f"s_prev: {s_prev}\n s_next: {s_next}\n text_tokens: {text_tokens}\n units: {units}")
        generated_speech_tokens = self._generate_from_units(s_prev, units)
        return self._save_audio(speech_token + generated_speech_tokens, output_path)


if __name__ == "__main__":
    inferencer = MultiSpeakerInferencer(
    "/mnt/workspace/liuhw/streaming_tts/exp/librispeech/m6/m6_librispeech_o19_wavlarge_length_next0_word_fw1_keepsil3.json"
    )

    # Cross task
    inferencer.infer_cross(
        wav_path="/mnt/workspace/liuhw/streaming_tts/test_3s.wav",
        prompt_text="Perhaps I am mistaken, answered she.",
        target_text="Oh my Lord, cried Miss Woodley, with a most forcible accent, You are the last person on earth she would pardon me for entrusting.",
        output_path="cross.wav",
    )

    # Continuation task
    inferencer.infer_continuation(
        wav_path="/mnt/workspace/liuhw/streaming_tts/test_8s.wav",
        target_text="Oh my Lord, cried Miss Woodley, with a most forcible accent, You are the last person on earth she would pardon me for entrusting.",
        output_path="continuous.wav",
    )
