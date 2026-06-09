import torch
from typing import Optional

from ctc_model.ctc_align import align_text_logits_and_speech_tokens, encode_text_with_separators
from ctc_model.encoder_loader import infer_ctc_logits, load_encoder_input_for_ctc, load_pretrained_encoder
from ctc_model.tokenizer import load_tknz
from src.gpt2 import GPTConfig, GPT
from src.nac import build_nac


class ModelHandler:
    def __init__(self, config, device_id: Optional[int] = None):
        self.config = config
        if torch.cuda.is_available() and device_id is not None:
            cuda_index = int(device_id)
            torch.cuda.set_device(cuda_index)
            device_str = f"cuda:{cuda_index}"
        else:
            device_str = "cpu"
            
        self.device = torch.device(device_str)
        print(f"Initializing model handler on {self.device}")

        self.nac = self.initialize_nac()
        self.encoder = self.initialize_encoder()
        self.tokenizer = self.initialize_tokenizer()
        self.whistle_encoder = self.encoder
        self.model = self.initialize_gpt_model()

    def initialize_nac(self):
        print(f"Loading NAC from {self.config['nac']['nac_model']} on {self.device}")
        return build_nac(self.config, device=self.device)

    def initialize_encoder(self):
        print(f"Loading encoder from {self.config['encoder']['model_path']} on {self.device}")
        return load_pretrained_encoder(self.config, device=self.device)

    def initialize_tokenizer(self):
        print(f"Loading tokenizer from {self.config['tokenizer']['file']}")
        return load_tknz(self.config["tokenizer"]["file"])


    def tokenize_text(self, text: str):
        tokenization = encode_text_with_separators(text, self.tokenizer, self.config)
        return text, tokenization

    def load_audio_for_ctc(self, audio_path: str, sample_rate: int = 16000):
        return load_encoder_input_for_ctc(
            audio_path,
            self.config,
            self.device,
            sample_rate=sample_rate,
        )

    def infer_ctc_logits(self, audio_path: str, sample_rate: int = 16000):
        return infer_ctc_logits(
            audio_path,
            self.encoder,
            self.config,
            self.device,
            sample_rate=sample_rate,
        )

    def align_text_and_speech(self, audio_path: str, text: str, speech_token_ids):
        logits = self.infer_ctc_logits(audio_path)
        # Keep encoder inference on GPU, but run forced alignment on CPU for stability.
        if isinstance(logits, torch.Tensor) and logits.is_cuda:
            logits = logits.detach().cpu()
        alignment_result = align_text_logits_and_speech_tokens(
            text=text,
            logits=logits,
            tokenizer=self.tokenizer,
            speech_token_ids=speech_token_ids,
            config=self.config,
        )
        return text, alignment_result

    def initialize_gpt_model(self) -> GPT:
        print(f"Loading GPT model from {self.config['inference']['model_path']}")
        checkpoint = torch.load(self.config['inference']['model_path'], map_location=self.device)
        checkpoint_model_args = checkpoint['model_args']
        model_args = {k: checkpoint_model_args[k] for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']}
        model_args["is_train"] = False
        model_args["pad_token"] = checkpoint_model_args.get(
            'pad_token',
            self.config["model"].get('pad_token', self.config["data"]["pad"]),
        )
        model_args['length_aligned_input'] = checkpoint_model_args.get('length_aligned_input', False)
        if model_args['length_aligned_input']:
            model_args['text_embd_dim'] = checkpoint_model_args.get(
                'text_embd_dim',
                self.config["model"].get('text_embd_dim', 256),
            )
            model_args['speech_embd_dim'] = checkpoint_model_args.get(
                'speech_embd_dim',
                self.config["model"].get('speech_embd_dim', 512),
            )
            model_args['speech_vocab_size'] = checkpoint_model_args.get(
                'speech_vocab_size',
                self.config["model"].get('speech_vocab_size', self.config["nac"]["codebook_size"]),
            )
            model_args['text_vocab_size'] = checkpoint_model_args.get(
                'text_vocab_size',
                self.config["model"].get('text_vocab_size', model_args['vocab_size'] - model_args['speech_vocab_size']),
            )
            model_args['text_pad_token'] = checkpoint_model_args.get(
                'text_pad_token',
                self.config["model"].get('text_pad_token', model_args['pad_token'] - model_args['speech_vocab_size']),
            )
        # Create model with loaded config
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # Load state dict from checkpoint
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
