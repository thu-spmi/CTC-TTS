import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Sequence, Dict

from .syllabify import syllabify_tokens

from .utils import build_length_aligned_text_speech_blockwise

class PhonemeAndSpeechDataset(Dataset):
    def __init__(self, data_path, future_speech_token_num=0, eob_token=4176, eos_token=4178, codebook_size=4096, silence_token=453, keep_silence_num=-1, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize trainging dataset
        Args:
            data_path (str): Path to a JSON file containing interleaved sequences of speech tokens and phonemes
            future_speech_token_num(int): Number of future speech tokens added at current block, default is 3
            eob_token (int): End of block token, default is 4176
            eos_token (int): End of text token, default is 4178
            codebook_size (int): Codebook size for wavtokenizer, default is 4096
            silence_token (int): Silence token
            keep_silence_num (int): If a block contains consecutive `silence_token`s, keep only `keep_silence_num` of them. The default `-1` means keep all of them
            device (str): Device to use for the dataset, default is CUDA if available, otherwise CPU
        """
        super(PhonemeAndSpeechDataset, self).__init__()

        with open(data_path, 'r') as f:
            self.list_data_dict = json.load(f)
        self.future_speech_token_num = future_speech_token_num
        self.eob_token = eob_token
        self.eos_token = eos_token
        self.codebook_size = codebook_size
        self.silence_token = silence_token
        self.keep_silence_num = keep_silence_num

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Get a single data item by index.
        
        Args:
            idx: Index of the data item
            
        Returns:
            seq: Tensor of shape (seq_len,) containing the interleaved sequences of speech  and phonemes
        """
        item = self.list_data_dict[idx]
        seq = item["seqs"]

        # At the end of the current block, before `eob`, append the first `future_speech_token_num` speech tokens from the next block
        if isinstance(seq, str):
            seq = [int(x) for x in seq.strip().split()]
        eob = self.eob_token
        cb = self.codebook_size
        out, i = [], 0
        while i < len(seq):
            j = i
            while j < len(seq) and seq[j] != eob:
                j += 1
            out.extend(seq[i:j])
            k, m = j + 1, j + 1
            while m < len(seq) and seq[m] != eob:
                m += 1
            out.extend([t for t in seq[k:m] if t < cb][:self.future_speech_token_num])
            if j < len(seq):
                out.append(eob)
            i = j + 1

        # If a block contains consecutive `silence_token`s, keep only `keep_silence_num` of them
        if self.keep_silence_num >= 0:
            compressed = []
            i = 0
            sil = self.silence_token
            eob = self.eob_token
            ksn = self.keep_silence_num
            while i < len(out):
                token = out[i]
                if token == eob:
                    compressed.append(token)
                    i += 1
                    continue
                if token == sil:
                    j = i
                    while j < len(out) and out[j] == sil:
                        j += 1
                    run_len = j - i
                    keep = run_len if run_len <= ksn else ksn
                    if keep > 0:
                        compressed.extend([sil] * keep)
                    i = j
                    continue
                compressed.append(token)
                i += 1
            out = compressed
        
        seq = torch.tensor(out, dtype=torch.long)

        return seq

@dataclass
class PhonemeAndSpeechDataCollator:
    """Collator for batching data examples."""

    def __init__(self, enable_length_align, eob_token=4176, pad_token=4177, silence_token=453, silence_num=0, codebook_size=4096, need_offset=True):

        """
        Initialize the collator.
        
        Args:
            enable_length_align (bool): Whether to enable length alignment for batching
            eob_token (int): End-of-block token, default is 4176
            pad_token (int): Pad token, default is 4177
            silence_token(int): silence token of wavtokenizer, default is 453 for large checkpoint
            silence_num (int): Number of silence tokens to insert at the beginning of the first block, default is 0
            codebook_size (int): codebook_size of wavtokenizer, default is 4096
            need_offset (bool): Whether to apply causal shift to targets, default is True
        """

        self.enable_length_align = enable_length_align
        self.eob_token = eob_token
        self.pad_token = pad_token
        self.silence_token = silence_token
        self.silence_num = silence_num
        self.codebook_size = codebook_size
        self.speech_upper = codebook_size - 1
        self.zero_speech_token = -1 # Marker for the all-zero tensor position when stacking text and speech embeddings
        self.need_offset = need_offset # Whether to shift
    
    def __call__(self, instances: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            instances: List of training sequences
            
        Returns:
            batch: Batched and padded data
        """

        if not self.enable_length_align:    # Concatenate text and speech along the length dimension
            # Build `targets` (speech tokens and `eob`) and return `seqs/targets`
            targets = []
            for seq in instances:
                mask = (seq < self.codebook_size) | (seq == self.eob_token) # | (seq == 4178)
                target = torch.where(mask, seq, torch.tensor(-100, dtype=seq.dtype))
                if self.need_offset:
                    target = torch.cat([target[1:], torch.tensor([-100], dtype=seq.dtype)])
                targets.append(target)
            targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
            seqs = torch.nn.utils.rnn.pad_sequence(instances, batch_first=True, padding_value=self.pad_token )
            batch = {"seqs": seqs, "targets": targets}
            return batch

        # Align text and speech by length, return `text_seqs` and `speech_seqs` with the same length, and also return `targets` (speech tokens and `eob` token)
        # Used during training when text and speech are concatenated on the feature dimension
        text_seqs_list = []
        speech_seqs_list = []
        text_seqs_raw = []
        speech_seqs_raw = []
        for seq in instances:
            text_seq, speech_seq = build_length_aligned_text_speech_blockwise(
                seq=[int(t) for t in seq.tolist()],
                eob_token=self.eob_token,
                pad_token=self.pad_token,
                speech_upper=self.speech_upper,
                zero_speech_token=self.zero_speech_token,
                silence_token=self.silence_token,
                silence_num=self.silence_num,
            )
            text_seqs_raw.append(text_seq)
            speech_seqs_raw.append(speech_seq)
            text_seqs_list.append(torch.tensor(text_seq, dtype=torch.long))
            speech_seqs_list.append(torch.tensor(speech_seq, dtype=torch.long))
        # Generate `targets` containing only speech tokens and the `eob` token
        targets_list = []
        for text_seq, speech_seq in zip(text_seqs_raw, speech_seqs_raw):
            L = min(len(text_seq), len(speech_seq))
            tgt_row = []
            for i in range(L):
                s = speech_seq[i]
                t = text_seq[i]
                if 0 <= s <= self.speech_upper:
                    tgt_row.append(s)
                elif t == self.eob_token:
                    tgt_row.append(self.eob_token)
                else:
                    tgt_row.append(-100)
            # Whether to shift by one position
            if self.need_offset and len(tgt_row) > 0:
                tgt_row = tgt_row[1:] + [-100]
            # If no shift is needed, keep the unshifted `tgt_row`
            targets_list.append(torch.tensor(tgt_row, dtype=torch.long))
        text_seqs = torch.nn.utils.rnn.pad_sequence(text_seqs_list, batch_first=True, padding_value=self.pad_token)
        speech_seqs = torch.nn.utils.rnn.pad_sequence(speech_seqs_list, batch_first=True, padding_value=self.zero_speech_token)
        targets = torch.nn.utils.rnn.pad_sequence(targets_list, batch_first=True, padding_value=-100)
        batch = {"text_seqs": text_seqs, "speech_seqs": speech_seqs, "targets": targets}
        return batch

def create_data_module(config, enable_length_align, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Create a data module for training.
    
    Args:
        config: Configuration object containing train_data and valid_data parameters
        enable_length_align: Whether to enable length alignment for data collation
        device: Device to use for the dataset, default is CUDA if available, otherwise CPU
        
    Returns:
        data_module: Dictionary containing dataset and collator
    """

    train_data_path = config["data"]["train_data"]
    valid_data_path = config["data"]["valid_data"]

    codebook_size = config["nac"]["codebook_size"]
    eob_token = config["data"]["eob"]
    pad_token = config["data"]["pad"]
    eos_token = config["data"]["eos"]

    future_speech_token_num = config["data"].get("future_speech_token_num", 0)

    keep_silence_num = config["data"].get("keep_silence_num", -1)
    silence_token = config["nac"]["SILENCE_TOKEN_ID"]
    silence_num = config["data"].get("silence_num", 0)

    train_dataset = PhonemeAndSpeechDataset(train_data_path, codebook_size=codebook_size, eob_token=eob_token, eos_token=eos_token, future_speech_token_num=future_speech_token_num, keep_silence_num=keep_silence_num, silence_token=silence_token, device=device)
    valid_dataset = PhonemeAndSpeechDataset(valid_data_path, codebook_size=codebook_size, eob_token=eob_token, eos_token=eos_token, future_speech_token_num=future_speech_token_num, keep_silence_num=keep_silence_num, silence_token=silence_token, device=device)

    if enable_length_align:
        data_collator = PhonemeAndSpeechDataCollator(enable_length_align=enable_length_align, eob_token=eob_token, pad_token=pad_token, silence_token=silence_token, codebook_size=codebook_size, silence_num=silence_num, need_offset=config["data"]["need_offset"])
    else:
        data_collator = PhonemeAndSpeechDataCollator(enable_length_align=enable_length_align, eob_token=eob_token, pad_token=pad_token, codebook_size=codebook_size, need_offset=config["data"]["need_offset"])
    return {
        'train_dataset': train_dataset,
        'valid_dataset': valid_dataset, 
        'data_collator': data_collator
    }


def create_dataloader(data_module, batch_size=2, shuffle=True):
    """
    Create a DataLoader from a data module.
    
    Args:
        data_module: Data module containing dataset and collator
        batch_size: Batch size for training
        shuffle: Whether to shuffle the tarin data
        
    Returns:
        train_dataloader: DataLoader for training
        valid_dataloader: DataLoader for validation
    """

    train_dataloader = DataLoader(
        data_module['train_dataset'],
        batch_size=batch_size,
        collate_fn=data_module['data_collator'],
        shuffle=shuffle
    )

    valid_dataloader = DataLoader(
        data_module['valid_dataset'],
        batch_size=batch_size,
        collate_fn=data_module['data_collator'],
        shuffle=shuffle
    )

    return train_dataloader, valid_dataloader
    
