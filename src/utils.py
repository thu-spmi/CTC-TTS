"""
Utility functions for speech prediction model training.
"""
import os
import re
import sys
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from contextlib import nullcontext
from typing import Dict, Optional, Sequence, List
from tqdm import tqdm
import numpy as np
import torchaudio

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


import editdistance

def compute_wer(pred_str: str, ref_str: str) -> float:
    pred_str = pred_str.strip()
    ref_str = ref_str.strip()
    pred_words = pred_str.split()
    ref_words = ref_str.split()
    if len(ref_words) == 0:
        return 0.0
    error = editdistance.eval(pred_words, ref_words)
    return error * 100.0 / len(ref_words)

def setup_environment(config):
    """Set up the training environment including DDP if applicable."""
    # Check if we're running with distributed data parallel
    ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1
    
    if ddp:
        init_process_group(backend=config['training']['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        
        # Adjust gradient accumulation steps proportionally 
        gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # Single GPU or CPU setup
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = 0
        device = config['training']['device']
        gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    
    # Calculate tokens per iteration for reporting
    tokens_per_iter = 0
    if config["model"]["model_type"] == "gpt2":
        tokens_per_iter = gradient_accumulation_steps * ddp_world_size * config['training']['batch_size'] * config["model"]['block_size']
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    # Create output directory
    if master_process:
        os.makedirs(config['out']['out_dir'], exist_ok=True)
    
    # Set up random seed
    torch.manual_seed(1337 + seed_offset) 
    
    # Enable TF32 precision for faster training on A100 GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Determine device type and precision
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['training']['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Return updated config along with environment variables
    config['training']['gradient_accumulation_steps'] = gradient_accumulation_steps
    
    return {
        'device': device,
        'device_type': device_type,
        'ctx': ctx,
        'master_process': master_process,
        'ddp': ddp,
        'ddp_local_rank': ddp_local_rank,
        'ddp_world_size': ddp_world_size,
        'tokens_per_iter': tokens_per_iter,
        'gradient_accumulation_steps': gradient_accumulation_steps
    }

def compute_block_speech_token_error_rate(predictions, targets, eob=4176, codebook_size=4096):
    """
    Compute the speech-token error rate for each sample’s word-speech alignment blocks and aggregate across the batch:
    - Use `eob` in `targets` to split blocks;
    - Only count positions in `targets` that correspond to speech tokens (`< codebook_size` and not `-100`);
    - For each block, call the official WER function to obtain the edit-distance ratio;
    - Finally, compute the arithmetic mean over all block error rates to get the overall average error rate, without weighting by block length.
    """

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().tolist()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().tolist()

    block_wers = []

    for pred_seq, tgt_seq in tqdm(zip(predictions, targets),
                                  total=min(len(predictions), len(targets)),
                                  desc="Block speech-token WER"):
        curr_pred_block = []
        curr_tgt_block = []
        for t_tok, p_tok in zip(tgt_seq, pred_seq):
            if t_tok == -100:
                continue
            if t_tok < codebook_size:
                # WER tokenizes by spaces, so token IDs are converted to strings and joined here
                curr_tgt_block.append(str(t_tok))
                if p_tok < codebook_size:
                    curr_pred_block.append(str(p_tok))
            if t_tok == eob:
                # Finalize the current block
                if len(curr_tgt_block) > 0:
                    # block_wer = wer_fn(" ".join(curr_pred_block), " ".join(curr_tgt_block)).item()
                    block_wer = compute_wer(" ".join(curr_pred_block), " ".join(curr_tgt_block))
                    block_wers.append(float(block_wer))
                curr_tgt_block = []
                curr_pred_block = []
        # Residual block at the end of the sequence that does not terminate with `eob`
        if len(curr_tgt_block) > 0:
            block_wer = compute_wer(" ".join(curr_pred_block), " ".join(curr_tgt_block))
            block_wers.append(float(block_wer))

    if len(block_wers) == 0:
        return 0.0
    return float(sum(block_wers) / len(block_wers))


def compute_sentence_speech_token_error_rate(predictions, targets, eob=4176, codebook_size=4096):
    """
    Compute the whole-utterance speech-token error rate (similar to WER), counting only positions in `targets` that correspond to speech tokens (`< codebook_size` and not `-100`).
    - For each sample, convert the valid speech-token sequence into a space-separated string;
    - Compute WER for each sample separately, then take the arithmetic mean over sample WERs to get the batch error rate.
    - `eob` is excluded from the error-rate calculation.
    """
    

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().tolist()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().tolist()

    sample_wers = []
    for pred_seq, tgt_seq in tqdm(zip(predictions, targets),
                                  total=min(len(predictions), len(targets)),
                                  desc="Sentence speech-token WER"):
        pred_tokens = []
        tgt_tokens = []
        for t_tok, p_tok in zip(tgt_seq, pred_seq):
            if t_tok == -100:
                continue
            if t_tok < codebook_size:
                # Only count speech positions; keep only speech tokens on the prediction side as well
                tgt_tokens.append(str(t_tok))
                if p_tok < codebook_size:
                    pred_tokens.append(str(p_tok))
        # Skip the sample if it has no valid reference tokens
        if len(tgt_tokens) == 0:
            continue
        pred_str = " ".join(pred_tokens)
        tgt_str = " ".join(tgt_tokens)
        # sample_wers.append(_wer_single(pred_str, tgt_str))
        sample_wers.append(compute_wer(pred_str, tgt_str))

    if len(sample_wers) == 0:
        return 0.0
    return float(sum(sample_wers) / len(sample_wers))


def _move_to_device(tensor, device, device_type):
    if device_type == 'cuda':
        return tensor.to(device, non_blocking=True)
    return tensor.to(device)


def prepare_batch(batch, context_length, device, device_type):
    has_dual_branch = ('text_seqs' in batch) and ('speech_seqs' in batch)
    targets = _move_to_device(batch['targets'], device, device_type)

    if has_dual_branch:
        text_seqs = _move_to_device(batch['text_seqs'], device, device_type)
        speech_seqs = _move_to_device(batch['speech_seqs'], device, device_type)
        effective_length = min(text_seqs.size(1), speech_seqs.size(1), targets.size(1), context_length)
        text_seqs = text_seqs[:, :effective_length]
        speech_seqs = speech_seqs[:, :effective_length]
        targets = targets[:, :effective_length]
        tokens = {'text_seqs': text_seqs, 'speech_seqs': speech_seqs}
    else:
        seqs = _move_to_device(batch['seqs'], device, device_type)
        effective_length = min(seqs.size(1), targets.size(1), context_length)
        seqs = seqs[:, :effective_length]
        targets = targets[:, :effective_length]
        tokens = seqs

    return tokens, targets, has_dual_branch


def fetch_batch(batch_iter, dataloader, context_length, device, device_type):
    while True:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)

        tokens, targets, has_dual_branch = prepare_batch(
            batch,
            context_length,
            device,
            device_type,
        )
        seq_len = tokens['text_seqs'].size(1) if has_dual_branch else tokens.size(1)
        if seq_len == 0 or targets.size(1) == 0:
            print("Skipping empty batch")
            continue
        return tokens, targets, batch_iter, has_dual_branch


@torch.no_grad()
def estimate_loss(model, valid_dataloader, context_length, eval_iters, ctx, device, device_type, codebook_size=4096, eob_token=4176):
    """Estimate loss over evaluation iterations."""
    print("===============begin to eval===============")
    out = {}
    model.eval()
    valid_iter = iter(valid_dataloader)
    for split in ['val']:
        losses = []
        accuracies = []
        speech_token_error_rates = []
        sentence_token_error_rates = []
        for k in range(eval_iters):
            tokens, targets, valid_iter, has_dual_branch = fetch_batch(
                valid_iter,
                valid_dataloader,
                context_length,
                device,
                device_type,
            )

            with ctx:
                try:
                    logits, loss, _ = model(tokens=tokens, targets=targets)
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
                    continue

            predictions = torch.argmax(logits, dim=-1)

            if k == 0 or k == eval_iters - 1:
                print(f"\n=== Verifying batch {k + 1} token comparison ===")
                print(f"Batch size: {targets.shape[0]}, Sequence length: {targets.shape[1]}")
                sample_idx = 0
                seq_len = min(100, targets.shape[1])
                print(f"\nSample {sample_idx + 1} first {seq_len} tokens:")
                if has_dual_branch:
                    print(f"Text tokens: {tokens['text_seqs'][sample_idx, :seq_len].tolist()}")
                    print(f"Speech tokens: {tokens['speech_seqs'][sample_idx, :seq_len].tolist()}")
                else:
                    print(f"Input X:     {tokens[sample_idx, :seq_len].tolist()}")

                print(f"Target Y:    {targets[sample_idx, :seq_len].tolist()}")
                print(f"Predicted:   {predictions[sample_idx, :seq_len].tolist()}")
                valid_positions = (targets[sample_idx, :seq_len] != -100)
                print(f"Valid mask:  {valid_positions.tolist()}")
                matches = (predictions[sample_idx, :seq_len] == targets[sample_idx, :seq_len]) & valid_positions
                print(f"Matches:     {matches.tolist()}")
                print(f"Valid token count: {valid_positions.sum().item()}, Correct token count: {matches.sum().item()}")

            valid_mask = (targets != -100)
            correct_tokens = (predictions == targets).float() * valid_mask.float()
            if valid_mask.sum() > 0:
                accuracy = (correct_tokens.sum() / valid_mask.sum()).item()
            else:
                accuracy = 0.0

            losses.append(loss.item())
            accuracies.append(accuracy)
            speech_token_error_rates.append(
                compute_block_speech_token_error_rate(
                    predictions,
                    targets,
                    eob=eob_token,
                    codebook_size=codebook_size,
                )
            )
            sentence_token_error_rates.append(
                compute_sentence_speech_token_error_rate(
                    predictions,
                    targets,
                    eob=eob_token,
                    codebook_size=codebook_size,
                )
            )

        out[split] = float(sum(losses) / len(losses)) if losses else 0.0
        out[f'{split}_accuracy'] = float(sum(accuracies) / len(accuracies)) if accuracies else 0.0
        out[f'{split}_speech_token_error_rate'] = float(sum(speech_token_error_rates) / len(speech_token_error_rates)) if speech_token_error_rates else 0.0
        out[f'{split}_sentence_token_error_rate'] = float(sum(sentence_token_error_rates) / len(sentence_token_error_rates)) if sentence_token_error_rates else 0.0
    model.train()
    return out


def save_checkpoint(config, model, optimizer, iter_num, model_args):
    """Save model checkpoint."""
    raw_model = model.module if isinstance(model, DDP) else model
    
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'config': config,
    }
    
    # Save to specified filename or default
    checkpoint_path = os.path.join(config['out']['out_dir'], config['out'].get('checkpoint_filename', 'ckpt.pt'))
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    
    # Save backup copy and maintain only latest 10 checkpoints
    if config['out'].get('always_save_checkpoint', False):
        # Create a new checkpoint
        backup_path = os.path.join(config['out']['out_dir'], f'ckpt_{iter_num}.pt')
        print(f"Saving backup checkpoint to {backup_path}")
        torch.save(checkpoint, backup_path)
        
        # Get all checkpoint files
        checkpoint_files = [f for f in os.listdir(config['out']['out_dir']) 
                          if f.startswith('ckpt_') and f.endswith('.pt')]
        
        # If the number of checkpoints exceeds 10, delete the oldest ones
        if len(checkpoint_files) > 10:
            # Sort checkpoints by iteration number
            checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            # Delete the oldest checkpoint
            for old_ckpt in checkpoint_files[:-10]:
                old_path = os.path.join(config['out']['out_dir'], old_ckpt)
                try:
                    os.remove(old_path)
                    print(f"Removed old checkpoint: {old_ckpt}")
                except Exception as e:
                    print(f"Error removing checkpoint {old_ckpt}: {str(e)}")

def build_length_aligned_text_speech_blockwise(
    seq: list,
    eob_token: int = 4176,
    pad_token: int = 4177,
    codebook_size: int = 4096,
    speech_upper: Optional[int] = None,
    zero_speech_token: int = -1,
    silence_token: int = 453,
    silence_num: int = 0,
) -> (list, list):
    """
    Split the text-speech sequence into blocks by `eob` and collect the text and speech in each block
    Args:
        seq (list): Text-speech sequence containing `eob`, speech tokens, and text tokens
        eob_token (int): End of block token, default is 4176
        pad_token (int): Padding token, default is 4177
        speech_upper (int): Upper bound for speech tokens, default is codebook_size - 1
        zero_speech_token (int): Zero speech token, default is -1
        silence_token (int): Silence token, inserted before the speech sequence of the first block
        silence_num (int): Number of silence tokens to insert; when `> 0`, this applies only to the first block
    Returns:
        text_aligned (list[int]): Length-aligned text sequence
        speech_aligned (list[int]): Length-aligned speech sequence
    """
    # tokens = [int(t) for t in seq.tolist()]
    # First split into blocks by `eob` and collect the text and speech for each block
    if speech_upper is None:
        speech_upper = codebook_size - 1
    blocks = []
    cur_text_block: list[int] = []
    cur_speech_block: list[int] = []
    for tok in seq:
        if tok == eob_token:
            blocks.append({"text": cur_text_block, "speech": cur_speech_block})
            cur_text_block = []
            cur_speech_block = []
        elif tok == pad_token:
            # Ignore internal padding inside the sample, which usually does not exist
            continue
        elif tok <= speech_upper:
            cur_speech_block.append(tok)
        else:
            cur_text_block.append(tok)
    if len(cur_text_block) > 0 or len(cur_speech_block) > 0:
        blocks.append({"text": cur_text_block, "speech": cur_speech_block})

    text_aligned: list[int] = []
    speech_aligned: list[int] = []
    first_initialized = False  # Initialize only once at the first phoneme of the whole sequence

    # Generate aligned output block by block; truncate if text is longer than speech
    for bi, blk in enumerate(blocks):
        t_block = blk["text"][:]
        s_block = blk["speech"][:]

        # Only for the first block: insert the specified number of `silence_token`s before the speech sequence
        if bi == 0 and silence_num > 0:
            s_block = [silence_token] * silence_num + s_block

        # First-time initialization: if not initialized yet and the block contains text, consume the first text token of the block and pair it with `zero_speech`
        if not first_initialized and len(t_block) > 0:
            t0 = t_block.pop(0)
            text_aligned.append(t0)
            speech_aligned.append(zero_speech_token)
            first_initialized = True

        # Pair one by one using the minimum length
        min_len = min(len(t_block), len(s_block))
        for i in range(min_len):
            text_aligned.append(t_block[i])
            speech_aligned.append(s_block[i])

        # If speech is longer than text, align the extra speech with padded text
        if len(s_block) > min_len:
            for i in range(min_len, len(s_block)):
                text_aligned.append(pad_token)
                speech_aligned.append(s_block[i])

        # If text is longer than speech, truncate the extra text
        # End-of-block pair
        text_aligned.append(eob_token)
        speech_aligned.append(zero_speech_token)

    return text_aligned, speech_aligned

def split_sequence_by_speech_tokens(seqs, speech_token_num=225, eob_token=4176, codebook_size=4096):
    """
    Split a sequence into two parts at the first separator token after a specified number of speech tokens
    
    Args:
        seqs: Input sequence
        speech_token_num: Target number of speech tokens (default 225, approximately 3 seconds × 75 token/s = 225)
        eob_token: End-of-block token (default 4176)
    
    Returns:
        s_prev: Prefix sequence
        s_next: Suffix sequence
        split_position: Split position
    """
    speech_token_count = 0
    split_position = -1
    
    # Traverse the sequence and count speech tokens
    for i, token in enumerate(seqs):
        if token < codebook_size:
            speech_token_count += 1
            
        # After the speech-token count reaches the target, search for the first separator token
        if speech_token_count >= speech_token_num and token == eob_token:
            split_position = i + 1 
            break
    
    # If no suitable split point is found, use the default strategy
    if split_position == -1:
        print(f"Warning: did not find {speech_token_num} after {eob_token}, using the middle of the sequence as the split point")
        split_position = len(seqs) // 2
    
    # Split the sequence
    s_prev = seqs[:split_position]
    s_next = seqs[split_position:]
    
    # print(f"Split information:")
    # print(f"  - Speech-token count: {speech_token_count}")
    # print(f"  - Split position: {split_position}")
    # print(f"  - s_prevLength: {len(s_prev)}")
    # print(f"  - s_nextLength: {len(s_next)}")
    
    return s_prev, s_next, split_position

def extract_unit_from_seqs(seqs: list, block_type: str, eob_token=4176, codebook_size=4096, unit_token_start=None, unit_token_end=None, separator_token_start=None, separator_token_end=None):
    """
    Extract text-side units from a block sequence.

    This function restores the text prefix at the start of each block from `seqs`, while ignoring the speech tokens appended afterward.
    Two block structures are currently supported:

    - `biword`:
      `current-word unit sequence + separator`
    - `biunit`:
      `current unit + optional separator`

    Args:
        seqs:
            Input 1-D block sequence.
            The sequence contains text tokens, separator tokens, speech tokens, and `eob_token`.
        block_type:
            Text organization mode for the block.
            - `"biword"`: Extract the `current-word unit sequence + separator` at the start of each block
            - `"biunit"`: Extract the `current unit + optional separator` at the start of each block
        eob_token:
            Block-end token. After this token is encountered, extraction restarts for the next text block.
        codebook_size:
            Upper bound of the speech-token vocabulary, kept as a compatibility parameter.
            The current implementation does not directly use this value for boundary detection; it mainly relies on the ranges of unit/separator tokens.
        unit_token_start:
            Inclusive start ID of text-unit tokens.
        unit_token_end:
            Inclusive end ID of text-unit tokens.
        separator_token_start:
            Inclusive start ID of separator tokens.
        separator_token_end:
            Inclusive end ID of separator tokens.

    Returns:
        units:
            Extracted 2-D list of units.
            - When `block_type == "biword"`, each element has the form
              `current-word unit sequence + separator`
            - When `block_type == "biunit"`, each element has the form
              `current unit + optional separator`
            The return value contains only the text prefix of each block, excluding following speech tokens, future-word tokens, and `eob_token`.
    """

    units = []
    current_unit = []
    append_flag = True
    if block_type == "biword":
        for i in seqs:
            if i == eob_token:
                append_flag = True
                continue
            if append_flag and unit_token_start <= i <= unit_token_end:
                current_unit.append(i)
            else:
                if separator_token_start <= i <= separator_token_end:
                    current_unit.append(i)
                    if current_unit:
                        units.append(current_unit)
                    current_unit = []
                append_flag = False
    elif block_type == "biunit":  
        for i, unit in enumerate(seqs):
            if unit == eob_token:
                append_flag = True
                continue
            if append_flag and unit_token_start <= unit <= unit_token_end:
                current_unit.append(unit)
                if i + 1 < len(seqs) and separator_token_start <= seqs[i+1] <= separator_token_end:
                    current_unit.append(seqs[i+1])
                if current_unit:
                    units.append(current_unit)
                current_unit = []
                append_flag = False
                continue
    else:
        raise ValueError(f"Invalid block type: {block_type} not supported")
            
    return units

def load_lexicon(lexicon_path):
    """
    Load a word-to-phoneme mapping from a lexicon file
    
    Args:
        lexicon_path (str): Lexicon file path, with one entry per line in the format `word\tphoneme1 phoneme2 ...`
    
    Returns:
        dict: Dictionary mapping words to phoneme strings
    """
    word_phoneme_mapping = {}
    
    if not os.path.exists(lexicon_path):
        print(f"Warning: Lexicon file not found at {lexicon_path}")
        return word_phoneme_mapping
    
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Split with tabs
                parts = line.split('\t')
                if len(parts) < 2:
                    print(f"Warning: Invalid line {line_num} in lexicon: {line}")
                    continue
                
                word = parts[0].strip()
                phonemes_str = parts[1].strip()

                word_phoneme_mapping[word] = phonemes_str
        
        print(f"Loaded {len(word_phoneme_mapping)} words from lexicon: {lexicon_path}")
        
    except Exception as e:
        print(f"Error loading lexicon from {lexicon_path}: {e}")
    
    return word_phoneme_mapping

def update_lexicon_file(lexicon_path, word_phoneme_mapping):
    """
    Update the lexicon file by adding new word-to-phoneme mappings
    
    Args:
        lexicon_path (str): Lexicon file path
        word_phoneme_mapping (dict): Word-to-phoneme dictionary containing new mappings
    
    Returns:
        int: Number of updated mappings
    """
    if not os.path.exists(lexicon_path):
        print(f"Warning: Lexicon file not found at {lexicon_path}, creating new file")
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
    
    # Read the existing lexicon
    existing_mappings = {}
    if os.path.exists(lexicon_path):
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        phonemes_str = parts[1].strip()
                        existing_mappings[word] = phonemes_str
        except Exception as e:
            print(f"Error reading existing lexicon: {e}")
    
    # Find new mappings
    new_mappings = {}
    for word, phonemes in word_phoneme_mapping.items():
        if word not in existing_mappings:
            new_mappings[word] = phonemes
    
    if not new_mappings:
        print("No new mappings to add to lexicon")
        return 0
    
    # Merge mappings
    all_mappings = {**existing_mappings, **new_mappings}
    
    # Sort alphabetically
    sorted_words = sorted(all_mappings.keys())
    
    # Write the updated lexicon file
    try:
        # Create a backup
        if os.path.exists(lexicon_path):
            backup_path = lexicon_path + '.backup'
            import shutil
            shutil.copy2(lexicon_path, backup_path)
            print(f"Created backup at: {backup_path}")
        
        with open(lexicon_path, 'w', encoding='utf-8') as f:
            # f.write(f"# Updated lexicon file with {len(new_mappings)} new mappings\n")
            # f.write(f"# Total entries: {len(all_mappings)}\n")
            # f.write(f"# Format: word\tphoneme1 phoneme2 ...\n\n")
            
            for word in sorted_words:
                f.write(f"{word}\t{all_mappings[word]}\n")
        
        print(f"Successfully updated lexicon file: {lexicon_path}")
        print(f"Added {len(new_mappings)} new mappings")
        print(f"Total mappings in file: {len(all_mappings)}")
        
        return len(new_mappings)
        
    except Exception as e:
        print(f"Error writing updated lexicon file: {e}")
        return 0

def get_overlap_weights(overlap_len: int, window: str = "equal_power"):
    """
    Generate fade-in/fade-out weights for the overlap region.
    Supported modes:
    - 'none'       Pure OLA (no windowing): `w_prev = 1`, `w_new = 1`
    - 'linear'     Linear cross-fade: `w_prev = 1 - t`, `w_new = t`
    - 'hann'       Half-Hann cross-fade: `w_prev = 0.5 * (1 + cos(pi * t))`, `w_new = 1 - w_prev`
    - 'equal_power'Equal-power cross-fade: `w_prev = cos(theta)`, `w_new = sin(theta)`, `theta ∈ [0, π/2]`
    """
    if overlap_len <= 0:
        return None, None
    t = np.linspace(0.0, 1.0, overlap_len, dtype=np.float32)
    if window == "equal_power":
        theta = t * (np.pi / 2.0)
        w_prev = np.cos(theta)
        w_new = np.sin(theta)
    elif window == "hann":
        w_prev = 0.5 * (1.0 + np.cos(np.pi * t))
        w_new = 1.0 - w_prev
    elif window == "linear":
        w_prev = 1.0 - t
        w_new = t
    elif window == "none":
        w_prev = np.ones_like(t)
        w_new = np.ones_like(t)
    else:
        w_prev = np.ones_like(t)
        w_new = np.ones_like(t)
    return w_prev.astype(np.float32), w_new.astype(np.float32)


def sample_token_from_probs(probs, config):
    strategy = config["inference"].get("sampling_strategy", "greedy")
    if strategy == "greedy":
        return probs.argmax(dim=-1).item()
    elif strategy == "topk":
        p = probs[0]
        k = int(config["inference"].get("k", 10))
        k = max(1, min(k, p.size(-1)))
        top_probs, top_indices = torch.topk(p, k)
        top_probs = top_probs / top_probs.sum()
        idx = torch.multinomial(top_probs, 1)
        return top_indices[idx].item()
    elif strategy == "topp":
        p = float(config["inference"].get("p", 0.9))
        probs_1 = probs[0]
        sorted_probs, sorted_indices = torch.sort(probs_1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative <= p
        if not mask.any():
            mask[0] = True
        filtered_probs = sorted_probs * mask
        filtered_probs = filtered_probs / filtered_probs.sum()
        idx = torch.multinomial(filtered_probs, 1)
        return sorted_indices[idx].item()
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

def init_condition(model_handler, config: dict, speech_prompt: str = None, text_prompt: str = None):
    """
    Initialize model input conditions.

    For single-speaker tasks, return an empty prefix directly; multi-speaker tasks consistently reuse `ctc_align.py`
    through the generic text-speech alignment interface via `ModelHandler.align_text_and_speech()`
    After obtaining `seqs_sequence`, trim the initialization prefix according to the specific task.

    Args:
        model_handler (ModelHandler): ModelHandler instance
        config (dict): Configuration dictionary
        speech_prompt (str): Speech prompt path
        text_prompt (str): Text prompt
    """

    codebook_size = config["nac"]["codebook_size"]
    tokens_per_second = config["nac"]["tokens_per_second"]
    silence_token = config["nac"]["SILENCE_TOKEN_ID"]
    eob_token = config["data"]["eob"]
    eos_token = config["data"]["eos"]
    pad_token = config["data"]["pad"]

    task_type = config["inference"].get("task_type", "single-speaker")
    if task_type == "single-speaker":
        if model_handler.model.config.length_aligned_input:
            return [], [], None
        return [], None

    if not speech_prompt:
        raise ValueError(f"`speech_prompt` is required for task_type='{task_type}'.")
    if not text_prompt:
        raise ValueError(f"`text_prompt` is required for task_type='{task_type}'.")

    kvcache = None
    speech_tokens = model_handler.nac.encode(speech_prompt).tolist()[0][0]
    _, alignment_result = model_handler.align_text_and_speech(
        speech_prompt,
        text_prompt,
        speech_tokens,
    )

    if task_type == "continuation":
        initial_seq, _, _ = split_sequence_by_speech_tokens(
            alignment_result.seqs_sequence,
            tokens_per_second * 3,
            eob_token,
            codebook_size=codebook_size,
        )
        initial_seq = list(initial_seq)
    elif task_type == "cross":
        if config["inference"].get("replace_cross_eos_with_target_prefix", False):
            # Legacy sequence: the future word for the last word of a sentence is replaced with `eos`, so it is not suitable for streaming inference.
            raise ValueError(
                "`replace_cross_eos_with_target_prefix=True` needs target-text prefix injection, "
                "which is not available in `init_condition()`. "
                "Please disable this legacy option or pass target prefix at the caller level."
            )
        initial_seq = [int(token) for token in alignment_result.seqs_sequence if int(token) != eos_token]
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    if model_handler.model.config.length_aligned_input:
        text_seq, speech_seq = build_length_aligned_text_speech_blockwise(
            seq=initial_seq,
            eob_token=eob_token,
            pad_token=pad_token,
            codebook_size=codebook_size,
            speech_upper=codebook_size - 1,
            silence_token=silence_token,
        )

    with torch.inference_mode():
        if model_handler.model.config.length_aligned_input:
            text_pre = []
            speech_pre = []
            for i in range(len(speech_seq)):
                text_pre.append(text_seq[i])
                speech_pre.append(speech_seq[i])
                text_tensor = torch.tensor([text_pre], dtype=torch.long, device=model_handler.device)
                speech_tensor = torch.tensor([speech_pre], dtype=torch.long, device=model_handler.device)
                speech_decoder_output, _, kvcache = model_handler.model(
                    tokens={"text_seqs": text_tensor, "speech_seqs": speech_tensor},
                    kvcache=kvcache,
                )
            logits = speech_decoder_output[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            current_token = sample_token_from_probs(probs, config)
            speech_pre.append(current_token)
            return text_pre, speech_pre, kvcache

        init_list = []
        for token in initial_seq:
            init_list.append(token)
            input_seq = torch.tensor(init_list, dtype=torch.long).unsqueeze(0).to(model_handler.device)
            speech_decoder_output, _, kvcache = model_handler.model(input_seq, kvcache=kvcache)
        return init_list, kvcache
    

def generate_unit_list(config, target_phonemes_with_separators, offset=0):
    """
    Generate the unit list according to the config and the target unit sequence.
    
    Args:
        config (dict): Configuration dictionary
        target_phonemes_with_separators (list): Target text-unit sequence, including separators such as `<space>`, `,`, `.`, `!`, and `?`
    
    Returns:
        list: Generated unit list
    """

    unit_vocab_size = config["data"]["unit_vocab_size"] # Vocabulary size of text units, excluding separators, `eob`, and `eos`

    unit_list = [] # list[list[int]]
    if config['data']['block_type'] == 'biunit':
        eos_token = config['data']['eos']
        unit_token_start = config["data"]["unit_token_start"]
        unit_token_end = config["data"]["unit_token_end"]
        separator_token_start = config["data"]["separator_token_start"]
        separator_token_end = config["data"]["separator_token_end"]
        sentence_end_tokens = {
            int(token_id)
            for separator, token_id in config["data"]["separators"].items()
            if separator in {".", "!", "?"}
        }

        tokens = [int(token) for token in target_phonemes_with_separators if int(token) != eos_token]
        token_count = len(tokens)

        for i, token in enumerate(tokens):
            if not (unit_token_start <= token <= unit_token_end):
                continue

            current_unit = [token]
            next_index = i + 1
            if next_index < token_count and separator_token_start <= tokens[next_index] <= separator_token_end:
                separator_token = tokens[next_index]
                current_unit.append(separator_token)
                if separator_token in sentence_end_tokens:
                    current_unit.append(eos_token)
                else:
                    next_token = eos_token
                    for candidate in tokens[next_index + 1:]:
                        if unit_token_start <= candidate <= unit_token_end:
                            next_token = candidate
                            break
                    current_unit.append(next_token)
            else:
                next_token = eos_token
                for candidate in tokens[next_index:]:
                    if unit_token_start <= candidate <= unit_token_end:
                        next_token = candidate
                        break
                current_unit.append(next_token)
            unit_list.append(current_unit)
    elif config['data']['block_type'] == 'biword':
        eos_token = config['data']['eos']
        unit_token_start = config["data"]["unit_token_start"]
        unit_token_end = config["data"]["unit_token_end"]
        separator_token_start = config["data"]["separator_token_start"]
        separator_token_end = config["data"]["separator_token_end"]
        sentence_end_tokens = {
            int(token_id)
            for separator, token_id in config["data"]["separators"].items()
            if separator in {".", "!", "?"}
        }

        tokens = [int(token) for token in target_phonemes_with_separators if int(token) != eos_token]
        word_list = []
        current_word = []

        for token in tokens:
            if unit_token_start <= token <= unit_token_end:
                current_word.append(token)
                continue

            if separator_token_start <= token <= separator_token_end:
                if current_word:
                    word_list.append((current_word, token))
                    current_word = []
                continue

        if current_word:
            word_list.append((current_word, None))

        for i, (current_word_units, separator_token) in enumerate(word_list):
            current_unit = list(current_word_units)

            if separator_token is not None:
                current_unit.append(separator_token)

            if separator_token in sentence_end_tokens or i == len(word_list) - 1:
                # Legacy training sequence: append `eos` to sentence-final words
                # current_unit.append(eos_token)
                pass
            else:
                next_word_units, _ = word_list[i + 1]
                current_unit.extend(next_word_units)

            unit_list.append(current_unit)
    else:
        raise ValueError(f"Unknown unit type: {config['data']['block_type']}")
    
    return unit_list

def load_mfa_phone2token(phone2token_path):
    """
        phone2token_pathFile format: first column is the phoneme and second column is the token
        a 0
        b 1
        ...
        Return the mapping dictionary from phonemes to tokens
    """
    phone2token = {}
    with open(phone2token_path, 'r', encoding='utf-8') as f:
        for line in f:
            phone, token = line.split(" ")
            phone2token[phone] = int(token.strip())
    return phone2token

def load_mfa_dict(mfa_dict_path):
    """
        mfa_dict_pathFile format: first column is the word and the last column is the phoneme string
        aachen	ɑ k ə n
        aae	ej ej iː
        ...
        Return the mapping dictionary from words to phonemes
    """
    word2phone = {}
    with open(mfa_dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_list = line.split("\t")
            if len(line_list) < 2:
                continue
            word = line_list[0].strip()
            phones = line_list[1].strip()
            word2phone[word] = phones
    
    return word2phone


def get_local_advance_seq(seq, local_advance, eob, eos, codebook_size=4096):
    """
    Move `eob` and the nearby phoneme tokens around `eob` in `seq` forward by `local_advance` positions
    `seq` is a 1-D list. When traversing `seq`, if `eob` is encountered, move `eob` and its neighboring phoneme tokens (>= `codebook_size` and <= `codebook_size + 79`, or equal to `eos`)
    forward by `local_advance` positions while keeping other tokens unchanged
    """
    seq = list(seq)
    i = 0
    while i < len(seq):
        if seq[i] == eob:
            if i + 1 >= len(seq):
                break
            next_token = seq[i + 1]
            if not ((codebook_size <= next_token <= codebook_size + 79) or next_token == eos):
                raise ValueError(f"Invalid token after `eob`: {next_token}")
            pair = [eob, next_token]
            del seq[i:i + 2]
            insert_pos = max(0, i - local_advance)
            seq[insert_pos:insert_pos] = pair
            i = insert_pos + 2
        else:
            i += 1
    return seq
