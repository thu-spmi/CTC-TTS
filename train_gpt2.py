"""
This training script can be run both on a single GPU in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import torch
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import json

# Import model components
from src.gpt2 import GPTConfig, GPT

# Import data loading utilities
from src.data import create_data_module, create_dataloader

# Import utility functions 
from src.utils import (
    setup_environment,
    save_checkpoint,
    estimate_loss,
    fetch_batch,
)
from src.scheduler import build_lr_scheduler
from src.optimizer import build_optimizer

from torch.utils.tensorboard import SummaryWriter  # Add TensorBoard


def initialize_model(config, device):
    """Initialize model from scratch, checkpoint or pretrained weights."""
    # Set up model arguments from config
    model_args = dict(
        n_layer=config['model']['n_layer'], 
        n_head=config['model']['n_head'], 
        n_embd=config['model']['n_embd'], 
        block_size=config['model']['block_size'],
        bias=config['model']['bias'], 
        vocab_size=None, 
        dropout=config['model']['dropout'],
    )
    meta_vocab_size = config["model"]["vocab_size"]  # default vocab size
    iter_num = 0  # default starting iteration

    # Pass dual-branch input parameters to `GPTConfig`
    model_args['length_aligned_input'] = config['model'].get('length_aligned_input', False)
    model_args['pad_token'] = config['model'].get('pad_token', config['data']['pad'])
    if model_args['length_aligned_input']:
        model_args['text_embd_dim'] = config['model'].get('text_embd_dim', 256)
        model_args['speech_embd_dim'] = config['model'].get('speech_embd_dim', 512)
        model_args['speech_vocab_size'] = config['model'].get('speech_vocab_size', config['nac']['codebook_size'])
        model_args['text_vocab_size'] = config['model'].get('text_vocab_size', meta_vocab_size - model_args['speech_vocab_size'])
        model_args['text_pad_token'] = config['model'].get('text_pad_token', model_args['pad_token'] - model_args['speech_vocab_size'])

    if config['training']['init_from'] == 'scratch':
        # Initialize new model from scratch
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = meta_vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
    elif config['training']['init_from'] == 'resume':
        # Resume from a checkpoint
        print(f"Resuming training from {config['out']['out_dir']}")
        ckpt_path = os.path.join(config['out']['out_dir'], config['out']['checkpoint_filename'])
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        
        # Ensure essential config attributes match the checkpoint
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size'
        ]:
            model_args[k] = checkpoint_model_args[k]
        model_args['length_aligned_input'] = checkpoint_model_args.get(
            'length_aligned_input',
            config['model'].get('length_aligned_input', False),
        )
        model_args['pad_token'] = checkpoint_model_args.get(
            'pad_token',
            config['model'].get('pad_token', config['data']['pad']),
        )
        
        if model_args['length_aligned_input']:
            model_args['text_embd_dim'] = checkpoint_model_args.get(
                'text_embd_dim',
                config['model'].get('text_embd_dim', 256),
            )
            model_args['speech_embd_dim'] = checkpoint_model_args.get(
                'speech_embd_dim',
                config['model'].get('speech_embd_dim', 768),
            )
            model_args['speech_vocab_size'] = checkpoint_model_args.get(
                'speech_vocab_size',
                config['model'].get('speech_vocab_size', config['nac']['codebook_size']),
            )
            model_args['text_vocab_size'] = checkpoint_model_args.get(
                'text_vocab_size',
                config['model'].get('text_vocab_size', model_args['vocab_size'] - model_args['speech_vocab_size']),
            )
            model_args['text_pad_token'] = checkpoint_model_args.get(
                'text_pad_token',
                config['model'].get('text_pad_token', model_args['pad_token'] - model_args['speech_vocab_size']),
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
        
        # Get iteration number from checkpoint
        iter_num = checkpoint['iter_num']
    
    # Move model to the specified device
    model.to(device)
    
    # Crop block size if needed
    if config['model']['block_size'] < model.config.block_size:
        model.crop_block_size(config['model']['block_size'])
        model_args['block_size'] = config['model']['block_size']
    
    return model, model_args, iter_num

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT-2 model for speech synthesis')
    parser.add_argument(
        '--config', 
        type=str, 
        default='/mnt/workspace/liuhw/streaming_tts/config/config.json',
        help='Path to configuration file'
    )
    return parser.parse_args()

def main():
    """Main training function."""

    # Parse command line arguments
    args = parse_args()
    # Load configuration from specified path
    print(f"Loading configuration from: {args.config}")
    # Get default config and update with command line arguments
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Set up training environment
    print("Setting up training environment")
    env = setup_environment(config)
    device = env['device']
    device_type = env['device_type']
    ctx = env['ctx']
    master_process = env['master_process']
    ddp = env['ddp']
    ddp_local_rank = env['ddp_local_rank']

    # Initialize model
    print("Initializing model")
    context_length = int(config['model']['block_size'])
    model, model_args, iter_num = initialize_model(config, device)

    # Initialize optimizer and gradient scaler
    print("Configuring optimizer")
    optimizer = build_optimizer(config, model, device_type)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(config['training']['dtype'] == 'float16'))
    
    # Compile model if enabled (requires PyTorch 2.0+)
    if config['training']['compile']:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # Wrap model in DDP if distributed training
    if ddp:
        print(f"Setting up Distributed Data Parallel (rank {ddp_local_rank})")
        model = DDP(model, device_ids=[ddp_local_rank])

    # Create data loader
    print("Creating data module and dataloader")
    data_module = create_data_module(config, enable_length_align=model_args['length_aligned_input'], device=device)
    print(f"train_dataset length: {len(data_module['train_dataset'])}")
    print(f"valid_dataset length: {len(data_module['valid_dataset'])}")
    train_dataloader, valid_dataloader = create_dataloader(
        data_module,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    
    tb_writer = SummaryWriter(config['out']['tensorboard_dir']) if master_process else None
    raw_model = model.module if ddp else model

    print("Fetching initial batch")
    train_iter = iter(train_dataloader)
    X, Y, train_iter, _ = fetch_batch(
        train_iter,
        train_dataloader,
        context_length,
        device,
        device_type,
    )

    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    lr_scheduler = build_lr_scheduler(config)
    optimizer.zero_grad(set_to_none=True)

    print(f"Beginning training from iteration {iter_num}")

    try:
        while True:
            lr = lr_scheduler.get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if iter_num % config['training']['eval_interval'] == 0 and master_process:
                losses = estimate_loss(
                    model,
                    valid_dataloader,
                    context_length,
                    config['training']['eval_iters'],
                    ctx,
                    device,
                    device_type,
                    codebook_size=config["nac"]["codebook_size"],
                    eob_token=config["data"]["eob"],
                )
                print(f"step {iter_num}: val loss {losses['val']:.8f} val accuracy {losses['val_accuracy']:.8f}")

                tb_writer.add_scalar('Loss/val', losses['val'], iter_num)
                tb_writer.add_scalar('Accuracy/val', losses['val_accuracy'], iter_num)
                tb_writer.add_scalar('Learning_rate', lr, iter_num)
                if running_mfu >= 0.0:
                    tb_writer.add_scalar('MFU', running_mfu * 100, iter_num)
                if 'val_speech_token_error_rate' in losses:
                    tb_writer.add_scalar('BlockTokenErrorRate/val', losses['val_speech_token_error_rate'], iter_num)
                if 'val_sentence_token_error_rate' in losses:
                    tb_writer.add_scalar('SentenceTokenErrorRate/val', losses['val_sentence_token_error_rate'], iter_num)

                save_checkpoint(config, model, optimizer, iter_num, model_args)

            if iter_num == 0 and config['training']['eval_only']:
                print("Evaluation complete, exiting")
                break

            lossf = 0.0
            skip_step = False

            for micro_step in range(env['gradient_accumulation_steps']):
                if ddp:
                    model.require_backward_grad_sync = (micro_step == env['gradient_accumulation_steps'] - 1)

                try:
                    with ctx:
                        _, loss, _ = model(tokens=X, targets=Y)
                except Exception as e:
                    print(f"Error in forward pass: {str(e)}")
                    optimizer.zero_grad(set_to_none=True)
                    X, Y, train_iter, _ = fetch_batch(
                        train_iter,
                        train_dataloader,
                        context_length,
                        device,
                        device_type,
                    )
                    skip_step = True
                    break

                lossf += loss.item()
                scaler.scale(loss / env['gradient_accumulation_steps']).backward()
                X, Y, train_iter, _ = fetch_batch(
                    train_iter,
                    train_dataloader,
                    context_length,
                    device,
                    device_type,
                )

            if skip_step:
                t0 = time.time()
                continue

            if config['training']['grad_clip'] != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            lossf /= env['gradient_accumulation_steps']
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(config['training']['batch_size'] * env['gradient_accumulation_steps'], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            if iter_num % config['out']['log_interval'] == 0 and master_process:
                tb_writer.add_scalar('Training/loss', lossf, iter_num)
                tb_writer.add_scalar('Training/learning_rate', lr, iter_num)
                tb_writer.add_scalar('Training/time_ms', dt * 1000, iter_num)
                if running_mfu >= 0.0:
                    tb_writer.add_scalar('Training/mfu', running_mfu * 100, iter_num)

                print(f"iter {iter_num}: loss {lossf:.8f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

            iter_num += 1
            local_iter_num += 1

            if iter_num > config['training']['max_iters']:
                print(f"Reached maximum iterations ({config['training']['max_iters']}), training complete")
                break
    finally:
        if tb_writer is not None:
            tb_writer.close()
        if ddp:
            destroy_process_group()
    
    print("Training finished successfully")


if __name__ == "__main__":
    main()
