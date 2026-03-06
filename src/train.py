#!/usr/bin/env python
"""Full training script for FloodLM with model and normalization checkpointing."""

import os
import sys
import json
import time
import pickle
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import wandb

# Setup paths (works whether launched via train.py wrapper or directly)
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import get_recurrent_dataloader, get_model_config, make_x_dyn
from model import FloodAutoregressiveHeteroModel
from data_lazy import initialize_data
from data_config import SELECTED_MODEL

# Configuration
CONFIG = {
    'history_len': 10,
    'forecast_len': 64,          # Max rollout horizon (curriculum will sample 1..max_h per batch)
    'batch_size': 24,
    'epochs': 24,                # Model_2: 3 epochs per stage (24 total); Model_1: 2 per stage (reduce manually if needed)
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'save_dir': 'checkpoints',
    'checkpoint_interval': 1,   # Save every N epochs
    'early_stopping_patience': 5,  # Only active once max_h == forecast_len; None to disable
    'early_stopping_min_rel_delta': 0.01,
    'curriculum_val_horizon': 32,  # Fixed horizon for multi-step val rollout
}

# Kaggle sigmas — used only for logging RMSE in meters and approx Kaggle score.
# Water level is normalized by these sigmas (meanstd), so sqrt(MSE_norm) == NRMSE directly.
KAGGLE_SIGMA = {
    (1, 1): 16.878,  # Model_1, 1D nodes
    (1, 2): 14.379,  # Model_1, 2D nodes
    (2, 1):  3.192,  # Model_2, 1D nodes
    (2, 2):  2.727,  # Model_2, 2D nodes
}

def save_normalization_stats(norm_stats, save_dir, model_id=None):
    """Save model-specific normalization statistics to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Use model ID from SELECTED_MODEL if not provided
    if model_id is None:
        model_id = SELECTED_MODEL
    
    # Extract serializable components
    stats_to_save = {
        'static_1d_params': norm_stats.get('static_1d_params', {}),
        'static_2d_params': norm_stats.get('static_2d_params', {}),
        'dynamic_1d_params': norm_stats.get('dynamic_1d_params', {}),
        'dynamic_2d_params': norm_stats.get('dynamic_2d_params', {}),
        'node1d_cols': norm_stats.get('node1d_cols', []),
        'node2d_cols': norm_stats.get('node2d_cols', []),
        'edge1_cols': norm_stats.get('edge1_cols', []),
        'edge2_cols': norm_stats.get('edge2_cols', []),
        'feature_type_1d': norm_stats.get('feature_type_1d', {}),
        'feature_type_2d': norm_stats.get('feature_type_2d', {}),
    }
    
    # Convert torch tensors to lists for JSON serialization
    for key in ['oneD_mu', 'oneD_sigma', 'twoD_mu', 'twoD_sigma', 
                'edge1_mu', 'edge1_sigma', 'edge2_mu', 'edge2_sigma']:
        if key in norm_stats and isinstance(norm_stats[key], torch.Tensor):
            stats_to_save[key] = norm_stats[key].cpu().numpy().tolist()
    
    # Save as JSON
    stats_path = os.path.join(save_dir, f'{model_id}_normalization_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    
    # Also save as pickle for full FeatureNormalizer objects (optional, for reference)
    normalizer_path = os.path.join(save_dir, f'{model_id}_normalizers.pkl')
    normalizers_data = {
        'normalizer_1d': norm_stats.get('normalizer_1d'),
        'normalizer_2d': norm_stats.get('normalizer_2d'),
    }
    with open(normalizer_path, 'wb') as f:
        pickle.dump(normalizers_data, f)
    
    print(f"[INFO] Saved normalization statistics to {stats_path}")
    print(f"[INFO] Saved normalizer objects to {normalizer_path}")

def save_checkpoint(model, epoch, loss, save_dir, config, model_id=None, global_step=None):
    """Save model checkpoint and related information."""
    os.makedirs(save_dir, exist_ok=True)

    # Use model ID from SELECTED_MODEL if not provided
    if model_id is None:
        model_id = SELECTED_MODEL

    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'config': config,
        'model_arch_config': {
            'h_dim': model.h_dim,
            'msg_dim': model.cell.msg_dim,
            'hidden_dim': model.cell._hidden_dim,
        },
        'loss': loss,
        'model_id': model_id,
        'global_step': global_step,
        'wandb_run_id': wandb.run.id if wandb.run is not None else None,
    }
    
    checkpoint_path = os.path.join(save_dir, f'{model_id}_epoch_{epoch:03d}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    
    return checkpoint_path

def evaluate_rollout(model, dataloader, criterion, device, norm_stats, rollout_steps, batched_static_graph=None, max_batches=None, use_mixed_precision=False, rain_1d_index=None):
    """Evaluate at a fixed multi-step rollout horizon.

    Returns (combined_norm, 1d_norm, 2d_norm, per_node_1d_mse) where
    per_node_1d_mse is a 1-D numpy array of shape [N_1d] with per-node
    mean MSE (normalized) averaged across all batches and timesteps.
    """
    model.eval()
    total, total_1d, total_2d = 0.0, 0.0, 0.0
    per_node_1d_accum = None  # [N_1d] accumulated MSE sum
    n = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if batch is None:
                continue
            avail = batch['y_future_1d'].shape[1]
            h = min(rollout_steps, avail)
            static_graph = batch['static_graph'].to(device)
            y_hist_1d     = batch['y_hist_1d'].to(device)
            y_hist_2d     = batch['y_hist_2d'].to(device)
            rain_hist_2d  = batch['rain_hist_2d'].to(device)
            y_future_1d   = batch['y_future_1d'].to(device)
            y_future_2d   = batch['y_future_2d'].to(device)
            rain_future_2d = batch['rain_future_2d'].to(device)
            _r1d = rain_1d_index if rain_1d_index is not None else getattr(static_graph, 'rain_1d_index', None)
            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                predictions = model.forward_unroll(
                    data=static_graph,
                    y_hist_1d=y_hist_1d, y_hist_2d=y_hist_2d,
                    rain_hist=rain_hist_2d, rain_future=rain_future_2d,
                    make_x_dyn=lambda y, r, d, _r=_r1d: make_x_dyn(
                        y['oneD'], y['twoD'], r, d,
                        rain_1d_index=_r,
                    ),
                    rollout_steps=h, device=device,
                    batched_data=batched_static_graph,
                )
                loss_1d = criterion(predictions['oneD'], y_future_1d[:, :h])
                loss_2d = criterion(predictions['twoD'], y_future_2d[:, :h])
                # Per-node 1D MSE: mean over batch (B) and time (h), keep node dim
                # predictions['oneD']: [B, h, N_1d, 1], y_future_1d: [B, h, N_1d, 1]
                node_mse_1d = ((predictions['oneD'] - y_future_1d[:, :h]) ** 2).mean(dim=(0, 1, 3))  # [N_1d]
            total_1d += loss_1d.item()
            total_2d += loss_2d.item()
            total += ((loss_1d + loss_2d) / 2).item()
            node_mse_cpu = node_mse_1d.float().cpu()
            if per_node_1d_accum is None:
                per_node_1d_accum = node_mse_cpu
            else:
                per_node_1d_accum += node_mse_cpu
            n += 1
    if n == 0:
        return float('nan'), float('nan'), float('nan'), None
    per_node_1d_mse = (per_node_1d_accum / n).numpy()
    return total / n, total_1d / n, total_2d / n, per_node_1d_mse


def train(resume_from=None, use_mixed_precision=False, skip_validation=False, pretrain_from=None, train_split='train', extra_epochs=None):
    """Main training loop.
    
    Args:
        resume_from: Path to checkpoint directory to resume from
        use_mixed_precision: Whether to use mixed precision (float16) training
    """
    
    # Determine if resuming
    resume_path = None
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise ValueError(f"Resume checkpoint not found: {resume_path}")
        print(f"\n[INFO] Resuming training from: {resume_path}")
    print("\n" + "="*70)
    print("FloodLM Training Script")
    print("="*70)
    
    # Peek at checkpoint to recover wandb run ID and global_step before init
    _wandb_resume_id = None
    _global_step_resume = 0
    if resume_path:
        try:
            _ckpt_files = sorted(resume_path.glob(f'{SELECTED_MODEL}*.pt'))
            if _ckpt_files:
                _peek = torch.load(_ckpt_files[-1], map_location='cpu')
                _wandb_resume_id = _peek.get('wandb_run_id', None)
                _global_step_resume = _peek.get('global_step', 0) or 0
        except Exception:
            pass

    run_name = f"{SELECTED_MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if _wandb_resume_id:
        wandb.init(project="floodlm", id=_wandb_resume_id, resume="must", config=CONFIG)
        print(f"[INFO] Resuming wandb run: {_wandb_resume_id} (global_step offset: {_global_step_resume})")
    else:
        wandb.init(project="floodlm", name=run_name, config=CONFIG)

    # Create a dated run subdirectory so each training run is isolated.
    # Also maintain a shared checkpoints/latest/ directory that always contains
    # the most recent best checkpoint + normalizers for EVERY model, so inference
    # can find Model_1 and Model_2 files in one place regardless of training order.
    run_dir = os.path.join(CONFIG['save_dir'], run_name)
    os.makedirs(run_dir, exist_ok=True)
    latest_dir = os.path.join(CONFIG['save_dir'], 'latest')
    os.makedirs(latest_dir, exist_ok=True)
    print(f"[INFO] Checkpoint dir: {run_dir}")
    print(f"[INFO] Latest dir:     {latest_dir}")

    device = torch.device(CONFIG['device'])
    print(f"[INFO] Device: {device}")

    # Initialize data and fetch preprocessing
    print(f"\n[INFO] Initializing data...")
    data = initialize_data()
    norm_stats = data['norm_stats']

    # Save normalization statistics into the run directory
    print(f"[INFO] Saving normalization statistics...")
    save_normalization_stats(norm_stats, run_dir)
    
    # Create dataloaders for train, val, test
    print(f"\n[INFO] Creating dataloaders...")
    print(f"  History: {CONFIG['history_len']}, Forecast: {CONFIG['forecast_len']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Split: train (data leakage prevention: normalization computed on train only)")
    
    train_dataloader = get_recurrent_dataloader(
        history_len=CONFIG['history_len'],
        forecast_len=CONFIG['forecast_len'],
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        split=train_split,
    )
    print(f"  Training split: {train_split}")

    if skip_validation:
        val_dataloader = None
        print(f"  Validation: DISABLED (--no-val)")
    else:
        val_dataloader = get_recurrent_dataloader(
            history_len=CONFIG['history_len'],
            forecast_len=CONFIG['forecast_len'],
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            split='val',
        )
    
    # Get model config
    print(f"\n[INFO] Getting model configuration...")
    model_config = get_model_config()
    print(f"  Node types: {model_config['node_types']}")
    print(f"  Node static dims: {model_config['node_static_dims']}")
    print(f"  Node dynamic dims: {model_config['node_dyn_input_dims']}")
    
    # Initialize model
    print(f"\n[INFO] Building model...")
    model_config.update({
        'h_dim': 96,     # GRU hidden size — kept large for temporal memory
        'msg_dim': 64,
        # Edge-type-specific hidden dims (reduced from 96/192 to 64/128)
        'hidden_dim': {
            'oneDedge':    64,
            'oneDedgeRev': 64,
            'twoDedge':    128,
            'twoDedgeRev': 128,
            # Cross-type edges: Model_2 uses StaticDynamicEdgeMP with only ~170 edges
            # and 2 edge features — hidden_dim=32 is sufficient.
            # Model_1 uses GATv2CrossTypeMP which ignores hidden_dim entirely.
            'twoDoneD':    32,
            'oneDtwoD':    32,
        },
    })
    model = FloodAutoregressiveHeteroModel(**model_config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    wandb.watch(model, log='all', log_freq=50)  # logs gradients + weights every 50 steps
    
    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()

    # Mixed precision: GradScaler for stable fp16 backward pass
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None

    # Load checkpoint if resuming
    start_epoch = 1
    if resume_path:
        try:
            # Try to find the checkpoint for this specific model
            checkpoint_files = sorted(resume_path.glob(f'{SELECTED_MODEL}*.pt'))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoints found for {SELECTED_MODEL} in {resume_path}")

            # Load the latest checkpoint for this model
            checkpoint_path = checkpoint_files[-1]
            print(f"[INFO] Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Restore model and optimizer states
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
                print(f"[INFO] Model weights restored")
            
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                print(f"[INFO] Optimizer state restored")
            
            # Get start epoch from checkpoint metadata
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"[INFO] Resuming from epoch {start_epoch} (last saved epoch: {checkpoint['epoch']})")
            
            print(f"[INFO] Checkpoint successfully loaded")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            print(f"[WARNING] Starting fresh from epoch 1")
            start_epoch = 1

    # If extra_epochs is set (used by fine-tune pipeline), override CONFIG['epochs'] so that
    # exactly extra_epochs epochs run starting from start_epoch, regardless of what was
    # stored in CONFIG. This prevents the common mistake of --epochs 4 meaning "stop at epoch 4"
    # when start_epoch is already 25.
    if extra_epochs is not None:
        CONFIG['epochs'] = start_epoch - 1 + extra_epochs
        print(f"[INFO] extra_epochs={extra_epochs}: running epochs {start_epoch}..{CONFIG['epochs']}")

    # Transfer learning: load weights from another model (e.g. Model_1 -> Model_2)
    # Resets epoch + optimizer — fresh training schedule with warm weights.
    if pretrain_from:
        pretrain_path = Path(pretrain_from)
        # Prefer best_h64, fall back to best, then any .pt
        for candidate in [
            pretrain_path / 'Model_1_best_h64.pt',
            pretrain_path / 'Model_1_best.pt',
            *sorted(pretrain_path.glob('Model_1*.pt')),
        ]:
            if candidate.exists():
                print(f"[INFO] Transfer learning: loading weights from {candidate}")
                ckpt = torch.load(candidate, map_location=device)
                if 'model_state' in ckpt:
                    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
                    if missing:
                        print(f"[INFO]   Missing keys (randomly initialized): {len(missing)}")
                    if unexpected:
                        print(f"[INFO]   Unexpected keys (ignored): {len(unexpected)}")
                    print(f"[INFO] Transfer weights loaded (epoch counter reset to 1, fresh optimizer)")
                break
        else:
            print(f"[WARNING] No Model_1 checkpoint found in {pretrain_from} — training from scratch")

    # Water level is normalized by kaggle_sigma (meanstd), so sqrt(MSE_norm) == NRMSE directly.
    # Loss = (loss_1d + loss_2d) / 2 — equal weight, directly Kaggle-aligned.
    model_id = int(SELECTED_MODEL.split('_')[-1])
    kaggle_sigma_1d = KAGGLE_SIGMA[(model_id, 1)]
    kaggle_sigma_2d = KAGGLE_SIGMA[(model_id, 2)]
    wl_1d = norm_stats['dynamic_1d_params']['water_level']
    wl_2d = norm_stats['dynamic_2d_params']['water_level']
    print(f"[INFO] Loss: (loss_1d + loss_2d) / 2  — water_level normalized by kaggle_sigma, so MSE_norm = NRMSE²")
    print(f"  1D: mean={wl_1d['mean']:.3f}m, sigma={wl_1d['sigma']:.3f}m (kaggle_σ={kaggle_sigma_1d})")
    print(f"  2D: mean={wl_2d['mean']:.3f}m, sigma={wl_2d['sigma']:.3f}m (kaggle_σ={kaggle_sigma_2d})")

    # Pre-build batched static graphs once — reused every forward pass to eliminate
    # per-batch CPU overhead from Batch.from_data_list.
    print(f"\n[INFO] Pre-building batched static graphs (B={CONFIG['batch_size']})...")
    _static_graph_cpu = next(iter(train_dataloader))['static_graph']
    train_batched_graph = model._make_batched_graph(_static_graph_cpu, CONFIG['batch_size']).to(device)
    val_batched_graph   = model._make_batched_graph(_static_graph_cpu, CONFIG['batch_size']).to(device) if not skip_validation else None
    # Capture rain_1d_index on device for use in make_x_dyn closures (graph-level attr
    # may not survive Batch.from_data_list so we pin it to a closure variable instead).
    _rain_1d_index = _static_graph_cpu.rain_1d_index.to(device) if hasattr(_static_graph_cpu, 'rain_1d_index') else None
    print(f"[INFO] Batched graphs ready.")

    # Node 197 loss mask (Model_2 only): node 197 is a confirmed data artifact
    # (depth=-1, base_area=0, physically impossible geometry). Its error is irreducible
    # from available features and injects misleading gradients. Masking only affects
    # training loss; predictions are still generated at inference.
    N_1d = _static_graph_cpu["oneD"].num_nodes
    if SELECTED_MODEL == "Model_2":
        _loss_mask_1d = torch.ones(N_1d, device=device)
        _loss_mask_1d[197] = 0.0
        print(f"[INFO] Model_2: node 197 masked from 1D training loss.")
    else:
        _loss_mask_1d = None

    print(f"\n[INFO] Training configuration:")
    print(f"  Learning rate: {CONFIG['lr']}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Checkpoint interval: {CONFIG['checkpoint_interval']}")

    # Training loop
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}\n")

    epoch_start_time = time.time()
    global_step = _global_step_resume  # Restored from checkpoint on resume; 0 for fresh runs

    # Mid-epoch validation disabled — at h=64 even 3-batch rollouts are expensive.
    # Full validation runs at end of each epoch instead.
    # Set VAL_CHECK_INTERVAL to an integer (e.g. 50) to re-enable.
    VAL_CHECK_INTERVAL = None
    VAL_SUBSET_BATCHES = 3

    best_kaggle_at_max_h = float('inf')   # best (NRMSE_1D + NRMSE_2D)/2 seen at full horizon
    best_kaggle_epoch = None
    prev_max_h = None  # Track curriculum jumps for LR reduction (Model_2 only)
    no_improve_count = 0  # Early stopping counter (only active at full horizon)
    best_train_loss_at_max_h = float('inf')  # fallback for early stopping when val disabled
    val_loss_norm = None  # Set in epoch loop; initialized here to avoid UnboundLocalError if loop is empty

    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        batch_start_time = time.time()

        # Curriculum: doubles every 3 epochs for Model_2 (slower convergence), 2 for Model_1
        # Model_2: epoch 1-3→1, 4-6→2, 7-9→4, 10-12→8, 13-15→16, 16-18→32, 19-24→64
        # Model_1: epoch 1-2→1, 3-4→2, ..., 13-14→64 (set epochs=16 manually)
        _stage_len = 3 if SELECTED_MODEL == 'Model_2' else 2
        max_h = min(CONFIG['forecast_len'], 2 ** ((epoch - 1) // _stage_len))

        # Epoch boundary marker — visible as a vertical annotation in wandb
        wandb.log({'epoch': epoch, 'curriculum/max_h': max_h}, step=global_step)

        # Model_2 only: drop LR by 3x at the three problematic curriculum jumps (h=8, 32, 64).
        # Skipping the early small-horizon jumps preserves learning capacity at h=64.
        # Model_1 gradients are well-behaved so this is skipped there.
        if SELECTED_MODEL == 'Model_2' and prev_max_h is not None and max_h != prev_max_h and max_h in {8, 32, 64}:
            for g in optimizer.param_groups:
                g['lr'] *= 0.3
            new_lr = optimizer.param_groups[0]['lr']
            print(f"[INFO] Model_2 curriculum jump {prev_max_h}→{max_h}: LR reduced to {new_lr:.2e}")
            wandb.log({'train/lr': new_lr}, step=global_step)
        prev_max_h = max_h

        print(f"[INFO] Curriculum: epoch={epoch}/{CONFIG['epochs']}, max_h={max_h}, stage_len={_stage_len} (all batches train at h={max_h})")

        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:
                continue

            # Train at the full curriculum horizon for this epoch — no sampling.
            # Each epoch IS the curriculum stage; always training at max_h maximizes
            # exposure to the deployment-length rollout (target ~50 steps).
            avail = batch['y_future_1d'].shape[1]
            rollout_steps = min(max_h, avail)

            # Extract batch data
            static_graph = batch['static_graph'].to(device, non_blocking=True)
            y_hist_1d = batch['y_hist_1d'].to(device, non_blocking=True)        # [B, H, N_1d, 1]
            y_hist_2d = batch['y_hist_2d'].to(device, non_blocking=True)        # [B, H, N_2d, 1]
            rain_hist_2d = batch['rain_hist_2d'].to(device, non_blocking=True)  # [B, H, N_2d, R]
            y_future_1d = batch['y_future_1d'].to(device, non_blocking=True)    # [B, T, N_1d, 1]
            y_future_2d = batch['y_future_2d'].to(device, non_blocking=True)    # [B, T, N_2d, 1]
            rain_future_2d = batch['rain_future_2d'].to(device, non_blocking=True)  # [B, T, N_2d, R]

            # Vectorized forward pass over all B samples at once
            # predictions: {'oneD': [B, rollout_steps, N_1d, 1], 'twoD': [B, rollout_steps, N_2d, 1]}
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                predictions = model.forward_unroll(
                    data=static_graph,
                    y_hist_1d=y_hist_1d,
                    y_hist_2d=y_hist_2d,
                    rain_hist=rain_hist_2d,
                    rain_future=rain_future_2d,
                    make_x_dyn=lambda y, r, d: make_x_dyn(
                        y['oneD'], y['twoD'], r, d,
                        rain_1d_index=_rain_1d_index,
                    ),
                    rollout_steps=rollout_steps,
                    device=device,
                    batched_data=train_batched_graph,
                    use_grad_checkpoint=use_mixed_precision,
                )

                # Sigma-weighted normalized loss: convex combo keeps scale ~1x raw MSE
                # Slice targets to match the sampled rollout_steps
                if _loss_mask_1d is not None:
                    # Per-node MSE for 1D, mask out node 197, then average over valid nodes
                    # predictions['oneD']: [B, T, N_1d, 1]; y_future_1d: [B, T, N_1d, 1]
                    sq_err_1d = (predictions['oneD'] - y_future_1d[:, :rollout_steps]) ** 2
                    loss_1d = (sq_err_1d.mean(dim=(0, 1, 3)) * _loss_mask_1d).sum() / _loss_mask_1d.sum()
                else:
                    loss_1d = criterion(predictions['oneD'], y_future_1d[:, :rollout_steps])
                loss_2d = criterion(predictions['twoD'], y_future_2d[:, :rollout_steps])
                loss = (loss_1d + loss_2d) / 2

            # Backward pass (scaler handles fp16 gradient scaling when enabled)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # must unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            avg_loss = epoch_loss / num_batches

            # Build the wandb log dict for this step — always includes train loss
            log_dict = {
                'loss/train_batch': loss.item(),
                'loss/train_avg': avg_loss,
                'loss/train_1d': loss_1d.item(),
                'loss/train_2d': loss_2d.item(),
                # Horizon-stratified: separate curve per epoch's rollout length in wandb
                f'loss/train_h{rollout_steps}': loss.item(),
                'curriculum/rollout_steps': rollout_steps,
                'curriculum/max_h': max_h,
                'epoch': epoch,
            }

            # Mid-epoch validation (disabled when VAL_CHECK_INTERVAL is None)
            if VAL_CHECK_INTERVAL is not None and (batch_idx + 1) % VAL_CHECK_INTERVAL == 0:
                val_combined_mid, val_1d_mid, val_2d_mid = evaluate_rollout(
                    model, val_dataloader, criterion, device, norm_stats,
                    rollout_steps=max_h,
                    batched_static_graph=val_batched_graph,
                    max_batches=VAL_SUBSET_BATCHES,
                    use_mixed_precision=use_mixed_precision,
                    rain_1d_index=_rain_1d_index,
                )
                # sqrt(MSE_norm) == NRMSE directly (water_level normalized by kaggle_sigma)
                val_nrmse_1d_mid = val_1d_mid ** 0.5
                val_nrmse_2d_mid = val_2d_mid ** 0.5
                val_rmse_1d_mid  = val_nrmse_1d_mid * kaggle_sigma_1d
                val_rmse_2d_mid  = val_nrmse_2d_mid * kaggle_sigma_2d
                val_kaggle_mid   = (val_nrmse_1d_mid + val_nrmse_2d_mid) / 2
                print(f"[INFO] Val-Subset (Epoch {epoch}, Batch {batch_idx+1}, h={max_h}):")
                print(f"  Combined (norm): {val_combined_mid:.6e}  "
                      f"1D RMSE={val_rmse_1d_mid:.4f}m NRMSE={val_nrmse_1d_mid:.4f}  "
                      f"2D RMSE={val_rmse_2d_mid:.4f}m NRMSE={val_nrmse_2d_mid:.4f}  "
                      f"approx_kaggle={val_kaggle_mid:.4f}")
                log_dict['loss/val_norm']          = val_combined_mid
                log_dict['loss/val_1d_norm']       = val_1d_mid
                log_dict['loss/val_2d_norm']       = val_2d_mid
                log_dict['loss/val_1d_rmse_m']     = val_rmse_1d_mid
                log_dict['loss/val_2d_rmse_m']     = val_rmse_2d_mid
                log_dict['loss/val_1d_nrmse']      = val_nrmse_1d_mid
                log_dict['loss/val_2d_nrmse']      = val_nrmse_2d_mid
                log_dict['loss/val_approx_kaggle'] = val_kaggle_mid
                model.train()  # Switch back to training mode

            wandb.log(log_dict, step=global_step)

            # Console progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - batch_start_time
                print(f"Epoch {epoch}/{CONFIG['epochs']} | "
                      f"Batch {batch_idx+1:3d} | "
                      f"Loss: {loss.item():.6f} | "
                      f"Avg: {avg_loss:.6f} | "
                      f"{elapsed:.1f}s")
        
        # End-of-epoch stats
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n[INFO] Epoch {epoch}/{CONFIG['epochs']}: Training Loss={avg_epoch_loss:.6f} | max_h={max_h} | Time: {epoch_time:.1f}s")
        
        # Full validation at current curriculum horizon — matches what we're training on.
        # This is used for early stopping and checkpointing.
        val_loss_norm = None
        val_nrmse_combined = None
        if not skip_validation:
            try:
                print(f"\n[INFO] Running full validation (h={max_h})...")
                val_combined, val_1d, val_2d, val_per_node_1d = evaluate_rollout(
                    model, val_dataloader, criterion, device, norm_stats,
                    rollout_steps=max_h, batched_static_graph=val_batched_graph,
                    use_mixed_precision=use_mixed_precision,
                    rain_1d_index=_rain_1d_index,
                )
                val_loss_norm = val_combined
                # sqrt(MSE_norm) == NRMSE directly; * kaggle_sigma gives RMSE in meters
                val_nrmse_1d = val_1d ** 0.5
                val_nrmse_2d = val_2d ** 0.5
                val_rmse_1d  = val_nrmse_1d * kaggle_sigma_1d
                val_rmse_2d  = val_nrmse_2d * kaggle_sigma_2d
                val_nrmse_combined = (val_nrmse_1d + val_nrmse_2d) / 2
                print(f"  h={max_h}  combined={val_combined:.6e}  "
                      f"1d={val_1d:.6e} (RMSE={val_rmse_1d:.4f}m, NRMSE={val_nrmse_1d:.4f})  "
                      f"2d={val_2d:.6e} (RMSE={val_rmse_2d:.4f}m, NRMSE={val_nrmse_2d:.4f})  "
                      f"approx_kaggle={val_nrmse_combined:.4f}")
                if max_h == CONFIG['forecast_len']:
                    if val_nrmse_combined < best_kaggle_at_max_h:
                        best_kaggle_at_max_h = val_nrmse_combined
                        best_kaggle_epoch = epoch
                        no_improve_count = 0
                        # Save best h=64 checkpoint — used by inference script
                        best_h64_src = os.path.join(run_dir, f'{SELECTED_MODEL}_epoch_{epoch:03d}.pt')
                        if os.path.exists(best_h64_src):
                            import shutil as _shutil
                            for _dst in [os.path.join(run_dir, f'{SELECTED_MODEL}_best_h64.pt'),
                                         os.path.join(latest_dir, f'{SELECTED_MODEL}_best_h64.pt')]:
                                _shutil.copy(best_h64_src, _dst)
                            print(f"[INFO] New best h=64 checkpoint saved (approx_kaggle={val_nrmse_combined:.4f})")
                    else:
                        no_improve_count += 1
                        patience = CONFIG['early_stopping_patience']
                        print(f"[INFO] No improvement at h={max_h}: {no_improve_count}/{patience} epochs without improvement")
                        if patience is not None and no_improve_count >= patience:
                            print(f"[INFO] Early stopping triggered after {no_improve_count} epochs without improvement at h={max_h}")
                            break
            except Exception as e:
                print(f"[WARNING] Validation failed (epoch {epoch}): {e} — skipping val this epoch")

        # Fallback early stopping on training loss when validation is disabled
        if skip_validation and max_h == CONFIG['forecast_len']:
            patience = CONFIG['early_stopping_patience']
            if avg_epoch_loss < best_train_loss_at_max_h:
                best_train_loss_at_max_h = avg_epoch_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"[INFO] No train loss improvement at h={max_h}: {no_improve_count}/{patience} epochs")
                if patience is not None and no_improve_count >= patience:
                    print(f"[INFO] Early stopping (train loss) triggered after {no_improve_count} epochs without improvement")
                    break

        model.train()

        wandb_payload = {'loss/train': avg_epoch_loss, 'curriculum/epoch_max_h': max_h, 'epoch': epoch, 'curriculum/max_h': max_h}
        if val_loss_norm is not None:
            wandb_payload.update({
                f'rollout_val/h{max_h}_combined': val_combined,
                f'rollout_val/h{max_h}_1d_mse_norm': val_1d,
                f'rollout_val/h{max_h}_2d_mse_norm': val_2d,
                f'rollout_val/h{max_h}_1d_rmse_m': val_rmse_1d,
                f'rollout_val/h{max_h}_2d_rmse_m': val_rmse_2d,
                f'rollout_val/h{max_h}_1d_nrmse': val_nrmse_1d,
                f'rollout_val/h{max_h}_2d_nrmse': val_nrmse_2d,
                f'rollout_val/h{max_h}_approx_kaggle': val_nrmse_combined,
            })
            # Per-node 1D MSE table — sortable in wandb UI to identify hard nodes
            if val_per_node_1d is not None:
                table = wandb.Table(columns=['node_id', 'mse_norm', 'nrmse', 'rmse_m'])
                for node_i, mse in enumerate(val_per_node_1d):
                    nrmse = float(mse) ** 0.5
                    table.add_data(node_i, float(mse), nrmse, nrmse * kaggle_sigma_1d)
                wandb_payload[f'rollout_val/h{max_h}_1d_per_node'] = table
        wandb.log(wandb_payload, step=global_step)

        # Checkpoint
        if epoch % CONFIG['checkpoint_interval'] == 0 or epoch == CONFIG['epochs']:
            save_checkpoint(model, epoch, avg_epoch_loss, run_dir, CONFIG, global_step=global_step)

        # Best checkpoint tracking — always save the last epoch's checkpoint to latest/
        # (Early stopping is disabled: val loss is incomparable across epochs with different max_h)
        import shutil
        best_checkpoint = os.path.join(run_dir, f'{SELECTED_MODEL}_epoch_{epoch:03d}.pt')
        if os.path.exists(best_checkpoint):
            best_path = os.path.join(run_dir, f'{SELECTED_MODEL}_best.pt')
            shutil.copy(best_checkpoint, best_path)
            shutil.copy(best_checkpoint, os.path.join(latest_dir, f'{SELECTED_MODEL}_best.pt'))
            for fname in [f'{SELECTED_MODEL}_normalizers.pkl',
                           f'{SELECTED_MODEL}_normalization_stats.json']:
                src = os.path.join(run_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(latest_dir, fname))
            _val_str = f"{val_loss_norm:.6e}" if val_loss_norm is not None else "N/A"
            print(f"[INFO] Checkpoint mirrored to latest/ (h={max_h} val_loss={_val_str})")

        print()

        epoch_start_time = time.time()

    # Final summary
    print("\n" + "="*70)
    print("Training Complete")
    print("="*70)
    if val_loss_norm is not None:
        print(f"Final epoch val loss (h={max_h}): {val_loss_norm:.6f}")
        print(f"  1D NRMSE={val_nrmse_1d:.4f}  2D NRMSE={val_nrmse_2d:.4f}")
        print(f"  Last epoch approx Kaggle score = {val_nrmse_combined:.4f}")
    else:
        print(f"Final epoch val loss: N/A (validation disabled)")
    if best_kaggle_epoch is not None:
        print(f"  *** {SELECTED_MODEL} best approx Kaggle score (h={CONFIG['forecast_len']}) = {best_kaggle_at_max_h:.4f}  (epoch {best_kaggle_epoch}) ***")
    else:
        print(f"  *** {SELECTED_MODEL} best approx Kaggle score: no full-horizon epochs completed ***")
    print(f"  (Competition score = mean of this over Model_1 and Model_2)")
    patience = CONFIG['early_stopping_patience']
    print(f"Early stopping: {'patience=' + str(patience) + ' (active at h=' + str(CONFIG['forecast_len']) + ')' if patience else 'disabled'}")
    print(f"Checkpoints saved to: {run_dir}")
    print(f"Latest dir:           {latest_dir}")
    final_model_path = os.path.join(run_dir, f'{SELECTED_MODEL}_epoch_{CONFIG["epochs"]:03d}.pt')
    print(f"Final model: {final_model_path}")
    print(f"Best model: {os.path.join(run_dir, f'{SELECTED_MODEL}_best.pt')}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FloodLM with optional resume from checkpoint")
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint directory to resume from (e.g., checkpoints/latest/ or checkpoints/Model_2_20260303_003721/)')
    parser.add_argument('--mixed-precision', action='store_true', 
                        help='Use mixed precision (float16) training to reduce GPU memory usage')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size for resume training')
    parser.add_argument('--learning-rate', '--lr', type=float, default=None,
                        help='Override learning rate for resume training')
    parser.add_argument('--max-h', type=int, default=None,
                        help='Override max curriculum horizon (default: 64)')
    parser.add_argument('--no-val', action='store_true',
                        help='Skip validation each epoch (faster, disables early stopping and best-h64 tracking)')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Load Model_1 weights as starting point for Model_2 (transfer learning). '
                             'Pass path to checkpoint dir (e.g. checkpoints/latest). '
                             'Epoch counter and optimizer are reset — full training schedule runs.')
    parser.add_argument('--train-split', type=str, default='train', choices=['train', 'all'],
                        help='Which data split to use for training. "all" = train+val+test (use only for final submission fine-tuning).')
    args = parser.parse_args()
    
    # Apply command-line overrides to CONFIG
    # Note: --epochs when combined with --resume is treated as *additional* epochs to run
    # (resolved inside train() after start_epoch is known). Without --resume it's absolute.
    extra_epochs = None
    if args.epochs is not None:
        if args.resume is not None:
            extra_epochs = args.epochs
        else:
            CONFIG['epochs'] = args.epochs
    if args.batch_size is not None:
        CONFIG['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        CONFIG['lr'] = args.learning_rate
    if args.max_h is not None:
        CONFIG['forecast_len'] = args.max_h
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        print("[INFO] Mixed precision training enabled")
        torch.set_float32_matmul_precision('medium')  # Speeds up matmuls on L40S/A100
        # Actual fp16 autocast + GradScaler is applied inside train()
    
    train(resume_from=args.resume, use_mixed_precision=args.mixed_precision, skip_validation=args.no_val, pretrain_from=args.pretrain, train_split=args.train_split, extra_epochs=extra_epochs)
