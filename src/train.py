#!/usr/bin/env python
"""Full training script for FloodLM with model and normalization checkpointing."""

import os
import sys
import json
import time
import pickle
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

from data import get_recurrent_dataloader, get_model_config, unnormalize_col
from model import FloodAutoregressiveHeteroModel
from data_lazy import initialize_data
from data_config import SELECTED_MODEL

# Configuration
CONFIG = {
    'history_len': 10,
    'forecast_len': 1,
    'batch_size': 32,
    'epochs': 4,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'save_dir': 'checkpoints',
    'checkpoint_interval': 1,  # Save every N epochs
    'early_stopping_patience': 3,   # Stop if val loss doesn't improve for N epochs
    'early_stopping_min_rel_delta': 0.01,  # Minimum 1% relative improvement to count
}

# Kaggle metric normalization sigmas: {(model_id, node_type): sigma}
# node_type 1 = 1D, node_type 2 = 2D
# Source: competition metric definition
KAGGLE_SIGMA = {
    (1, 1): 16.878,  # Model_1, 1D nodes
    (1, 2): 14.379,  # Model_1, 2D nodes
    (2, 1):  3.192,  # Model_2, 1D nodes
    (2, 2):  2.727,  # Model_2, 2D nodes
}

def get_sigma_weights(model_id: int, norm_stats: dict):
    """Return (w_1d, w_2d) for NRMSE-aligned loss in normalized space.

    Our normalization is min-max:  x_norm = (x - vmin) / (vmax - vmin)
    So:  MSE_raw = MSE_norm * (vmax - vmin)^2

    The Kaggle metric computes NRMSE = RMSE_raw / kaggle_sigma, so:
      NRMSE^2 = MSE_raw / kaggle_sigma^2
              = MSE_norm * (vmax - vmin)^2 / kaggle_sigma^2

    To make the training loss proportional to NRMSE^2 in normalized space:
      w = (vmax - vmin)^2 / kaggle_sigma^2

    Combined loss = (w_1d * loss_1d + w_2d * loss_2d) / 2
    gives equal 50/50 contribution per node type, scaled to match the metric.
    """
    kaggle_sigma_1d = KAGGLE_SIGMA[(model_id, 1)]
    kaggle_sigma_2d = KAGGLE_SIGMA[(model_id, 2)]

    wl_params_1d = norm_stats['dynamic_1d_params']['water_level']
    wl_params_2d = norm_stats['dynamic_2d_params']['water_level']
    range_1d = wl_params_1d['max'] - wl_params_1d['min']
    range_2d = wl_params_2d['max'] - wl_params_2d['min']

    w_1d = (range_1d / kaggle_sigma_1d) ** 2
    w_2d = (range_2d / kaggle_sigma_2d) ** 2
    return w_1d, w_2d

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

def save_checkpoint(model, epoch, loss, save_dir, config, model_id=None):
    """Save model checkpoint and related information."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Use model ID from SELECTED_MODEL if not provided
    if model_id is None:
        model_id = SELECTED_MODEL
    
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'config': config,
        'loss': loss,
        'model_id': model_id,
    }
    
    checkpoint_path = os.path.join(save_dir, f'{model_id}_epoch_{epoch:03d}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    
    return checkpoint_path

def evaluate(model, dataloader, criterion, device, norm_stats, split_name='Validation', debug=False, max_batches=None, batched_static_graph=None, w_1d=None, w_2d=None):
    """Evaluate model on a dataset with both normalized and denormalized losses.

    Args:
        max_batches: If set, only evaluate on the first N batches (useful for faster validation)
        batched_static_graph: Pre-built PyG Batch of B copies of the static graph (avoids rebuild each call)
    """
    model.eval()
    total_loss_norm = 0.0
    total_loss_denorm = 0.0
    total_loss_1d_norm = 0.0
    total_loss_2d_norm = 0.0
    total_loss_1d_denorm = 0.0
    total_loss_2d_denorm = 0.0
    num_batches = 0

    # Debug statistics
    all_errors_norm = []
    all_errors_denorm = []
    
    wl_col_1d = norm_stats['node1d_cols'].index('water_level')
    wl_col_2d = norm_stats['node2d_cols'].index('water_level')

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Stop early if max_batches is set
            if max_batches is not None and batch_idx >= max_batches:
                break

            if batch is None:
                continue

            static_graph = batch['static_graph'].to(device)
            y_hist_1d = batch['y_hist_1d'].to(device)        # [B, H, N_1d, 1]
            y_hist_2d = batch['y_hist_2d'].to(device)        # [B, H, N_2d, 1]
            rain_hist_2d = batch['rain_hist_2d'].to(device)  # [B, H, N_2d, R]
            y_future_1d = batch['y_future_1d'].to(device)    # [B, T, N_1d, 1]
            y_future_2d = batch['y_future_2d'].to(device)    # [B, T, N_2d, 1]
            rain_future_2d = batch['rain_future_2d'].to(device)

            # Vectorized forward pass over all B samples at once
            # predictions: {'oneD': [B, T, N_1d, 1], 'twoD': [B, T, N_2d, 1]}
            predictions = model.forward_unroll(
                data=static_graph,
                y_hist_1d=y_hist_1d,
                y_hist_2d=y_hist_2d,
                rain_hist=rain_hist_2d,
                rain_future=rain_future_2d,
                make_x_dyn=lambda y, r, data: {
                    'oneD': y['oneD'],
                    'twoD': torch.cat([y['twoD'], r], dim=-1),
                },
                rollout_steps=1,
                device=device,
                batched_data=batched_static_graph,
            )

            # Sigma-weighted 50/50 loss: (w_1d * loss_1d + w_2d * loss_2d) / 2
            # Matches Kaggle metric: equal weight per node type, scaled by 1/sigma^2
            loss_1d = criterion(predictions['oneD'], y_future_1d)
            loss_2d = criterion(predictions['twoD'], y_future_2d)
            loss_norm = (w_1d * loss_1d + w_2d * loss_2d) / 2.0

            # Denormalize for interpretable loss
            pred_1d_denorm = unnormalize_col(predictions['oneD'], norm_stats, col=wl_col_1d, node_type='oneD')
            target_1d_denorm = unnormalize_col(y_future_1d, norm_stats, col=wl_col_1d, node_type='oneD')
            pred_2d_denorm = unnormalize_col(predictions['twoD'], norm_stats, col=wl_col_2d, node_type='twoD')
            target_2d_denorm = unnormalize_col(y_future_2d, norm_stats, col=wl_col_2d, node_type='twoD')
            loss_1d_denorm = criterion(pred_1d_denorm, target_1d_denorm)
            loss_2d_denorm = criterion(pred_2d_denorm, target_2d_denorm)
            loss_denorm = (loss_1d_denorm + loss_2d_denorm) / 2.0

            total_loss_norm += loss_norm.item()
            total_loss_denorm += loss_denorm.item()
            total_loss_1d_norm += loss_1d.item()
            total_loss_2d_norm += loss_2d.item()
            total_loss_1d_denorm += loss_1d_denorm.item()
            total_loss_2d_denorm += loss_2d_denorm.item()
            num_batches += 1

            # Debug: collect error statistics
            if debug and batch_idx < 5:
                all_errors_norm.append((predictions['twoD'] - y_future_2d).abs().cpu().numpy())
                all_errors_denorm.append((pred_2d_denorm - target_2d_denorm).abs().cpu().numpy())

            # Debug print for first few batches
            if debug and batch_idx < 3:
                print(f"  [DEBUG] Batch {batch_idx}: norm_loss={loss_norm.item():.9f}, denorm_loss={loss_denorm.item():.9f}")
                print(f"    1D pred range (norm):   [{predictions['oneD'].min():.6f}, {predictions['oneD'].max():.6f}]")
                print(f"    1D target range (norm): [{y_future_1d.min():.6f}, {y_future_1d.max():.6f}]")
                print(f"    1D pred range (denorm): [{pred_1d_denorm.min():.6f}, {pred_1d_denorm.max():.6f}]")
                print(f"    1D target range (denorm):[{target_1d_denorm.min():.6f}, {target_1d_denorm.max():.6f}]")
                print(f"    2D pred range (norm):   [{predictions['twoD'].min():.6f}, {predictions['twoD'].max():.6f}]")
                print(f"    2D target range (norm): [{y_future_2d.min():.6f}, {y_future_2d.max():.6f}]")
                print(f"    2D pred range (denorm): [{pred_2d_denorm.min():.6f}, {pred_2d_denorm.max():.6f}]")
                print(f"    2D target range (denorm):[{target_2d_denorm.min():.6f}, {target_2d_denorm.max():.6f}]")
    
    n = num_batches if num_batches > 0 else 1
    avg_loss_norm    = total_loss_norm / n
    avg_loss_denorm  = total_loss_denorm / n
    avg_1d_norm      = total_loss_1d_norm / n
    avg_2d_norm      = total_loss_2d_norm / n
    avg_1d_denorm    = total_loss_1d_denorm / n
    avg_2d_denorm    = total_loss_2d_denorm / n

    # Approx NRMSE per node type (sqrt of weighted normalized MSE)
    wl_1d = norm_stats['dynamic_1d_params']['water_level']
    wl_2d = norm_stats['dynamic_2d_params']['water_level']
    range_1d = wl_1d['max'] - wl_1d['min']
    range_2d = wl_2d['max'] - wl_2d['min']
    model_num = int(SELECTED_MODEL.split('_')[-1])
    nrmse_1d = (avg_1d_norm ** 0.5) * range_1d / KAGGLE_SIGMA[(model_num, 1)]
    nrmse_2d = (avg_2d_norm ** 0.5) * range_2d / KAGGLE_SIGMA[(model_num, 2)]
    approx_kaggle = (nrmse_1d + nrmse_2d) / 2.0

    print(f"[INFO] {split_name}:")
    print(f"  Combined loss (norm):      {avg_loss_norm:.6e}")
    print(f"  1D MSE (norm):  {avg_1d_norm:.6e}  |  RMSE (m): {(avg_1d_denorm**0.5):.4f}  |  NRMSE≈{nrmse_1d:.4f}")
    print(f"  2D MSE (norm):  {avg_2d_norm:.6e}  |  RMSE (m): {(avg_2d_denorm**0.5):.4f}  |  NRMSE≈{nrmse_2d:.4f}")
    print(f"  Approx Kaggle score:       {approx_kaggle:.4f}")

    if debug and len(all_errors_norm) > 0:
        errors_norm = np.concatenate(all_errors_norm)
        errors_denorm = np.concatenate(all_errors_denorm)
        print(f"[DEBUG] Error distribution (normalized): mean={errors_norm.mean():.9f}, std={errors_norm.std():.9f}, max={errors_norm.max():.9f}")
        print(f"[DEBUG] Error distribution (denormalized): mean={errors_denorm.mean():.9f}, std={errors_denorm.std():.9f}, max={errors_denorm.max():.9f}")

    return avg_loss_norm, avg_loss_denorm, avg_1d_norm, avg_2d_norm, avg_1d_denorm, avg_2d_denorm

def train():
    """Main training loop."""
    print("\n" + "="*70)
    print("FloodLM Training Script")
    print("="*70)
    
    run_name = f"{SELECTED_MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        split='train',
    )

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

    # Sigma weights for metric-aligned loss in normalized space:
    # w = (our_range / kaggle_sigma)^2 accounts for min-max normalization
    model_id = int(SELECTED_MODEL.split('_')[-1])
    w_1d, w_2d = get_sigma_weights(model_id, norm_stats)
    wl_1d = norm_stats['dynamic_1d_params']['water_level']
    wl_2d = norm_stats['dynamic_2d_params']['water_level']
    print(f"[INFO] Loss weights (normalized space):")
    print(f"  1D: range={wl_1d['max']-wl_1d['min']:.3f}m, kaggle_σ={KAGGLE_SIGMA[(model_id,1)]}, w={w_1d:.6f}")
    print(f"  2D: range={wl_2d['max']-wl_2d['min']:.3f}m, kaggle_σ={KAGGLE_SIGMA[(model_id,2)]}, w={w_2d:.6f}")

    # Pre-build batched static graphs once — reused every forward pass to eliminate
    # per-batch CPU overhead from Batch.from_data_list.
    print(f"\n[INFO] Pre-building batched static graphs (B={CONFIG['batch_size']})...")
    _static_graph_cpu = next(iter(train_dataloader))['static_graph']
    train_batched_graph = model._make_batched_graph(_static_graph_cpu, CONFIG['batch_size']).to(device)
    val_batched_graph   = model._make_batched_graph(_static_graph_cpu, CONFIG['batch_size']).to(device)
    print(f"[INFO] Batched graphs ready.")

    print(f"\n[INFO] Training configuration:")
    print(f"  Learning rate: {CONFIG['lr']}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Checkpoint interval: {CONFIG['checkpoint_interval']}")

    # Training loop
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    early_stopping_counter = 0
    epoch_start_time = time.time()
    global_step = 0  # Single monotonic step counter for wandb

    # Configuration for frequent monitoring
    VAL_CHECK_INTERVAL = 50  # Every N training batches, do lightweight validation
    VAL_SUBSET_BATCHES = 3   # Use only 3 batches for lightweight validation (very fast)

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        batch_start_time = time.time()

        # Epoch boundary marker — visible as a vertical annotation in wandb
        wandb.log({'epoch': epoch}, step=global_step)
        
        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:
                continue
            
            # Extract batch data
            static_graph = batch['static_graph'].to(device)
            y_hist_1d = batch['y_hist_1d'].to(device)        # [B, H, N_1d, 1]
            y_hist_2d = batch['y_hist_2d'].to(device)        # [B, H, N_2d, 1]
            rain_hist_2d = batch['rain_hist_2d'].to(device)  # [B, H, N_2d, R]
            y_future_1d = batch['y_future_1d'].to(device)    # [B, T, N_1d, 1]
            y_future_2d = batch['y_future_2d'].to(device)    # [B, T, N_2d, 1]
            rain_future_2d = batch['rain_future_2d'].to(device)  # [B, T, N_2d, R]

            # Vectorized forward pass over all B samples at once
            # predictions: {'oneD': [B, T, N_1d, 1], 'twoD': [B, T, N_2d, 1]}
            predictions = model.forward_unroll(
                data=static_graph,
                y_hist_1d=y_hist_1d,
                y_hist_2d=y_hist_2d,
                rain_hist=rain_hist_2d,
                rain_future=rain_future_2d,
                make_x_dyn=lambda y, r, data: {
                    'oneD': y['oneD'],
                    'twoD': torch.cat([y['twoD'], r], dim=-1),
                },
                rollout_steps=CONFIG['forecast_len'],
                device=device,
                batched_data=train_batched_graph,
            )

            # Sigma-weighted 50/50 loss: (w_1d * loss_1d + w_2d * loss_2d) / 2
            # Matches Kaggle metric: equal weight per node type, scaled by 1/sigma^2
            loss_1d = criterion(predictions['oneD'], y_future_1d)
            loss_2d = criterion(predictions['twoD'], y_future_2d)
            loss = (w_1d * loss_1d + w_2d * loss_2d) / 2.0
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
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
            }

            # Lightweight validation every N batches — logged at the same step as train
            if (batch_idx + 1) % VAL_CHECK_INTERVAL == 0:
                val_loss_norm, val_loss_denorm, val_1d_norm, val_2d_norm, val_1d_denorm, val_2d_denorm = evaluate(
                    model, val_dataloader, criterion, device, norm_stats,
                    split_name=f"Val-Subset (Epoch {epoch}, Batch {batch_idx+1})",
                    debug=False,
                    max_batches=VAL_SUBSET_BATCHES,
                    batched_static_graph=val_batched_graph,
                    w_1d=w_1d, w_2d=w_2d,
                )
                log_dict['loss/val_norm']    = val_loss_norm
                log_dict['loss/val_denorm']  = val_loss_denorm
                log_dict['loss/val_1d_norm'] = val_1d_norm
                log_dict['loss/val_2d_norm'] = val_2d_norm
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
        
        print(f"\n[INFO] Epoch {epoch}/{CONFIG['epochs']}: Training Loss={avg_epoch_loss:.6f} | Time: {epoch_time:.1f}s")
        
        # Full validation evaluation (now less frequent due to quick checks)
        print(f"\n[INFO] Running full validation...")
        val_loss_norm, val_loss_denorm, val_1d_norm, val_2d_norm, val_1d_denorm, val_2d_denorm = evaluate(
            model, val_dataloader, criterion, device, norm_stats,
            split_name="Validation", debug=False,
            batched_static_graph=val_batched_graph, w_1d=w_1d, w_2d=w_2d,
        )

        # Approx Kaggle NRMSE for wandb
        range_1d = norm_stats['dynamic_1d_params']['water_level']['max'] - norm_stats['dynamic_1d_params']['water_level']['min']
        range_2d = norm_stats['dynamic_2d_params']['water_level']['max'] - norm_stats['dynamic_2d_params']['water_level']['min']
        nrmse_1d = (val_1d_norm ** 0.5) * range_1d / KAGGLE_SIGMA[(model_id, 1)]
        nrmse_2d = (val_2d_norm ** 0.5) * range_2d / KAGGLE_SIGMA[(model_id, 2)]

        wandb.log({
            'loss/train': avg_epoch_loss,
            'loss/val_norm': val_loss_norm,
            'loss/val_denorm': val_loss_denorm,
            'loss/val_1d_norm': val_1d_norm,
            'loss/val_2d_norm': val_2d_norm,
            'nrmse/val_1d': nrmse_1d,
            'nrmse/val_2d': nrmse_2d,
            'nrmse/val_kaggle_approx': (nrmse_1d + nrmse_2d) / 2.0,
        }, step=global_step)

        # Checkpoint
        if epoch % CONFIG['checkpoint_interval'] == 0 or epoch == CONFIG['epochs']:
            save_checkpoint(model, epoch, avg_epoch_loss, run_dir, CONFIG)

        # Early stopping + best checkpoint: check if val loss improved by at least min_rel_delta
        min_rel_delta = CONFIG['early_stopping_min_rel_delta']
        if val_loss_norm < best_val_loss * (1.0 - min_rel_delta):
            best_val_loss = val_loss_norm
            early_stopping_counter = 0
            best_checkpoint = os.path.join(run_dir, f'{SELECTED_MODEL}_epoch_{epoch:03d}.pt')
            best_path = os.path.join(run_dir, f'{SELECTED_MODEL}_best.pt')
            if os.path.exists(best_checkpoint):
                import shutil
                shutil.copy(best_checkpoint, best_path)
                print(f"[INFO] New best model saved: {best_path} (val_loss={val_loss_norm:.6e})")
                # Mirror into latest/ so inference always finds all models in one place
                shutil.copy(best_checkpoint, os.path.join(latest_dir, f'{SELECTED_MODEL}_best.pt'))
                for fname in [f'{SELECTED_MODEL}_normalizers.pkl',
                               f'{SELECTED_MODEL}_normalization_stats.json']:
                    src = os.path.join(run_dir, fname)
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(latest_dir, fname))
                print(f"[INFO] Mirrored best checkpoint + normalizers to {latest_dir}")
        else:
            early_stopping_counter += 1
            print(f"[INFO] No significant val improvement ({early_stopping_counter}/{CONFIG['early_stopping_patience']})")

        print()

        # Early stopping trigger
        if early_stopping_counter >= CONFIG['early_stopping_patience']:
            print(f"[INFO] Early stopping triggered after {epoch} epochs (no >{min_rel_delta*100:.0f}% val improvement for {CONFIG['early_stopping_patience']} epochs)")
            wandb.log({'early_stopped_epoch': epoch}, step=global_step)
            break

        epoch_start_time = time.time()

    # Final summary
    print("\n" + "="*70)
    print("Training Complete")
    print("="*70)
    print(f"Best validation loss (Normalized): {best_val_loss:.6f}")
    print(f"\nInterpretation: Denormalized loss is in original water level units (meters)")
    print(f"Checkpoints saved to: {run_dir}")
    print(f"Latest dir:           {latest_dir}")
    final_model_path = os.path.join(run_dir, f'{SELECTED_MODEL}_epoch_{CONFIG["epochs"]:03d}.pt')
    print(f"Final model: {final_model_path}")
    print(f"Best model: {os.path.join(run_dir, f'{SELECTED_MODEL}_best.pt')}")
    print("="*70 + "\n")

if __name__ == "__main__":
    train()
