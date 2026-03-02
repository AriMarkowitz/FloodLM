#!/usr/bin/env python
"""Full training script for FloodLM with model and normalization checkpointing."""

import os
import sys
import json
import time
import pickle
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
    'epochs': 10,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'save_dir': 'checkpoints',
    'checkpoint_interval': 1,  # Save every N epochs
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

def evaluate(model, dataloader, criterion, device, norm_stats, split_name='Validation', debug=False, max_batches=None, batched_static_graph=None):
    """Evaluate model on a dataset with both normalized and denormalized losses.

    Args:
        max_batches: If set, only evaluate on the first N batches (useful for faster validation)
        batched_static_graph: Pre-built PyG Batch of B copies of the static graph (avoids rebuild each call)
    """
    model.eval()
    total_loss_norm = 0.0
    total_loss_denorm = 0.0
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

            # Node-count-weighted normalized loss (MSELoss averages over all B*T*N elements)
            n_1d = predictions['oneD'].shape[2]
            n_2d = predictions['twoD'].shape[2]
            n_total = n_1d + n_2d
            loss_1d = criterion(predictions['oneD'], y_future_1d)
            loss_2d = criterion(predictions['twoD'], y_future_2d)
            loss_norm = (n_1d * loss_1d + n_2d * loss_2d) / n_total

            # Denormalize for interpretable loss
            pred_1d_denorm = unnormalize_col(predictions['oneD'], norm_stats, col=wl_col_1d, node_type='oneD')
            target_1d_denorm = unnormalize_col(y_future_1d, norm_stats, col=wl_col_1d, node_type='oneD')
            pred_2d_denorm = unnormalize_col(predictions['twoD'], norm_stats, col=wl_col_2d, node_type='twoD')
            target_2d_denorm = unnormalize_col(y_future_2d, norm_stats, col=wl_col_2d, node_type='twoD')
            loss_1d_denorm = criterion(pred_1d_denorm, target_1d_denorm)
            loss_2d_denorm = criterion(pred_2d_denorm, target_2d_denorm)
            loss_denorm = (n_1d * loss_1d_denorm + n_2d * loss_2d_denorm) / n_total

            total_loss_norm += loss_norm.item()
            total_loss_denorm += loss_denorm.item()
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
    
    avg_loss_norm = total_loss_norm / num_batches if num_batches > 0 else 0
    avg_loss_denorm = total_loss_denorm / num_batches if num_batches > 0 else 0
    
    print(f"[INFO] {split_name} Loss (Normalized): {avg_loss_norm:.9f}")
    print(f"[INFO] {split_name} Loss (Denormalized): {avg_loss_denorm:.9f}")
    
    if debug and len(all_errors_norm) > 0:
        errors_norm = np.concatenate(all_errors_norm)
        errors_denorm = np.concatenate(all_errors_denorm)
        print(f"[DEBUG] Error distribution (normalized): mean={errors_norm.mean():.9f}, std={errors_norm.std():.9f}, max={errors_norm.max():.9f}")
        print(f"[DEBUG] Error distribution (denormalized): mean={errors_denorm.mean():.9f}, std={errors_denorm.std():.9f}, max={errors_denorm.max():.9f}")
    
    return avg_loss_norm, avg_loss_denorm

def train():
    """Main training loop."""
    print("\n" + "="*70)
    print("FloodLM Training Script")
    print("="*70)
    
    wandb.init(project="floodlm", config=CONFIG)

    device = torch.device(CONFIG['device'])
    print(f"[INFO] Device: {device}")
    
    # Initialize data and fetch preprocessing
    print(f"\n[INFO] Initializing data...")
    data = initialize_data()
    norm_stats = data['norm_stats']
    
    # Save normalization statistics
    print(f"[INFO] Saving normalization statistics...")
    save_normalization_stats(norm_stats, CONFIG['save_dir'])
    
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
    
    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()

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

    best_loss = float('inf')
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

            # Node-count-weighted loss (MSELoss averages over all B*T*N elements)
            n_1d = predictions['oneD'].shape[2]
            n_2d = predictions['twoD'].shape[2]
            n_total = n_1d + n_2d
            loss_1d = criterion(predictions['oneD'], y_future_1d)
            loss_2d = criterion(predictions['twoD'], y_future_2d)
            loss = (n_1d * loss_1d + n_2d * loss_2d) / n_total
            
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
            }

            # Lightweight validation every N batches — logged at the same step as train
            if (batch_idx + 1) % VAL_CHECK_INTERVAL == 0:
                val_loss_norm, val_loss_denorm = evaluate(
                    model, val_dataloader, criterion, device, norm_stats,
                    split_name=f"Val-Subset (Epoch {epoch}, Batch {batch_idx+1})",
                    debug=False,
                    max_batches=VAL_SUBSET_BATCHES,
                    batched_static_graph=val_batched_graph,
                )
                print(f"  → Quick Val Loss (Norm): {val_loss_norm:.9f}")
                log_dict['loss/val_norm'] = val_loss_norm
                log_dict['loss/val_denorm'] = val_loss_denorm
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
        val_loss_norm, val_loss_denorm = evaluate(model, val_dataloader, criterion, device, norm_stats, split_name="Validation", debug=False, batched_static_graph=val_batched_graph)

        wandb.log({
            'loss/train': avg_epoch_loss,
            'loss/val_norm': val_loss_norm,
            'loss/val_denorm': val_loss_denorm,
        }, step=global_step)
        
        print()
        
        # Checkpoint
        if epoch % CONFIG['checkpoint_interval'] == 0 or epoch == CONFIG['epochs']:
            save_checkpoint(model, epoch, avg_epoch_loss, CONFIG['save_dir'], CONFIG)
        
        # Track best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_checkpoint = os.path.join(CONFIG['save_dir'], f'{SELECTED_MODEL}_epoch_{epoch:03d}.pt')
            best_path = os.path.join(CONFIG['save_dir'], f'{SELECTED_MODEL}_best.pt')
            if os.path.exists(best_checkpoint):
                import shutil
                shutil.copy(best_checkpoint, best_path)
                print(f"[INFO] New best model saved: {best_path}")
        
        epoch_start_time = time.time()
    
    # Final summary
    print("\n" + "="*70)
    print("Training Complete")
    print("="*70)
    print(f"Best training loss (Normalized): {best_loss:.6f}")
    print(f"\nInterpretation: Denormalized loss is in original water level units (meters)")
    print(f"Checkpoints saved to: {CONFIG['save_dir']}")
    final_model_path = os.path.join(CONFIG['save_dir'], f'{SELECTED_MODEL}_epoch_{CONFIG["epochs"]:03d}.pt')
    print(f"Final model: {final_model_path}")
    print(f"Best model: {os.path.join(CONFIG['save_dir'], f'{SELECTED_MODEL}_best.pt')}")
    print("="*70 + "\n")

if __name__ == "__main__":
    train()
