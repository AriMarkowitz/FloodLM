#!/usr/bin/env python
"""
Model_3 training script — non-autoregressive encoder-decoder.

No curriculum — trains on full events from epoch 1.
Uses gradient accumulation over multiple events to compensate for batch_size=1.

Usage:
    python src/model3/train.py
    python src/model3/train.py --resume checkpoints/Model_3_20260308_120000
    python src/model3/train.py --resume checkpoints/Model_3_20260308_120000/Model_3_epoch_010.pt
    python src/model3/train.py --mixed-precision
"""

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
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import wandb

# Allow running directly as a script
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model3.config import MODEL_ID, CONFIG, HIDDEN_DIMS, KAGGLE_SIGMA_1D, KAGGLE_SIGMA_2D
from model3.data import (
    initialize_data, build_static_graph, get_graph_config,
    get_full_event_dataloader, make_x_dyn,
)
from model3.model import HeteroEncoderDecoderModel


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(model, epoch, loss, save_dir, optimizer=None, scheduler=None,
                    global_step=None, wandb_run_id=None, best=False):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'model_arch_config': {
            'h_dim': model.h_dim,
            'msg_dim': model.cell.msg_dim,
            'hidden_dim': model.cell._hidden_dim,
            'decoder_hidden_dim': CONFIG['decoder_hidden_dim'],
            'T_max': CONFIG['T_max'],
            'dec_d_model': CONFIG['dec_d_model'],
            'dec_nhead': CONFIG['dec_nhead'],
            'dec_num_layers': CONFIG['dec_num_layers'],
            'dec_ffn_dim': CONFIG['dec_ffn_dim'],
            'dec_dropout': CONFIG['dec_dropout'],
        },
        'config': CONFIG,
        'loss': loss,
        'model_id': MODEL_ID,
        'global_step': global_step,
        'wandb_run_id': wandb_run_id,
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
    }
    tag = 'best' if best else f'epoch_{epoch:03d}'
    path = os.path.join(save_dir, f'{MODEL_ID}_{tag}.pt')
    torch.save(checkpoint, path)
    return path


def save_normalization_stats(norm_stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    stats = {
        'static_1d_params':  norm_stats.get('static_1d_params', {}),
        'static_2d_params':  norm_stats.get('static_2d_params', {}),
        'dynamic_1d_params': norm_stats.get('dynamic_1d_params', {}),
        'dynamic_2d_params': norm_stats.get('dynamic_2d_params', {}),
        'node1d_cols': norm_stats.get('node1d_cols', []),
        'node2d_cols': norm_stats.get('node2d_cols', []),
    }
    json_path = os.path.join(save_dir, f'{MODEL_ID}_normalization_stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    pkl_path = os.path.join(save_dir, f'{MODEL_ID}_normalizers.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'normalizer_1d': norm_stats['normalizer_1d'],
            'normalizer_2d': norm_stats['normalizer_2d'],
        }, f)
    print(f"[INFO] Saved norm stats: {json_path}")
    print(f"[INFO] Saved normalizers: {pkl_path}")


# ============================================================
# Validation
# ============================================================

def kaggle_score_from_preds(pred_1d, pred_2d, true_1d, true_2d):
    """
    Replicate Kaggle's per-node RMSE hierarchy (Model_2 sigmas).

    pred_1d, true_1d: [T, N_1d, 1]  (normalized space)
    pred_2d, true_2d: [T, N_2d, 1]

    Kaggle hierarchy:
      1. RMSE per node timeseries / sigma  →  per-node standardized RMSE
      2. Mean over 1D nodes, mean over 2D nodes  →  two group scores
      3. Mean of the two group scores  →  event score

    Since data is already normalized by sigma (meanstd with sigma as std),
    RMSE in normalized space == RMSE_raw / sigma == standardized RMSE.
    """
    # [T, N, 1] → [N, T]
    p1 = pred_1d.squeeze(-1).T  # [N_1d, T]
    t1 = true_1d.squeeze(-1).T
    p2 = pred_2d.squeeze(-1).T  # [N_2d, T]
    t2 = true_2d.squeeze(-1).T

    # Per-node RMSE in normalized space = standardized RMSE
    rmse_per_node_1d = ((p1 - t1) ** 2).mean(dim=1).sqrt()  # [N_1d]
    rmse_per_node_2d = ((p2 - t2) ** 2).mean(dim=1).sqrt()  # [N_2d]

    score_1d = rmse_per_node_1d.mean().item()
    score_2d = rmse_per_node_2d.mean().item()
    return (score_1d + score_2d) / 2, score_1d, score_2d


def evaluate(model, val_loader, static_graph, rain_1d_index, device, use_mixed_precision):
    """
    Run full-event decoding on all val events.
    Returns (kaggle_combined, kaggle_1d, kaggle_2d) — Kaggle-aligned per-node hierarchy.
    """
    model.eval()
    event_scores, event_1d, event_2d = [], [], []

    def _make_x_dyn(y1d, y2d, rain2):
        return make_x_dyn(y1d, y2d, rain2, rain_1d_index, device)

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            y_hist_1d    = batch['y_hist_1d'].to(device)
            y_hist_2d    = batch['y_hist_2d'].to(device)
            rain_hist_2d = batch['rain_hist_2d'].to(device)
            y_future_1d  = batch['y_future_1d'].to(device)     # [T, N_1d, 1]
            y_future_2d  = batch['y_future_2d'].to(device)
            rain_future  = batch['rain_future_2d'].to(device)

            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                preds = model(
                    data=static_graph,
                    y_hist_1d=y_hist_1d, y_hist_2d=y_hist_2d,
                    rain_hist_2d=rain_hist_2d, rain_future_2d=rain_future,
                    make_x_dyn=_make_x_dyn, rain_1d_index=rain_1d_index,
                    history_len=CONFIG['history_len'], device=device,
                )

            ev, e1, e2 = kaggle_score_from_preds(
                preds['oneD'].float(), preds['twoD'].float(),
                y_future_1d, y_future_2d,
            )
            event_scores.append(ev)
            event_1d.append(e1)
            event_2d.append(e2)

    if not event_scores:
        return float('nan'), float('nan'), float('nan')
    return float(np.mean(event_scores)), float(np.mean(event_1d)), float(np.mean(event_2d))


# ============================================================
# Training
# ============================================================

def train(resume_from=None, use_mixed_precision=False):
    print("\n" + "=" * 70)
    print(f"Model_3 Training — Non-Autoregressive Encoder-Decoder")
    print("=" * 70)

    # Peek at checkpoint for wandb resume
    _wandb_run_id = None
    _global_step_resume = 0
    resume_path = None
    resume_file = None
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise ValueError(f"Resume path not found: {resume_path}")
        if resume_path.suffix == '.pt':
            resume_file = resume_path
            resume_path = resume_path.parent
        try:
            f = resume_file or sorted(resume_path.glob(f'{MODEL_ID}*.pt'))[-1]
            ckpt = torch.load(f, map_location='cpu')
            _wandb_run_id = ckpt.get('wandb_run_id')
            _global_step_resume = ckpt.get('global_step', 0) or 0
        except Exception:
            pass

    run_name = f"{MODEL_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if _wandb_run_id:
        wandb.init(project="floodlm", group="model3", id=_wandb_run_id, resume="must", config=CONFIG)
    else:
        wandb.init(project="floodlm", group="model3", name=run_name, config=CONFIG)

    run_dir = os.path.join(CONFIG['save_dir'], run_name)
    latest_dir = os.path.join(CONFIG['save_dir'], 'latest')  # checkpoints/model3/latest/
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # Initialize data
    print("[INFO] Initializing data...")
    data = initialize_data()
    norm_stats = data['norm_stats']
    save_normalization_stats(norm_stats, run_dir)
    save_normalization_stats(norm_stats, latest_dir)

    # Build static graph
    static_graph = build_static_graph(
        data['static_1d_sorted'], data['static_2d_sorted'],
        data['edges1d'], data['edges2d'], data['edges1d2d'],
        data['edges1dfeats'], data['edges2dfeats'],
        data['static_1d_cols'], data['static_2d_cols'],
        data['edge1_cols'], data['edge2_cols'],
    ).to(device)
    rain_1d_index = static_graph.rain_1d_index.to(device) if hasattr(static_graph, 'rain_1d_index') else None

    # Dataloaders
    train_loader = get_full_event_dataloader(split='train', shuffle=True)
    val_loader   = get_full_event_dataloader(split='val',   shuffle=False)

    # Model
    graph_config = get_graph_config(
        data['static_1d_cols'], data['static_2d_cols'],
        data['edge1_cols'], data['edge2_cols'],
    )
    graph_config.update({
        'h_dim':           CONFIG['h_dim'],
        'msg_dim':         CONFIG['msg_dim'],
        'hidden_dim':      HIDDEN_DIMS,
        'decoder_hidden_dim': CONFIG['decoder_hidden_dim'],
        'T_max':           CONFIG['T_max'],
        'dec_d_model':     CONFIG['dec_d_model'],
        'dec_nhead':       CONFIG['dec_nhead'],
        'dec_num_layers':  CONFIG['dec_num_layers'],
        'dec_ffn_dim':     CONFIG['dec_ffn_dim'],
        'dec_dropout':     CONFIG['dec_dropout'],
    })
    model = HeteroEncoderDecoderModel(**graph_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {total_params:,}")
    wandb.watch(model, log='all', log_freq=50)

    optimizer = Adam(model.parameters(), lr=CONFIG['lr'])
    _lr_ratio = CONFIG['lr_final'] / CONFIG['lr']
    _total_epochs = CONFIG['epochs']
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: _lr_ratio ** (e / max(_total_epochs - 1, 1)))
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None

    # Load checkpoint
    start_epoch = 1
    if resume_path:
        try:
            ckpt_path = resume_file or sorted(resume_path.glob(f'{MODEL_ID}*.pt'))[-1]
            print(f"[INFO] Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            if 'optimizer_state' in ckpt and ckpt['optimizer_state']:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            if 'scheduler_state' in ckpt and ckpt['scheduler_state']:
                scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"[INFO] Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}. Starting fresh.")

    criterion = nn.MSELoss()
    global_step = _global_step_resume
    best_val = float('inf')
    no_improve = 0

    def _make_x_dyn(y1d, y2d, rain2):
        return make_x_dyn(y1d, y2d, rain2, rain_1d_index, device)

    print(f"\n[INFO] Training config:")
    print(f"  Epochs: {CONFIG['epochs']}, LR: {CONFIG['lr']} → {CONFIG['lr_final']:.2e}")
    print(f"  Batch: 1 event, grad_accum: {CONFIG['grad_accum_steps']} events")
    print(f"  Mixed precision: {use_mixed_precision}")
    print(f"\n{'=' * 70}\nTraining\n{'=' * 70}\n")

    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        n_events = 0
        epoch_start = time.time()

        wandb.log({'epoch': epoch}, step=global_step)
        optimizer.zero_grad()

        for event_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            y_hist_1d    = batch['y_hist_1d'].to(device)       # [H, N_1d, 1]
            y_hist_2d    = batch['y_hist_2d'].to(device)
            rain_hist_2d = batch['rain_hist_2d'].to(device)
            y_future_1d  = batch['y_future_1d'].to(device)     # [T, N_1d, 1]
            y_future_2d  = batch['y_future_2d'].to(device)
            rain_future  = batch['rain_future_2d'].to(device)

            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                preds = model(
                    data=static_graph,
                    y_hist_1d=y_hist_1d, y_hist_2d=y_hist_2d,
                    rain_hist_2d=rain_hist_2d, rain_future_2d=rain_future,
                    make_x_dyn=_make_x_dyn, rain_1d_index=rain_1d_index,
                    history_len=CONFIG['history_len'], device=device,
                )
                loss_1d = criterion(preds['oneD'], y_future_1d)
                loss_2d = criterion(preds['twoD'], y_future_2d)
                loss = (loss_1d + loss_2d) / 2
                # Scale by grad_accum so effective lr is stable regardless of accum steps
                loss_scaled = loss / CONFIG['grad_accum_steps']

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            nrmse_1d = loss_1d.item() ** 0.5
            nrmse_2d = loss_2d.item() ** 0.5
            epoch_loss += loss.item()
            n_events += 1
            global_step += 1
            horizon = y_future_1d.shape[0]

            wandb.log({
                'train/loss': loss.item(),
                'train/nrmse_1d': nrmse_1d,
                'train/nrmse_2d': nrmse_2d,
                'lr': optimizer.param_groups[0]['lr'],
            }, step=global_step)

            # Gradient accumulation: step every N events
            if n_events % CONFIG['grad_accum_steps'] == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                    optimizer.step()
                optimizer.zero_grad()

        # Final gradient step for remaining accumulation
        if n_events % CONFIG['grad_accum_steps'] != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(n_events, 1)
        print(f"[Epoch {epoch:03d}/{CONFIG['epochs']}] loss={mean_loss:.4f}  events={n_events}  time={epoch_time:.0f}s")

        # Validation
        val_start = time.time()
        val_combined, val_1d, val_2d = evaluate(
            model, val_loader, static_graph, rain_1d_index, device, use_mixed_precision)
        val_time = time.time() - val_start
        print(f"  Val NRMSE: combined={val_combined:.4f}  1D={val_1d:.4f}  2D={val_2d:.4f}  val_t={val_time:.0f}s")
        wandb.log({
            'val/nrmse_combined': val_combined,
            'val/nrmse_1d': val_1d,
            'val/nrmse_2d': val_2d,
            'train/epoch_loss': mean_loss,
            'epoch_time_s': epoch_time,
            'val/time_s': val_time,
        }, step=global_step)

        # Save periodic checkpoint
        if epoch % CONFIG['checkpoint_interval'] == 0:
            ckpt_path = save_checkpoint(
                model, epoch, mean_loss, run_dir,
                optimizer=optimizer, scheduler=scheduler,
                global_step=global_step, wandb_run_id=wandb.run.id if wandb.run else None)

        # Best checkpoint
        if not np.isnan(val_combined) and val_combined < best_val:
            best_val = val_combined
            no_improve = 0
            save_checkpoint(
                model, epoch, mean_loss, run_dir,
                optimizer=optimizer, scheduler=scheduler,
                global_step=global_step, wandb_run_id=wandb.run.id if wandb.run else None,
                best=True)
            # Mirror to latest/
            save_checkpoint(
                model, epoch, mean_loss, latest_dir,
                optimizer=optimizer, scheduler=scheduler,
                global_step=global_step, wandb_run_id=wandb.run.id if wandb.run else None,
                best=True)
        else:
            no_improve += 1

        # Early stopping
        if (CONFIG['early_stopping_patience'] is not None
                and no_improve >= CONFIG['early_stopping_patience']):
            print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {no_improve} epochs)")
            break

    print(f"\n[INFO] Training complete. Best val NRMSE: {best_val:.4f}")
    wandb.finish()


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model_3')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint dir or .pt file to resume from')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use AMP mixed precision training')
    args = parser.parse_args()
    train(resume_from=args.resume, use_mixed_precision=args.mixed_precision)
