#!/usr/bin/env python
"""Full training script for FloodLM with model and normalization checkpointing."""

import os
import sys
import json
import time
import pickle
import random
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import wandb

# Setup paths (works whether launched via train.py wrapper or directly)
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import get_recurrent_dataloader, get_model_config, make_x_dyn, NonBatchableGraph
from model import FloodAutoregressiveHeteroModel
from data_lazy import initialize_data
from data_config import SELECTED_MODEL

# Configuration
CONFIG = {
    'history_len': 10,
    'forecast_len': 64,          # Max rollout horizon (curriculum will sample 1..max_h per batch)
    'batch_size': 24,
    'epochs': 46,                # Model_2: 6@h1, 4@h2, 4@h4, 4@h6, 4@h8, 4@h16, 4@h24, 4@h32, 4@h48, 8@h64 (46 total); Model_1: 2 per stage
    'lr': 1e-3,
    'lr_final': 10**-4.5,          # ~3.16e-5; log-linear decay over all epochs (1.5 decades)
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'save_dir': 'checkpoints',
    'checkpoint_interval': 1,   # Save every N epochs
    'early_stopping_patience': 5,  # Only active once max_h == forecast_len; None to disable
    'early_stopping_min_rel_delta': 0.01,
    'curriculum_val_horizon': 32,  # Fixed horizon for multi-step val rollout
    # Noise-injection warm-start curriculum (Model_2 only)
    # Stage k trains with:  10 clean warm-start steps + k perturbed steps → predict step 11+k
    # Smart-detect advance: after each epoch, scan per-step train NRMSE up to noise_max_extra_steps.
    # K advances to the largest contiguous prefix where nrmse[k] <= min(nrmse[:5]) * noise_smart_alpha.
    # noise_smart_init_epochs: epochs of pure h=1 training before smart detection activates
    # noise_smart_probe_batches: train batches used for the per-step NRMSE scan (keep small)
    # noise_smart_alpha: relative threshold — include step k if nrmse[k] <= alpha * min(nrmse[:5])
    # noise_max_extra_steps:  maximum extra perturbed warm-start steps (K_max)
    'noise_smart_alpha': 1.3,
    'noise_smart_init_epochs': 2,
    'noise_smart_probe_batches': 8,
    'noise_max_extra_steps': 54,   # 10 + 54 = 64 total warm-start, then predict step 65 (or clamp)
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

def save_checkpoint(model, epoch, loss, save_dir, config, model_id=None, global_step=None, scheduler=None, optimizer=None):
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
            'h_dim': model.cell._h_dim,  # may be dict (per node type) or int
            'msg_dim': model.cell.msg_dim,
            'hidden_dim': model.cell._hidden_dim,
            'num_1d_extra_hops': model.cell.num_1d_extra_hops,
            'node_dyn_input_dims': CONFIG.get('node_dyn_input_dims'),
        },
        'loss': loss,
        'model_id': model_id,
        'global_step': global_step,
        'wandb_run_id': wandb.run.id if wandb.run is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
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


def evaluate_full_event_rollout(model, val_event_file_list, data, norm_stats, static_graph, device,
                                history_len=10, use_mixed_precision=False, rain_1d_index=None,
                                n_rain_channels=None):
    """Run full autoregressive rollout (t=0 to T) on each val event and compute NRMSE.

    Mirrors the inference loop exactly — warm-starts on history_len true timesteps,
    then predicts autoregressively for the rest of the event.

    Returns (combined_nrmse, nrmse_1d, nrmse_2d, per_event_results) averaged over all val events.
    per_event_results: list of dicts with keys 'event_name', 'combined_nrmse', 'nrmse_1d', 'nrmse_2d', 'T',
        't_worst' (timestep of max combined error), 't_worst_frac' (as fraction of T),
        'rain_total', 'rain_peak', 't_rain_peak_frac', 'rain_at_worst' (spatially-averaged rain channel 0)
    """
    from autoregressive_inference import prepare_event_tensors, autoregressive_rollout_both

    model.eval()
    total_1d, total_2d = 0.0, 0.0
    n_events = 0
    per_event_results = []

    with torch.no_grad():
        for _, event_path, _ in val_event_file_list:
            try:
                from autoregressive_inference import load_event_data
                node_1d, node_2d = load_event_data(event_path)
                y1_all, y2_all, rain2_all, _, _, _ = prepare_event_tensors(node_1d, node_2d, norm_stats, device)
                if n_rain_channels is not None:
                    rain2_all = rain2_all[:, :, :n_rain_channels]
                T = y1_all.size(0)
                if T <= history_len:
                    continue
                y1_hist = y1_all[:history_len]   # [H, N1, 1]
                y2_hist = y2_all[:history_len]   # [H, N2, 1]
                with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                    pred_1d, pred_2d = autoregressive_rollout_both(
                        model, static_graph, y1_hist, y2_hist, rain2_all, device, history_len=history_len
                    )
                # pred_1d: [T-H, N1, 1], y1_all[H:]: [T-H, N1, 1]
                T_pred = T - history_len
                # Per-timestep MSE: [T_pred]
                err_1d_t = ((pred_1d - y1_all[history_len:]) ** 2).mean(dim=(1, 2))  # [T_pred]
                err_2d_t = ((pred_2d - y2_all[history_len:]) ** 2).mean(dim=(1, 2))
                mse_1d = err_1d_t.mean().item()
                mse_2d = err_2d_t.mean().item()
                total_1d += mse_1d
                total_2d += mse_2d
                n_events += 1

                # Combined per-timestep instantaneous NRMSE
                combined_t = (err_1d_t ** 0.5 + err_2d_t ** 0.5) / 2  # [T_pred]
                # t_worst = timestep where error grows fastest (largest single-step jump)
                if T_pred > 1:
                    delta_t = combined_t[1:] - combined_t[:-1]  # [T_pred-1]
                    t_worst = delta_t.argmax().item() + 1        # +1 since diff is shifted
                else:
                    t_worst = 0
                t_worst_frac = t_worst / max(T_pred - 1, 1)

                # Rainfall stats over prediction window (rain2_all: [T, N_rain, C])
                rain_pred = rain2_all[history_len:, :, 0]  # [T_pred, N_rain] — first rain channel
                rain_mean_t = rain_pred.mean(dim=1)        # [T_pred] — spatial mean per timestep
                rain_total = rain_mean_t.sum().item()
                rain_peak = rain_mean_t.max().item()
                t_rain_peak = rain_mean_t.argmax().item()
                t_rain_peak_frac = t_rain_peak / max(T_pred - 1, 1)
                rain_at_worst = rain_mean_t[t_worst].item()

                event_name = Path(str(event_path)).name
                per_event_results.append({
                    'event_name': event_name,
                    'nrmse_1d': mse_1d ** 0.5,
                    'nrmse_2d': mse_2d ** 0.5,
                    'combined_nrmse': (mse_1d ** 0.5 + mse_2d ** 0.5) / 2,
                    'T': T_pred,
                    't_worst': t_worst,
                    't_worst_frac': round(t_worst_frac, 3),
                    'rain_total': round(rain_total, 4),
                    'rain_peak': round(rain_peak, 4),
                    't_rain_peak_frac': round(t_rain_peak_frac, 3),
                    'rain_at_worst': round(rain_at_worst, 4),
                })
            except Exception as e:
                print(f"[WARNING] Full-event val failed for {event_path}: {e}")

    if n_events == 0:
        return float('nan'), float('nan'), float('nan'), []
    nrmse_1d = (total_1d / n_events) ** 0.5
    nrmse_2d = (total_2d / n_events) ** 0.5
    return (nrmse_1d + nrmse_2d) / 2, nrmse_1d, nrmse_2d, per_event_results


def _measure_rollout_nrmse_per_step(model, train_dataloader, max_steps, device,
                                     batched_static_graph=None, use_mixed_precision=False,
                                     rain_1d_index=None, max_batches=8):
    """
    Cheap per-step NRMSE scan for smart curriculum advance.

    Runs a true-fed rollout on up to max_batches train batches and returns
    scalar NRMSE for each step 0..max_steps-1.  Step k NRMSE is the sqrt of
    the mean squared normalised error when predicting step k from the true
    history (i.e. no error accumulation — each step is independent).

    Returns:
        nrmse_per_step: list of floats, length = actual steps measured (<= max_steps)
    """
    model.eval()
    # sq_err[k] accumulates sum of squared errors; count[k] counts samples
    sq_err = [0.0] * max_steps
    counts = [0]   * max_steps

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:
                continue
            if batch_idx >= max_batches:
                break

            avail = batch['y_future_1d'].shape[1]
            steps = min(max_steps, avail)

            static_graph   = batch['static_graph'].to(device)
            y_hist_1d      = batch['y_hist_1d'].to(device)
            y_hist_2d      = batch['y_hist_2d'].to(device)
            rain_hist_2d   = batch['rain_hist_2d'].to(device)
            y_future_1d    = batch['y_future_1d'].to(device)
            y_future_2d    = batch['y_future_2d'].to(device)
            rain_future_2d = batch['rain_future_2d'].to(device)

            B    = y_hist_1d.size(0)
            N_1d = y_hist_1d.size(2)
            N_2d = y_hist_2d.size(2)
            H    = y_hist_1d.size(1)

            bsg = batched_static_graph
            if bsg is None:
                bsg = model._make_batched_graph(static_graph, B)

            _r1d = rain_1d_index if rain_1d_index is not None else getattr(static_graph, 'rain_1d_index', None)

            def _make_x(y, r, d, _r=_r1d):
                return make_x_dyn(y['oneD'], y['twoD'], r, d, rain_1d_index=_r)

            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                # Clean warm-start
                h = model.init_hidden(static_graph, B, device=device)
                for k in range(H):
                    y1d_k = y_hist_1d[:, k].reshape(B * N_1d, 1)
                    y2d_k = y_hist_2d[:, k].reshape(B * N_2d, 1)
                    r_k   = rain_hist_2d[:, k].reshape(B * N_2d, -1)
                    y_t   = {'oneD': y1d_k, 'twoD': y2d_k}
                    x_dyn = _make_x(y_t, r_k, bsg)
                    for _ctx in ('global', 'ctx1d', 'ctx2d'):
                        if _ctx in model.node_types:
                            x_dyn[_ctx] = torch.zeros(B, 1, device=device, dtype=y1d_k.dtype)
                    h = model.cell(bsg, h, x_dyn)

                # True-fed rollout: predict each step from true history
                for k in range(steps):
                    y_pred_dict = model.predict_water_levels(h, B, {'oneD': N_1d, 'twoD': N_2d})
                    # Combined MSE (both node types equally weighted)
                    mse_1d = float(((y_pred_dict['oneD'] - y_future_1d[:, k]) ** 2).mean())
                    mse_2d = float(((y_pred_dict['twoD'] - y_future_2d[:, k]) ** 2).mean())
                    sq_err[k] += (mse_1d + mse_2d) / 2.0
                    counts[k] += 1

                    # Feed TRUE value for isolation
                    y1d_true = y_future_1d[:, k].reshape(B * N_1d, 1)
                    y2d_true = y_future_2d[:, k].reshape(B * N_2d, 1)
                    r_k      = rain_future_2d[:, k].reshape(B * N_2d, -1)
                    y_t      = {'oneD': y1d_true, 'twoD': y2d_true}
                    x_dyn    = _make_x(y_t, r_k, bsg)
                    for _ctx in ('global', 'ctx1d', 'ctx2d'):
                        if _ctx in model.node_types:
                            x_dyn[_ctx] = torch.zeros(B, 1, device=device, dtype=y1d_true.dtype)
                    h = model.cell(bsg, h, x_dyn)

    nrmse_per_step = [
        (sq_err[k] / counts[k]) ** 0.5 if counts[k] > 0 else float('inf')
        for k in range(max_steps)
    ]
    model.train()
    return nrmse_per_step


def collect_per_lag_noise_stats(model, val_dataloader, max_lag, device,
                                 batched_static_graph=None, use_mixed_precision=False,
                                 rain_1d_index=None):
    """
    Run a single-step rollout over the val set to collect per-node, per-lag
    signed prediction errors.

    For lag k (0-indexed), we measure err[b, n] = pred[k] - true[k]  when the
    model is fed true water levels for steps 0..k-1 and predicts step k.
    This gives the distribution of errors that arise when a (k+1)-step-ahead
    prediction is fed back as input.

    Returns:
        noise_mu_1d:    [max_lag, N_1d]  — per-lag per-node mean signed error
        noise_sigma_1d: [max_lag, N_1d]  — per-lag per-node std of signed error
        noise_mu_2d:    [max_lag, N_2d]
        noise_sigma_2d: [max_lag, N_2d]
    """
    model.eval()
    # Accumulators: list of lists — outer = lag, inner = per-batch error tensors
    errs_1d = [[] for _ in range(max_lag)]
    errs_2d = [[] for _ in range(max_lag)]
    N_1d = N_2d = 1  # fallback; overwritten from first batch

    with torch.no_grad():
        for batch in val_dataloader:
            if batch is None:
                continue
            avail = batch['y_future_1d'].shape[1]
            lags_to_use = min(max_lag, avail)

            static_graph  = batch['static_graph'].to(device)
            y_hist_1d     = batch['y_hist_1d'].to(device)      # [B, H, N_1d, 1]
            y_hist_2d     = batch['y_hist_2d'].to(device)
            rain_hist_2d  = batch['rain_hist_2d'].to(device)
            y_future_1d   = batch['y_future_1d'].to(device)    # [B, T, N_1d, 1]
            y_future_2d   = batch['y_future_2d'].to(device)
            rain_future_2d = batch['rain_future_2d'].to(device)

            B    = y_hist_1d.size(0)
            N_1d = y_hist_1d.size(2)
            N_2d = y_hist_2d.size(2)
            H    = y_hist_1d.size(1)

            bsg = batched_static_graph
            if bsg is None:
                bsg = model._make_batched_graph(static_graph, B)

            _r1d = rain_1d_index if rain_1d_index is not None else getattr(static_graph, 'rain_1d_index', None)

            def _make_x(y, r, d, _r=_r1d):
                return make_x_dyn(y['oneD'], y['twoD'], r, d, rain_1d_index=_r)

            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                # Clean warm-start
                h = model.init_hidden(static_graph, B, device=device)
                for k in range(H):
                    y1d_k = y_hist_1d[:, k].reshape(B * N_1d, 1)
                    y2d_k = y_hist_2d[:, k].reshape(B * N_2d, 1)
                    r_k   = rain_hist_2d[:, k].reshape(B * N_2d, -1)
                    y_t   = {'oneD': y1d_k, 'twoD': y2d_k}
                    x_dyn = _make_x(y_t, r_k, bsg)
                    for _ctx in ('global', 'ctx1d', 'ctx2d'):
                        if _ctx in model.node_types:
                            x_dyn[_ctx] = torch.zeros(B, 1, device=device, dtype=y1d_k.dtype)
                    h = model.cell(bsg, h, x_dyn)

                # Roll out one step at a time, always feeding TRUE water level at each step
                # so we measure the error at lag k independently of prior errors.
                # At each lag k:  predict from h_k (warmed up on true[0..k-1]), compare to true[k]
                for k in range(lags_to_use):
                    y_pred_dict = model.predict_water_levels(h, B, {'oneD': N_1d, 'twoD': N_2d})
                    # signed error: pred - true, shape [B, N, 1]
                    err_1d = (y_pred_dict['oneD'] - y_future_1d[:, k]).squeeze(-1)  # [B, N_1d]
                    err_2d = (y_pred_dict['twoD'] - y_future_2d[:, k]).squeeze(-1)  # [B, N_2d]
                    errs_1d[k].append(err_1d.float().cpu())
                    errs_2d[k].append(err_2d.float().cpu())

                    # Feed TRUE value for next step (not prediction) to isolate each lag
                    y1d_true = y_future_1d[:, k].reshape(B * N_1d, 1)
                    y2d_true = y_future_2d[:, k].reshape(B * N_2d, 1)
                    r_k      = rain_future_2d[:, k].reshape(B * N_2d, -1)
                    y_t      = {'oneD': y1d_true, 'twoD': y2d_true}
                    x_dyn    = _make_x(y_t, r_k, bsg)
                    for _ctx in ('global', 'ctx1d', 'ctx2d'):
                        if _ctx in model.node_types:
                            x_dyn[_ctx] = torch.zeros(B, 1, device=device, dtype=y1d_true.dtype)
                    h = model.cell(bsg, h, x_dyn)

    # Aggregate across batches
    noise_mu_1d    = torch.zeros(max_lag, N_1d)
    noise_sigma_1d = torch.zeros(max_lag, N_1d)
    noise_mu_2d    = torch.zeros(max_lag, N_2d)
    noise_sigma_2d = torch.zeros(max_lag, N_2d)
    for k in range(max_lag):
        if errs_1d[k]:
            e1 = torch.cat(errs_1d[k], dim=0)   # [total_B, N_1d]
            noise_mu_1d[k]    = e1.mean(dim=0)
            noise_sigma_1d[k] = e1.std(dim=0).clamp(min=1e-6)
        if errs_2d[k]:
            e2 = torch.cat(errs_2d[k], dim=0)
            noise_mu_2d[k]    = e2.mean(dim=0)
            noise_sigma_2d[k] = e2.std(dim=0).clamp(min=1e-6)

    model.train()
    return noise_mu_1d, noise_sigma_1d, noise_mu_2d, noise_sigma_2d


def train(resume_from=None, use_mixed_precision=False, skip_validation=False, pretrain_from=None, train_split='train', extra_epochs=None, mirror_latest=True, cold_start_passes=None):
    """Main training loop.
    
    Args:
        resume_from: Path to checkpoint directory to resume from
        use_mixed_precision: Whether to use mixed precision (float16) training
    """
    
    # Determine if resuming
    resume_path = None
    resume_specific_file = None  # Set when --resume points to a .pt file directly
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise ValueError(f"Resume checkpoint not found: {resume_path}")
        if resume_path.suffix == '.pt':
            # Specific file provided — use it directly
            resume_specific_file = resume_path
            resume_path = resume_path.parent
        print(f"\n[INFO] Resuming training from: {resume_specific_file or resume_path}")
    print("\n" + "="*70)
    print("FloodLM Training Script")
    print("="*70)

    # Peek at checkpoint to recover wandb run ID and global_step before init
    _wandb_resume_id = None
    _global_step_resume = 0
    if resume_path:
        try:
            if resume_specific_file:
                _peek_file = resume_specific_file
            else:
                _ckpt_files = sorted(resume_path.glob(f'{SELECTED_MODEL}*.pt'))
                _peek_file = _ckpt_files[-1] if _ckpt_files else None
            if _peek_file:
                _peek = torch.load(_peek_file, map_location='cpu')
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
    if cold_start_passes is not None:
        print(f"  Cold-start mode: full-event rollout per event, {cold_start_passes} passes")

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
    CONFIG['node_dyn_input_dims'] = model_config['node_dyn_input_dims']
    _n_rain_channels = model_config['node_dyn_input_dims']['twoD'] - 1  # twoD = wl + rain_channels
    
    # Initialize model
    print(f"\n[INFO] Building model...")
    model_config.update({
        # Per-node-type GRU hidden size: Model_2 gives 1D nodes extra capacity
        # since its 190-node channel network is the hard bottleneck (order-of-mag harder).
        # Model_1 uses scalar 96 (17 nodes, shared dim is fine).
        'h_dim': {'oneD': 192, 'twoD': 96, 'global': 32} if SELECTED_MODEL == 'Model_2' else 96,
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
        # Model_2 has 190 1D nodes (vs 17 for Model_1) — extra hops propagate
        # information further along the larger channel network each timestep.
        'num_1d_extra_hops': 4 if SELECTED_MODEL == 'Model_2' else 0,
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

    # Log-linear LR decay: lr_init → lr_final over CONFIG['epochs'] epochs
    # At epoch e (1-indexed), multiplier = (lr_final/lr_init)^((e-1)/(N-1))
    _lr_ratio = CONFIG['lr_final'] / CONFIG['lr']
    _total_epochs = CONFIG['epochs']
    def _lr_lambda(epoch_0indexed):
        if _total_epochs <= 1:
            return 1.0
        return _lr_ratio ** (epoch_0indexed / (_total_epochs - 1))
    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # Mixed precision: GradScaler for stable fp16 backward pass
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None

    # Load checkpoint if resuming
    start_epoch = 1
    if resume_path:
        try:
            # Use specific file if provided, otherwise pick latest in directory
            if resume_specific_file:
                checkpoint_path = resume_specific_file
            else:
                checkpoint_files = sorted(resume_path.glob(f'{SELECTED_MODEL}*.pt'))
                if not checkpoint_files:
                    raise FileNotFoundError(f"No checkpoints found for {SELECTED_MODEL} in {resume_path}")
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

            if 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
                # base_lrs is saved in scheduler state from the original run.
                # After load, base_lrs may reflect the old lr (e.g. 1e-3) — override
                # with the current CONFIG lr so scheduler.step() doesn't jump to old scale.
                scheduler.base_lrs = [CONFIG['lr']] * len(scheduler.base_lrs)
                print(f"[INFO] Scheduler state restored (base_lrs overridden to {CONFIG['lr']:.2e})")
            
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
    _single_static_graph = _static_graph_cpu  # used by full-event rollout val (B=1 inference)
    _val_event_file_list = data.get('val_event_file_list', [])
    # Capture rain_1d_index on device for use in make_x_dyn closures (graph-level attr
    # may not survive Batch.from_data_list so we pin it to a closure variable instead).
    _rain_1d_index = _static_graph_cpu.rain_1d_index.to(device) if hasattr(_static_graph_cpu, 'rain_1d_index') else None
    print(f"[INFO] Batched graphs ready.")

    # No per-node loss mask: masking node 197 from training loss left it undertrained
    # at inference → outlier prediction that hurts Kaggle RMSE more than noisy gradients.
    _loss_mask_1d = None

    print(f"\n[INFO] Training configuration:")
    print(f"  Learning rate: {CONFIG['lr']} → {CONFIG['lr_final']:.2e} (log-linear over {CONFIG['epochs']} epochs)")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Checkpoint interval: {CONFIG['checkpoint_interval']}")

    # Training loop
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}\n")

    epoch_start_time = time.time()
    global_step = _global_step_resume  # Restored from checkpoint on resume; 0 for fresh runs

    # -----------------------------------------------------------------------
    # Cold-start fine-tuning mode: train exclusively on the first window of
    # each event (matching the inference condition) for N shuffled passes.
    # Replaces the standard epoch loop entirely when --cold-start-passes is set.
    # -----------------------------------------------------------------------
    if cold_start_passes is not None:
        from fullevent.data import get_full_event_dataset
        print(f"[INFO] Cold-start fine-tuning: loading full-event dataset (train split={train_split})...")
        cs_dataset = get_full_event_dataset(split=train_split, shuffle=True,
                                            history_len=CONFIG['history_len'])
        n_events = len(cs_dataset)
        print(f"[INFO] Cold-start pool: {n_events} events, batch_size={CONFIG['batch_size']}, "
              f"{cold_start_passes} passes (grouped by T_future for memory efficiency)")

        # Move static graph to device once — reused for every cold-start batch
        cs_static_graph_dev = _static_graph_cpu.to(device)

        total_loss = 0.0
        total_batches = 0
        pass_start = time.time()

        for pass_idx in range(1, cold_start_passes + 1):
            cs_dataset.shuffle = True  # re-shuffle each pass
            for batch in cs_dataset.iter_grouped(CONFIG['batch_size']):
                y_hist_1d_b      = batch['y_hist_1d'].to(device, non_blocking=True)
                y_hist_2d_b      = batch['y_hist_2d'].to(device, non_blocking=True)
                rain_hist_2d_b   = batch['rain_hist_2d'].to(device, non_blocking=True)
                y_future_1d_b    = batch['y_future_1d'].to(device, non_blocking=True)
                y_future_2d_b    = batch['y_future_2d'].to(device, non_blocking=True)
                rain_future_2d_b = batch['rain_future_2d'].to(device, non_blocking=True)
                rollout_steps    = y_future_1d_b.shape[1]  # full event length (variable)
                B = y_hist_1d_b.shape[0]

                # Build batched static graph for this batch size (may differ from train_batched_graph)
                cs_batched_graph = model._make_batched_graph(_static_graph_cpu, B).to(device)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                    predictions = model.forward_unroll(
                        data=cs_static_graph_dev,
                        y_hist_1d=y_hist_1d_b,
                        y_hist_2d=y_hist_2d_b,
                        rain_hist=rain_hist_2d_b,
                        rain_future=rain_future_2d_b,
                        make_x_dyn=lambda y, r, d: make_x_dyn(
                            y['oneD'], y['twoD'], r, d,
                            rain_1d_index=_rain_1d_index,
                        ),
                        rollout_steps=rollout_steps,
                        device=device,
                        batched_data=cs_batched_graph,
                        use_grad_checkpoint=use_mixed_precision,
                    )
                    loss_1d = criterion(predictions['oneD'], y_future_1d_b)
                    loss_2d = criterion(predictions['twoD'], y_future_2d_b)
                    loss = (loss_1d + loss_2d) / 2

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item()
                total_batches += 1
                global_step += 1
                wandb.log({'loss/train': loss.item(), 'coldstart/rollout_steps': rollout_steps}, step=global_step)

            if pass_idx % 50 == 0 or pass_idx == cold_start_passes:
                elapsed = time.time() - pass_start
                avg = total_loss / total_batches if total_batches > 0 else 0.0
                print(f"[INFO] Cold-start pass {pass_idx}/{cold_start_passes} | "
                      f"Avg loss: {avg:.6f} | {elapsed:.1f}s")

        # Save final checkpoint and mirror to latest
        save_checkpoint(model, start_epoch, total_loss / max(total_batches, 1),
                        run_dir, CONFIG, global_step=global_step, scheduler=scheduler, optimizer=optimizer)
        import shutil
        best_checkpoint = os.path.join(run_dir, f'{SELECTED_MODEL}_epoch_{start_epoch:03d}.pt')
        if os.path.exists(best_checkpoint) and mirror_latest:
            shutil.copy(best_checkpoint, os.path.join(latest_dir, f'{SELECTED_MODEL}_best.pt'))
            shutil.copy(best_checkpoint, os.path.join(latest_dir, f'{SELECTED_MODEL}_best_h64.pt'))
            for fname in [f'{SELECTED_MODEL}_normalizers.pkl', f'{SELECTED_MODEL}_normalization_stats.json']:
                src = os.path.join(run_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(latest_dir, fname))
            print(f"[INFO] Cold-start checkpoint mirrored to latest/")
        print(f"\n[INFO] Cold-start fine-tuning complete: {cold_start_passes} passes, "
              f"{total_batches} batches, avg loss={total_loss/max(total_batches,1):.6f}")
        return

    # Mid-epoch validation disabled — at h=64 even 3-batch rollouts are expensive.
    # Full validation runs at end of each epoch instead.
    # Set VAL_CHECK_INTERVAL to an integer (e.g. 50) to re-enable.
    VAL_CHECK_INTERVAL = None
    VAL_SUBSET_BATCHES = 3

    best_kaggle_at_max_h = float('inf')   # best (NRMSE_1D + NRMSE_2D)/2 seen at full horizon
    best_kaggle_epoch = None

    no_improve_count = 0  # Early stopping counter (only active at full horizon)
    best_train_loss_at_max_h = float('inf')  # fallback for early stopping when val disabled
    val_loss_norm = None  # Set in epoch loop; initialized here to avoid UnboundLocalError if loop is empty

    # -----------------------------------------------------------------------
    # Noise-injection warm-start curriculum state (Model_2 only)
    # -----------------------------------------------------------------------
    # noise_extra_steps (K): number of perturbed steps appended beyond the 10-step clean warm-start
    # Smart-detect advance: each epoch, scan per-step train NRMSE up to noise_max_extra_steps.
    # K advances to the largest contiguous prefix where nrmse[k] <= min(nrmse[:5]) * noise_smart_alpha.
    # noise_smart_init_epochs: pure h=1 warmup epochs before smart detection activates.
    _use_noise_curriculum = (SELECTED_MODEL == 'Model_2')
    _noise_extra_steps = 0          # K — starts at 0 (no extra steps); advances via smart detect
    _noise_max_extra = CONFIG['noise_max_extra_steps']
    _noise_smart_alpha = CONFIG['noise_smart_alpha']
    _noise_smart_init_epochs = CONFIG['noise_smart_init_epochs']
    _noise_smart_probe_batches = CONFIG['noise_smart_probe_batches']
    _noise_init_epochs_done = 0     # counts pure h=1 warmup epochs
    # noise_stats: None until first smart-detect advance
    _noise_mu_1d    = None   # [K_cur, N_1d]
    _noise_sigma_1d = None   # [K_cur, N_1d]
    _noise_mu_2d    = None   # [K_cur, N_2d]
    _noise_sigma_2d = None   # [K_cur, N_2d]
    if _use_noise_curriculum:
        print(f"[INFO] Noise-injection curriculum enabled (Model_2) — smart-detect mode")
        print(f"  alpha={_noise_smart_alpha}, init_epochs={_noise_smart_init_epochs}, "
              f"probe_batches={_noise_smart_probe_batches}, max_extra={_noise_max_extra}")
        print(f"  Init phase: {_noise_smart_init_epochs} epoch(s) of pure h=1 before smart detection")

    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        batch_start_time = time.time()

        # Curriculum schedule
        # Model_1: doubles every 2 epochs (power-of-2); epoch 1-2→1, 3-4→2, ..., 13+→64
        # Model_2: noise-injection warm-start curriculum — each stage k adds one more
        #   perturbed warm-start step (beyond the 10 clean steps) and predicts 1 step ahead.
        #   After noise_max_extra_steps stages, falls back to full h=1 horizon-expanding schedule.
        if _use_noise_curriculum and _noise_extra_steps <= _noise_max_extra:
            # Noise curriculum is active — fixed 1-step rollout target, expanding warm-start
            max_h = 1  # predict only the next step (step 11+K)
            noise_K = _noise_extra_steps
        elif SELECTED_MODEL == 'Model_2':
            _m2_boundaries = [6, 10, 14, 18, 22, 26, 30, 34, 38, 46]  # last epoch of each stage; 46 total
            _m2_horizons   = [1,  2,  4,  6,  8, 16, 24, 32, 48, 64]
            _stage_idx = next((i for i, b in enumerate(_m2_boundaries) if epoch <= b), len(_m2_boundaries) - 1)
            max_h = _m2_horizons[_stage_idx]
            noise_K = 0
        else:
            _stage_len = 2
            max_h = min(CONFIG['forecast_len'], 2 ** ((epoch - 1) // _stage_len))
            noise_K = 0

        # Epoch boundary marker — visible as a vertical annotation in wandb
        wandb.log({'epoch': epoch, 'curriculum/max_h': max_h,
                   'curriculum/noise_extra_steps': noise_K}, step=global_step)

        if _use_noise_curriculum and noise_K <= _noise_max_extra:
            _phase = "init" if _noise_init_epochs_done < _noise_smart_init_epochs else "smart-detect"
            print(f"[INFO] Noise curriculum [{_phase}]: epoch={epoch}/{CONFIG['epochs']}, "
                  f"noise_K={noise_K} (10 clean + {noise_K} perturbed → predict step {11+noise_K})")
        else:
            print(f"[INFO] Curriculum: epoch={epoch}/{CONFIG['epochs']}, max_h={max_h}")

        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:
                continue

            # Extract batch data
            static_graph = batch['static_graph'].to(device, non_blocking=True)
            y_hist_1d = batch['y_hist_1d'].to(device, non_blocking=True)        # [B, H, N_1d, 1]
            y_hist_2d = batch['y_hist_2d'].to(device, non_blocking=True)        # [B, H, N_2d, 1]
            rain_hist_2d = batch['rain_hist_2d'].to(device, non_blocking=True)  # [B, H, N_2d, R]
            y_future_1d = batch['y_future_1d'].to(device, non_blocking=True)    # [B, T, N_1d, 1]
            y_future_2d = batch['y_future_2d'].to(device, non_blocking=True)    # [B, T, N_2d, 1]
            rain_future_2d = batch['rain_future_2d'].to(device, non_blocking=True)  # [B, T, N_2d, R]

            avail = y_future_1d.shape[1]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                # Noise-injection curriculum: K extra perturbed warm-start steps, rollout=1
                if _use_noise_curriculum and noise_K > 0 and _noise_mu_1d is not None and avail > noise_K:
                    # Slice extra steps from y_future: steps 0..K-1 are the perturbed warm-start
                    # Target: step K (y_future[:, K:K+1])
                    y_extra_1d  = y_future_1d[:, :noise_K]        # [B, K, N_1d, 1]
                    y_extra_2d  = y_future_2d[:, :noise_K]
                    rain_extra  = rain_future_2d[:, :noise_K]
                    # Remaining future for autoregressive rollout (just 1 step target)
                    rain_fut    = rain_future_2d[:, noise_K:noise_K+1]
                    # noise stats: slice to current K lags (mu/sigma are [K_max, N])
                    nm1 = _noise_mu_1d[:noise_K].to(device)
                    ns1 = _noise_sigma_1d[:noise_K].to(device)
                    nm2 = _noise_mu_2d[:noise_K].to(device)
                    ns2 = _noise_sigma_2d[:noise_K].to(device)
                    rollout_steps = 1

                    predictions = model.forward_unroll_with_noise(
                        data=static_graph,
                        y_hist_1d=y_hist_1d, y_hist_2d=y_hist_2d,
                        rain_hist=rain_hist_2d,
                        y_extra_1d=y_extra_1d, y_extra_2d=y_extra_2d,
                        rain_extra=rain_extra,
                        noise_mu_1d=nm1, noise_sigma_1d=ns1,
                        noise_mu_2d=nm2, noise_sigma_2d=ns2,
                        rain_future=rain_fut,
                        make_x_dyn=lambda y, r, d: make_x_dyn(
                            y['oneD'], y['twoD'], r, d,
                            rain_1d_index=_rain_1d_index,
                        ),
                        rollout_steps=rollout_steps,
                        device=device,
                        batched_data=train_batched_graph,
                        use_grad_checkpoint=use_mixed_precision,
                    )
                    loss_1d = criterion(predictions['oneD'], y_future_1d[:, noise_K:noise_K+1])
                    loss_2d = criterion(predictions['twoD'], y_future_2d[:, noise_K:noise_K+1])
                else:
                    # Standard horizon-expanding curriculum (or noise K=0 / no stats yet)
                    rollout_steps = min(max_h, avail)
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
                    if _loss_mask_1d is not None:
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
                f'loss/train_h{rollout_steps}': loss.item(),
                'curriculum/rollout_steps': rollout_steps,
                'curriculum/max_h': max_h,
                'curriculum/noise_extra_steps': noise_K,
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
                # Full-event rollout val — mirrors inference exactly, much cheaper than
                # windowed val and a far better proxy for the actual Kaggle score.
                full_event_nrmse_combined = None
                if _val_event_file_list:
                    try:
                        print(f"[INFO] Running full-event rollout val ({len(_val_event_file_list)} events)...")
                        fe_combined, fe_1d, fe_2d, fe_per_event = evaluate_full_event_rollout(
                            model, _val_event_file_list, data, norm_stats,
                            _single_static_graph, device,
                            history_len=CONFIG['history_len'],
                            use_mixed_precision=use_mixed_precision,
                            rain_1d_index=_rain_1d_index,
                            n_rain_channels=_n_rain_channels,
                        )
                        full_event_nrmse_combined = fe_combined
                        fe_rmse_1d = fe_1d * kaggle_sigma_1d
                        fe_rmse_2d = fe_2d * kaggle_sigma_2d
                        print(f"  full_event  combined={fe_combined:.4f}  "
                              f"1d={fe_1d:.4f} (RMSE={fe_rmse_1d:.4f}m)  "
                              f"2d={fe_2d:.4f} (RMSE={fe_rmse_2d:.4f}m)")
                        fe_log = {
                            'full_event_val/combined_nrmse': fe_combined,
                            'full_event_val/1d_nrmse': fe_1d,
                            'full_event_val/2d_nrmse': fe_2d,
                            'full_event_val/1d_rmse_m': fe_rmse_1d,
                            'full_event_val/2d_rmse_m': fe_rmse_2d,
                        }
                        # Per-event breakdown table — sortable in wandb, reveals hard events
                        if fe_per_event:
                            fe_sorted = sorted(fe_per_event, key=lambda x: x['combined_nrmse'], reverse=True)
                            fe_table = wandb.Table(columns=[
                                'rank', 'event_name', 'combined_nrmse', 'nrmse_1d', 'nrmse_2d', 'T_steps',
                                't_worst', 't_worst_frac', 'rain_total', 'rain_peak',
                                't_rain_peak_frac', 'rain_at_worst',
                            ])
                            for rank, ev in enumerate(fe_sorted, 1):
                                fe_table.add_data(
                                    rank, ev['event_name'],
                                    round(ev['combined_nrmse'], 5), round(ev['nrmse_1d'], 5), round(ev['nrmse_2d'], 5),
                                    ev['T'],
                                    ev.get('t_worst', -1), ev.get('t_worst_frac', -1.0),
                                    ev.get('rain_total', 0.0), ev.get('rain_peak', 0.0),
                                    ev.get('t_rain_peak_frac', -1.0), ev.get('rain_at_worst', 0.0),
                                )
                            fe_log['full_event_val/per_event_table'] = fe_table
                            # Std across events — how uneven the difficulty is
                            _fe_nrmses = [ev['combined_nrmse'] for ev in fe_per_event]
                            fe_log['full_event_val/combined_nrmse_std'] = float(np.std(_fe_nrmses))
                            fe_log['full_event_val/combined_nrmse_max'] = float(max(_fe_nrmses))
                            # Top-5 worst events printed to console
                            print(f"  Top-5 hardest val events:")
                            for ev in fe_sorted[:5]:
                                print(f"    {ev['event_name']:40s}  combined={ev['combined_nrmse']:.4f}  "
                                      f"1d={ev['nrmse_1d']:.4f}  2d={ev['nrmse_2d']:.4f}  T={ev['T']}  "
                                      f"t_jump={ev.get('t_worst_frac',-1):.2f}  "
                                      f"rain_peak={ev.get('rain_peak',0):.3f}@{ev.get('t_rain_peak_frac',-1):.2f}  "
                                      f"rain@jump={ev.get('rain_at_worst',0):.3f}")
                        wandb.log(fe_log, step=global_step)
                    except Exception as e:
                        print(f"[WARNING] Full-event rollout val failed: {e}")

                if max_h == CONFIG['forecast_len']:
                    # Use full-event NRMSE for best-checkpoint tracking when available
                    _ckpt_metric = full_event_nrmse_combined if full_event_nrmse_combined is not None else val_nrmse_combined
                    if _ckpt_metric < best_kaggle_at_max_h:
                        best_kaggle_at_max_h = _ckpt_metric
                        best_kaggle_epoch = epoch
                        no_improve_count = 0
                        # Save best h=64 checkpoint — used by inference script
                        best_h64_src = os.path.join(run_dir, f'{SELECTED_MODEL}_epoch_{epoch:03d}.pt')
                        if os.path.exists(best_h64_src):
                            import shutil as _shutil
                            dst_list = [os.path.join(run_dir, f'{SELECTED_MODEL}_best_h64.pt')]
                            if mirror_latest:
                                dst_list.append(os.path.join(latest_dir, f'{SELECTED_MODEL}_best_h64.pt'))
                            for _dst in dst_list:
                                _shutil.copy(best_h64_src, _dst)
                            print(f"[INFO] New best h=64 checkpoint saved (metric={_ckpt_metric:.4f})")
                    else:
                        no_improve_count += 1
                        patience = CONFIG['early_stopping_patience']
                        print(f"[INFO] No improvement at h={max_h}: {no_improve_count}/{patience} epochs without improvement")
                        if patience is not None and no_improve_count >= patience:
                            print(f"[INFO] Early stopping triggered after {no_improve_count} epochs without improvement at h={max_h}")
                            break
            except Exception as e:
                print(f"[WARNING] Validation failed (epoch {epoch}): {e} — skipping val this epoch")

        # -----------------------------------------------------------------------
        # Noise curriculum smart-detect advance (Model_2 only)
        # Each epoch: scan per-step train NRMSE up to noise_max_extra_steps.
        # K advances to the largest contiguous prefix where nrmse[k] <= alpha * nrmse[0].
        # First noise_smart_init_epochs epochs are a pure h=1 warmup (no detection).
        # -----------------------------------------------------------------------
        if _use_noise_curriculum and noise_K <= _noise_max_extra:
            if _noise_init_epochs_done < _noise_smart_init_epochs:
                # Init phase: just count epochs, don't scan yet
                _noise_init_epochs_done += 1
                print(f"[INFO] Noise curriculum init phase: {_noise_init_epochs_done}/{_noise_smart_init_epochs} epochs done")
            else:
                # Smart-detect phase: measure per-step NRMSE on train, find new K
                try:
                    # Probe enough ahead to detect a jump, but don't scan all 54 steps early on.
                    # Headroom: 2× current K + 10, capped at noise_max_extra.
                    probe_steps = min(max(_noise_extra_steps * 2 + 10, 10), _noise_max_extra)
                    print(f"[INFO] Smart-detect: probing {probe_steps} steps (K={_noise_extra_steps}) on {_noise_smart_probe_batches} train batches...")
                    nrmse_steps = _measure_rollout_nrmse_per_step(
                        model, train_dataloader, max_steps=probe_steps, device=device,
                        batched_static_graph=train_batched_graph,
                        use_mixed_precision=use_mixed_precision,
                        rain_1d_index=_rain_1d_index,
                        max_batches=_noise_smart_probe_batches,
                    )
                    # Baseline = min over first 5 steps (true-fed dips early, use the best
                    # the model can do rather than the noisy step-0 warm-start artifact).
                    # Threshold relative to that minimum: stricter than step-0 baseline.
                    baseline = min(nrmse_steps[:min(5, len(nrmse_steps))])
                    threshold = baseline * _noise_smart_alpha
                    new_K = 0
                    for k, nrmse_k in enumerate(nrmse_steps):
                        if nrmse_k <= threshold:
                            new_K = k + 1
                        else:
                            break  # stop at first breach (contiguous prefix only)
                    new_K = min(new_K, _noise_max_extra)

                    # Log per-step NRMSE to wandb for visibility
                    wandb.log({
                        'noise_curriculum/smart_K': new_K,
                        'noise_curriculum/baseline_nrmse': baseline,
                        'noise_curriculum/threshold_nrmse': threshold,
                        **{f'noise_curriculum/nrmse_step_{k}': v for k, v in enumerate(nrmse_steps[:min(10, len(nrmse_steps))])},
                    }, step=global_step)
                    print(f"[INFO] Smart-detect: baseline={baseline:.5f}, threshold={threshold:.5f} (alpha={_noise_smart_alpha})")
                    print(f"  Per-step NRMSE: {[f'{v:.4f}' for v in nrmse_steps[:min(15, len(nrmse_steps))]]}")
                    print(f"  K: {_noise_extra_steps} → {new_K}")

                    if new_K > _noise_extra_steps:
                        # Collect full per-node noise stats up to new_K
                        try:
                            print(f"[INFO] Collecting per-lag noise stats for K={new_K} lags (from train set)...")
                            _nm1, _ns1, _nm2, _ns2 = collect_per_lag_noise_stats(
                                model, train_dataloader, max_lag=new_K, device=device,
                                batched_static_graph=train_batched_graph,
                                use_mixed_precision=use_mixed_precision,
                                rain_1d_index=_rain_1d_index,
                            )
                            _noise_mu_1d    = _nm1
                            _noise_sigma_1d = _ns1
                            _noise_mu_2d    = _nm2
                            _noise_sigma_2d = _ns2
                            wandb.log({
                                'noise_curriculum/K': new_K,
                                'noise_curriculum/mu_1d_mean': float(_nm1.abs().mean()),
                                'noise_curriculum/sigma_1d_mean': float(_ns1.mean()),
                                'noise_curriculum/mu_2d_mean': float(_nm2.abs().mean()),
                                'noise_curriculum/sigma_2d_mean': float(_ns2.mean()),
                            }, step=global_step)
                            print(f"[INFO] Noise stats collected: "
                                  f"|mu_1d|={float(_nm1.abs().mean()):.4f}  sig_1d={float(_ns1.mean()):.4f}  "
                                  f"|mu_2d|={float(_nm2.abs().mean()):.4f}  sig_2d={float(_ns2.mean()):.4f}")
                            _noise_extra_steps = new_K
                            print(f"[INFO] Noise curriculum: K advanced to {_noise_extra_steps}")
                        except Exception as e:
                            print(f"[WARNING] Noise stats collection failed: {e} — keeping K={_noise_extra_steps}")
                    else:
                        print(f"[INFO] Noise curriculum: K stays at {_noise_extra_steps} (model not ready to advance)")
                except Exception as e:
                    print(f"[WARNING] Smart-detect probe failed: {e} — K unchanged")

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

        # Step LR scheduler (log-linear decay, once per epoch)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({'train/lr': current_lr}, step=global_step)

        # Checkpoint
        if epoch % CONFIG['checkpoint_interval'] == 0 or epoch == CONFIG['epochs']:
            save_checkpoint(model, epoch, avg_epoch_loss, run_dir, CONFIG, global_step=global_step, scheduler=scheduler, optimizer=optimizer)

        # Best checkpoint tracking — always save the last epoch's checkpoint to latest/
        # (Early stopping is disabled: val loss is incomparable across epochs with different max_h)
        import shutil
        best_checkpoint = os.path.join(run_dir, f'{SELECTED_MODEL}_epoch_{epoch:03d}.pt')
        if os.path.exists(best_checkpoint):
            best_path = os.path.join(run_dir, f'{SELECTED_MODEL}_best.pt')
            shutil.copy(best_checkpoint, best_path)
            if mirror_latest:
                shutil.copy(best_checkpoint, os.path.join(latest_dir, f'{SELECTED_MODEL}_best.pt'))
                for fname in [f'{SELECTED_MODEL}_normalizers.pkl',
                               f'{SELECTED_MODEL}_normalization_stats.json']:
                    src = os.path.join(run_dir, fname)
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(latest_dir, fname))
                _val_str = f"{val_loss_norm:.6e}" if val_loss_norm is not None else "N/A"
                print(f"[INFO] Checkpoint mirrored to latest/ (h={max_h} val_loss={_val_str})")
            else:
                print(f"[INFO] Checkpoint saved to run_dir only (--no-mirror-latest active)")

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
    parser.add_argument('--no-mirror-latest', action='store_true',
                        help='Skip mirroring checkpoints to checkpoints/latest/. Use for probe/experimental runs to avoid overwriting the best saved model.')
    parser.add_argument('--cold-start-passes', type=int, default=None,
                        help='Fine-tune on cold-start windows only (first window per event). '
                             'Runs N shuffled passes over the pool instead of the standard epoch loop. '
                             'Intended for use with --resume and --max-h 64 at a low LR.')
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
        CONFIG['lr_final'] = args.learning_rate  # flat LR when explicitly overridden (e.g. finetune)
    if args.max_h is not None:
        CONFIG['forecast_len'] = args.max_h
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        print("[INFO] Mixed precision training enabled")
        torch.set_float32_matmul_precision('medium')  # Speeds up matmuls on L40S/A100
        # Actual fp16 autocast + GradScaler is applied inside train()
    
    train(resume_from=args.resume, use_mixed_precision=args.mixed_precision, skip_validation=args.no_val, pretrain_from=args.pretrain, train_split=args.train_split, extra_epochs=extra_epochs, mirror_latest=not args.no_mirror_latest, cold_start_passes=args.cold_start_passes)
