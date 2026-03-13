#!/usr/bin/env python
"""
Noise-injection training for FloodLM Model_2.

Strategy:
  1. Load a pretrained h=1 checkpoint (Model_2, fully converged).
  2. Train in a curriculum where K advances incrementally:
     - At K=1: predict step 1 from clean GT. After epoch, AR-rollout 1 step,
       collect error stats (mu[1], sigma[1]).
     - At K=2: feed step 1 as GT+noise(1), predict step 2. After epoch,
       AR-rollout 2 steps, collect stats at lags 1-2.
     - At K=n: feed steps 1..n-1 as GT+noise(1..n-1), predict step n.
       After epoch, AR-rollout n steps, update stats 1..n.
  3. Noise stats at each lag always reflect the model's actual AR error
     at that lag, collected after the most recent epoch at that K.
  4. K advances by k_per_advance every epochs_per_k epochs.

Cost per epoch == h=1 training cost. No BPTT through horizon.
"""

import os
import sys
import json
import time
import argparse
import pickle
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import get_model_config, make_x_dyn
from model import FloodAutoregressiveHeteroModel
from data_lazy import initialize_data
from data_config import SELECTED_MODEL
from autoregressive_inference import (
    load_event_data,
    build_static_graph_from_cache,
    prepare_event_tensors,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    'history_len':          10,
    'lr':                   1e-3,
    'lr_high_k':            1e-4,        # LR once K >= lr_drop_at_k
    'lr_drop_at_k':         20,          # K threshold at which LR drops to lr_high_k
    'batch_size':           32,          # events per batch
    'epochs':               500,
    'K_MAX':                450,         # final max lag
    'epochs_per_k':         1,           # int or 'auto'; 'auto' = advance when loss converges
    'k_per_advance':        1,           # how many K steps to advance each time
    'converge_patience':    5,           # (auto mode) advance after this many epochs w/o improvement
    'converge_threshold':   0.005,       # (auto mode) relative improvement threshold (0.5%)
    'max_epochs_per_k':     40,          # (auto mode) force advance after this many epochs at one K
    'save_dir':             'checkpoints',
    'checkpoint_interval':  10,
    'wandb_project':        'floodlm',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

KAGGLE_SIGMA = {
    (2, 1): 3.192,
    (2, 2): 2.727,
}

# ---------------------------------------------------------------------------
# Dataset: one sample per train event = first window tensors
# ---------------------------------------------------------------------------

class FirstWindowDataset(Dataset):
    """
    Returns pre-loaded first-window tensors for each train event.
    Each sample: dict with y_hist_1d, y_hist_2d, rain_hist_2d,
                             y_future_1d, y_future_2d, rain_future_2d,
                             event_path
    history_len warm-start steps + at least 1 future step required.
    """
    def __init__(self, event_paths, norm_stats, history_len, device='cpu'):
        self.samples = []
        H = history_len
        skipped = 0
        print(f"[INFO] Loading first-window tensors for {len(event_paths)} events...")
        for ep in event_paths:
            try:
                n1d, n2d = load_event_data(ep)
                y1, y2, rain2, _, _, _ = prepare_event_tensors(n1d, n2d, norm_stats, device='cpu')
                T = y1.shape[0]
                if T <= H:
                    skipped += 1
                    continue
                self.samples.append({
                    'y_hist_1d':      y1[:H],           # [H, N1, 1]
                    'y_hist_2d':      y2[:H],           # [H, N2, 1]
                    'rain_hist_2d':   rain2[:H],         # [H, N2, C]
                    'y_future_1d':    y1[H:],            # [T-H, N1, 1]
                    'y_future_2d':    y2[H:],            # [T-H, N2, 1]
                    'rain_future_2d': rain2[H:],          # [T-H, N2, C]
                    'event_path':     str(ep),
                })
            except Exception as e:
                print(f"[WARN] Skipping {ep}: {e}")
                skipped += 1
        print(f"[INFO] Loaded {len(self.samples)} events ({skipped} skipped).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# AR rollout noise collection — roll out exactly K steps, collect errors
# ---------------------------------------------------------------------------

def collect_ar_noise_stats(model, samples, static_graph, rain_1d_index,
                            K, device, use_mixed_precision=False):
    """
    AR-rollout the model for exactly K steps from each event's warm-start.
    Collect per-node per-lag signed errors: err[k] = pred[k] - true[k].

    Returns:
        mu_1d:    [K, N1]  mean signed error per node per lag
        sigma_1d: [K, N1]  std of signed error per node per lag
        mu_2d:    [K, N2]
        sigma_2d: [K, N2]
    """
    model.eval()
    H = samples[0]['y_hist_1d'].shape[0]
    N1 = samples[0]['y_hist_1d'].shape[1]
    N2 = samples[0]['y_hist_2d'].shape[1]

    err1_by_lag = [[] for _ in range(K)]
    err2_by_lag = [[] for _ in range(K)]

    with torch.no_grad():
        for s in samples:
            y1 = s['y_hist_1d'].to(device)
            y2 = s['y_hist_2d'].to(device)
            rain = s['rain_hist_2d'].to(device)
            y1_fut = s['y_future_1d'].to(device)
            y2_fut = s['y_future_2d'].to(device)
            rain_fut = s['rain_future_2d'].to(device)
            T_fut = y1_fut.shape[0]

            if T_fut < K:
                continue  # event too short for this K

            B = 1
            bsg = model._make_batched_graph(static_graph, B)
            _r1d = rain_1d_index

            def _make_x(y1k, y2k, rk):
                return make_x_dyn(y1k, y2k, rk, bsg, rain_1d_index=_r1d)

            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                # Warm-start with GT history
                h = model.init_hidden(static_graph, B, device=device)
                for t in range(H):
                    y1k = y1[t].reshape(N1, 1)
                    y2k = y2[t].reshape(N2, 1)
                    rk  = rain[t].reshape(N2, -1)
                    x = _make_x(y1k, y2k, rk)
                    for _ctx in ('global', 'ctx1d', 'ctx2d'):
                        if _ctx in model.node_types:
                            x[_ctx] = torch.zeros(B, 1, device=device, dtype=y1k.dtype)
                    h = model.cell(bsg, h, x)

                # AR rollout K steps — feed predictions back
                for k in range(K):
                    pred = model.predict_water_levels(h, B, {'oneD': N1, 'twoD': N2})
                    p1 = pred['oneD'].reshape(N1)
                    p2 = pred['twoD'].reshape(N2)
                    t1 = y1_fut[k].reshape(N1)
                    t2 = y2_fut[k].reshape(N2)
                    err1_by_lag[k].append((p1 - t1).cpu())
                    err2_by_lag[k].append((p2 - t2).cpu())

                    # Feed prediction back as input
                    y1k = p1.reshape(N1, 1).detach()
                    y2k = p2.reshape(N2, 1).detach()
                    rk  = rain_fut[k].reshape(N2, -1)
                    x = _make_x(y1k, y2k, rk)
                    for _ctx in ('global', 'ctx1d', 'ctx2d'):
                        if _ctx in model.node_types:
                            x[_ctx] = torch.zeros(B, 1, device=device, dtype=y1k.dtype)
                    h = model.cell(bsg, h, x)

    # Compute mu/sigma per lag
    mu_1d    = torch.zeros(K, N1)
    sigma_1d = torch.ones(K, N1)
    mu_2d    = torch.zeros(K, N2)
    sigma_2d = torch.ones(K, N2)

    for k in range(K):
        if len(err1_by_lag[k]) == 0:
            continue
        e1 = torch.stack(err1_by_lag[k], dim=0)
        e2 = torch.stack(err2_by_lag[k], dim=0)
        mu_1d[k]    = e1.mean(0)
        sigma_1d[k] = e1.std(0).clamp(min=1e-6)
        mu_2d[k]    = e2.mean(0)
        sigma_2d[k] = e2.std(0).clamp(min=1e-6)

    print(f"  [noise stats K={K}] lag 0: sig_1d={sigma_1d[0].mean():.4f} sig_2d={sigma_2d[0].mean():.4f}"
          f"  |  lag {K-1}: sig_1d={sigma_1d[K-1].mean():.4f} sig_2d={sigma_2d[K-1].mean():.4f}"
          f"  |  n_events={len(err1_by_lag[0])}")

    model.train()
    return mu_1d, sigma_1d, mu_2d, sigma_2d


def ar_eval(model, samples, static_graph, rain_1d_index,
            K, device, use_mixed_precision=False):
    """
    AR-rollout K steps on all events, return mean per-step weighted loss
    and the overall mean loss across all steps.
    """
    model.eval()
    H = samples[0]['y_hist_1d'].shape[0]
    N1 = samples[0]['y_hist_1d'].shape[1]
    N2 = samples[0]['y_hist_2d'].shape[1]

    # per-step accumulators
    step_loss_sum = [0.0] * K
    step_count = [0] * K

    with torch.no_grad():
        for s in samples:
            y1 = s['y_hist_1d'].to(device)
            y2 = s['y_hist_2d'].to(device)
            rain_h = s['rain_hist_2d'].to(device)
            y1_fut = s['y_future_1d'].to(device)
            y2_fut = s['y_future_2d'].to(device)
            rain_fut = s['rain_future_2d'].to(device)
            T_fut = y1_fut.shape[0]

            if T_fut < K:
                continue

            B = 1
            bsg = model._make_batched_graph(static_graph, B)
            _r1d = rain_1d_index

            def _make_x(y1k, y2k, rk):
                return make_x_dyn(y1k, y2k, rk, bsg, rain_1d_index=_r1d)

            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                h = model.init_hidden(static_graph, B, device=device)
                for t in range(H):
                    y1k = y1[t].reshape(N1, 1)
                    y2k = y2[t].reshape(N2, 1)
                    rk = rain_h[t].reshape(N2, -1)
                    x = _make_x(y1k, y2k, rk)
                    for _ctx in ('global', 'ctx1d', 'ctx2d'):
                        if _ctx in model.node_types:
                            x[_ctx] = torch.zeros(B, 1, device=device, dtype=y1k.dtype)
                    h = model.cell(bsg, h, x)

                # AR rollout
                for k in range(K):
                    pred = model.predict_water_levels(h, B, {'oneD': N1, 'twoD': N2})
                    p1 = pred['oneD'].reshape(N1, 1)
                    p2 = pred['twoD'].reshape(N2, 1)
                    t1 = y1_fut[k].reshape(N1, 1)
                    t2 = y2_fut[k].reshape(N2, 1)
                    loss_k = compute_weighted_loss(p1, p2, t1, t2)
                    step_loss_sum[k] += loss_k.item()
                    step_count[k] += 1

                    # Feed prediction back
                    rk = rain_fut[k].reshape(N2, -1)
                    x = _make_x(p1.detach(), p2.detach(), rk)
                    for _ctx in ('global', 'ctx1d', 'ctx2d'):
                        if _ctx in model.node_types:
                            x[_ctx] = torch.zeros(B, 1, device=device, dtype=p1.dtype)
                    h = model.cell(bsg, h, x)

    per_step = [step_loss_sum[k] / max(step_count[k], 1) for k in range(K)]
    mean_loss = sum(per_step) / K if K > 0 else 0.0
    model.train()
    return mean_loss, per_step


# ---------------------------------------------------------------------------
# Weighted loss (same as main training)
# ---------------------------------------------------------------------------

def compute_weighted_loss(pred_1d, pred_2d, true_1d, true_2d):
    """Sigma-weighted normalized convex combination MSE."""
    w1d = (1.0 / KAGGLE_SIGMA[(2, 1)]) ** 2
    w2d = (1.0 / KAGGLE_SIGMA[(2, 2)]) ** 2
    loss_1d = ((pred_1d - true_1d) ** 2).mean()
    loss_2d = ((pred_2d - true_2d) ** 2).mean()
    return (w1d * loss_1d + w2d * loss_2d) / (w1d + w2d)


# ---------------------------------------------------------------------------
# Training step: feed K-1 noisy GT steps, predict step K
# ---------------------------------------------------------------------------

def train_step_noise_injection(model, samples, static_graph, rain_1d_index,
                                mu_1d, sigma_1d, mu_2d, sigma_2d,
                                current_K, history_len, device,
                                use_mixed_precision=False):
    """
    For each sample in the batch:
      - Warm-start with clean history (history_len steps, ground truth)
      - Feed K-1 steps of GT + noise(lag) (no grad, builds corrupted h)
      - Feed step K as GT + noise(K) with grad, predict step K+1, loss vs GT
    Returns scalar loss.

    current_K: the target prediction step (1-indexed).
      K=1 means predict step 1 from clean warm-start (no noise).
      K=2 means feed step 1 with noise(lag=0), predict step 2.
    """
    H = history_len
    total_loss = 0.0
    n = 0

    for s in samples:
        y1 = s['y_hist_1d'].to(device)
        y2 = s['y_hist_2d'].to(device)
        rain_h = s['rain_hist_2d'].to(device)
        y1_fut = s['y_future_1d'].to(device)
        y2_fut = s['y_future_2d'].to(device)
        rain_fut = s['rain_future_2d'].to(device)
        T_fut = y1_fut.shape[0]

        # Need at least current_K future steps
        if T_fut < current_K:
            continue

        B = 1
        N1 = y1.shape[1]
        N2 = y2.shape[1]
        bsg = model._make_batched_graph(static_graph, B)
        _r1d = rain_1d_index

        def _make_x(y1k, y2k, rk):
            return make_x_dyn(y1k, y2k, rk, bsg, rain_1d_index=_r1d)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_mixed_precision):
            # Clean warm-start
            h = model.init_hidden(static_graph, B, device=device)
            for t in range(H):
                y1k = y1[t].reshape(N1, 1)
                y2k = y2[t].reshape(N2, 1)
                rk  = rain_h[t].reshape(N2, -1)
                x = _make_x(y1k, y2k, rk)
                for _ctx in ('global', 'ctx1d', 'ctx2d'):
                    if _ctx in model.node_types:
                        x[_ctx] = torch.zeros(B, 1, device=device, dtype=y1k.dtype)
                h = model.cell(bsg, h, x)

            # K-1 noisy steps (no grad): simulate corrupted hidden state
            # For K=1, this loop doesn't execute (predict from clean h)
            for step in range(current_K - 1):
                lag_idx = min(step, mu_1d.shape[0] - 1)
                noise1 = torch.randn(N1, 1, device=device) * sigma_1d[lag_idx].unsqueeze(1) \
                         + mu_1d[lag_idx].unsqueeze(1)
                noise2 = torch.randn(N2, 1, device=device) * sigma_2d[lag_idx].unsqueeze(1) \
                         + mu_2d[lag_idx].unsqueeze(1)
                y1_noisy = y1_fut[step].reshape(N1, 1) + noise1
                y2_noisy = y2_fut[step].reshape(N2, 1) + noise2
                rk = rain_fut[step].reshape(N2, -1)
                x = _make_x(y1_noisy, y2_noisy, rk)
                for _ctx in ('global', 'ctx1d', 'ctx2d'):
                    if _ctx in model.node_types:
                        x[_ctx] = torch.zeros(B, 1, device=device, dtype=y1_noisy.dtype)
                h = model.cell(bsg, h, x)

        # Final step WITH gradients: feed GT[K-1] + noise, predict, loss vs GT[K]
        # For K=1: feed GT[0] clean (no noise stats yet), predict step 1
        with torch.amp.autocast('cuda', enabled=use_mixed_precision):
            target_input_idx = current_K - 1  # 0-indexed future step to feed
            if current_K == 1:
                # K=1: feed clean GT[0], predict step 1
                y1_in = y1_fut[0].reshape(N1, 1)
                y2_in = y2_fut[0].reshape(N2, 1)
            else:
                lag_idx = min(current_K - 1, mu_1d.shape[0] - 1)
                noise1 = torch.randn(N1, 1, device=device) * sigma_1d[lag_idx].unsqueeze(1) \
                         + mu_1d[lag_idx].unsqueeze(1)
                noise2 = torch.randn(N2, 1, device=device) * sigma_2d[lag_idx].unsqueeze(1) \
                         + mu_2d[lag_idx].unsqueeze(1)
                y1_in = y1_fut[target_input_idx].reshape(N1, 1) + noise1
                y2_in = y2_fut[target_input_idx].reshape(N2, 1) + noise2

            rk = rain_fut[target_input_idx].reshape(N2, -1)
            x = _make_x(y1_in, y2_in, rk)
            for _ctx in ('global', 'ctx1d', 'ctx2d'):
                if _ctx in model.node_types:
                    x[_ctx] = torch.zeros(B, 1, device=device, dtype=rk.dtype)
            h_grad = model.cell(bsg, {ntype: v.detach() for ntype, v in h.items()}, x)
            pred = model.predict_water_levels(h_grad, B, {'oneD': N1, 'twoD': N2})

            # Target is the NEXT step after what we fed
            target_idx = current_K  # 0-indexed: GT[current_K]
            if target_idx >= T_fut:
                continue
            true_1d = y1_fut[target_idx].reshape(N1, 1)
            true_2d = y2_fut[target_idx].reshape(N2, 1)
            loss = compute_weighted_loss(pred['oneD'].reshape(N1, 1), pred['twoD'].reshape(N2, 1),
                                         true_1d, true_2d)
            total_loss = total_loss + loss
            n += 1

    if n == 0:
        return None
    return total_loss / n


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_pretrained(checkpoint_path, model, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state') or ckpt.get('model_state_dict') or ckpt
    model.load_state_dict(state, strict=True)
    print(f"[INFO] Loaded pretrained weights from {checkpoint_path}")
    return ckpt


def save_checkpoint(model, optimizer, epoch, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to pretrained h=1 Model_2 checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to noise-injection checkpoint to resume from')
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--no-mirror-latest', action='store_true')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override CONFIG epochs')
    parser.add_argument('--epochs-per-k', type=str, default=None,
                        help='Override epochs_per_k: integer or "auto" for convergence-based')
    parser.add_argument('--k-per-advance', type=int, default=None,
                        help='Override k_per_advance (how many K steps per advance)')
    parser.add_argument('--wandb-run-id', type=str, default=None)
    args = parser.parse_args()

    device = CONFIG['device']
    use_amp = args.mixed_precision
    H = CONFIG['history_len']
    if args.epochs is not None:
        CONFIG['epochs'] = args.epochs
    if args.epochs_per_k is not None:
        if args.epochs_per_k == 'auto':
            CONFIG['epochs_per_k'] = 'auto'
        else:
            CONFIG['epochs_per_k'] = int(args.epochs_per_k)
    if args.k_per_advance is not None:
        CONFIG['k_per_advance'] = args.k_per_advance

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("[INFO] Initializing data...")
    data = initialize_data()
    norm_stats = data['norm_stats']

    train_event_paths = [ep for _, ep, _ in data['train_event_file_list']]
    val_event_paths   = [ep for _, ep, _ in data['val_event_file_list']]

    print(f"[INFO] Train events: {len(train_event_paths)}, Val events: {len(val_event_paths)}")

    # Build static graph
    static_graph = build_static_graph_from_cache(data).to(device)

    # -----------------------------------------------------------------------
    # Model — build from checkpoint arch config
    # -----------------------------------------------------------------------
    ckpt_peek = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    model_cfg = get_model_config()
    arch_config = ckpt_peek.get('model_arch_config', {
        'h_dim': 96, 'msg_dim': 64,
        'hidden_dim': {'oneDedge': 64, 'oneDedgeRev': 64, 'twoDedge': 128,
                       'twoDedgeRev': 128, 'twoDoneD': 64, 'oneDtwoD': 64},
    })
    if 'node_dyn_input_dims' in arch_config:
        model_cfg['node_dyn_input_dims'] = arch_config['node_dyn_input_dims']
    else:
        model_cfg['node_dyn_input_dims'] = {'oneD': 3, 'twoD': 2}
        if 'global' in model_cfg.get('node_types', []):
            model_cfg['node_dyn_input_dims']['global'] = 1
    model_cfg.update({k: v for k, v in arch_config.items() if k != 'node_dyn_input_dims'})
    model = FloodAutoregressiveHeteroModel(**model_cfg).to(device)
    load_pretrained(args.pretrained, model, device)

    rain_1d_index = getattr(static_graph, 'rain_1d_index', None)

    # -----------------------------------------------------------------------
    # First-window dataset
    # -----------------------------------------------------------------------
    dataset = FirstWindowDataset(train_event_paths, norm_stats, H, device='cpu')
    samples_all = dataset.samples
    val_dataset = FirstWindowDataset(val_event_paths, norm_stats, H, device='cpu')
    val_samples = val_dataset.samples

    # -----------------------------------------------------------------------
    # Optimizer + scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    total_epochs = CONFIG['epochs']

    start_epoch = 0
    start_K = 1
    run_id = args.wandb_run_id

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        start_K = ckpt.get('current_K', 1)
        run_id = ckpt.get('wandb_run_id', run_id)
        print(f"[INFO] Resumed from epoch {start_epoch}, K={start_K}")

    # -----------------------------------------------------------------------
    # wandb
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name  = f"Model_2_NI_{timestamp}"
    save_dir  = Path(CONFIG['save_dir']) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb_init_kwargs = dict(
        project=CONFIG['wandb_project'],
        name=run_name,
        config=CONFIG,
    )
    if run_id:
        wandb_init_kwargs.update(id=run_id, resume='must')
    run = wandb.init(**wandb_init_kwargs)
    actual_run_id = run.id

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    epochs_per_k     = CONFIG['epochs_per_k']
    auto_advance     = (epochs_per_k == 'auto')
    fixed_epochs_per_k = 1 if auto_advance else epochs_per_k
    k_per_advance    = CONFIG['k_per_advance']
    converge_patience  = CONFIG['converge_patience']
    converge_threshold = CONFIG['converge_threshold']
    max_epochs_per_k   = CONFIG['max_epochs_per_k']
    K_MAX            = CONFIG['K_MAX']
    batch_size       = CONFIG['batch_size']
    global_step      = 0
    current_K        = start_K

    # Noise stats — start empty, will be collected before first epoch that needs them
    mu_1d = torch.zeros(0)
    sigma_1d = torch.zeros(0)
    mu_2d = torch.zeros(0)
    sigma_2d = torch.zeros(0)

    epochs_at_this_k = 0
    best_loss_at_k = float('inf')
    epochs_since_improve = 0
    lr_drop_at_k = CONFIG['lr_drop_at_k']
    lr_high_k    = CONFIG['lr_high_k']
    _lr_dropped  = False  # track so we only log the drop once

    for epoch in range(start_epoch, total_epochs):
        if current_K > K_MAX:
            print(f"[INFO] Reached K_MAX={K_MAX}, stopping.")
            break

        # --- Collect noise stats when K advances ---
        # At K=1 we don't need noise (clean prediction), but we collect after
        # training at K=1 so we have stats for K=2.
        # At K>1 we need stats [0..K-2] for the noisy warm-up steps.
        if current_K > 1 and (mu_1d.shape[0] < current_K - 1 or epochs_at_this_k == 0):
            # Need to collect/re-collect stats up to current_K - 1
            # We AR-rollout current_K-1 steps to get error stats at each lag
            print(f"[INFO] Collecting noise stats for K={current_K} "
                  f"(AR rollout {current_K - 1} steps)...")
            mu_1d, sigma_1d, mu_2d, sigma_2d = collect_ar_noise_stats(
                model, samples_all, static_graph, rain_1d_index,
                K=current_K - 1, device=device, use_mixed_precision=use_amp,
            )
            mu_1d = mu_1d.to(device)
            sigma_1d = sigma_1d.to(device)
            mu_2d = mu_2d.to(device)
            sigma_2d = sigma_2d.to(device)

        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        indices = torch.randperm(len(samples_all)).tolist()

        for batch_start in range(0, len(samples_all), batch_size):
            batch_idx  = indices[batch_start:batch_start + batch_size]
            batch_samp = [samples_all[i] for i in batch_idx]

            optimizer.zero_grad()
            loss = train_step_noise_injection(
                model, batch_samp, static_graph, rain_1d_index,
                mu_1d, sigma_1d, mu_2d, sigma_2d,
                current_K=current_K,
                history_len=H,
                device=device,
                use_mixed_precision=use_amp,
            )
            if loss is None:
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches  += 1
            global_step += 1

            if n_batches % 10 == 0:
                print(f"Epoch {epoch+1}/{total_epochs} | Batch {n_batches} | "
                      f"Loss: {loss.item():.6f} | K={current_K} | "
                      f"{time.time()-t0:.1f}s")

        epochs_at_this_k += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now   = optimizer.param_groups[0]['lr']
        elapsed  = time.time() - t0

        # AR eval on val events — rollout current_K steps feeding predictions back (no noise)
        ar_mean, ar_per_step = ar_eval(
            model, val_samples, static_graph, rain_1d_index,
            K=current_K, device=device, use_mixed_precision=use_amp,
        )

        sig_1d_mean = sigma_1d.mean().item() if sigma_1d.numel() > 0 else 0.0
        sig_2d_mean = sigma_2d.mean().item() if sigma_2d.numel() > 0 else 0.0

        # Convergence signal: train loss (smoother than val_ar; val_ar used for monitoring only)
        if auto_advance:
            if avg_loss < best_loss_at_k * (1 - converge_threshold):
                best_loss_at_k = avg_loss
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        advance_info = (f"patience={epochs_since_improve}/{converge_patience}"
                        if auto_advance else f"epochs_at_K={epochs_at_this_k}/{fixed_epochs_per_k}")
        print(f"[INFO] Epoch {epoch+1}/{total_epochs}: train={avg_loss:.6f}  val_ar={ar_mean:.6f}  "
              f"K={current_K}  {advance_info}  LR={lr_now:.2e}  Time={elapsed:.1f}s")

        log_dict = {
            'train/loss':          avg_loss,
            'val/ar_eval':         ar_mean,
            'train/K':             current_K,
            'train/lr':            lr_now,
            'train/sigma_1d_mean': sig_1d_mean,
            'train/sigma_2d_mean': sig_2d_mean,
            'epoch':               epoch + 1,
        }
        for _k, _v in enumerate(ar_per_step):
            log_dict[f'ar_step/step_{_k+1:03d}'] = _v
        wandb.log(log_dict, step=global_step)

        # Checkpoint
        if (epoch + 1) % CONFIG['checkpoint_interval'] == 0 or epoch == total_epochs - 1:
            ckpt_path = save_dir / f"Model_2_NI_epoch_{epoch+1:04d}.pt"
            save_checkpoint(model, optimizer, epoch, str(ckpt_path), extra={
                'wandb_run_id': actual_run_id,
                'global_step':  global_step,
                'current_K':    current_K,
            })
            print(f"[INFO] Saved {ckpt_path}")

            if not args.no_mirror_latest:
                latest_dir = Path(CONFIG['save_dir']) / 'latest'
                latest_dir.mkdir(parents=True, exist_ok=True)
                torch.save(torch.load(ckpt_path, weights_only=False),
                           latest_dir / 'Model_2_NI_latest.pt')

        # Advance K?
        should_advance = False
        if auto_advance:
            if epochs_since_improve >= converge_patience:
                should_advance = True
                print(f"[INFO] Loss converged at K={current_K} "
                      f"(no {converge_threshold*100:.1f}% improvement for {converge_patience} epochs)")
            elif epochs_at_this_k >= max_epochs_per_k:
                should_advance = True
                print(f"[INFO] Hit max_epochs_per_k={max_epochs_per_k} at K={current_K}")
        else:
            if epochs_at_this_k >= fixed_epochs_per_k:
                should_advance = True

        if should_advance:
            current_K = min(current_K + k_per_advance, K_MAX)
            epochs_at_this_k = 0
            best_loss_at_k = float('inf')
            epochs_since_improve = 0
            print(f"[INFO] Advanced to K={current_K}")
            # Drop LR once K crosses threshold
            if not _lr_dropped and current_K >= lr_drop_at_k:
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_high_k
                _lr_dropped = True
                print(f"[INFO] LR dropped to {lr_high_k:.1e} (K={current_K} >= lr_drop_at_k={lr_drop_at_k})")

    wandb.finish()
    print("[INFO] Training complete.")


if __name__ == '__main__':
    main()
