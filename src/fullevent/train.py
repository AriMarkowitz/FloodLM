#!/usr/bin/env python
"""
Full-event autoregressive training for FloodLM.

Each training iteration:
  1. Load one entire event (variable T timesteps)
  2. Warm-start hidden state with H=10 ground-truth timesteps
  3. Autoregressively roll out for n = T-H timesteps using predicted water levels
  4. Compute per-node RMSE over the n rollout timesteps, averaged across nodes & groups

Loss = Kaggle-aligned per-node RMSE hierarchy:
  - Per-node RMSE over rollout timesteps (already in normalized space → NRMSE)
  - Mean over 1D nodes → group score 1D
  - Mean over 2D nodes → group score 2D
  - (group_1d + group_2d) / 2 → event loss
  This naturally accounts for variable n: RMSE divides by n, so longer/shorter
  events contribute equally to the gradient signal.

Usage:
    python -m fullevent.train
    python -m fullevent.train --resume checkpoints/fullevent/FullEvent_Model_1_20260308_120000
    python -m fullevent.train --resume checkpoints/fullevent/FullEvent_Model_1_best.pt
    python -m fullevent.train --pretrain-from checkpoints/latest/Model_1_best.pt
    SELECTED_MODEL=Model_2 python -m fullevent.train
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
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import wandb

# Allow running from src/ or project root
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
ROOT_DIR = SRC_DIR.parent
for _p in (str(SRC_DIR), str(ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fullevent.config import CONFIG, KAGGLE_SIGMA, SAVE_DIR, SELECTED_MODEL
from fullevent.data import get_full_event_dataloader, get_full_event_dataset
from data_lazy import initialize_data
from data import get_model_config, make_x_dyn, create_static_hetero_graph
from model import FloodAutoregressiveHeteroModel
import math

# ----------  GPU performance knobs  ----------
torch.backends.cudnn.benchmark = True          # auto-tune conv kernels for fixed graph sizes
torch.backends.cuda.matmul.allow_tf32 = True   # TF32 for matmuls on Ampere+
torch.backends.cudnn.allow_tf32 = True          # TF32 for cuDNN on Ampere+


# ============================================================
# Sinusoidal relative time embedding
# ============================================================

def sinusoidal_time_embedding(t: int, dim: int, max_period: float = 512.0,
                              device: torch.device = None) -> torch.Tensor:
    """
    Compute a sinusoidal positional embedding for timestep t.

    Returns a [1, dim] tensor of (sin, cos) pairs at geometrically-spaced
    frequencies, identical to the standard transformer PE but scaled for
    flood-event timescales (~50-500 timesteps).

    The embedding tells the model "how far into the event" the current
    timestep is, without requiring knowledge of total event length.

    Args:
        t: Integer timestep index (0-based from event start)
        dim: Embedding dimension (must be even)
        max_period: Controls the longest wavelength
        device: Target device

    Returns:
        [1, dim] tensor — broadcast-ready for [N, dim] concatenation
    """
    assert dim % 2 == 0, f"time_embed_dim must be even, got {dim}"
    half = dim // 2
    # Geometric frequency series: freq_k = 1 / max_period^(2k/dim)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=device) / half
    )  # [half]
    angles = t * freqs  # [half]
    return torch.cat([angles.sin(), angles.cos()], dim=-1).unsqueeze(0)  # [1, dim]


def precompute_time_embeddings(T_total: int, dim: int, max_period: float = 512.0,
                               device: torch.device = None) -> torch.Tensor:
    """
    Batch-compute sinusoidal time embeddings for all timesteps at once.

    Much faster than calling sinusoidal_time_embedding() in a loop because
    we issue a single fused GPU kernel instead of T separate ones.

    Args:
        T_total: Total number of timesteps (H + T_future)
        dim: Embedding dimension (must be even)
        max_period: Controls the longest wavelength
        device: Target device

    Returns:
        [T_total, dim] tensor — index with te_all[t] to get [dim] for timestep t
    """
    assert dim % 2 == 0, f"time_embed_dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=device) / half
    )  # [half]
    ts = torch.arange(T_total, dtype=torch.float32, device=device)  # [T_total]
    angles = ts.unsqueeze(1) * freqs.unsqueeze(0)  # [T_total, half]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [T_total, dim]


# ============================================================
# Horizon curriculum loss weighting (expanding-horizon exponential decay)
# ============================================================

def horizon_curriculum_weights(T_future: int, T_max_global: int,
                               epoch: int, total_epochs: int,
                               L_start: float = 2.0,
                               power: float = 1.0,
                               device: torch.device = None) -> torch.Tensor:
    """
    Exponential-decay weights with a geometrically expanding effective horizon.

    At each epoch, an effective horizon L determines how far into the future
    the model "cares about".  Weights decay as exp(-t/L), so timesteps
    much beyond L contribute near-zero loss.

    L grows geometrically over training:
        L(epoch) = L_start * (T_max_global / L_start) ^ progress^power
        progress = (epoch - 1) / (total_epochs - 1)       (0 at start, 1 at end)

    The `power` parameter controls how quickly the horizon expands:
        power=1: linear progress (default, original behavior)
        power=3: 75% of training stays on horizons ≤90 steps (T_max=435)

    Example (T_max=435, L_start=2, 100 epochs, power=3):
        Epoch  1: L≈  2 → only first ~3 steps get meaningful weight
        Epoch 25: L≈  2 → first ~11 steps
        Epoch 50: L≈  4 → first ~18 steps
        Epoch 75: L≈ 18 → first ~90 steps
        Epoch 90: L≈ 80 → first ~400 steps
        Epoch100: L=435 → nearly uniform

    Within each epoch's "visible" window, later timesteps still get
    exponentially less weight — matching the user's intent of
    "care about them decreasingly".

    Normalized so mean(w) = 1 to keep loss magnitude stable.

    Args:
        T_future:     Number of rollout timesteps for this event
        T_max_global: Maximum T_future across all events (absolute scale)
        epoch:        Current epoch (1-based)
        total_epochs: Total training epochs
        L_start:      Initial effective horizon (in timesteps). 0 disables.
        power:        Power applied to progress (>1 = slower expansion).
        device:       Target device

    Returns:
        [T_future] tensor of per-timestep weights (mean ≈ 1)
    """
    if L_start <= 0 or total_epochs <= 1:
        # Disabled — uniform weights
        return torch.ones(T_future, dtype=torch.float32, device=device)

    progress = (epoch - 1) / (total_epochs - 1)  # 0 at epoch 1, 1 at last epoch
    progress = progress ** power  # Slow down early expansion
    L = L_start * (T_max_global / max(L_start, 1e-6)) ** progress

    t = torch.arange(T_future, dtype=torch.float32, device=device)
    w = torch.exp(-t / L)  # [T_future]
    w = w / w.mean()  # Normalize so mean = 1
    return w


def effective_rollout_length(L: float, T_future: int, n_lifetimes: float = 5.0,
                             min_steps: int = 3) -> int:
    """
    Compute the number of rollout steps needed given curriculum horizon L.

    We roll out to n_lifetimes * L, which captures
    1 - exp(-n_lifetimes) of the total weight mass:
        n_lifetimes=5  →  99.3%
        n_lifetimes=4  →  98.2%
        n_lifetimes=6  →  99.8%

    Timesteps beyond this contribute < 0.7% of loss — not worth computing.
    Savings are dramatic early in training (L=2 → 10 steps instead of 435).

    Returns:
        int in [min_steps, T_future]
    """
    T_eff = int(math.ceil(n_lifetimes * L))
    return max(min_steps, min(T_eff, T_future))


# ============================================================
# Loss function: Kaggle-aligned per-node RMSE
# ============================================================

def kaggle_nrmse_loss(pred_1d, pred_2d, true_1d, true_2d, time_weights=None):
    """
    Kaggle-aligned per-node RMSE loss that naturally accounts for variable n,
    with optional horizon curriculum weighting.

    All inputs are in normalized space (water_level / kaggle_sigma), so
    RMSE in normalized space == NRMSE == RMSE_raw / sigma.

    Hierarchy (mirrors Kaggle scoring):
      1. Per-node weighted RMSE over T rollout timesteps:
         sqrt( weighted_mean_t (pred - true)^2 )
         When time_weights is None (eval), this is standard RMSE (uniform).
         During training, early timesteps can be emphasised via curriculum.
      2. Mean over 1D nodes → group NRMSE for 1D
      3. Mean over 2D nodes → group NRMSE for 2D
      4. (group_1d + group_2d) / 2 → event score

    Args:
        pred_1d: [T, N_1d, 1] predicted water levels (normalized)
        pred_2d: [T, N_2d, 1]
        true_1d: [T, N_1d, 1] ground truth (normalized)
        true_2d: [T, N_2d, 1]
        time_weights: [T] per-timestep weights (mean≈1), or None for uniform

    Returns:
        loss: scalar — differentiable event-level NRMSE
        nrmse_1d: float — 1D group NRMSE
        nrmse_2d: float — 2D group NRMSE
    """
    # [T, N, 1] → [N, T]
    p1 = pred_1d.squeeze(-1).T  # [N_1d, T]
    t1 = true_1d.squeeze(-1).T
    p2 = pred_2d.squeeze(-1).T  # [N_2d, T]
    t2 = true_2d.squeeze(-1).T

    eps = 1e-8
    sq_err_1d = (p1 - t1) ** 2  # [N_1d, T]
    sq_err_2d = (p2 - t2) ** 2  # [N_2d, T]

    if time_weights is not None:
        # time_weights: [T] → broadcast over nodes
        # Weighted mean: sum(w * x) / sum(w), but since mean(w)≈1, sum(w)≈T
        w = time_weights.unsqueeze(0)  # [1, T]
        wmse_1d = (sq_err_1d * w).sum(dim=1) / w.sum()  # [N_1d]
        wmse_2d = (sq_err_2d * w).sum(dim=1) / w.sum()  # [N_2d]
    else:
        wmse_1d = sq_err_1d.mean(dim=1)  # [N_1d]
        wmse_2d = sq_err_2d.mean(dim=1)  # [N_2d]

    rmse_per_node_1d = wmse_1d.clamp(min=eps).sqrt()  # [N_1d]
    rmse_per_node_2d = wmse_2d.clamp(min=eps).sqrt()  # [N_2d]

    # Group means
    nrmse_1d = rmse_per_node_1d.mean()
    nrmse_2d = rmse_per_node_2d.mean()

    # Combined score
    loss = (nrmse_1d + nrmse_2d) / 2

    return loss, nrmse_1d.item(), nrmse_2d.item()


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(model, epoch, loss, save_dir, optimizer=None, scheduler=None,
                    global_step=None, wandb_run_id=None, best=False, config=None):
    os.makedirs(save_dir, exist_ok=True)
    tag = 'best' if best else f'epoch_{epoch:03d}'
    model_id = f"FullEvent_{SELECTED_MODEL}"
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'model_arch_config': {
            'h_dim': model.h_dim,
            'msg_dim': model.cell.msg_dim,
            'hidden_dim': model.cell._hidden_dim,
        },
        'config': config or CONFIG,
        'loss': loss,
        'model_id': model_id,
        'selected_model': SELECTED_MODEL,
        'global_step': global_step,
        'wandb_run_id': wandb_run_id,
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
    }
    path = os.path.join(save_dir, f'{model_id}_{tag}.pt')
    torch.save(checkpoint, path)
    return path


def save_normalization_stats(norm_stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_id = f"FullEvent_{SELECTED_MODEL}"
    stats = {
        'static_1d_params':  norm_stats.get('static_1d_params', {}),
        'static_2d_params':  norm_stats.get('static_2d_params', {}),
        'dynamic_1d_params': norm_stats.get('dynamic_1d_params', {}),
        'dynamic_2d_params': norm_stats.get('dynamic_2d_params', {}),
        'node1d_cols': norm_stats.get('node1d_cols', []),
        'node2d_cols': norm_stats.get('node2d_cols', []),
    }
    json_path = os.path.join(save_dir, f'{model_id}_normalization_stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    pkl_path = os.path.join(save_dir, f'{model_id}_normalizers.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'normalizer_1d': norm_stats.get('normalizer_1d'),
            'normalizer_2d': norm_stats.get('normalizer_2d'),
        }, f)
    print(f"[INFO] Saved norm stats: {json_path}")


# ============================================================
# Full-event autoregressive forward pass (B≥1)
# ============================================================

def full_event_forward(
    model,
    static_graph_cpu,
    y_hist_1d,       # [B, H, N_1d, 1]
    y_hist_2d,       # [B, H, N_2d, 1]
    rain_hist_2d,    # [B, H, N_2d, 1]
    y_future_1d,     # [B, T_future, N_1d, 1]  (ground truth for loss)
    y_future_2d,     # [B, T_future, N_2d, 1]
    rain_future_2d,  # [B, T_future, N_2d, 1]
    rain_1d_index,
    device,
    graph_cache=None,
    time_embed_dim: int = 0,
    time_embed_max_period: float = 512.0,
):
    """
    Run full autoregressive rollout over B events in parallel.

    All B events MUST share the same T_future (grouped batching — no padding).
    When B=1, behaviour is identical to the old single-event forward.

    Uses PyG Batch to replicate the static graph B times, giving
    B*N_1d / B*N_2d total nodes. The GRU cell processes all B events
    simultaneously, yielding B× more work per kernel launch → higher
    GPU utilization.

    graph_cache: optional dict mapping B → pre-built batched graph on device.
                 Avoids rebuilding the PyG Batch every call.

    Returns:
        preds_1d: [B, T_future, N_1d, 1]
        preds_2d: [B, T_future, N_2d, 1]
    """
    B = y_hist_1d.size(0)
    H = y_hist_1d.size(1)
    N_1d = y_hist_1d.size(2)
    N_2d = y_hist_2d.size(2)
    T_future = y_future_1d.size(1)
    T_total = H + T_future

    # Build or fetch cached batched graph
    if graph_cache is not None and B in graph_cache:
        batched_graph = graph_cache[B]
    else:
        batched_graph = model._make_batched_graph(static_graph_cpu, B).to(device)
        if graph_cache is not None:
            graph_cache[B] = batched_graph

    h = model.init_hidden(static_graph_cpu, B, device=device)

    _has_global = 'global' in model.cell.node_types
    _use_time_embed = time_embed_dim > 0

    # Pre-compute all time embeddings once
    te_all = None
    if _use_time_embed:
        te_all = precompute_time_embeddings(
            T_total, time_embed_dim, time_embed_max_period, device=device
        )  # [T_total, dim]

    # Pre-allocate dummy global input (reused every step)
    _global_zero = None
    if _has_global:
        _global_zero = torch.zeros(B, 1, device=device, dtype=y_hist_1d.dtype)

    def _build_x_dyn(y1d_flat, y2d_flat, rain2d_flat, abs_t):
        """
        Build x_dyn dict for the cell.  All flat tensors are [B*N, feat].
        y1d_flat: [B*N_1d, 1], y2d_flat: [B*N_2d, 1],
        rain2d_flat: [B*N_2d, 1], abs_t: int
        """
        if rain_1d_index is not None:
            # rain_1d_index maps 1D node → nearest 2D node (per-graph).
            # For B>1 we need B copies of the index, offset by per-copy N_2d.
            # Build once on first call and cache.
            rain_1d_flat = rain2d_flat[_rain_idx_batched]    # [B*N_1d, 1]
            wl_2d_for_1d = y2d_flat[_rain_idx_batched]       # [B*N_1d, 1]
            dyn_1d = torch.cat([y1d_flat, rain_1d_flat, wl_2d_for_1d], dim=-1)
        else:
            dyn_1d = y1d_flat
        dyn_2d = torch.cat([y2d_flat, rain2d_flat], dim=-1)

        if _use_time_embed:
            te = te_all[abs_t]  # [dim]
            # Broadcast to [B*N, dim] — single expand, no copy
            dyn_1d = torch.cat([dyn_1d, te.unsqueeze(0).expand(B * N_1d, -1)], dim=-1)
            dyn_2d = torch.cat([dyn_2d, te.unsqueeze(0).expand(B * N_2d, -1)], dim=-1)

        x_dyn = {'oneD': dyn_1d, 'twoD': dyn_2d}
        if _has_global:
            x_dyn['global'] = _global_zero
        return x_dyn

    # Build batched rain_1d_index: offset each copy's indices by b*N_2d
    _rain_idx_batched = None
    if rain_1d_index is not None:
        if B == 1:
            _rain_idx_batched = rain_1d_index
        else:
            offsets = torch.arange(B, device=device) * N_2d  # [B]
            _rain_idx_batched = (
                rain_1d_index.unsqueeze(0) + offsets.unsqueeze(1)
            ).reshape(-1)  # [B*N_1d]

    # ---- Warm-start with ground truth ----
    for t in range(H):
        y1d_t = y_hist_1d[:, t, :, :].reshape(B * N_1d, -1)  # [B*N_1d, 1]
        y2d_t = y_hist_2d[:, t, :, :].reshape(B * N_2d, -1)
        r2d_t = rain_hist_2d[:, t, :, :].reshape(B * N_2d, -1)
        x_dyn_t = _build_x_dyn(y1d_t, y2d_t, r2d_t, abs_t=t)
        h = model.cell(batched_graph, h, x_dyn_t)

    # ---- Autoregressive rollout ----
    node_counts = {'oneD': N_1d, 'twoD': N_2d}

    # Pre-allocate output: [B, T_future, N, 1]
    preds_1d = torch.empty(B, T_future, N_1d, 1, device=device, dtype=y_hist_1d.dtype)
    preds_2d = torch.empty(B, T_future, N_2d, 1, device=device, dtype=y_hist_2d.dtype)

    for t in range(T_future):
        y_next = model.predict_water_levels(h, B, node_counts)
        # y_next['oneD']: [B, N_1d, 1]
        preds_1d[:, t] = y_next['oneD']
        preds_2d[:, t] = y_next['twoD']

        # Flatten predictions for next step input: [B, N, 1] → [B*N, 1]
        y1_flat = y_next['oneD'].reshape(B * N_1d, -1)
        y2_flat = y_next['twoD'].reshape(B * N_2d, -1)
        r2_flat = rain_future_2d[:, t, :, :].reshape(B * N_2d, -1)
        x_dyn_next = _build_x_dyn(y1_flat, y2_flat, r2_flat, abs_t=H + t)
        h = model.cell(batched_graph, h, x_dyn_next)

    return preds_1d, preds_2d


# ============================================================
# Validation
# ============================================================

def evaluate(model, val_dataset, static_graph_cpu, rain_1d_index, device,
             use_mixed_precision, graph_cache=None,
             time_embed_dim=0, time_embed_max_period=512.0,
             max_batch_size=4, max_rollout=None):
    """
    Evaluate on all val events using full autoregressive rollout.
    Uses grouped batching for throughput.

    If max_rollout is set, truncate each event's rollout to that many steps
    (used for curriculum-aligned validation).

    Returns (combined_nrmse, nrmse_1d, nrmse_2d).
    """
    model.eval()
    scores, scores_1d, scores_2d = [], [], []

    with torch.inference_mode():
        for batch in val_dataset.iter_grouped(max_batch_size):
            B = batch['y_hist_1d'].size(0)

            # Truncate BEFORE moving to GPU to save VRAM
            if max_rollout is not None:
                T_full = batch['y_future_1d'].size(1)
                T_use = min(max_rollout, T_full)
            else:
                T_use = batch['y_future_1d'].size(1)

            y_hist_1d    = batch['y_hist_1d'].to(device, non_blocking=True)
            y_hist_2d    = batch['y_hist_2d'].to(device, non_blocking=True)
            rain_hist_2d = batch['rain_hist_2d'].to(device, non_blocking=True)
            y_future_1d  = batch['y_future_1d'][:, :T_use].to(device, non_blocking=True)
            y_future_2d  = batch['y_future_2d'][:, :T_use].to(device, non_blocking=True)
            rain_future  = batch['rain_future_2d'][:, :T_use].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                pred_1d, pred_2d = full_event_forward(
                    model, static_graph_cpu,
                    y_hist_1d, y_hist_2d, rain_hist_2d,
                    y_future_1d, y_future_2d, rain_future,
                    rain_1d_index, device, graph_cache=graph_cache,
                    time_embed_dim=time_embed_dim,
                    time_embed_max_period=time_embed_max_period,
                )
                # Compute per-event loss (iterate over B) so each event
                # contributes equally regardless of batch size
                for b in range(B):
                    loss, nrmse_1d, nrmse_2d = kaggle_nrmse_loss(
                        pred_1d[b].float(), pred_2d[b].float(),
                        y_future_1d[b], y_future_2d[b],
                    )
                    scores.append(loss.item())
                    scores_1d.append(nrmse_1d)
                    scores_2d.append(nrmse_2d)

    if not scores:
        return float('nan'), float('nan'), float('nan')
    return float(np.mean(scores)), float(np.mean(scores_1d)), float(np.mean(scores_2d))


# ============================================================
# Training
# ============================================================

def train(resume_from=None, use_mixed_precision=None, pretrain_from=None):
    if use_mixed_precision is None:
        use_mixed_precision = CONFIG['mixed_precision']

    model_id = f"FullEvent_{SELECTED_MODEL}"
    print("\n" + "=" * 70)
    print(f"Full-Event Autoregressive Training — {SELECTED_MODEL}")
    print(f"  Each training step = autoregressive rollout over one entire event")
    print(f"  Loss = Kaggle-aligned per-node RMSE (accounts for variable n)")
    print("=" * 70)

    # ---- Handle resume / wandb ----
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
            f = resume_file or sorted(resume_path.glob(f'{model_id}*.pt'))[-1]
            ckpt = torch.load(f, map_location='cpu', weights_only=False)
            _wandb_run_id = ckpt.get('wandb_run_id')
            _global_step_resume = ckpt.get('global_step', 0) or 0
        except Exception:
            pass

    run_name = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if _wandb_run_id:
        wandb.init(project="floodlm", group="fullevent", id=_wandb_run_id,
                   resume="must", config=CONFIG)
    else:
        wandb.init(project="floodlm", group="fullevent", name=run_name, config=CONFIG)

    run_dir = os.path.join(SAVE_DIR, run_name)
    latest_dir = os.path.join(SAVE_DIR, 'latest')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    if device.type == 'cuda':
        # Reduce CUDA memory fragmentation
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}")
        print(f"[INFO] cudnn.benchmark={torch.backends.cudnn.benchmark}, "
              f"TF32 matmul={torch.backends.cuda.matmul.allow_tf32}")

    # ---- Initialize data ----
    print("[INFO] Initializing data...")
    data = initialize_data()
    norm_stats = data['norm_stats']
    save_normalization_stats(norm_stats, run_dir)
    save_normalization_stats(norm_stats, latest_dir)

    # ---- Build static graph ----
    model_config = get_model_config()
    static_graph_cpu = create_static_hetero_graph(
        data['static_1d_sorted'], data['static_2d_sorted'],
        data['edges1d'], data['edges2d'], data['edges1d2d'],
        data['edges1dfeats'], data['edges2dfeats'],
        data['static_1d_cols'], data['static_2d_cols'],
        data['edge1_cols'], data['edge2_cols'],
        node_id_col=data['NODE_ID_COL'],
    )
    static_graph = static_graph_cpu.to(device)
    rain_1d_index = (
        static_graph_cpu.rain_1d_index.to(device)
        if hasattr(static_graph_cpu, 'rain_1d_index')
        else None
    )

    # Graph cache: maps B → pre-built batched graph on device.
    # Avoids rebuilding PyG Batch every forward call. Populated lazily.
    graph_cache = {}

    # ---- Datasets (grouped batching) ----
    print("[INFO] Building datasets...")
    train_dataset = get_full_event_dataset(
        split='train', shuffle=True, history_len=CONFIG['history_len'])
    val_dataset = get_full_event_dataset(
        split='val', shuffle=False, history_len=CONFIG['history_len'])
    _max_batch_size = CONFIG.get('max_batch_size', 4)

    # Discover T_max_global across all training events for curriculum weighting
    _T_max_global = train_dataset._max_future()
    _horizon_L_start = CONFIG.get('horizon_L_start', 0)
    _horizon_power = CONFIG.get('horizon_power', 1)
    _max_BxT = CONFIG.get('max_BxT', 0)  # 0 = disabled (use fixed max_batch_size)
    print(f"[INFO] T_max_global={_T_max_global}, horizon_L_start={_horizon_L_start}, horizon_power={_horizon_power}")
    print(f"[INFO] max_batch_size={_max_batch_size}, max_BxT={_max_BxT}")
    print(f"[INFO] T_future distribution: {train_dataset._t_future_counts()}")

    # ---- Model ----
    # Bump dynamic input dims to account for time embedding
    _time_embed_dim = CONFIG.get('time_embed_dim', 0)
    _time_embed_max_period = CONFIG.get('time_embed_max_period', 512.0)
    if _time_embed_dim > 0:
        for nt in model_config['node_dyn_input_dims']:
            if nt not in ('global', 'ctx1d', 'ctx2d'):  # Don't add to virtual nodes
                model_config['node_dyn_input_dims'][nt] += _time_embed_dim
        print(f"[INFO] Time embedding: dim={_time_embed_dim}, max_period={_time_embed_max_period}")
        print(f"[INFO] Adjusted node_dyn_input_dims: {model_config['node_dyn_input_dims']}")

    print("[INFO] Building model...")
    model_config.update({
        'h_dim': CONFIG['h_dim'],
        'msg_dim': CONFIG['msg_dim'],
        'hidden_dim': CONFIG['hidden_dim'],
    })
    model = FloodAutoregressiveHeteroModel(**model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Pre-build B=1 graph into cache (always needed)
    graph_cache[1] = model._make_batched_graph(static_graph_cpu, 1).to(device)

    wandb.watch(model, log='all', log_freq=50)

    # ---- Optimizer / Scheduler ----
    optimizer = Adam(model.parameters(), lr=CONFIG['lr'])
    _lr_ratio = CONFIG['lr_final'] / CONFIG['lr']
    _total_epochs = CONFIG['epochs']
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda e: _lr_ratio ** (e / max(_total_epochs - 1, 1))
    )
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None

    # ---- Load checkpoint or pretrained weights ----
    start_epoch = 1
    if resume_path:
        try:
            ckpt_path = resume_file or sorted(resume_path.glob(f'{model_id}*.pt'))[-1]
            print(f"[INFO] Resuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            if ckpt.get('optimizer_state'):
                optimizer.load_state_dict(ckpt['optimizer_state'])
            if ckpt.get('scheduler_state'):
                scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"[INFO] Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}. Starting fresh.")

    if pretrain_from and not resume_path:
        print(f"[INFO] Loading pretrained weights from: {pretrain_from}")
        ckpt = torch.load(pretrain_from, map_location=device, weights_only=False)
        state = ckpt.get('model_state', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        print(f"[INFO] Pretrained weights loaded (fresh optimizer & scheduler)")

    # ---- Logging setup ----
    sigma_1d = KAGGLE_SIGMA[SELECTED_MODEL]['oneD']
    sigma_2d = KAGGLE_SIGMA[SELECTED_MODEL]['twoD']

    print(f"\n[INFO] Training config:")
    print(f"  Epochs: {CONFIG['epochs']}, LR: {CONFIG['lr']} → {CONFIG['lr_final']:.2e}")
    print(f"  Batch: up to {_max_batch_size} same-length events, "
          f"grad_accum: {CONFIG['grad_accum_steps']} batches")
    print(f"  Loss: Kaggle-aligned per-node RMSE (normalizes by n automatically)")
    print(f"  Kaggle sigma 1D={sigma_1d}, 2D={sigma_2d}")
    print(f"  Mixed precision: {use_mixed_precision}")
    print(f"\n{'=' * 70}\nTraining\n{'=' * 70}\n")

    global_step = _global_step_resume
    best_val = float('inf')
    no_improve = 0
    _train_start_time = time.time()  # Cumulative wall-clock tracking

    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_nrmse_1d = 0.0
        epoch_nrmse_2d = 0.0
        n_events = 0
        epoch_start = time.time()

        # Curriculum: compute effective horizon for this epoch
        # power > 1 slows down early expansion (more time at short horizons)
        _progress = ((epoch - 1) / max(_total_epochs - 1, 1)) ** _horizon_power
        _effective_L = _horizon_L_start * (
            _T_max_global / max(_horizon_L_start, 1e-6)
        ) ** _progress if _horizon_L_start > 0 else _T_max_global

        _effective_T = effective_rollout_length(_effective_L, _T_max_global)

        # Adaptive batch size: reduce B for longer rollouts to stay within GPU memory
        if _max_BxT > 0 and _effective_T > 0:
            _adaptive_B = max(1, min(_max_batch_size, _max_BxT // _effective_T))
        else:
            _adaptive_B = _max_batch_size

        wandb.log({
            'epoch': epoch,
            'train/curriculum_L': _effective_L,
            'train/effective_T': _effective_T,
            'train/adaptive_B': _adaptive_B,
        }, step=global_step)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_dataset.iter_grouped(_adaptive_B)):
            B = batch['y_hist_1d'].size(0)
            T_full = batch['y_future_1d'].size(1)

            # Truncate rollout BEFORE moving to GPU (saves VRAM)
            T_future = effective_rollout_length(_effective_L, T_full)

            y_hist_1d    = batch['y_hist_1d'].to(device, non_blocking=True)
            y_hist_2d    = batch['y_hist_2d'].to(device, non_blocking=True)
            rain_hist_2d = batch['rain_hist_2d'].to(device, non_blocking=True)
            y_future_1d  = batch['y_future_1d'][:, :T_future].to(device, non_blocking=True)
            y_future_2d  = batch['y_future_2d'][:, :T_future].to(device, non_blocking=True)
            rain_future  = batch['rain_future_2d'][:, :T_future].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                pred_1d, pred_2d = full_event_forward(
                    model, static_graph_cpu,
                    y_hist_1d, y_hist_2d, rain_hist_2d,
                    y_future_1d, y_future_2d, rain_future,
                    rain_1d_index, device, graph_cache=graph_cache,
                    time_embed_dim=_time_embed_dim,
                    time_embed_max_period=_time_embed_max_period,
                )

                # Horizon curriculum weighting (expanding-horizon exponential decay)
                _tw = None
                if _horizon_L_start > 0 and epoch < _total_epochs:
                    _tw = horizon_curriculum_weights(
                        T_future, _T_max_global, epoch, _total_epochs,
                        L_start=_horizon_L_start, power=_horizon_power,
                        device=device)

                # Per-event loss: compute for each event in batch, average
                batch_loss = 0.0
                batch_nrmse_1d = 0.0
                batch_nrmse_2d = 0.0
                for b in range(B):
                    loss_b, nrmse_1d_b, nrmse_2d_b = kaggle_nrmse_loss(
                        pred_1d[b], pred_2d[b],
                        y_future_1d[b], y_future_2d[b],
                        time_weights=_tw,
                    )
                    batch_loss = batch_loss + loss_b
                    batch_nrmse_1d += nrmse_1d_b
                    batch_nrmse_2d += nrmse_2d_b

                loss = batch_loss / B
                nrmse_1d = batch_nrmse_1d / B
                nrmse_2d = batch_nrmse_2d / B

                # Scale by grad_accum so effective LR is stable
                loss_scaled = loss / CONFIG['grad_accum_steps']

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            epoch_loss += loss.item()
            epoch_nrmse_1d += nrmse_1d
            epoch_nrmse_2d += nrmse_2d
            n_events += B
            global_step += 1

            wandb.log({
                'epoch': epoch,
                'train/loss': loss.item(),
                'train/nrmse_1d': nrmse_1d,
                'train/nrmse_2d': nrmse_2d,
                'train/rollout_steps': T_future,
                'train/batch_size': B,
                'train/rmse_1d_m': nrmse_1d * sigma_1d,
                'train/rmse_2d_m': nrmse_2d * sigma_2d,
                'lr': optimizer.param_groups[0]['lr'],
            }, step=global_step)

            # Gradient accumulation: step every N batches
            if (batch_idx + 1) % CONFIG['grad_accum_steps'] == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Console progress every 3 batches
            if (batch_idx + 1) % 3 == 0:
                avg = epoch_loss / max(n_events / B, 1)
                print(f"  Epoch {epoch} | Batch {batch_idx+1} | "
                      f"B={B} n={T_future}/{T_full} | loss={loss.item():.4f} | avg={avg:.4f}")

        # Flush remaining accumulated gradients
        if (batch_idx + 1) % CONFIG['grad_accum_steps'] != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        cumulative_time = time.time() - _train_start_time
        n_batches = batch_idx + 1
        mean_loss = epoch_loss / max(n_batches, 1)
        mean_1d = epoch_nrmse_1d / max(n_batches, 1)
        mean_2d = epoch_nrmse_2d / max(n_batches, 1)
        print(f"\n[Epoch {epoch:03d}/{CONFIG['epochs']}] "
              f"loss={mean_loss:.4f}  NRMSE 1D={mean_1d:.4f} 2D={mean_2d:.4f}  "
              f"events={n_events} batches={n_batches}  "
              f"time={epoch_time:.0f}s  cumulative={cumulative_time/60:.1f}min")

        # ---- Validation (truncated = curriculum-aligned, full = final metric) ----
        val_start = time.time()
        # Shared eval kwargs (batch size varies between truncated and full)
        _eval_base = dict(
            model=model, val_dataset=val_dataset,
            static_graph_cpu=static_graph_cpu,
            rain_1d_index=rain_1d_index, device=device,
            use_mixed_precision=use_mixed_precision,
            graph_cache=graph_cache,
            time_embed_dim=_time_embed_dim,
            time_embed_max_period=_time_embed_max_period,
        )
        # Full val needs smaller B since it rolls out to T_max_global
        if _max_BxT > 0 and _T_max_global > 0:
            _val_full_B = max(1, min(_max_batch_size, _max_BxT // _T_max_global))
        else:
            _val_full_B = _max_batch_size
        # Truncated val (matches training horizon — the metric that should improve)
        val_trunc_combined, val_trunc_1d, val_trunc_2d = evaluate(
            **_eval_base, max_batch_size=_adaptive_B, max_rollout=_effective_T)
        # Full val (the real Kaggle-style metric)
        val_combined, val_1d, val_2d = evaluate(
            **_eval_base, max_batch_size=_val_full_B, max_rollout=None)
        val_time = time.time() - val_start

        # Free GPU memory after validation (reduces fragmentation before next epoch)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        val_rmse_1d = val_1d * sigma_1d
        val_rmse_2d = val_2d * sigma_2d
        print(f"  Val (trunc n={_effective_T}): combined={val_trunc_combined:.4f}  "
              f"1D={val_trunc_1d:.4f}  2D={val_trunc_2d:.4f}")
        print(f"  Val (full):   combined={val_combined:.4f}  "
              f"1D={val_1d:.4f} ({val_rmse_1d:.3f}m)  "
              f"2D={val_2d:.4f} ({val_rmse_2d:.3f}m)  "
              f"val_t={val_time:.0f}s")

        wandb.log({
            'epoch': epoch,
            'val/nrmse_combined': val_combined,
            'val/nrmse_1d': val_1d,
            'val/nrmse_2d': val_2d,
            'val/rmse_1d_m': val_rmse_1d,
            'val/rmse_2d_m': val_rmse_2d,
            'val_trunc/nrmse_combined': val_trunc_combined,
            'val_trunc/nrmse_1d': val_trunc_1d,
            'val_trunc/nrmse_2d': val_trunc_2d,
            'train/epoch_loss': mean_loss,
            'train/epoch_nrmse_1d': mean_1d,
            'train/epoch_nrmse_2d': mean_2d,
            'epoch_time_s': epoch_time,
            'cumulative_time_min': cumulative_time / 60,
        }, step=global_step)

        # ---- Checkpoint ----
        if epoch % CONFIG['checkpoint_interval'] == 0:
            save_checkpoint(
                model, epoch, mean_loss, run_dir,
                optimizer=optimizer, scheduler=scheduler,
                global_step=global_step,
                wandb_run_id=wandb.run.id if wandb.run else None,
            )

        # During curriculum ramp-up, use truncated val for early stopping
        # (full val is meaningless when the model hasn't trained on long horizons).
        # Once the effective horizon covers most of the longest event, switch to full val.
        _curriculum_mostly_done = (_effective_T >= 0.8 * _T_max_global)
        _val_for_stopping = val_combined if _curriculum_mostly_done else val_trunc_combined

        # Best checkpoint (based on appropriate val metric)
        if not np.isnan(_val_for_stopping) and _val_for_stopping < best_val:
            best_val = _val_for_stopping
            no_improve = 0
            for save_dir in (run_dir, latest_dir):
                save_checkpoint(
                    model, epoch, mean_loss, save_dir,
                    optimizer=optimizer, scheduler=scheduler,
                    global_step=global_step,
                    wandb_run_id=wandb.run.id if wandb.run else None,
                    best=True,
                )
            _metric_label = 'full' if _curriculum_mostly_done else f'trunc(n={_effective_T})'
            print(f"  *** New best val NRMSE ({_metric_label}): {best_val:.4f} ***")
        else:
            no_improve += 1

        # Early stopping — only active once curriculum horizon is substantial
        patience = CONFIG['early_stopping_patience']
        if patience is not None and _curriculum_mostly_done and no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch} "
                  f"(no improvement for {no_improve} epochs)")
            break

        print()

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(f"Training Complete — {model_id}")
    print(f"  Best val NRMSE: {best_val:.4f}")
    print(f"  Checkpoints: {run_dir}")
    print(f"  Latest: {latest_dir}")
    print("=" * 70)
    wandb.finish()


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full-event autoregressive training for FloodLM')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint dir or .pt file to resume from')
    parser.add_argument('--pretrain-from', type=str, default=None,
                        help='Path to pretrained .pt weights (fresh optimizer, warm weights)')
    parser.add_argument('--mixed-precision', action='store_true', default=None,
                        help='Use AMP mixed precision training')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision')
    args = parser.parse_args()

    mp = args.mixed_precision
    if args.no_mixed_precision:
        mp = False

    train(
        resume_from=args.resume,
        use_mixed_precision=mp,
        pretrain_from=args.pretrain_from,
    )
