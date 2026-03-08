"""
Full-event autoregressive training configuration.

Reuses the same FloodAutoregressiveHeteroModel architecture from src/model.py,
but trains on whole events (variable-length autoregressive rollout) instead of
fixed-length curriculum windows.

Data source is selected by SELECTED_MODEL env var (same as main training).
"""

import os
from pathlib import Path

# Inherit model selection from environment / data_config
SELECTED_MODEL = os.environ.get("SELECTED_MODEL", "Model_1")

# Kaggle scoring sigmas — water_level is normalized by these, so sqrt(MSE_norm) == NRMSE
KAGGLE_SIGMA = {
    'Model_1': {'oneD': 16.878, 'twoD': 14.379},
    'Model_2': {'oneD':  3.192, 'twoD':  2.727},
}

# Training hyperparameters
CONFIG = {
    'history_len': 10,          # Warm-start timesteps (teacher forcing with ground truth)
    'max_batch_size': 4,        # Max events per batch (grouped by same T_future, no padding)
    'max_BxT': 450,             # Memory budget: adaptive B = min(max_batch_size, max_BxT // T_eff)
    'grad_accum_steps': 4,      # Accumulate gradients over N batches before optimizer step
    'epochs': 1000,
    'lr': 5e-4,
    'lr_final': 1e-5,
    'grad_clip': 1.0,
    'early_stopping_patience': 50,
    'checkpoint_interval': 1,
    'mixed_precision': True,

    # Relative time embedding: sinusoidal PE appended to dynamic inputs
    # at every timestep so the model knows "how far into the event" it is.
    # 8 dims = 4 sin + 4 cos at geometrically-spaced frequencies.
    'time_embed_dim': 8,
    'time_embed_max_period': 512,  # Max event length for frequency scaling

    # Horizon curriculum loss weighting (expanding-horizon exponential decay).
    # At each epoch an "effective horizon" L grows geometrically from
    # L_start to T_max_global over the full training run.
    # Weight at timestep t = exp(-t / L), normalized to mean=1.
    # Early epochs: only the first few timesteps matter (L≈2).
    # Late epochs:  nearly uniform (L≈T_max).
    # Set horizon_L_start=0 to disable (uniform from the start).
    'horizon_L_start': 2.0,     # Initial effective horizon (in timesteps)
    'horizon_power': 3,          # Power applied to progress curve (>1 = more time at short horizons)

    # Model architecture — larger dims than base training to use available
    # GPU memory (base uses h=96/msg=64; we have ~38% VRAM headroom at B=1).
    'h_dim': 128,
    'msg_dim': 96,
    'hidden_dim': {             # Per edge-type hidden dims
        'oneDedge':    96,
        'oneDedgeRev': 96,
        'twoDedge':    192,
        'twoDedgeRev': 192,
        'twoDoneD':    48,
        'oneDtwoD':    48,
    },
}

# Model_2 adds global-node edge hidden dims
if SELECTED_MODEL == 'Model_2':
    CONFIG['hidden_dim'].update({
        'oneDglobal':  48,
        'globaloneD':  48,
        'twoDglobal':  48,
        'globaltwoD':  48,
    })

# Checkpoint directory
SAVE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'fullevent'))
