"""
Full-event fine-tuning configuration.

Fine-tunes an existing h=64 checkpoint with:
  - Horizon curriculum starting at eff_T=64 (matches trained model), growing to T_max
  - Random warm-start history augmentation (h ~ Uniform(10, T_total - eff_T))
  - Lower learning rate suitable for fine-tuning
  - Fresh wandb run (new run_name, no resume)

Usage:
    python -m fullevent.train --finetune --pretrain-from checkpoints/latest/Model_1_best.pt
    SELECTED_MODEL=Model_2 python -m fullevent.train --finetune --pretrain-from checkpoints/latest/Model_2_best.pt
"""

import os
import copy
from fullevent.config import CONFIG as _BASE_CONFIG, KAGGLE_SIGMA, SELECTED_MODEL

# Start from base config, override fine-tune specific keys
CONFIG = copy.deepcopy(_BASE_CONFIG)
CONFIG.update({
    'epochs': 40,
    'lr': 1e-4,
    'lr_final': 1e-6,
    'early_stopping_patience': 20,

    # Horizon curriculum: start at eff_T=64, grow to T_max over training
    # ceil(5 * 12.8) = 64
    'horizon_L_start': 12.8,    # eff_T = 64 at epoch 1
    'horizon_power': 2,         # moderate growth rate

    # No time embedding — base checkpoint was trained without it
    'time_embed_dim': 0,

    # Random warm-start: always at least 10 steps, up to T_total - eff_T
    'random_history': True,
    'random_history_min_hist': 10,
    'random_history_max_K': 85,

    # Match base model arch (h_dim=96/msg_dim=64) so pretrained weights load cleanly
    'h_dim': 96,
    'msg_dim': 64,
    'hidden_dim': {
        'oneDedge':    64,
        'oneDedgeRev': 64,
        'twoDedge':    128,
        'twoDedgeRev': 128,
        'twoDoneD':    32,
        'oneDtwoD':    32,
    },
})

# Model_2 adds global-node edge hidden dims
if SELECTED_MODEL == 'Model_2':
    CONFIG['hidden_dim'].update({
        'oneDglobal':  32,
        'globaloneD':  32,
        'twoDglobal':  32,
        'globaltwoD':  32,
    })

# Checkpoint directory (separate from scratch training)
SAVE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'fullevent_finetune'))
