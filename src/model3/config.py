"""
Model_3 configuration — all hyperparameters and paths in one place.
Model_3 uses the same raw data as Model_2 (data/Model_2/) but has its own
cache, checkpoints, and training logic.
"""

import os

MODEL_ID = "Model_3"

# Data paths — reuse Model_2's raw event data
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Model_2")
BASE_PATH = os.path.normpath(BASE_PATH)
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH  = os.path.join(BASE_PATH, "test")
CACHE_PATH = os.path.join(BASE_PATH, ".cache", "Model_3_preprocessed.pkl")

# Train/val split (same random seed as Model_2 for comparable splits)
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Kaggle scoring sigmas for Model_2 data
# Water level is normalized by these sigmas so sqrt(MSE_norm) == NRMSE
KAGGLE_SIGMA_1D = 3.192
KAGGLE_SIGMA_2D = 2.727

# Columns to exclude from dynamic features
EXCLUDE_1D_DYNAMIC = ['inlet_flow']
EXCLUDE_2D_DYNAMIC = ['water_volume']

# Training hyperparameters
CONFIG = {
    # Encoder
    'h_dim': 192,              # GRU hidden size (2× previous)
    'msg_dim': 128,            # Message passing dimension (2× previous)
    'history_len': 20,         # Warm-start steps (2× previous — richer h_enc)

    # Transformer decoder
    'dec_d_model': 256,        # Transformer model dimension
    'dec_nhead': 8,            # Attention heads
    'dec_num_layers': 4,       # Transformer encoder layers
    'dec_ffn_dim': 512,        # Feedforward dim inside transformer
    'dec_dropout': 0.1,

    # Kept for checkpoint compat (no longer used by decoder)
    'decoder_hidden_dim': 256,

    'T_max': 512,              # Time embedding normalization constant (> max event length)
    'batch_size': 1,
    'grad_accum_steps': 8,
    'epochs': 1000,
    'lr': 3e-4,                # Lower LR — transformer decoders are more sensitive
    'lr_final': 1e-5,
    'grad_clip': 1.0,
    'early_stopping_patience': 15,
    'mixed_precision': True,
    'checkpoint_interval': 1,
    'save_dir': os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "model3")),
}

# Edge-type-specific hidden dims (same as Model_2)
HIDDEN_DIMS = {
    'oneDedge':    128,
    'oneDedgeRev': 128,
    'twoDedge':    256,
    'twoDedgeRev': 256,
    'twoDoneD':    64,
    'oneDtwoD':    64,
    # Global node edges (GATv2)
    'oneDglobal':  64,
    'globaloneD':  64,
    'twoDglobal':  64,
    'globaltwoD':  64,
}
