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

# ── Scale knobs ────────────────────────────────────────────────────────────────
# To grow the model, increase these tiers:
#   encoder:  ENC_SCALE  s → (s*96, s*64, s*10)  for (h_dim, msg_dim, history_len)
#   decoder:  DEC_SCALE  s → (s*64, s*2, s*2, s*128)  for (d_model, nhead, num_layers, ffn_dim)
#   edges:    EDGE_SCALE s → s*base  for all HIDDEN_DIMS
# MVP baseline = scale 1.
ENC_SCALE  = 1   # 1 → h_dim=96, msg_dim=64, history_len=10
DEC_SCALE  = 1   # 1 → d_model=64, nhead=2, num_layers=2, ffn_dim=128
EDGE_SCALE = 1   # 1 → edge hidden dims as defined in HIDDEN_DIMS below

def _enc(base): return int(base * ENC_SCALE)
def _dec(base): return int(base * DEC_SCALE)
# ───────────────────────────────────────────────────────────────────────────────

# Training hyperparameters
CONFIG = {
    # Encoder
    'h_dim':        _enc(96),   # GRU hidden size
    'msg_dim':      _enc(64),   # Message passing dimension
    'history_len':  _enc(10),   # Warm-start steps
    'num_mp_rounds': 2,         # Message passing rounds per GRU step (multi-hop spatial reasoning)
                                # 1 = original (1-hop), 2 = 2-hop, 3 = 3-hop receptive field per step

    # Transformer decoder
    'dec_d_model':   _dec(64),  # Transformer model dimension
    'dec_nhead':     _dec(2),   # Attention heads  (must divide dec_d_model)
    'dec_num_layers':_dec(4),   # Transformer encoder layers (increased from 2 for deeper temporal reasoning)
    'dec_ffn_dim':   _dec(128), # Feedforward dim inside transformer
    'dec_dropout': 0.1,

    # Node-chunking in transformer forward (lower = less peak VRAM, more passes)
    'dec_node_chunk': 512,

    'T_max': 512,              # Time embedding normalization constant (> max event length)
    'batch_size': 1,
    'grad_accum_steps': 8,
    'epochs': 1000,
    'lr': 3e-4,
    'lr_final': 1e-5,
    'grad_clip': 1.0,
    'early_stopping_patience': 15,
    'mixed_precision': True,
    'checkpoint_interval': 1,
    'save_dir': os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "model3")),
}

def _edge(base): return int(base * EDGE_SCALE)

# Edge-type-specific hidden dims
HIDDEN_DIMS = {
    'oneDedge':    _edge(64),
    'oneDedgeRev': _edge(64),
    'twoDedge':    _edge(128),
    'twoDedgeRev': _edge(128),
    'twoDoneD':    _edge(32),
    'oneDtwoD':    _edge(32),
    # Global node edges (GATv2)
    'oneDglobal':  _edge(32),
    'globaloneD':  _edge(32),
    'twoDglobal':  _edge(32),
    'globaltwoD':  _edge(32),
}
