# FloodLM Architecture

FloodLM is a heterogeneous graph neural network that autoregressively predicts water levels across a coupled 1D/2D flood hydraulic system. Two separate models (Model_1, Model_2) are trained on different simulation domains and their predictions are combined for the Kaggle submission.

---

## Problem Formulation

Given 10 timesteps of history (water levels + rainfall), predict the next 64 timesteps of water level at every node in the graph. This is done autoregressively: each predicted timestep becomes the input for the next.

**Inputs per timestep:**
- 1D nodes (channels): `[water_level, rainfall_2d, water_level_2d]` — shape `[N_1d, 3]` (rainfall and water level of each channel's connected 2D node)
- 2D nodes (floodplain cells): `[water_level, rainfall]` — shape `[N_2d, 2]`
- Static node/edge features: precomputed at graph construction, fixed across time

**Static node features:**
- 1D nodes (7): `position_x, position_y, depth, invert_elevation, surface_elevation, base_area, channel_2d_elev_diff`
  - `channel_2d_elev_diff` = connected 2D cell elevation − channel invert elevation; captures how deeply incised the channel is relative to its floodplain (strong RMSE predictor, r=0.58)
- 2D nodes (10): `position_x, position_y, area, roughness, min_elevation, elevation, curvature, flow_accumulation, aspect_sin, aspect_cos`
  - Aspect encoded as (sin, cos) to handle circularity; aspect=-1 sentinel (flat/undefined) → (0, 0)

**Outputs per timestep:**
- `Δwater_level` at every node (1D and 2D separately), denormalized to meters

---

## Graph Structure

The computational graph is a static `HeteroData` object with two node types and six directed edge types.

### Node Types

| Type | Count | Role |
|------|-------|------|
| `oneD` | 17 (Model_1) / 150 (Model_2) | 1D hydraulic channels |
| `twoD` | ~thousands | 2D floodplain cells |

**Static features per node:** elevation, area, geometry, hydraulic properties (varies by model and node type).

### Edge Types

| Edge | Direction | Module (Model_1) | Module (Model_2) | Purpose |
|------|-----------|-----------------|-----------------|---------|
| `oneDedge` | 1D → 1D | `StaticDynamicEdgeMP` | `StaticDynamicEdgeMP` | Downstream channel flow |
| `oneDedgeRev` | 1D ← 1D | `StaticDynamicEdgeMP` | `StaticDynamicEdgeMP` | Backwater / reverse flow |
| `twoDedge` | 2D → 2D | `StaticDynamicEdgeMP` | `StaticDynamicEdgeMP` | Floodplain propagation |
| `twoDedgeRev` | 2D ← 2D | `StaticDynamicEdgeMP` | `StaticDynamicEdgeMP` | Bidirectional inundation |
| `twoDoneD` | 2D → 1D | `GATv2CrossTypeMP` | `StaticDynamicEdgeMP` | Drainage into channels |
| `oneDtwoD` | 1D → 2D | `GATv2CrossTypeMP` | `StaticDynamicEdgeMP` | Channel overflow onto floodplain |

**Cross-type edge features:**
- **Model_1**: zero placeholder (dim=1) — GATv2 attention learns purely from hidden states
- **Model_2**: `[distance, elev_diff]` (dim=2, z-scored) — `elev_diff = 2D_elevation − 1D_invert_elevation`; positive = deeply incised channel. For `oneDtwoD` the sign is flipped (directional). Allows `StaticDynamicEdgeMP.base_weight` to learn static suppression of deeply incised connections.

Directional edge features (`relative_position_x`, `relative_position_y`, `slope`) are negated for reverse edges at graph construction time, encoding physical asymmetry.

---

## Model Architecture

### Top Level: `FloodAutoregressiveHeteroModel`

```
FloodAutoregressiveHeteroModel
├── cell: HeteroTransportCell      ← single RNN step
└── heads: ModuleDict
    ├── oneD: LayerNorm → Linear(64→128) → ReLU → Linear(128→1)
    └── twoD: LayerNorm → Linear(64→128) → ReLU → Linear(128→1)
```

The model runs `forward_unroll()`, which:
1. **Warm start** (H=10 steps): teacher-force the cell with true historical water levels to build up hidden state
2. **Autoregressive rollout** (T≤64 steps): predict → feed prediction as input → repeat

### `HeteroTransportCell` — One Timestep

For each timestep the cell performs:

```
1. Dynamic projection
   dyn_emb[nt] = dyn_norm[nt]( dyn_proj[nt]( x_dyn_t[nt] ) )
                                                              [N_nt, msg_dim]

2. Heterogeneous message passing
   msg[nt] = HeteroConv( all 6 edge types )( h_t, static_graph )
                                                              [N_nt, msg_dim]

3. GRU update
   h_next[nt] = h_norm[nt]( GRUCell( cat(dyn_emb, msg), h_t[nt] ) )
                                                              [N_nt, h_dim]

4. Output head
   pred[nt] = head[nt]( h_next[nt] )                        [N_nt, 1]
```

**Hidden state dimensions:** `h_dim=96` (GRU hidden, kept large for temporal memory), `msg_dim=64`, MLP hidden `=64–128` (edge-type-specific).
**Normalization:** LayerNorm on hidden state, messages, and dynamic inputs (prevents magnitude explosion across 64 rollout steps).

### Message Passing Modules

#### `StaticDynamicEdgeMP` (5 edge types)

Computes a gated message from source to destination:

```
edge_emb  = MLP( [edge_static || src_static || dst_static] )
base_wt   = softplus( w^T edge_emb )          ← static coupling strength
dyn_gate  = sigmoid( MLP( [h_src || h_dst] ) ) ← dynamic temporal gate
payload   = MLP( h_src )                       ← message content

message   = (base_wt × dyn_gate) × payload
```

The `softplus` ensures positive coupling; the `sigmoid` gate lets the model suppress messages when the hydraulic system is inactive.

#### `GATv2CrossTypeMP` (cross-type edges, Model_1 only)

Wraps `torch_geometric.nn.GATv2Conv` with 4 attention heads, followed by a two-layer feed-forward network (FFN) for nonlinear capacity:

```
attn_out     = GATv2Conv(
                   (h_src, h_dst),
                   in_channels  = (h_dim, h_dim),
                   out_channels = msg_dim // heads,
                   heads        = 4,
                   concat       = True,
               )                                   # [N_dst, msg_dim]

message[dst] = FFN( attn_out )                     # Linear(msg_dim→64)→ReLU→Linear(64→msg_dim)
```

Attention is computed jointly over source and destination hidden states, allowing the model to learn which channel nodes most influence each floodplain cell. The FFN adds nonlinear expressivity that a single GATv2Conv layer lacks.

**Model_2 uses `StaticDynamicEdgeMP` for cross-type edges instead**, because GATv2 cannot suppress deeply incised channels (15–34m elevation gap) using only hidden states. The `[distance, elev_diff]` edge features allow `base_weight` to learn static suppression directly from geometry.

---

## Normalization

All water levels are normalized using **meanstd normalization scaled by the Kaggle competition sigma**:

```
y_norm = (y - mean) / kaggle_sigma
```

| Model | Node Type | Kaggle σ (m) |
|-------|-----------|--------------|
| Model_1 | 1D | 16.878 |
| Model_1 | 2D | 14.379 |
| Model_2 | 1D | 3.192 |
| Model_2 | 2D | 2.727 |

This means `sqrt(MSE_norm) == NRMSE` directly in the competition metric space, so the training loss is directly interpretable as normalized RMSE.

---

## Loss Function

Sigma-weighted convex combination of 1D and 2D MSE:

```
w_1d  = (range_1d / sigma_1d)²
w_2d  = (range_2d / sigma_2d)²

loss  = (w_1d × MSE_1d  +  w_2d × MSE_2d) / (w_1d + w_2d)
```

This down-weights node types whose predictions are naturally high-variance relative to the competition metric, and keeps loss scale ~1× raw MSE so `lr=1e-3` stays well-calibrated.

---

## Training

### Curriculum Learning

Training gradually increases the rollout horizon to prevent gradient explosion from backpropagating through 64 steps from epoch 1.

**Model_1** uses a power-of-2 schedule (2 epochs per stage):

| Epochs | max_h |
|--------|-------|
| 1–2 | 1 |
| 3–4 | 2 |
| 5–6 | 4 |
| 7–8 | 8 |
| 9–10 | 16 |
| 11–12 | 32 |
| 13+ | 64 |

**Model_2** uses a custom schedule with intermediary steps at h=24 and h=48 to smooth the large h=16→32→64 jumps. Earlier stages get 4 epochs; later stages get 3; h=64 gets 6:

| Epochs | max_h |
|--------|-------|
| 1–4 | 1 |
| 5–8 | 2 |
| 9–12 | 4 |
| 13–16 | 8 |
| 17–19 | 16 |
| 20–22 | 24 |
| 23–25 | 32 |
| 26–28 | 48 |
| 29–34 | 64 |

### Stability Measures

- **Gradient clipping**: `clip_grad_norm_(params, max_norm=1.0)` every step
- **LR reduction at curriculum jumps (Model_2 only)**: LR × 0.3 at h=8, 24, 32, 48, 64
  - Schedule: `1e-3 → 3e-4 → 9e-5 → 2.7e-5 → 8.1e-6 → 2.4e-6`
- **Mixed precision**: `torch.amp.autocast` + `GradScaler` (via `--mixed-precision` flag)
- **Gradient checkpointing**: auto-enabled with `--mixed-precision`; recomputes activations during backward to avoid storing the full 64-step rollout graph in memory
- **Early stopping**: patience=5 epochs, active only at `max_h=64`

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| `history_len` | 10 |
| `forecast_len` | 64 |
| `batch_size` | 24 |
| `epochs` | Model_1: ~16+; Model_2: 34 |
| `lr` | 1e-3 |
| `h_dim` | 96 |
| `msg_dim` | 64 |
| `hidden_dim` (1D homogeneous edges) | 64 |
| `hidden_dim` (2D homogeneous edges) | 128 |
| `hidden_dim` (cross-type edges) | 32 |
| `dropout` | 0.0 |

Note: `hidden_dim` is edge-type-specific. Cross-type edges (`twoDoneD`, `oneDtwoD`) use 32 because there are only ~170 connections — a smaller MLP is sufficient and avoids overfitting. 1D homogeneous edges use 64; 2D homogeneous edges use 128 because the 2D mesh has thousands of edges. `h_dim` is kept at 96 (larger than `msg_dim`) because the GRU hidden state is the primary temporal memory.

---

## Inference

Inference is fully autoregressive over 64 steps with no teacher forcing. The `autoregressive_inference.py` script:

1. Loads both Model_1 and Model_2 from `checkpoints/latest/` (prefers `best_h64.pt` over `best.pt`)
2. Loads test events and runs `forward_unroll()` with `rollout_steps=64`
3. Denormalizes predictions: `y = y_norm × sigma + mean`, then clamps `water_level ≥ 0`
4. Writes Kaggle submission CSV

---

## Data Layout

```
data/{SELECTED_MODEL}/train/
├── 1d_nodes_static.csv          # static node features for 1D nodes
├── 2d_nodes_static.csv          # static node features for 2D nodes
├── 1d_edges_static.csv          # static edge features for 1D→1D edges
├── 2d_edges_static.csv          # static edge features for 2D→2D edges
├── 1d_edge_index.csv            # connectivity for 1D→1D edges
├── 2d_edge_index.csv            # connectivity for 2D→2D edges
├── 1d2d_connections.csv         # connectivity for cross-type edges
└── event_{i}/
    ├── 1d_nodes_dynamic_all.csv  # time series: water_level per 1D node
    └── 2d_nodes_dynamic_all.csv  # time series: water_level + rainfall per 2D node
```

Events are split 80/20 train/val by `data_config.py`. During the final fine-tune stage, all events (train+val) are used.

---

## Pipeline

```
run/pipeline.sh
  Stage 1: Training          (src/train.py, both models sequentially)
  Stage 2: Inference         (src/autoregressive_inference.py)
  Stage 3: RMSE evaluation   (kaggle/calculate_rmse.py)
  Stage 4: Architecture snapshot (run/snapshot_arch.sh)
  Stage 5: Kaggle submission (kaggle/submit_to_kaggle.py)

run/pipeline_finetune_submit.sh
  Stage 1: Fine-tune on train+val (--no-val, --max-h 64, reduced LR)
  Stage 2–5: same as above
```

SLURM jobs: `sbatch slurm/submit_slurm.sh` (training, 12h) and `sbatch slurm/submit_inference_slurm.sh` (inference+submit, 2h).
