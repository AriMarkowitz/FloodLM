# FloodLM Architecture

FloodLM is a heterogeneous graph neural network that autoregressively predicts water levels across a coupled 1D/2D flood hydraulic system. Two separate models (Model_1, Model_2) are trained on different simulation domains and their predictions are combined for the Kaggle submission.

---

## Problem Formulation

Given 10 timesteps of history (water levels + rainfall), predict the next 64 timesteps of water level at every node in the graph. This is done autoregressively: each predicted timestep becomes the input for the next.

**Inputs per timestep:**
- 1D nodes (channels): `[water_level]` — shape `[N_1d, 1]`
- 2D nodes (floodplain cells): `[water_level, rainfall]` — shape `[N_2d, 2]`
- Static node/edge features: precomputed at graph construction, fixed across time

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

| Edge | Direction | Module | Purpose |
|------|-----------|--------|---------|
| `oneDedge` | 1D → 1D | `StaticDynamicEdgeMP` | Downstream channel flow |
| `oneDedgeRev` | 1D ← 1D | `StaticDynamicEdgeMP` | Backwater / reverse flow |
| `twoDedge` | 2D → 2D | `StaticDynamicEdgeMP` | Floodplain propagation |
| `twoDedgeRev` | 2D ← 2D | `StaticDynamicEdgeMP` | Bidirectional inundation |
| `twoDoneD` | 2D → 1D | `StaticDynamicEdgeMP` | Drainage into channels |
| `oneDtwoD` | 1D → 2D | `GATv2CrossTypeMP` | Channel overflow onto floodplain |

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

**Hidden state dimensions:** `h_dim=96`, `msg_dim=96`, MLP hidden `=96–192` (edge-type-specific).
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

#### `GATv2CrossTypeMP` (1D → 2D edge)

Wraps `torch_geometric.nn.GATv2Conv` with 4 attention heads, followed by a two-layer feed-forward network (FFN) for nonlinear capacity:

```
attn_out     = GATv2Conv(
                   (h_src, h_dst),
                   in_channels  = (h_dim, h_dim),
                   out_channels = msg_dim // heads,
                   heads        = 4,
                   concat       = True,
               )                                   # [N_dst, msg_dim]

message[dst] = FFN( attn_out )                     # Linear(msg_dim→96)→ReLU→Linear(96→msg_dim)
```

Attention is computed jointly over source and destination hidden states, allowing the model to learn which channel nodes most influence each floodplain cell. The FFN adds nonlinear expressivity that a single GATv2Conv layer lacks (GATv2Conv has no internal hidden dimension).

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

Training uses an exponential horizon curriculum. `max_h` doubles every `_stage_len` epochs:

```python
_stage_len = 3 if Model_2 else 2
max_h = min(64, 2 ** ((epoch - 1) // _stage_len))
```

| Epochs | Model_1 max_h | Model_2 max_h |
|--------|--------------|--------------|
| 1–2 | 1 | — |
| 1–3 | — | 1 |
| 3–4 | 2 | — |
| 4–6 | — | 2 |
| 5–6 | 4 | — |
| 7–9 | — | 4 |
| 7–8 | 8 | — |
| 10–12 | — | 8 |
| 9–10 | 16 | — |
| 13–15 | — | 16 |
| 11–12 | 32 | — |
| 16–18 | — | 32 |
| 13–24 | 64 | — |
| 19–24 | — | 64 |

This prevents gradient explosion from backpropagating through 64 steps from epoch 1.

### Stability Measures

- **Gradient clipping**: `clip_grad_norm_(params, max_norm=1.0)` every step
- **LR reduction at curriculum jumps (Model_2 only)**: LR × 0.3 at h=8, 32, 64
  - Schedule: `1e-3 → 3e-4 → 9e-5 → 2.7e-5`
- **Mixed precision**: `torch.amp.autocast` + `GradScaler` (via `--mixed-precision` flag)
- **Gradient checkpointing**: auto-enabled with `--mixed-precision`; recomputes activations during backward to avoid storing the full 64-step rollout graph in memory
- **Early stopping**: patience=5 epochs, active only at `max_h=64`

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| `history_len` | 10 |
| `forecast_len` | 64 |
| `batch_size` | 16 |
| `epochs` | 24 |
| `lr` | 1e-3 |
| `h_dim` | 96 |
| `msg_dim` | 96 |
| `hidden_dim` (1D homogeneous + cross-type edges) | 96 |
| `hidden_dim` (2D homogeneous edges) | 192 |
| `dropout` | 0.0 |
| Total parameters | ~628K |

Note: `hidden_dim` is edge-type-specific. 1D homogeneous edges (`oneDedge`, `oneDedgeRev`) and cross-type edges (`twoDoneD`, `oneDtwoD`) use 96 because those graphs are sparse (≤200 edges). 2D homogeneous edges use 192 because the 2D mesh has thousands of edges.

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
