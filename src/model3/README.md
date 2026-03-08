# Model_3 — Non-Autoregressive Encoder-Decoder GNN

## Overview

Model_3 is a fully self-contained, non-autoregressive flood forecasting model. It was designed to
address two core weaknesses of Model_1/2's autoregressive rollout:

1. **Epoch cost**: Model_2 at h=64 takes ~1600s/epoch due to BPTT through 64 sequential GRU steps.
2. **Train/inference gap**: Model_1/2 train on 64-step windows; inference runs on ~200–400-step
   events. The model generalises across that gap but never trains on it.

Model_3 eliminates both: the encoder runs exactly 10 steps, the decoder predicts the entire
remaining event in one parallel forward pass. Backprop only flows through the 10 encoder steps.

---

## Architecture

### Graph topology (same as Model_2)
- **1D nodes** (~198): river channel cross-sections. Static features: invert elevation, slope, width, etc.
- **2D nodes** (~4299): floodplain cells. Static features: elevation, aspect (sin/cos), area, etc.
- **Global context node** (1): virtual node connected to all 1D and 2D nodes via GATv2.
- **Edge types** (10): `oneD↔oneD`, `twoD↔twoD`, `twoD↔oneD`, `oneD↔twoD`, `*↔global`.
  Cross-type edges carry [distance, elev_diff] as static features (StaticDynamicEdgeMP).

### Encoder (GRU × 10 steps, teacher-forced)
```
for t in 0..9:
    x_dyn_t[oneD] = [water_level_1d_t, rainfall_2d_t, water_level_2d_t]  # [N_1d, 3]
    x_dyn_t[twoD] = [water_level_2d_t, rainfall_2d_t]                    # [N_2d, 2]
    x_dyn_t[global] = [0]                                                 # [1, 1]
    h_t = HeteroTransportCell(graph, h_{t-1}, x_dyn_t)
h_enc = h_10  # {nt: [N_nt, 96]}
```
`HeteroTransportCell`: message passing (StaticDynamicEdgeMP / GATv2) → GRU update → LayerNorm.
Result: each node holds a 96-dim hidden state summarising the basin at t=10.

### Decoder (parallel MLP, no recurrence)
```
for each future timestep t ∈ {0, …, T_future-1}:
    time_emb_t = [sin(t/512), cos(t/512)]                     # [2]
    rain_t[node] = rain_future_t[node]                        # [1]
    input_t[node] = cat(h_enc[node], rain_t[node], time_emb_t)  # [99]
    water_level_t[node] = MLP(input_t[node])                  # [1]
```
All T_future timesteps computed in one batched MLP call via broadcasting:
`[T, N, 99] → Linear(99→128) → ReLU → LayerNorm → Linear(128→1) → [T, N, 1]`

Separate MLP heads for `oneD` and `twoD`. No spatial propagation in decoder.

### Parameter count
| Component | Params |
|---|---|
| Encoder HeteroTransportCell | ~478K |
| Decoder MLP (oneD) | ~13K |
| Decoder MLP (twoD) | ~13K |
| **Total** | **~529K** |

---

## Training

- **Dataset**: `FullEventFloodDataset` — one sample per event (no sliding windows).
  56 train events, 13 val events (80/20 split, same seed as Model_2).
- **Batch size**: 1 event per step (T_future varies per event ~190–400 steps).
- **Gradient accumulation**: 8 events before `optimizer.step()`.
- **Loss**: unweighted mean of MSE_1d and MSE_2d (normalized space, so ~= NRMSE²).
- **LR schedule**: log-linear decay from 1e-3 → 1e-4 over 30 epochs.
- **Mixed precision**: AMP autocast + GradScaler on CUDA.
- **Val metric**: NRMSE (sqrt of mean per-event MSE in normalized space) — directly comparable
  to the Kaggle NRMSE metric after denormalization with Kaggle sigmas.
- **Epoch 1 val**: combined=0.51, 1D=0.72, 2D=0.30 (cold start, no training yet comparable).

### Slurm
```bash
sbatch slurm/submit_slurm_model3.sh          # resume from checkpoints/model3/latest
sbatch slurm/submit_slurm_model3.sh scratch  # train from scratch
```

### Checkpoints
- Run checkpoints: `checkpoints/model3/Model_3_<timestamp>/`
- Best checkpoint (mirrored each time val improves): `checkpoints/model3/latest/Model_3_best.pt`

---

## Known Weaknesses

### 1. Spatial propagation freezes after t=10
The GNN runs for 10 steps then stops. Any flood dynamics that require spatial propagation
*after* t=10 — e.g. a wave travelling downstream over 50+ steps — cannot be captured. The
decoder predicts each node independently conditioned only on its own `h_enc` vector.

### 2. No temporal memory in decoder
Each future timestep is decoded independently. Prediction at t=50 is blind to prediction at
t=49. Real flood hydrographs have strong autocorrelation. The sinusoidal time embedding is a
weak proxy — it tells the MLP "you are at step 50 of 200" but carries no actual state.

### 3. h_enc must compress the entire event into 96 dims
A 96-dim vector must summarise what will happen over the next ~200–400 steps. For slow-
responding catchments or complex spatial dynamics, this bottleneck may be too tight.

### 4. Decoder sees no spatial context
Downstream nodes have no way of "knowing" what an upstream node predicted. The h_enc from
t=10 carries some upstream signal, but nothing propagates through the decoder.

---

## Ideas for Next Steps

### Quick wins (low risk, low cost)

**A. More encoder steps (history_len: 10 → 20)**
Gives h_enc more time to encode complex initial conditions. Each extra step is one more
GRU + message-passing round. Adds ~10% training cost, potentially meaningful h_enc improvement.

**B. Deeper decoder MLP (2 hidden layers, residual)**
Current: 1 hidden layer (128 units). Add a second hidden layer or residual connections.
Adds ~13K params per node type. Cheap, and may help model nonlinear temporal dynamics.

**C. Predict deltas, not absolutes**
Change decoder target from `water_level_t` to `water_level_t - water_level_{t=10}`.
Removes the DC offset problem — the MLP only needs to learn how much water level *changes*
from the known state at t=10. Likely the highest-value free change.

**D. Pass last-known water level as decoder input**
Add `water_level_{t=10}[node]` as a static feature to the decoder input:
`input = cat(h_enc, wl_at_t10, rain_t, time_emb)` → `[100]`.
Explicit anchor to the known initial condition at the end of the encoder.

---

### Architectural upgrades (medium cost)

**E. Temporal self-attention decoder**
Instead of an independent MLP per timestep, apply a Transformer encoder over the time axis:
```
decoder_in:  [N, T_future, h_dim + 1 + 2]   (h_enc broadcast + rain + time_emb)
→ Transformer(d_model=64, nhead=4, num_layers=2)
→ Linear → [N, T_future, 1]
```
Each timestep can attend to all others → captures autocorrelation, rising/falling limb,
lag effects. Still O(T²) attention but T≤400 and N nodes are independent → parallelisable.
This is probably the most natural improvement: adds temporal reasoning without recurrence.

**F. Periodic decoder GNN rounds (semi-autoregressive)**
Every K future steps (e.g. K=20), run one round of message passing using current predictions
as node features, updating all node states. Restores spatial propagation at low cost:
~T_future/K extra GNN rounds instead of T_future.

**G. Multi-step encoder output (not just final hidden state)**
Instead of only using h_{t=10}, concatenate or attend over h_{t=1..10}.
Gives the decoder access to the temporal trajectory of the basin, not just its endpoint.

---

### Generative / probabilistic decoder (high cost, high upside)

**H. Diffusion model decoder**
Frame the decoder as a conditional diffusion process:
- Condition: `h_enc` (basin state), `rain_future` (forcing), `time_emb`
- Denoising target: the full `[T_future, N, 1]` water level tensor
- Score network: U-Net or Transformer operating over (time × nodes)

**Pros**: naturally probabilistic — generates an ensemble of plausible futures.
Captures multimodal outcomes (e.g. "flood happens or it doesn't") that a point-estimate
MLP cannot. Calibrated uncertainty quantification essentially for free.

**Cons**: slow at inference (many denoising steps), complex to train, overkill if the
competition scoring is purely deterministic NRMSE.

**Realistic variant**: use a very small DDPM (10–20 denoising steps) with the score
network being a small Transformer conditioning on `h_enc + rain_future`. May be viable
if the Kaggle metric rewards calibration.

**I. Flow matching / consistency model**
Similar motivation to diffusion but trains with a single-step regression objective (flow
matching) and can be distilled to 1–4 inference steps. Much cheaper than DDPM at inference.
Still adds probabilistic expressiveness. Interesting middle ground.

---

## Recommendation for Next Experiment

Start with **C + D + B** (delta prediction + anchor water level input + deeper MLP) — all
three are 1–2 line changes and together address the biggest decoder weaknesses. Then run
a full 30-epoch training and compare val NRMSE vs the current baseline.

If that stalls, try **E** (temporal attention decoder) — it addresses the fundamental
independence-of-timesteps problem in a principled way with moderate added complexity.

Diffusion/flow matching is worth keeping in mind but is a large investment and should only
be pursued if the deterministic approaches plateau well above the competition leaderboard.
