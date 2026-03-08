# FloodLM — ML Next Steps

Items are ordered roughly by expected impact-to-effort ratio (highest first).

---

## 🔥 Quick Wins / High Impact

### Feed aggregated rainfall to 1D nodes — fix information starvation ⭐ TOP PRIORITY
**Root cause**: `make_x_dyn` gives 2D nodes `[water_level, rainfall]` but 1D nodes only `[water_level]`. Rainfall is the primary causal driver, and 1D nodes are completely blind to it as direct input. The only path for rainfall to reach a 1D node is: `rainfall → 2D state → twoD→oneD message → 1D GRU` — a mandatory one-step lag that compounds over 64 rollout steps.

**Fix**: at each timestep, aggregate rainfall from each 1D node's neighboring 2D cells and inject it directly into the 1D dynamic input. Two options:

- **Simple mean**: `rain_1d = scatter_mean(r[ei[0]], ei[1], dim=0, dim_size=N_1d)` where `ei = data[('twoD','twoDoneD','oneD')].edge_index`. One line.
- **Distance-weighted mean** (preferred): use the `edges1d2d` connection data, which already exists and encodes the physical connectivity between 1D channels and 2D cells. Weight each neighbor's rainfall by `1/distance` (or a learned soft weight) before aggregating — gives closer cells more influence, matching the physical intuition that nearby floodplain cells contribute more runoff to a channel. The edge_index is already precomputed in the static graph; just need to attach distance as an edge weight during graph construction.

**Implementation changes**:
- `src/data.py` `get_model_config()`: `node_dyn_input_dims['oneD']` 1 → 2
- `src/train.py` both `make_x_dyn` lambdas: compute `rain_1d` via scatter and `cat([y['oneD'], rain_1d])`
- For distance-weighted: add distance edge weights to `data[('twoD','twoDoneD','oneD')]` at graph build time in `create_static_hetero_graph`
- **Requires training from scratch** (dyn_proj['oneD'] input dim changes)

---

### Fix Model_2 1D instability
**Diagnosis from logs (train_Model_2_20260302_223946.log):**
- Model_1 h=64 1D NRMSE: **0.0088–0.0123** (stable, converged)
- Model_2 h=64 1D NRMSE: **0.220–0.297** (oscillating, never converging) — ~25x worse
- Model_2 2D NRMSE at h=64: 0.081–0.097 — actually fine, not the problem
- Best combined estimate: Model_1=0.0097, Model_2=0.151 → **(0.0097+0.151)/2 = 0.080**
- Training loss at h=32 (epochs 11-12): batch spikes to 0.077, 0.083 — model never stabilized before jumping to h=64
- **Root cause**: curriculum advances too fast for Model_2 1D nodes; they diverge at each horizon jump and never recover before the next one

**Fixes applied:**
- [x] **1. Gradient clipping** — `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` added before `optimizer.step()` in both AMP and non-AMP paths. AMP path correctly calls `scaler.unscale_()` first. Applied to both models (no-op for Model_1 whose gradients are already small).
- [x] **2. LR reduction at curriculum jumps (Model_2 only)** — LR drops 3x (`lr *= 0.3`) at h=8, 32, 64 only. Schedule: `1e-3 → 3e-4 (h=8) → 9e-5 (h=32) → 2.7e-5 (h=64)`. Logged to wandb as `train/lr`.

**If instability persists — next to try:**
- [ ] **3. Slower curriculum for Model_2**: Change `2 ** ((epoch-1) // 2)` → `2 ** ((epoch-1) // 3)`, or increase epochs to 24 (3 per stage). Model_1 doesn't need this.
- [ ] **4. Separate LR for 1D head**: Use `param_groups` to give `model.heads['oneD']` a lower lr (e.g. 3e-4 vs 1e-3 for the rest).

**Infrastructure added:**
- `slurm/submit_slurm_model2.sh` — trains Model_2 **from scratch**. Use: `sbatch slurm/submit_slurm_model2.sh`

---

### Mask node 197 from Model_2 training loss
Node 197 is a confirmed data artifact: `depth=-1`, `base_area=0`, `surface_elevation < invert_elevation` (physically impossible), no 1d2d connection, and water level capped at a fixed boundary (~32m). It is permanently the worst-predicted node (mean RMSE=0.64m, present in top-20 worst in 12/15 evals) and its error is irreducible from the available features. Including it in the loss introduces misleading gradients that may harm the other 197 nodes.
- **Implementation**: create a `loss_mask_1d` boolean tensor of shape `[N_1d]` with node 197 set to False; apply before averaging the per-node MSE in the loss computation. Predictions are still generated for node 197 at inference (required for submission) — masking only affects gradient flow during training.
- **Caveat**: if node 197's boundary forcing occasionally provides useful signal to neighboring nodes via message passing, masking its loss could slightly hurt those neighbors. Monitor neighbor node RMSE before/after.

---

### Scheduled sampling — closes the teacher-forcing gap
Highest-leverage fix for val→Kaggle gap. During rollout, replace teacher-forced inputs with model predictions with probability `p_sample`, ramped from 0→~0.5 over the h=64 epochs. Trains the model to be robust to its own errors rather than always seeing ground truth. Implement in `forward_unroll` by mixing `y_true_t` and `y_pred_t` at each step based on a `sample_prob` argument passed from the train loop.

---

### Multi-step rollout loss — train on full autoregressive trajectory
Instead of computing loss only at the final predicted timestep (or uniformly across steps with teacher forcing), unroll the model autoregressively for the full horizon and compute loss at *every* step of the rollout using the model's own predictions as inputs. This directly trains the model on the distribution it will face at inference — the compounding error distribution — rather than the teacher-forced distribution.

**Why it helps**: teacher-forced training sees ground truth at every step, so the model never learns to recover from its own errors. Multi-step rollout loss closes the train/test mismatch caused by recursive prediction — empirically shown to yield 20–30% improvement in long-range R² on noisy benchmarks (Benechehab et al., 2024) and 3× reduction in max relative error in PDE-based reduced-order models (Stephany et al., 2025).

**Formal objective**: `L(θ) = Σⱼ αⱼ · E[‖s_{t+j} − f_θʲ(s_t)‖²]` where `f_θʲ` is the j-step rollout using model predictions as intermediate inputs and `αⱼ` are horizon weights. Setting `αⱼ ∝ βʲ` with `β > 1` upweights tail errors (pairs with timestep-weighted loss); `β < 1` emphasizes near-term errors. Normalization: `Σαⱼ = 1`.

**Bias–variance tradeoff**: one-step loss (α₁=1) is unbiased but high-variance at long horizons; multi-step loss introduces bias into one-step predictions but significantly reduces variance across the rollout. The optimal α depends on system noise — intermediate weighting provides the best overall tradeoff, especially as noise increases.

**Interaction with curriculum**: already partially addressed by the horizon curriculum (h=1→64), but within each curriculum stage all steps still use teacher forcing. True multi-step rollout loss removes teacher forcing *within* each stage. Consider annealing rollout horizon during early curriculum stages for stability (start short, extend as training stabilizes).

**Adaptive horizon scheduling**: rather than fixed αⱼ, sample rollout horizons randomly each batch (uniform or curriculum-guided). This encourages robustness across arbitrary prediction lengths, not just the fixed max_h.

**Cost**: BPTT through all rollout steps — pairs with gradient checkpointing (memory) and truncated BPTT / detach-every-K (gradient explosion). Overhead is training-only; inference is unchanged.

**Combine with scheduled sampling**: scheduled sampling is a softer version (mix true/predicted inputs with probability p); full rollout loss is the hard version. Recommended path: implement scheduled sampling first (lower risk), graduate to full rollout loss at h=64 once training is stable.

---

### Annealed timestep-weighted loss as horizon curriculum replacement
Instead of advancing `max_h` epoch-by-epoch, train at h=64 from epoch 1 but front-load the loss onto early timesteps and anneal toward uniform weighting over training. This eliminates gradient shocks from horizon jumps while still giving the model a structured learning signal.

**Schedule**: exponential decay weights `w_t = exp(-λ · t/T)`, annealed linearly to uniform (`λ=0`) over `anneal_epochs`:
```python
lam = lam_max * max(0.0, 1.0 - (epoch - 1) / anneal_epochs)
t_idx = torch.arange(T, device=device).float()
w_t = torch.exp(-lam * t_idx / T)          # [T]
w_t = w_t / w_t.sum()                       # normalize to sum=1
# Apply: sq_err [B,T,N,1] → weighted mean
loss = (sq_err.mean(dim=(0,2,3)) * w_t).sum()
```
- **`lam_max=3`**: at epoch 1, t=1 gets ~e^0=1 vs t=64 gets ~e^{-3}≈0.05 (20× front-loaded)
- **`anneal_epochs=15`**: fully uniform by epoch 16; remaining epochs see full h=64 loss
- **Advantages over horizon curriculum**: no gradient shock from horizon jumps; model always sees h=64 signal and learns global structure from epoch 1; smoother optimization landscape
- **Suggested total epochs**: ~20 (15 anneal + 5 uniform at h=64 with early stopping)
- **LR**: no need for LR reductions at curriculum jumps — single LR schedule (e.g. cosine decay or fixed 1e-3)
- Log `curriculum/lambda` and both weighted + unweighted loss to wandb to track annealing progress
- **Cost**: every epoch runs the full h=64 rollout (vs h=1 in early curriculum stages) — ~10-15× more compute per early epoch. Mitigate with truncated BPTT (detach every K=8 steps) or accept the cost if total epoch count is lower (20 vs 34)

### Timestep-weighted loss — penalize tail drift
Instead of uniform MSE over all rollout steps, weight later timesteps more heavily so the model prioritizes not drifting at the end of the rollout.
- Simple linear: `w_t = t / T` → tail gets T× more weight than step 1
- Exponential: `w_t = exp(α·t/T)` with α~1-2 → stronger tail emphasis
- Implementation: compute per-step MSE (keep `reduction='none'`), apply weights, sum/mean. Weights tensor of shape `[1, T, 1, 1]` broadcasts against `[B, T, N, 1]`. One-line change in the loss computation.
- Note: changes loss scale so may need lr adjustment; log both weighted and unweighted loss to wandb to track.
- **Combine with multi-step rollout loss**: tail-weighting is most meaningful when the tail steps are computed autoregressively (compounding error), not teacher-forced.

---

### Detach intermediate rollout steps — reduce gradient explosion at large h
For large rollout horizons (h=32, h=64), gradients backprop through all T steps and can explode or cause instability. Detach every K steps (e.g. K=8 or K=16) so gradients only flow within windows rather than through the full sequence. This is truncated BPTT.
- Implementation: in `forward_unroll`, call `hidden_state = hidden_state.detach()` (or `h.detach()`) every K rollout steps.
- K is a hyperparameter; K=8 is a reasonable start. Larger K = more accurate gradients but more risk of explosion.
- Pairs well with gradient clipping — use both together.
- Complements gradient checkpointing (which reduces memory but doesn't truncate gradients).

---

### Fix metric alignment: switch training loss from MSE to RMSE
The Kaggle metric is RMSE not MSE. Minor for optimization direction but worth aligning.

---

## Training Strategy

### Stability-gated curriculum (replace fixed epoch schedule)
Rather than advancing `max_h` on a fixed epoch clock, only jump to the next horizon once loss has stabilized.
- **Stability criterion**: track a rolling average of the last N batches' loss (e.g. N=50). Advance `max_h` when `|rolling_avg - prev_rolling_avg| < threshold` for M consecutive checks (e.g. M=3 checks spaced 50 batches apart).
- **Implementation sketch**: maintain a `deque(maxlen=50)` of recent batch losses; after every 50 batches compute `rolling_avg`; if it has changed by less than `stability_thresh` (e.g. 1% relative) for 3 consecutive checks, double `max_h`. Impose a minimum batches-per-stage floor to avoid lucky early jumps.
- **Safety ceiling**: max epoch budget so training doesn't run indefinitely.
- **LR interaction**: pair with LR reduction — when a jump is triggered, also apply `lr *= 0.3`.
- **Calibrate threshold from Model 1**: extract rolling loss CV from Model 1's stable inter-spike regions at each curriculum stage — that CV is the target Model 2 must hit before advancing.
- **Wandb logging**: log `curriculum/max_h` and `curriculum/rolling_loss` per check.

---

### Fine-tune final epochs on train+val data before Kaggle submission
Once the model is fully trained and evaluated, re-run a few final epochs with the validation events added back into training. This maximizes the data the final submission model has seen. Only do this after architecture and hyperparameters are locked in — no further tuning after this point.
- Keep a snapshot of the pre-fine-tune checkpoint for comparison.
- Suggested: 2–4 epochs at h=64, reduced LR (e.g. 1/3 of final training LR).

---

## Architecture

### ⭐ Replace GATv2CrossTypeMP with StaticDynamicEdgeMP + richer edge features for cross-type edges — TOP ARCHITECTURE PRIORITY
**Finding**: The best-performing submission (`Model_2_20260303_121200`, epoch 16, loss=0.028) used plain `StaticDynamicEdgeMP` for both `twoDoneD` and `oneDtwoD` edges. All subsequent runs that introduced `GATv2CrossTypeMP` (attention-based) have shown degrading validation performance at h=32+. The GATv2 attention is likely failing because the 2D→1D signal is structurally misleading for deeply incised channels (nodes 99, 85, 84, 132) where the connected 2D node sits 15–34m above the channel invert — the attention mechanism cannot overcome this with hidden states alone.

**Root cause of hard nodes (99, 85, 84, 132)**: These are deeply incised channels that fill from near-dry to near-full during flood events (15–34m dynamic range). Their connected 2D nodes sit 15–34m *above* the channel invert — the 2D hidden state encodes floodplain dynamics at 42–58m elevation while the channel fills from 23–27m. The 2D→1D messages are structurally irrelevant for these nodes; the GATv2 attention mechanism cannot overcome this because it only sees hidden states, not the physical elevation gap.

**Fix**: Revert both cross-type edge directions to `StaticDynamicEdgeMP`, but enrich the edge feature set so the MLP can learn to gate messages appropriately:
- **Inter-node distance** (Euclidean distance between 1D and 2D node positions) — encodes physical proximity of the connection
- **`channel_2d_elev_diff`** (connected 2D elevation − 1D invert elevation) — the strongest RMSE predictor found (r=0.58); this is the critical feature that lets the `base_weight` scalar and `dynamic_gate` MLP in `StaticDynamicEdgeMP` learn to suppress 2D→1D messages when the elevation gap is large

**Learned per-node suppression**: The `StaticDynamicEdgeMP` architecture is well-suited for this — its `base_weight = softplus(w^T u_e)` is a learned static coupling strength computed from edge+node static features. With `elev_diff` in the edge features, `base_weight` can learn to approach zero for large-gap connections, effectively silencing the 2D→1D message for deeply incised channels without any architectural changes beyond the edge feature. This is cleaner than a separate gating module because it's already end-to-end differentiable and tied to the physical feature.

**Implementation**:
- `src/model.py`: remove `GATv2CrossTypeMP` class; use `StaticDynamicEdgeMP` for `oneDtwoD` and `twoDoneD` in `HeteroTransportCell`
- `src/data.py` `create_static_hetero_graph()`: add `distance` and `elev_diff` as edge features to the 1d2d edge store (both directions; negate `elev_diff` for `oneDtwoD`)
- Update `edge_static_dims` accordingly in `get_model_config()`
- **Requires cache invalidation and training from scratch**

### GATv2Conv for 1D→2D connections
- [x] Added `GATv2CrossTypeMP` (4 heads) for `oneD→twoD` edges. Old `StaticDynamicEdgeMP` kept for all other edge types.
- [ ] **Revert** — see above priority item.

### ✅ Dual context nodes (ctx1d + ctx2d) + relative position embedding — IMPLEMENTED
Replace the single global context node with two domain-specific context nodes that communicate with each other. Each node's GRU input is augmented with the context hidden state concatenated with its own static features, giving each node a sense of "where it is" in the global context.

**Edge types** (all use GATv2CrossTypeMP, e_dim=1):
```
oneD  → ctx1d  → oneD     (1D domain context: aggregates all channels, broadcasts back)
twoD  → ctx2d  → twoD     (2D domain context: aggregates all floodplain, broadcasts back)
ctx1d → ctx2d             (channel context informs floodplain context)
ctx2d → ctx1d             (floodplain context informs channel context)
```

**Relative position embedding**: at each timestep, concatenate context hidden state(s) with each node's own static features before the GRU update — lets the model learn "given global state X and I am at position/elevation Y, how should I update?" This is the key difference from just receiving a context message via edges.

**Why dual > single global**: the single global node is dominated by 2D signal (~thousands of nodes vs ~200 1D nodes). Splitting ensures 1D nodes get a channel-network summary that isn't washed out by floodplain dynamics.

**After confirming this works**: try K=3-5 parallel context nodes per domain (replace ctx1d with ctx1d_0..ctx1d_K). GATv2 attention will learn soft regional assignments without hard boundaries. Check learned attention weights for spatial clustering — if present, emergent regions appear for free.

### Multiple learned context nodes per domain (K=3-5, soft regional embeddings)
Extension of dual context nodes. Replace single ctx1d/ctx2d with K parallel nodes each. No hard region boundaries — GATv2 attention over all K context nodes learns soft assignments. Nodes that are hydraulically similar will naturally attend to the same context node. Only try after dual context nodes are confirmed to help; inspect attention weights for spatial structure before adding more K.

### Investigate attention-based MP for 1D→1D and 2D→2D edges
Replace `StaticDynamicEdgeMP` with GAT-style attention for the within-type edges.

### PE-GNN-style positional encodings
Add Laplacian eigenvector or random walk PE to encode structural roles (confluence nodes, end-of-chain nodes, catchment hierarchy) that 3D coordinates don't capture. Most likely to help with error propagation in long rollouts where upstream/downstream position matters.

### Larger h_dim for 1D nodes — increase GRU memory capacity
Current `h_dim=64` means the GRU compresses `[dyn_emb(64) || msg_emb(64)]` → 64 floats of memory per node per step. For 2D cells this is likely adequate — dynamics are local and diffusive, neighbors tell you most of what you need. For 1D channels the hidden state has to carry much more longitudinal memory: flood wave timing, rise/recession shape, upstream propagation context — all in 64 floats over 64 steps.

- **Diagnostic signal**: after fixing the rainfall input, if 1D val loss plateaus while 2D keeps improving, that's the sign capacity is the constraint.
- **Option 1 — symmetric increase**: raise `h_dim` globally (64→128). Quadruples GRU parameter count; risks overfitting with limited training events.
- **Option 2 — type-specific h_dim** (preferred): give 1D nodes a larger hidden state (e.g. 128) while keeping 2D at 64. Requires making `h_dim` a per-node-type dict in `HeteroTransportCell`. Surgical and cost-efficient since the 1D graph is much smaller than 2D.
- Do this *after* the rainfall input fix so the two changes can be evaluated independently.

### Alternatives to GRU for temporal modeling
GRU compresses all history into a fixed hidden state. Alternatives:
- **Hybrid — GRU + local attention window**: keep GRU recurrence but add a short causal attention window over the last K hidden states (K=8) before the output head. Low additional cost, directly targets forgetting in long rollouts. Lowest-risk first experiment.
- **Mamba / State Space Models**: O(1) memory per step like GRU, but achieves transformer-quality long-range modeling. `mamba-ssm` provides a drop-in layer.
- **Linear attention**: approximates full softmax attention via kernel trick; O(1) memory, faster, less expressive.
- **Full self-attention over rollout history**: richest but O(T²) memory. Most suitable if memory is not a bottleneck.

### Transfer learning: Model_1 → Model_2 weight initialization
Use Model_1 weights to warm-start Model_2 training, giving it a strong prior on general hydraulic message-passing before it encounters Model_2's trickier 150-node 1D regime.

**Which checkpoint to use — h=4, NOT h=64:**
- Use the Model_1 checkpoint at the *end of the h=4 stage* (e.g. `Model_1_epoch_006.pt` from the dated run directory, or whichever epoch corresponds to the last h=4 epoch in the curriculum).
- **Do NOT use the h=64 best checkpoint.** By h=64, Model_1's weights are heavily specialized for 17-node, σ≈17m, ~300m-elevation dynamics. Transferring those to Model_2 (150 nodes, σ≈3m, different slope regime) means the GRU hidden state encoding and output head scaling are all miscalibrated — the model would need to unlearn them before it can start learning Model_2's regime, likely causing instability at early curriculum stages.
- At h=4, the shared message-passing primitives (flood wave propagation, edge convolution, GRU gating) are well-learned, but long-horizon specialization has not yet accumulated. This is the sweet spot for transfer.

**Implementation:**
- Load `Model_1_epoch_006.pt` (or equivalent) via `--resume`, then immediately switch to `SELECTED_MODEL=Model_2`.
- Heads and `dyn_proj` input dimensions differ between models if 1D input dim changes — either: (a) skip loading mismatched layers (`strict=False` + manual key filtering), or (b) keep dims identical for the initial transfer experiment.
- Use a **compressed curriculum** (`_stage_len=2` instead of 3) since mechanical GNN priors are pre-learned; the model only needs to adapt to Model_2's hydraulic scale and node count.
- Start at a reduced LR (e.g. 3e-4 instead of 1e-3) to avoid catastrophic forgetting of the shared primitives.

---

### Joint model training — shared trunk + model_id conditioning
Train a single model on both Model_1 and Model_2 data simultaneously.
- *Shared trunk* learns general hydro-transport primitives; 2x data → better generalization.
- *model_id embedding* (dim ~8–16) broadcast to all nodes and concatenated into each GRU input — lets trunk condition on hydraulic regime.
- *Model-specific heads* (`nn.ModuleDict`) handle the large scale difference (Model_1 σ ~17m vs Model_2 σ ~3m).
- *Optional*: replace model_id with event-level rainfall stats (mean/max/quantiles over nodes and time) — continuous, more informative, available at inference.

---

## Metric Alignment
- [x] **Fix 1D/2D loss weighting + σ alignment**: Replaced node-count-weighted loss with `(w_1d * loss_1d + w_2d * loss_2d) / 2` where `w = (our_range / kaggle_σ)²`. Significant correction (~10x more 2D nodes).
- [ ] **Switch training loss from MSE to RMSE**: Minor alignment fix.

---

## Model Evaluation
- [x] **Multi-step autoregressive rollout evaluation**: Added `evaluate_rollout()` — runs fixed-horizon val at h=1 and h=32. Logged as `rollout_val/h1_*` and `rollout_val/h32_*`.

---

## Data
- [ ] **Validate Model 2 data — raw and preprocessed**: Sanity-check for outliers, NaNs, unexpected value ranges, and whether flood event distribution is comparable to Model 1. The 1D loss spike magnitude (~10x worse) could partly be a data issue.
- [ ] Evaluate whether 20% val split is sufficient or if more events are needed.
- [ ] Consider adding a held-out test set once architecture is finalized.

---

## Infrastructure
- [ ] Profile data loading bottleneck — check whether `num_workers > 0` improves GPU utilization.
- [x] **Remove dead cumulative/mean water level features**: Removed `add_temporal_features()`, all Pass 2 loops, and `_process_event_for_engineered_pass`. `.cache/` deleted to force rebuild.
