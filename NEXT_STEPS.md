# FloodLM — ML Next Steps

## 🔥 TOP PRIORITY: Fix Model_2 1D instability

**Diagnosis from logs (train_Model_2_20260302_223946.log):**
- Model_1 h=64 1D NRMSE: **0.0088–0.0123** (stable, converged)
- Model_2 h=64 1D NRMSE: **0.220–0.297** (oscillating, never converging) — ~25x worse
- Model_2 2D NRMSE at h=64: 0.081–0.097 — actually fine, not the problem
- Best combined estimate: Model_1=0.0097, Model_2=0.151 → **(0.0097+0.151)/2 = 0.080**
- Training loss at h=32 (epochs 11-12): batch spikes to 0.077, 0.083 — model never stabilized before jumping to h=64
- **Root cause**: curriculum advances too fast for Model_2 1D nodes; they diverge at each horizon jump and never recover before the next one

**Fixes applied:**

- [x] **1. Gradient clipping** — `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` added before `optimizer.step()` in both AMP and non-AMP paths. AMP path correctly calls `scaler.unscale_()` first. Applied to both models (no-op for Model_1 whose gradients are already small).

- [x] **2. LR reduction at curriculum jumps (Model_2 only)** — LR drops 3x (`lr *= 0.3`) at h=8, 32, 64 only (not every jump, to preserve learning capacity at h=64). Gated on `SELECTED_MODEL == 'Model_2'`. LR schedule: `1e-3 → 3e-4 (h=8) → 9e-5 (h=32) → 2.7e-5 (h=64)`. Logged to wandb as `train/lr`. Note: current run (job 275441+) is a hybrid — LR was also reduced at h=2 and h=4 under the old logic, so effective LR is lower than intended through h=8.

**If instability persists after the above — next to try:**

- [ ] **3. Slower curriculum for Model_2**:
  Change `2 ** ((epoch-1) // 2)` → `2 ** ((epoch-1) // 3)` in train.py when running Model_2, or increase epochs to 24 (3 per stage). Model_1 doesn't need this.

- [ ] **4. Separate LR for 1D head**:
  Use `param_groups` to give `model.heads['oneD']` a lower lr (e.g. 3e-4 vs 1e-3 for the rest). Most targeted fix if the output head is still the unstable component after clipping.

**Infrastructure added:**
- `slurm/submit_slurm_model2.sh` — trains Model_2 **from scratch** (no resume — don't inherit unstable weights). Use: `sbatch slurm/submit_slurm_model2.sh`

## Metric Alignment
- [x] **Fix 1D/2D loss weighting + σ alignment**: Replaced node-count-weighted loss with `(w_1d * loss_1d + w_2d * loss_2d) / 2` where `w = (our_range / kaggle_σ)²` (accounts for min-max normalization). Gives equal 50/50 weight per node type AND scales each term to match NRMSE metric. With ~10x more 2D nodes, this was a significant correction.
- [ ] **Switch training loss from MSE to RMSE**: The metric is RMSE not MSE. Minor for optimization direction but worth aligning once other changes are in.

## Model Evaluation
- [x] **Multi-step autoregressive rollout evaluation**: Added `evaluate_rollout()` — runs fixed-horizon val at h=1 (always) and h=32 (once `max_h` reaches 32). Logged as `rollout_val/h1_*` and `rollout_val/h32_*` in wandb.

## Training
- [ ] **Scheduled sampling** (highest leverage for closing val→Kaggle gap): During the rollout, replace teacher-forced inputs with model predictions with probability `p_sample`, ramped from 0→~0.5 over the h=64 epochs. Trains the model to be robust to its own errors rather than always seeing ground truth. Implement in `forward_unroll` by mixing `y_true_t` and `y_pred_t` at each step based on a `sample_prob` argument passed from the train loop.

- [ ] **Learning rate reduction at curriculum jumps (Model 2)**: Rather than LR warmup, consider dropping LR by ~3-5x at the h=8 and h=32 boundaries specifically for Model 2 — the gradient report shows these are the stages where instability compounds. Could be as simple as a step LR scheduler keyed to `max_h` thresholds: `if max_h != prev_max_h and max_h in {8, 32, 64}: lr *= 0.3`. This is cleaner than warmup since the model has already learned at the previous horizon and just needs a calmer adjustment rather than a full reset.

- [ ] **Gradient clipping for Model 2**: The gradient report shows spikes are almost entirely localized to `heads.oneD.2` (±0.4) and `cell.dyn_proj.twoD` (±0.05). The message passing layers (`mp_mod`) are well-behaved throughout. Adding `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` would have real impact for Model 2. Alternatively, clip just the output head parameters for a more targeted fix. Note: the very small 2D mp_mod gradients (~1e-4) at h=64 may indicate the 2D message passing is undertraining — worth watching 2D val loss separately.

- [ ] **Timestep-weighted loss** (addresses error accumulation at the tail): Instead of uniform MSE over all rollout steps, weight later timesteps more heavily so the model prioritizes getting the tail right.
  - Simple linear: `w_t = t / T` → tail gets T× more weight than step 1
  - Exponential: `w_t = exp(α·t/T)` with α~1-2 → stronger emphasis
  - Implementation: compute per-step MSE (keep `reduction='none'`), apply weights, sum/mean. One line change in the loss computation — weights tensor of shape `[1, T, 1, 1]` broadcast against `[B, T, N, 1]`.
  - Note: changes loss scale so may need lr adjustment; log both weighted and unweighted loss to wandb to track.

- [x] **Curriculum horizon training**: Implemented. `forecast_len=64`, `max_h = min(64, 2**((epoch-1)//_stage_len))` where `_stage_len=3` for Model_2 (slower) and `2` for Model_1. `epochs=24` (3 per stage × 8 stages for Model_2). Horizon-stratified wandb logging.

- [ ] **Stability-gated curriculum (replace fixed epoch schedule)**: Rather than advancing `max_h` on a fixed epoch clock, only jump to the next horizon once the loss at the current horizon has stabilized. Model 2 1D/2D losses are visibly unstable at h=4 and much worse at h=8 — the model is never converging before the next jump, compounding instability across stages. A fixed slower schedule (e.g. `// 4` instead of `// 2`) is better than current but still arbitrary.
  - **Stability criterion**: track a rolling average of the last N batches' loss (e.g. N=50). Advance `max_h` when `|rolling_avg - prev_rolling_avg| < threshold` for M consecutive checks (e.g. M=3 checks spaced 50 batches apart). This is essentially the same logic as early stopping but applied per curriculum stage.
  - **Implementation sketch**: maintain a `deque(maxlen=50)` of recent batch losses; after every 50 batches compute `rolling_avg`; if `rolling_avg` has changed by less than `stability_thresh` (e.g. 1% relative) for 3 consecutive checks, double `max_h`; also impose a minimum number of batches per stage (e.g. 200) to avoid jumping too early on a lucky streak.
  - **Safety ceiling**: still set a max epoch budget so training doesn't run indefinitely if a stage never fully stabilizes.
  - **LR interaction**: pair this with the LR reduction idea — when a jump is triggered, also apply the `lr *= 0.3` step-down so the model enters the new horizon with a calmer update size.
  - **Wandb logging**: log `curriculum/max_h` and `curriculum/rolling_loss` per check so you can visually verify the gating is working as intended.
  - **Calibrate threshold from Model 1**: Rather than guessing the stability threshold, extract it empirically from Model 1's training run. Compute the rolling loss CV (std/mean over the last 50 batches) in Model 1's stable inter-spike regions at each curriculum stage — that CV value is the target Model 2 must hit before advancing. Model 1's post-jump recovery takes ~100-200 batches and the baseline stays flat between spikes; if Model 2 hasn't returned to a comparable CV within that window, it hasn't converged. This grounds the threshold in observed behavior rather than an arbitrary constant.

  **Implementation plan (uniform-per-batch sampling):**

  1. **Prerequisite — increase `forecast_len` in dataloader to 64** (or your target max). This is the critical infrastructure change; nothing else works without it.

  2. **Epoch → max_horizon schedule** — use `max_h = min(64, 2 ** (epoch // 3))` instead of fixed brackets. Doubles every 3 epochs, starts at 1, hits 64 at epoch 18. Adapts automatically to whatever total epoch count you set and doesn't need updating when you change `epochs`.
     - Original fixed-bracket suggestion was `{0-4: 1, 5-9: 8, 10-19: 16, 20-29: 32, 30+: 64}` — fine if you prefer explicit control, but the formula is simpler.
     - **Watch out**: with `epochs=4` + early stopping patience=3, you'd never leave `max_h=1`. Either increase epochs significantly (20+) or trigger horizon increases on *val plateau* rather than epoch number.

  3. **Per-batch rollout sampling**: `rollout_steps = min(h, y_future_1d.shape[1])` where `h ~ Uniform{1, ..., max_h}`. The cap is just `forecast_len` (all samples in a batch have the same future length from the dataloader), no per-sample min needed.

  4. **Keep `history_len=10`**. Call `model.forward_unroll(..., rollout_steps=rollout_steps)` — already matches deployment.

  5. **Loss**: `nn.MSELoss` already averages over time steps, so loss scale stays comparable across different `h` values. Apply the same σ-weighted 50/50 formula.

  6. **Horizon-stratified wandb logging** — two complementary signals:
     - `curriculum/rollout_steps` (scalar per batch): lets you verify the horizon distribution is uniform and advancing as expected.
     - `loss/train_h{rollout_steps}` (e.g. `loss/train_h1`, `loss/train_h8`, `loss/train_h32`): log the batch loss under a horizon-specific key so wandb shows separate curves per rollout length. This answers "is h=32 loss improving while h=1 stays flat?" — the key curriculum training signal. Simple to implement: `wandb.log({f"loss/train_h{rollout_steps}": loss.item(), "curriculum/rollout_steps": rollout_steps})`.

  7. **Validation**: evaluate at a fixed horizon (e.g. h=32) AND h=1, so you can track both rollout performance and regression on the 1-step baseline.

## Architecture
- [x] **GATv2Conv for 1D→2D connections**: Added `GATv2CrossTypeMP` (4 heads, `msg_dim//heads` per head) for `oneD→twoD` edges in `model.py`. Edge type added to `create_static_hetero_graph` and `get_model_config`. Old `StaticDynamicEdgeMP` kept for all other edge types.
- [ ] Investigate attention-based message passing (e.g., GAT) for 1D→1D and 2D→2D edges
- [ ] **PE-GNN-style positional encodings** (try if autoregressive rollout struggles): add Laplacian eigenvector or random walk PE to encode structural roles (confluence nodes, end-of-chain nodes, catchment hierarchy) that 3D coordinates don't capture. Most likely to help with error propagation in long rollouts where upstream/downstream position in the network matters.
- [ ] **Joint model training — shared trunk + model_id conditioning + model-specific heads**: Train a single model on both Model_1 and Model_2 data simultaneously.
  - *Shared trunk* (`HeteroTransportCell`) learns general hydro-transport primitives; 2x data means better generalization.
  - *model_id embedding* (learnable vector, dim ~8–16) broadcast to all nodes and concatenated into each GRU input at every step — lets the trunk condition on which hydraulic regime it's in without separate weights.
  - *Model-specific heads* (`nn.ModuleDict` keyed by model_id) handle the large scale difference (Model_1 σ ~17m vs Model_2 σ ~3m).
  - *Optional richer conditioning*: replace or augment model_id with event-level rainfall stats computed from `rain_hist` (mean/max/quantiles over nodes and time) — continuous, more informative, and available at inference.
  - Key challenge: graphs have different node counts and topology, so batching requires routing each sample through its model's static graph; loss weighting already handles the σ difference.

## Data
- [ ] **Validate Model 2 data — raw and preprocessed**: Given the training instability, sanity-check the data at both stages. Raw: check for outliers, NaNs, unexpected value ranges, and whether the distribution of flood events is comparable to Model 1. Preprocessed: verify normalization stats (min/max per feature), check that the sigma-weighted loss scaling is correct for Model 2's actual value ranges, and confirm no data leakage between train/val splits. The 1D loss spike magnitude (~10x worse than Model 1) could partly be a data issue rather than purely a model/training issue.
- [ ] Evaluate whether 20% val split is sufficient or if more events are needed
- [ ] Consider adding a held-out test set once the model architecture is finalized

## Infrastructure
- [ ] Profile data loading bottleneck — check whether `num_workers > 0` in DataLoader improves GPU utilization further
- [x] **Remove dead cumulative/mean water level features from data pipeline**: Removed `add_temporal_features()`, all Pass 2 normalization loops, and `_process_event_for_engineered_pass` from `data_lazy.py` and `data.py`. `.cache/` deleted to force rebuild.
