# FloodLM — ML Next Steps

## Metric Alignment
- [x] **Fix 1D/2D loss weighting + σ alignment**: Replaced node-count-weighted loss with `(w_1d * loss_1d + w_2d * loss_2d) / 2` where `w = (our_range / kaggle_σ)²` (accounts for min-max normalization). Gives equal 50/50 weight per node type AND scales each term to match NRMSE metric. With ~10x more 2D nodes, this was a significant correction.
- [ ] **Switch training loss from MSE to RMSE**: The metric is RMSE not MSE. Minor for optimization direction but worth aligning once other changes are in.

## Model Evaluation
- [ ] **Multi-step autoregressive rollout evaluation**: Add fixed-horizon val pass (e.g. h=32 or 64) that unrolls autoregressively. Requires `forecast_len >= fixed_horizon` in the dataloader. Currently val uses `rollout_steps=1`.

## Training
- [ ] **Curriculum horizon training (top priority)**: Train with gradually increasing autoregressive rollout length to push the model past its 1-step plateau.

  **Implementation plan (uniform-per-batch sampling):**

  1. **Prerequisite — increase `forecast_len` in dataloader to 64** (or your target max). This is the critical infrastructure change; nothing else works without it.

  2. **Epoch → max_horizon schedule** — use `max_h = min(64, 2 ** (epoch // 3))` instead of fixed brackets. Doubles every 3 epochs, starts at 1, hits 64 at epoch 18. Adapts automatically to whatever total epoch count you set and doesn't need updating when you change `epochs`.
     - Original fixed-bracket suggestion was `{0-4: 1, 5-9: 8, 10-19: 16, 20-29: 32, 30+: 64}` — fine if you prefer explicit control, but the formula is simpler.
     - **Watch out**: with `epochs=4` + early stopping patience=3, you'd never leave `max_h=1`. Either increase epochs significantly (20+) or trigger horizon increases on *val plateau* rather than epoch number.

  3. **Per-batch rollout sampling**: `rollout_steps = min(h, y_future_1d.shape[1])` where `h ~ Uniform{1, ..., max_h}`. The cap is just `forecast_len` (all samples in a batch have the same future length from the dataloader), no per-sample min needed.

  4. **Keep `history_len=10`**. Call `model.forward_unroll(..., rollout_steps=rollout_steps)` — already matches deployment.

  5. **Loss**: `nn.MSELoss` already averages over time steps, so loss scale stays comparable across different `h` values. Apply the same σ-weighted 50/50 formula.

  6. **Log `rollout_steps` to wandb** each batch so you can see the horizon distribution during training.

  7. **Validation**: evaluate at a fixed horizon (e.g. h=32) AND h=1, so you can track both rollout performance and regression on the 1-step baseline.

## Architecture
- [ ] Investigate attention-based message passing (e.g., GAT) vs. current GNN conv layers
- [ ] **PE-GNN-style positional encodings** (try if autoregressive rollout struggles): add Laplacian eigenvector or random walk PE to encode structural roles (confluence nodes, end-of-chain nodes, catchment hierarchy) that 3D coordinates don't capture. Most likely to help with error propagation in long rollouts where upstream/downstream position in the network matters.

## Data
- [ ] Evaluate whether 20% val split is sufficient or if more events are needed
- [ ] Consider adding a held-out test set once the model architecture is finalized

## Infrastructure
- [ ] Profile data loading bottleneck — check whether `num_workers > 0` in DataLoader improves GPU utilization further
