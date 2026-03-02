# FloodLM — ML Next Steps

## Metric Alignment
- [ ] **Fix 1D/2D loss weighting (high priority)**: Current training loss weights 1D and 2D contributions by node count. The Kaggle metric averages 1D and 2D node-type scores equally (50/50) regardless of node count. Change training loss to `(loss_1d + loss_2d) / 2` to match. This is a one-line fix with potentially significant impact.
- [ ] **Align standardization constants**: The metric standardizes RMSE by 4 fixed global σ values per `(model_id, node_type)`: `{(1,1): 16.878, (1,2): 14.379, (2,1): 3.192, (2,2): 2.727}`. Consider computing our normalization σ values to match, or scaling the loss terms accordingly during training.
- [ ] **Switch training loss from MSE to RMSE**: The metric is RMSE not MSE. Minor for optimization direction but worth aligning once other changes are in.

## Model Evaluation
- [ ] **Multi-step autoregressive rollout evaluation**: Add a separate evaluation pass that unrolls the model autoregressively for N steps (e.g., 10 or 24), feeding predictions back as inputs. This stress-tests error accumulation and gives a more realistic measure of real-world performance. Currently both training and validation use `rollout_steps=1` (1-step-ahead only).

## Training
- [ ] Experiment with longer `forecast_len` (multi-step training target) once baseline 1-step model converges
- [ ] Explore curriculum learning: start with 1-step rollout, gradually extend rollout horizon during training

## Architecture
- [ ] Investigate attention-based message passing (e.g., GAT) vs. current GNN conv layers

## Data
- [ ] Evaluate whether 20% val split is sufficient or if more events are needed
- [ ] Consider adding a held-out test set once the model architecture is finalized

## Infrastructure
- [ ] Profile data loading bottleneck — check whether `num_workers > 0` in DataLoader improves GPU utilization further
