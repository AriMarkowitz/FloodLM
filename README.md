# FloodLM - Recurrent Flood Model with Static-Dynamic Separation

This folder contains the updated flood prediction model with a new architecture that separates static and dynamic features.

## Architecture Overview

### Key Changes from Previous Version

1. **Static Graph Structure**: The model now uses a single static heterogeneous graph containing only time-invariant features (topology, elevation, slopes, etc.)

2. **Dynamic Time Series**: Water levels and rainfall are passed as separate time series tensors rather than being baked into the graph

3. **Recurrent Processing**: The model processes time by maintaining hidden states that are updated at each timestep, rather than creating temporal graphs

4. **Autoregressive Rollout**: Predictions are made autoregressively by feeding predicted water levels forward at each step

## Data Model

### Static Graph (`HeteroData`)
```python
data["oneD"].x_static       # [N_1d, static_dim_1d] static features for 1D nodes
data["twoD"].x_static       # [N_2d, static_dim_2d] static features for 2D nodes
data["oneD", "oneDedge", "oneD"].edge_index         # Edge connectivity
data["oneD", "oneDedge", "oneD"].edge_attr_static   # Static edge features
# ... similar for other edge types
```

### Dynamic Time Series
```python
y_hist_1d:     [H, N_1d, 1]  # Historical water levels for 1D nodes
y_hist_2d:     [H, N_2d, 1]  # Historical water levels for 2D nodes
rain_hist_2d:  [H, N_2d, 1]  # Historical rainfall
rain_future_2d:[T, N_2d, 1]  # Future rainfall forecast
```

## Key Functions

### 1. `create_static_hetero_graph()`
Creates a single static graph with node and edge features that don't change over time.

```python
static_graph = create_static_hetero_graph(
    static_1d_norm, static_2d_norm,
    edges1d, edges2d, edges1d2d,
    edges1dfeats_norm, edges2dfeats_norm,
    static_1d_cols, static_2d_cols,
    edge1_cols, edge2_cols,
)
```

### 2. `make_x_dyn()`
Constructs the dynamic input dictionary at each timestep. This function is called by the model to inject dynamic data into the recurrent cell.

```python
def make_x_dyn(
    y_pred_1d: torch.Tensor,    # [N_1d, 1]
    y_pred_2d: torch.Tensor,    # [N_2d, 1]
    rain_2d: torch.Tensor,      # [N_2d, R]
    data: HeteroData,
) -> dict[str, torch.Tensor]:
    """
    Returns:
        x_dyn["oneD"]: [N_1d, dyn_dim_1d] 
        x_dyn["twoD"]: [N_2d, dyn_dim_2d]
    """
```

### 3. `RecurrentFloodDataset`
New dataset class that yields static graph + dynamic time series for training.

```python
dataset = RecurrentFloodDataset(
    event_file_list=event_file_list,
    static_1d_norm=static_1d_sorted,
    static_2d_norm=static_2d_sorted,
    # ... other parameters
    history_len=10,
    forecast_len=1,
    shuffle=True,
)
```

### 4. `get_recurrent_dataloader()`
Convenience function to create a dataloader for the recurrent model.

```python
dataloader = get_recurrent_dataloader(
    history_len=10,
    forecast_len=1,
    batch_size=8,
    shuffle=True,
)
```

### 5. `get_model_config()`
Returns configuration dictionary for model initialization.

```python
config = get_model_config()
# Returns:
# {
#     'node_types': ['oneD', 'twoD'],
#     'edge_types': [('oneD', 'oneDedge', 'oneD'), ...],
#     'node_static_dims': {'oneD': 10, 'twoD': 15},
#     'node_dyn_input_dims': {'oneD': 1, 'twoD': 2},
#     'edge_static_dims': {...},
#     'pred_node_type': 'twoD'
# }
```

## Model Architecture

### `FloodAutoregressiveHeteroModel`

The model consists of:

1. **HeteroTransportCell**: Recurrent cell that processes the graph at each timestep
   - Uses static features via message passing
   - Uses dynamic features via input projection
   - Maintains hidden states as memory

2. **StaticDynamicEdgeMP**: Message passing that combines:
   - Static coupling (learned from edge/node static features)
   - Dynamic gating (based on current hidden states)
   - Message payload (information transfer)

3. **Prediction Head**: Decodes hidden state to predicted water level

### Training Loop

```python
predictions = model.forward_unroll(
    data=static_graph,              # Static graph (shared)
    y_hist_true=y_hist[H, N, 1],   # Historical water levels
    rain_hist=rain_hist[H, N, R],   # Historical rainfall
    rain_future=rain_future[T, N, R],  # Future rainfall
    make_x_dyn=make_x_dyn_fn,       # Dynamic input function
    rollout_steps=forecast_len,
    device=device,
)

loss = criterion(predictions, ground_truth)
```

### Curriculum Learning

Rather than training directly on the full 64-step autoregressive rollout from epoch 1 (which causes gradient explosion and unstable early training), we use an **exponential horizon curriculum**: the rollout length doubles every few epochs, starting at 1 step and reaching 64 by the final training stage.

This was a deliberate design choice — autoregressive models are notoriously hard to train end-to-end at long horizons because errors compound and gradients vanish or explode through many unrolled steps. The curriculum lets the model first learn short-range hydraulic dynamics well, then gradually extend its temporal reach.

```
Model_1: h=1 (ep 1-2) → h=2 (3-4) → h=4 (5-6) → h=8 (7-8) → h=16 (9-10) → h=32 (11-12) → h=64 (13-24)
Model_2: h=1 (ep 1-3) → h=2 (4-6) → h=4 (7-9) → h=8 (10-12) → h=16 (13-15) → h=32 (16-18) → h=64 (19-24)
```

Model_2 uses a longer stage length (3 epochs per stage vs 2) because its domain is harder (smaller Kaggle sigmas = tighter tolerances). See `ARCHITECTURE.md` for full details.

### Training Order

When running both models (`pipeline.sh all`), **Model_2 trains first**. This is intentional — Model_2 is harder (higher-resolution domain, stricter Kaggle metric) and we want early visibility into its performance during a run.

## Usage

See `example_usage.py` for a complete example showing:
- Data loading
- Model initialization
- Training loop
- Inference

Run the example:
```bash
cd FloodLM
python example_usage.py
```

## Files

- `src/data.py`: Data loading and preprocessing
  - `create_static_hetero_graph()`: Static graph builder
  - `make_x_dyn()`: Dynamic input constructor
  - `RecurrentFloodDataset`: Dataset class
  - `get_recurrent_dataloader()`: Dataloader factory
  - `get_model_config()`: Model config helper

- `src/model.py`: Model architecture
  - `StaticDynamicEdgeMP`: Message passing module
  - `HeteroTransportCell`: Recurrent graph cell
  - `FloodAutoregressiveHeteroModel`: Full model

- `example_usage.py`: Complete usage example

## Benefits of New Architecture

1. **Efficiency**: Static graph is created once and reused
2. **Clarity**: Clear separation between static structure and dynamic processes
3. **Flexibility**: Easy to plug in different dynamic features
4. **Scalability**: Recurrent processing handles arbitrary sequence lengths
5. **Interpretability**: Static coupling vs. dynamic gating are explicit

## Migration from Old Architecture

The old architecture created temporal graphs with all features combined. To use the new architecture:

1. Use `get_recurrent_dataloader()` instead of `get_dataloader()`
2. Initialize model with `get_model_config()`
3. Use `forward_unroll()` method for training
4. Provide `make_x_dyn` function to inject dynamic data

The old data loading functions (`get_dataloader()`, `MultiEventGraphStream`) remain available for backward compatibility.
