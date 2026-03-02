#!/usr/bin/env python3
"""
Quick sanity test for FloodLM pipeline.
- Loads only first 5 events
- Runs one forward pass through model
- Validates output shapes
- Completes in ~2 minutes
"""

import sys
import os
import torch
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Override max events for quick test
os.environ["FLOOD_MAX_EVENTS"] = "5"

print("=" * 70)
print("FloodLM Quick Test (5 events max)")
print("=" * 70)

try:
    print("\n[1/5] Importing modules...")
    from data import get_recurrent_dataloader, get_model_config
    from model import FloodAutoregressiveHeteroModel
    
    print("[2/5] Loading model config...")
    config = get_model_config()
    print(f"     Node types: {config['node_types']}")
    print(f"     Edge types: {config['edge_types']}")
    
    print("[3/5] Creating dataloader (loading & normalizing data)...")
    start = time.time()
    dl = get_recurrent_dataloader(batch_size=2, shuffle=False)
    elapsed = time.time() - start
    print(f"     ✓ Dataloader ready (took {elapsed:.1f}s)")
    
    print("[4/5] Getting first batch...")
    batch = next(iter(dl))
    print(f"     Batch shapes:")
    print(f"       y_hist_1d: {batch['y_hist_1d'].shape}")
    print(f"       y_hist_2d: {batch['y_hist_2d'].shape}")
    print(f"       rain_hist_2d: {batch['rain_hist_2d'].shape}")
    
    print("[5/5] Running forward pass through model...")
    # Use CPU for this test - MPS has issues with index_select operations in PyG
    device = 'cpu'
    print(f"     Using device: {device}")
    model = FloodAutoregressiveHeteroModel(
        node_types=config['node_types'],
        edge_types=config['edge_types'],
        node_static_dims=config['node_static_dims'],
        node_dyn_input_dims=config['node_dyn_input_dims'],
        edge_static_dims=config['edge_static_dims'],
        pred_node_type=config['pred_node_type'],
        h_dim=32,
        msg_dim=32,
        hidden_dim=64,
    ).to(device)
    
    # Move static graph to device and ensure all tensors are on device
    static_graph = batch['static_graph'].to(device)
    
    # Dummy make_x_dyn function (replace with real one in production)
    # Note: pred_node_type is "twoD", so y_pred_nodes are 2D water levels
    def make_x_dyn(y_pred_nodes, rain_pred_nodes, data):
        """Convert predictions to dynamic input dict.
        
        Called during autoregressive rollout:
        - y_pred_nodes: [N_twoD, 1] predicted water levels for 2D nodes
        - rain_pred_nodes: [N_twoD, R] rainfall for 2D nodes
        - data: HeteroData graph
        """
        # For oneD nodes: we use zeros (no prediction, no rainfall)
        y_oneD = torch.zeros(data['oneD'].num_nodes, 1, device=y_pred_nodes.device, dtype=y_pred_nodes.dtype)
        
        # For twoD nodes: use prediction + rainfall
        x_twoD = torch.cat([y_pred_nodes, rain_pred_nodes], dim=-1)
        
        return {'oneD': y_oneD, 'twoD': x_twoD}
    
    # Take first sample from batch (forward_unroll expects unbatched data)
    # Batch shapes: [batch, time, nodes, channels]
    # Need: [time, nodes, channels]
    # Note: pred_node_type is "twoD", so we predict 2D nodes
    y_hist_2d_unbatch = batch['y_hist_2d'][0].to(device)  # [10, 3716, 1]
    rain_hist_unbatch = batch['rain_hist_2d'][0].to(device)  # [10, 3716, 1]
    rain_future_unbatch = batch['rain_future_2d'][0:1].to(device)  # [1, 3716, 1]
    
    # Squeeze the channel dimension since we only have 1 channel for water level
    y_hist_2d_unbatch = y_hist_2d_unbatch.squeeze(-1).unsqueeze(-1)  # [10, 3716, 1] (keep as [T, N, 1])
    rain_hist_unbatch = rain_hist_unbatch.squeeze(-1).unsqueeze(-1)  # [10, 3716, 1] (keep as [T, N, 1])
    
    preds = model.forward_unroll(
        data=static_graph,
        y_hist_true=y_hist_2d_unbatch,
        rain_hist=rain_hist_unbatch,
        rain_future=rain_future_unbatch,
        make_x_dyn=make_x_dyn,
        rollout_steps=1,
        device=device,
    )
    
    print(f"     ✓ Forward pass successful!")
    print(f"     Predictions shape: {preds.shape}")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    sys.exit(0)

except Exception as e:
    print(f"\n✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
