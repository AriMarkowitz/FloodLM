#!/usr/bin/env python
"""Inference script that loads saved model and applies cached normalization statistics."""

import os
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, 'src')

from model import FloodAutoregressiveHeteroModel
from data import get_recurrent_dataloader

def load_normalization_stats(stats_path):
    """Load saved normalization statistics."""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    print(f"[INFO] Loaded normalization statistics from {stats_path}")
    return stats

def load_checkpoint(checkpoint_path, device):
    """Load trained model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Rebuild model with saved config
    model = FloodAutoregressiveHeteroModel(
        node_types=config.get('node_types', ['oneD', 'twoD']),
        edge_types=config.get('edge_types', []),
        static_dims=config.get('static_dims', {}),
        dynamic_dims=config.get('dynamic_dims', {}),
        edge_static_dims=config.get('edge_static_dims', {}),
        h_dim=config.get('h_dim', 64),
        msg_dim=config.get('msg_dim', 64),
        predict_delta=config.get('predict_delta', True),
        predict_node_type=config.get('predict_node_type', 'twoD'),
    )
    
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    
    print(f"[INFO] Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")
    
    return model

def run_inference(model, checkpoint_dir, num_samples=5):
    """Run inference on validation data."""
    device = next(model.parameters()).device
    
    # Load dataloader
    print(f"\n[INFO] Loading validation data...")
    dataloader = get_recurrent_dataloader(
        history_len=10,
        forecast_len=1,
        batch_size=4,
        shuffle=False,
    )
    
    print(f"[INFO] Running inference on {num_samples} samples...\n")
    
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for sample_idx, batch in enumerate(dataloader):
            if sample_idx >= num_samples:
                break
            
            if batch is None:
                continue
            
            static_graph = batch['static_graph'].to(device)
            y_hist_2d = batch['y_hist_2d'].to(device)
            rain_hist_2d = batch['rain_hist_2d'].to(device)
            y_future_2d = batch['y_future_2d'].to(device)
            rain_future_2d = batch['rain_future_2d'].to(device)
            
            # Forward pass
            predictions = model.forward_unroll(
                data=static_graph,
                y_hist_true=y_hist_2d,
                rain_hist=rain_hist_2d,
                rain_future=rain_future_2d,
                make_x_dyn=lambda y, r, data: {
                    'oneD': torch.zeros((data['oneD'].num_nodes, 1), device=device),
                    'twoD': torch.cat([y, r], dim=-1),
                },
                rollout_steps=1,
                device=device,
            )
            
            # Compute MSE
            mse = torch.mean((predictions - y_future_2d) ** 2)
            total_loss += mse.item()
            
            # Print sample stats
            print(f"Sample {sample_idx+1}:")
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Targets shape: {y_future_2d.shape}")
            print(f"  MSE: {mse.item():.6f}")
            print(f"  Pred range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            print(f"  True range: [{y_future_2d.min():.4f}, {y_future_2d.max():.4f}]")
            print()
    
    avg_loss = total_loss / num_samples
    print(f"[INFO] Average MSE over {num_samples} samples: {avg_loss:.6f}")

def main():
    """Main inference flow."""
    print("\n" + "="*70)
    print("FloodLM Inference Script")
    print("="*70 + "\n")
    
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint_dir = 'checkpoints'
    
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint directory: {checkpoint_dir}\n")
    
    # Find best checkpoint
    best_checkpoint = os.path.join(checkpoint_dir, 'model_best.pt')
    if not os.path.exists(best_checkpoint):
        print(f"[ERROR] Best checkpoint not found: {best_checkpoint}")
        print(f"[INFO] Available files in {checkpoint_dir}:")
        if os.path.exists(checkpoint_dir):
            for f in sorted(os.listdir(checkpoint_dir)):
                print(f"  - {f}")
        return
    
    # Load normalization stats
    stats_file = os.path.join(checkpoint_dir, 'normalization_stats.json')
    if os.path.exists(stats_file):
        norm_stats = load_normalization_stats(stats_file)
        print(f"  1D nodes: {len(norm_stats['node1d_cols'])} features")
        print(f"  2D nodes: {len(norm_stats['node2d_cols'])} features\n")
    else:
        print(f"[WARN] Normalization stats not found: {stats_file}")
        print(f"[INFO] Available files in {checkpoint_dir}:")
        for f in sorted(os.listdir(checkpoint_dir)):
            print(f"  - {f}")
    
    # Load model
    print(f"[INFO] Loading model...")
    model = load_checkpoint(best_checkpoint, device)
    
    # Run inference
    run_inference(model, checkpoint_dir, num_samples=5)
    
    print("\n" + "="*70)
    print("Inference Complete")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
