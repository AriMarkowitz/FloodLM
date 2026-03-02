#!/usr/bin/env python
"""
Inference script for FloodLM model with two modes:
1. Teacher Forcing: Use real water levels at each step (best-case one-step-ahead prediction)
2. Autoregressive: Use predicted water levels for multi-step rollout (real-world scenario)

Usage:
    # Teacher forcing mode (one-step ahead with real data)
    python inference.py --checkpoint checkpoints/model_best.pt --mode teacher --split test
    
    # Autoregressive mode (multi-step with predictions)
    python inference.py --checkpoint checkpoints/model_best.pt --mode autoregressive --rollout_steps 50 --split test
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Setup paths
sys.path.insert(0, 'src')

from data import get_recurrent_dataloader, get_model_config, unnormalize_col
from model import FloodAutoregressiveHeteroModel
from data_lazy import initialize_data


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    model_config = get_model_config()
    
    # Initialize model
    model = FloodAutoregressiveHeteroModel(**model_config)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Loaded model from epoch {checkpoint['epoch']}")
    print(f"[INFO] Training loss: {checkpoint['loss']:.6f}")
    
    return model, checkpoint


def inference_teacher_forcing(
    model, 
    dataloader, 
    norm_stats,
    device,
    max_samples=None,
    denormalize=True,
):
    """
    Teacher forcing inference: Use real water levels at each step to predict one step ahead.
    This shows best-case performance when the model has access to perfect historical data.
    
    For each window, predict t+1 from [t-9, t-8, ..., t].
    """
    model.eval()
    
    predictions_all = []
    targets_all = []
    metadata_all = []
    
    print(f"\n[INFO] Running TEACHER FORCING inference...")
    print(f"[INFO] Mode: One-step-ahead prediction with real historical data")
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            static_graph = batch['static_graph'].to(device)
            y_hist_2d = batch['y_hist_2d'].to(device)  # [B, H, N, 1]
            rain_hist_2d = batch['rain_hist_2d'].to(device)
            y_future_2d = batch['y_future_2d'].to(device)  # [B, T, N, 1]
            rain_future_2d = batch['rain_future_2d'].to(device)
            
            batch_size = y_hist_2d.size(0)
            forecast_len = y_future_2d.size(1)  # T timesteps available
            
            # Process each sample in batch
            for i in range(batch_size):
                if max_samples is not None and sample_count >= max_samples:
                    break
                
                # For teacher forcing: predict each timestep independently using real history
                sample_predictions = []
                sample_targets = []
                
                # For each future timestep, use real history up to that point
                for t in range(forecast_len):
                    # History: original history + real future up to t-1
                    if t == 0:
                        history = y_hist_2d[i]  # [H, N, 1]
                        rain_history = rain_hist_2d[i]
                    else:
                        # Concatenate original history with real future [:t]
                        history = torch.cat([y_hist_2d[i], y_future_2d[i, :t]], dim=0)  # [H+t, N, 1]
                        rain_history = torch.cat([rain_hist_2d[i], rain_future_2d[i, :t]], dim=0)
                        
                        # Keep only last H timesteps
                        history = history[-10:]
                        rain_history = rain_history[-10:]
                    
                    # Predict one step ahead
                    pred = model.forward_unroll(
                        data=static_graph,
                        y_hist_true=history,
                        rain_hist=rain_history,
                        rain_future=rain_future_2d[i, t:t+1],  # Just next rainfall
                        make_x_dyn=lambda y, r, data: {
                            'oneD': torch.zeros((data['oneD'].num_nodes, 1), device=device),
                            'twoD': torch.cat([y, r], dim=-1),
                        },
                        rollout_steps=1,
                        device=device,
                    )
                    
                    sample_predictions.append(pred[0])  # [N, 1]
                    sample_targets.append(y_future_2d[i, t])  # [N, 1]
                
                # Stack all predictions for this sample
                pred_np = torch.stack(sample_predictions, dim=0).cpu().numpy()  # [T, N, 1]
                target_np = torch.stack(sample_targets, dim=0).cpu().numpy()
                
                # Denormalize if requested
                if denormalize:
                    pred_denorm = []
                    target_denorm = []
                    for t in range(forecast_len):
                        pred_t = torch.from_numpy(pred_np[t]).to(device)
                        target_t = torch.from_numpy(target_np[t]).to(device)
                        
                        pred_t_denorm = unnormalize_col(pred_t, norm_stats, col=0, node_type='twoD')
                        target_t_denorm = unnormalize_col(target_t, norm_stats, col=0, node_type='twoD')
                        
                        pred_denorm.append(pred_t_denorm.cpu().numpy())
                        target_denorm.append(target_t_denorm.cpu().numpy())
                    
                    pred_np = np.stack(pred_denorm, axis=0)
                    target_np = np.stack(target_denorm, axis=0)
                
                predictions_all.append(pred_np)
                targets_all.append(target_np)
                metadata_all.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'mode': 'teacher_forcing',
                    'timesteps': forecast_len,
                    'n_nodes': pred_np.shape[1],
                })
                
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    break
            
            if (batch_idx + 1) % 10 == 0:
                print(f"[INFO] Processed {batch_idx + 1} batches, {len(predictions_all)} samples")
            
            if max_samples is not None and sample_count >= max_samples:
                break
    
    print(f"[INFO] Total samples processed: {len(predictions_all)}")
    return predictions_all, targets_all, metadata_all


def inference_autoregressive(
    model, 
    dataloader, 
    norm_stats,
    device,
    rollout_steps=50,
    max_samples=None,
    denormalize=True,
):
    """
    Autoregressive inference: Use model predictions as input for subsequent predictions.
    This shows real-world performance when predicting multiple steps into the future.
    
    Start with first 10 real timesteps, then predict rollout_steps ahead autoregressively.
    """
    model.eval()
    
    predictions_all = []
    targets_all = []
    metadata_all = []
    
    print(f"\n[INFO] Running AUTOREGRESSIVE inference...")
    print(f"[INFO] Mode: Multi-step prediction using model's own predictions")
    print(f"[INFO] Rollout steps: {rollout_steps}")
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            static_graph = batch['static_graph'].to(device)
            y_hist_2d = batch['y_hist_2d'].to(device)
            rain_hist_2d = batch['rain_hist_2d'].to(device)
            y_future_2d = batch['y_future_2d'].to(device)
            rain_future_2d = batch['rain_future_2d'].to(device)
            
            batch_size = y_hist_2d.size(0)
            available_steps = min(rollout_steps, y_future_2d.size(1))
            
            # Process each sample in batch
            for i in range(batch_size):
                if max_samples is not None and sample_count >= max_samples:
                    break
                
                # Autoregressive prediction: model uses its own predictions
                predictions = model.forward_unroll(
                    data=static_graph,
                    y_hist_true=y_hist_2d[i],  # Use initial 10 real timesteps
                    rain_hist=rain_hist_2d[i],
                    rain_future=rain_future_2d[i, :available_steps],  # Known future rainfall
                    make_x_dyn=lambda y, r, data: {
                        'oneD': torch.zeros((data['oneD'].num_nodes, 1), device=device),
                        'twoD': torch.cat([y, r], dim=-1),
                    },
                    rollout_steps=available_steps,
                    device=device,
                )
                
                # Convert to numpy
                pred_np = predictions.cpu().numpy()  # [rollout_steps, N, 1]
                target_np = y_future_2d[i, :available_steps].cpu().numpy()
                
                # Denormalize if requested
                if denormalize:
                    pred_denorm = []
                    target_denorm = []
                    
                    for t in range(available_steps):
                        pred_t = torch.from_numpy(pred_np[t]).to(device)
                        target_t = torch.from_numpy(target_np[t]).to(device)
                        
                        pred_t_denorm = unnormalize_col(pred_t, norm_stats, col=0, node_type='twoD')
                        target_t_denorm = unnormalize_col(target_t, norm_stats, col=0, node_type='twoD')
                        
                        pred_denorm.append(pred_t_denorm.cpu().numpy())
                        target_denorm.append(target_t_denorm.cpu().numpy())
                    
                    pred_np = np.stack(pred_denorm, axis=0)
                    target_np = np.stack(target_denorm, axis=0)
                
                predictions_all.append(pred_np)
                targets_all.append(target_np)
                metadata_all.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'mode': 'autoregressive',
                    'rollout_steps': available_steps,
                    'n_nodes': pred_np.shape[1],
                })
                
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    break
            
            if (batch_idx + 1) % 10 == 0:
                print(f"[INFO] Processed {batch_idx + 1} batches, {len(predictions_all)} samples")
            
            if max_samples is not None and sample_count >= max_samples:
                break
    
    print(f"[INFO] Total samples processed: {len(predictions_all)}")
    return predictions_all, targets_all, metadata_all


def compute_metrics(predictions, targets, mode_name):
    """Compute evaluation metrics."""
    # Convert lists to arrays
    preds = np.array(predictions)  # [N_samples, timesteps, N_nodes, 1]
    targs = np.array(targets)
    
    # Compute metrics
    mae = np.mean(np.abs(preds - targs))
    rmse = np.sqrt(np.mean((preds - targs) ** 2))
    mape = np.mean(np.abs((preds - targs) / (np.abs(targs) + 1e-8))) * 100
    
    # Per-timestep metrics
    mae_per_step = np.mean(np.abs(preds - targs), axis=(0, 2, 3))  # [timesteps]
    rmse_per_step = np.sqrt(np.mean((preds - targs) ** 2, axis=(0, 2, 3)))
    
    metrics = {
        'mode': mode_name,
        'n_samples': len(predictions),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'mae_per_step': mae_per_step.tolist(),
        'rmse_per_step': rmse_per_step.tolist(),
    }
    
    print(f"\n[METRICS] Overall ({mode_name}):")
    print(f"  Samples: {len(predictions)}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    
    print(f"\n[METRICS] First 10 timesteps:")
    for t in range(min(10, len(mae_per_step))):
        print(f"  Step {t+1:2d}: MAE={mae_per_step[t]:.6f}, RMSE={rmse_per_step[t]:.6f}")
    
    if len(mae_per_step) > 10:
        print(f"\n[METRICS] Every 10 timesteps:")
        for t in range(9, len(mae_per_step), 10):
            print(f"  Step {t+1:2d}: MAE={mae_per_step[t]:.6f}, RMSE={rmse_per_step[t]:.6f}")
    
    return metrics


def save_results(predictions, targets, metadata, metrics, output_dir):
    """Save predictions and metrics to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions and targets as numpy arrays
    np.savez_compressed(
        os.path.join(output_dir, 'predictions.npz'),
        predictions=np.array(predictions),
        targets=np.array(targets),
    )
    print(f"\n[INFO] Saved predictions to {output_dir}/predictions.npz")
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata to {output_dir}/metadata.json")
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics to {output_dir}/metrics.json")


def main():
    parser = argparse.ArgumentParser(description='Run inference with FloodLM model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, required=True, choices=['teacher', 'autoregressive'],
                        help='Inference mode: teacher (one-step with real data) or autoregressive (multi-step with predictions)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', 'all'],
                        help='Data split to run inference on')
    parser.add_argument('--rollout_steps', type=int, default=50, 
                        help='Number of autoregressive rollout steps (only for autoregressive mode)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to process')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Device to run inference on')
    parser.add_argument('--no_denormalize', action='store_true', help='Keep predictions normalized')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print("FloodLM Inference Script")
    print(f"{'='*70}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Mode: {args.mode.upper()}")
    print(f"[INFO] Split: {args.split}")
    if args.mode == 'autoregressive':
        print(f"[INFO] Rollout steps: {args.rollout_steps}")
    print(f"[INFO] Denormalize: {not args.no_denormalize}")
    
    # Initialize data to get normalization stats
    print(f"\n[INFO] Initializing data...")
    data = initialize_data()
    norm_stats = data['norm_stats']
    
    # Load model
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    
    # Determine forecast length based on mode
    if args.mode == 'teacher':
        # For teacher forcing, we need enough future data to evaluate
        forecast_len = args.rollout_steps if args.rollout_steps > 1 else 50
    else:
        # For autoregressive, forecast_len must be >= rollout_steps
        forecast_len = args.rollout_steps
    
    # Create dataloader
    print(f"\n[INFO] Creating dataloader for {args.split} split...")
    dataloader = get_recurrent_dataloader(
        history_len=10,
        forecast_len=forecast_len,
        batch_size=args.batch_size,
        shuffle=False,
        split=args.split,
    )
    
    # Run inference based on mode
    if args.mode == 'teacher':
        predictions, targets, metadata = inference_teacher_forcing(
            model=model,
            dataloader=dataloader,
            norm_stats=norm_stats,
            device=device,
            max_samples=args.max_samples,
            denormalize=not args.no_denormalize,
        )
        mode_name = 'teacher_forcing'
    else:  # autoregressive
        predictions, targets, metadata = inference_autoregressive(
            model=model,
            dataloader=dataloader,
            norm_stats=norm_stats,
            device=device,
            rollout_steps=args.rollout_steps,
            max_samples=args.max_samples,
            denormalize=not args.no_denormalize,
        )
        mode_name = f'autoregressive_{args.rollout_steps}steps'
    
    # Compute metrics
    print(f"\n[INFO] Computing metrics...")
    metrics = compute_metrics(predictions, targets, mode_name)
    
    # Save results
    output_dir = os.path.join(args.output_dir, f"{args.split}_{mode_name}")
    print(f"\n[INFO] Saving results to {output_dir}...")
    save_results(predictions, targets, metadata, metrics, output_dir)
    
    print(f"\n{'='*70}")
    print("Inference Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
