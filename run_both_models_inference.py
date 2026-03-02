#!/usr/bin/env python
"""
Wrapper script to run inference for both Model_1 and Model_2 and combine results.

Usage:
    python run_both_models_inference.py --checkpoint checkpoints/model_best.pt --output final_submission.csv
"""

import os
import sys
import subprocess
import tempfile
import argparse
from pathlib import Path

import pandas as pd


def run_model_inference(model_id, checkpoint, temp_output, device='auto', max_events=None):
    """
    Run inference for a single model.
    
    Args:
        model_id: Model ID (1 or 2)
        checkpoint: Path to checkpoint file
        temp_output: Temporary output file for this model
        device: Device to use ('auto', 'cpu', 'mps', 'cuda')
        max_events: Max events to process (None for all)
    
    Returns:
        Path to output file if successful, None otherwise
    """
    print(f"\n{'='*70}")
    print(f"Running inference for Model_{model_id}")
    print(f"{'='*70}")
    
    # Build command
    cmd = [
        sys.executable,
        'autoregressive_inference.py',
        '--checkpoint', checkpoint,
        '--output', temp_output,
        '--model-id', str(model_id),
        '--device', device,
    ]
    
    if max_events is not None:
        cmd.extend(['--max-events', str(max_events)])
    
    # Set environment variable to select model
    env = os.environ.copy()
    env['SELECTED_MODEL'] = f'Model_{model_id}'
    
    print(f"[INFO] Command: {' '.join(cmd)}")
    
    # Run inference
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print(f"[INFO] Model_{model_id} inference completed successfully")
        return temp_output
    else:
        print(f"[ERROR] Model_{model_id} inference failed (exit code: {result.returncode})")
        return None


def combine_submissions(model1_csv, model2_csv, output_csv):
    """
    Combine predictions from Model_1 and Model_2 into single submission.
    
    Args:
        model1_csv: CSV with model_id=1 predictions
        model2_csv: CSV with model_id=2 predictions
        output_csv: Output combined CSV
    """
    print(f"\n{'='*70}")
    print("Combining submissions")
    print(f"{'='*70}")
    
    # Read both submissions
    df1 = pd.read_csv(model1_csv)
    df2 = pd.read_csv(model2_csv)
    
    print(f"[INFO] Model_1 submission: {len(df1)} rows")
    print(f"[INFO] Model_2 submission: {len(df2)} rows")
    
    # Verify model_id columns
    if df1['model_id'].unique().tolist() != [1]:
        print(f"[WARN] Model_1 CSV has unexpected model_ids: {df1['model_id'].unique().tolist()}")
    if df2['model_id'].unique().tolist() != [2]:
        print(f"[WARN] Model_2 CSV has unexpected model_ids: {df2['model_id'].unique().tolist()}")
    
    # Combine
    combined = pd.concat([df1, df2], ignore_index=True)
    
    # Sort by row_id if present, else by (model_id, event_id, node_type, node_id, timestep)
    if 'row_id' in combined.columns:
        combined = combined.sort_values('row_id').reset_index(drop=True)
    else:
        # Re-add row_id
        combined.insert(0, 'row_id', range(len(combined)))
    
    # Save combined
    combined.to_csv(output_csv, index=False)
    
    print(f"[INFO] Combined submission: {len(combined)} rows")
    print(f"[INFO] Saved to: {output_csv}")
    
    # Show statistics
    print(f"\n[INFO] Summary:")
    print(f"  Total rows: {len(combined)}")
    print(f"  Model IDs: {sorted(combined['model_id'].unique().tolist())}")
    print(f"  Events per model: {combined.groupby('model_id')['event_id'].nunique()}")
    print(f"  Water level range: [{combined['water_level'].min():.6f}, {combined['water_level'].max():.6f}]")
    print(f"  Water level mean: {combined['water_level'].mean():.6f}")
    print(f"  Water level std: {combined['water_level'].std():.6f}")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(description='Run inference for both models and combine results')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='final_submission.csv',
                        help='Output CSV file path for combined submission')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Device to run inference on')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Maximum number of events to process (for testing)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary submission files (for debugging)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("FloodLM Inference - Both Models")
    print(f"{'='*70}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Output: {args.output}")
    print(f"[INFO] Device: {args.device}")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_model1 = os.path.join(tmpdir, 'model1_submission.csv')
        temp_model2 = os.path.join(tmpdir, 'model2_submission.csv')
        
        # Run Model_1 inference
        model1_result = run_model_inference(1, args.checkpoint, temp_model1, args.device, args.max_events)
        if model1_result is None:
            print("[ERROR] Model_1 inference failed, aborting")
            return 1
        
        # Run Model_2 inference
        model2_result = run_model_inference(2, args.checkpoint, temp_model2, args.device, args.max_events)
        if model2_result is None:
            print("[ERROR] Model_2 inference failed, aborting")
            return 1
        
        # Combine submissions
        combine_submissions(temp_model1, temp_model2, args.output)
        
        # Optionally keep temp files
        if args.keep_temp:
            keep_model1 = args.output.replace('.csv', '_model1.csv')
            keep_model2 = args.output.replace('.csv', '_model2.csv')
            
            import shutil
            shutil.copy(temp_model1, keep_model1)
            shutil.copy(temp_model2, keep_model2)
            print(f"\n[INFO] Temporary files kept:")
            print(f"  {keep_model1}")
            print(f"  {keep_model2}")
    
    print(f"\n{'='*70}")
    print("Complete!")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
