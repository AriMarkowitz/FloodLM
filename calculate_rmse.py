#!/usr/bin/env python3
"""
Calculate RMSE between two prediction CSVs.

Compares final_submission.csv against predictions.csv to evaluate model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_rmse(pred_file1, pred_file2):
    """
    Calculate RMSE between two prediction files.
    
    Args:
        pred_file1: Path to first prediction CSV (usually the newer one)
        pred_file2: Path to reference prediction CSV (usually the baseline)
    
    Returns:
        RMSE value
    """
    print(f"[INFO] Loading {pred_file1}...")
    df1 = pd.read_csv(pred_file1)
    print(f"       Shape: {df1.shape}, Columns: {list(df1.columns)}")
    
    print(f"[INFO] Loading {pred_file2}...")
    df2 = pd.read_csv(pred_file2)
    print(f"       Shape: {df2.shape}, Columns: {list(df2.columns)}")
    
    # Determine key columns (common to both)
    common_cols = set(df1.columns) & set(df2.columns)
    print(f"\n[INFO] Common columns: {sorted(common_cols)}")
    
    # Identify key columns (all except water_level)
    key_cols = sorted([c for c in common_cols if c != 'water_level'])
    print(f"[INFO] Key columns for alignment: {key_cols}")
    
    # Check if both have water_level
    if 'water_level' not in df1.columns or 'water_level' not in df2.columns:
        raise ValueError("Both files must have 'water_level' column")
    
    # Merge on key columns
    print(f"\n[INFO] Merging on key columns...")
    merged = pd.merge(
        df1, df2,
        on=key_cols,
        how='inner',
        suffixes=('_new', '_ref')
    )
    print(f"       Merged shape: {merged.shape}")
    
    if len(merged) == 0:
        raise ValueError("No matching rows found after merge!")
    
    # Calculate RMSE
    pred_new = merged['water_level_new']
    pred_ref = merged['water_level_ref']
    
    # Handle NaN values
    valid_mask = pred_new.notna() & pred_ref.notna()
    n_valid = valid_mask.sum()
    print(f"\n[INFO] Valid row pairs: {n_valid} / {len(merged)}")
    
    if n_valid == 0:
        raise ValueError("No valid pairs with both water_level values!")
    
    # Calculate RMSE
    residuals = (pred_new[valid_mask] - pred_ref[valid_mask]).values
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Print statistics
    mae = np.mean(np.abs(residuals))
    print(f"\n[INFO] RMSE: {rmse:.6f}")
    print(f"[INFO] MAE:  {mae:.6f}")
    print(f"[INFO] Min residual:  {residuals.min():.6f}")
    print(f"[INFO] Max residual:  {residuals.max():.6f}")
    print(f"[INFO] Mean residual: {residuals.mean():.6f}")
    print(f"[INFO] Std residual:  {residuals.std():.6f}")
    
    return rmse


if __name__ == "__main__":
    # File paths
    floodlm_dir = Path(__file__).parent
    submission_file = floodlm_dir / "kaggle_submission.csv"
    reference_file = Path("/Users/Lion/Desktop/UrbanFloodModeling/FloodModel/predictions.csv")
    
    if not submission_file.exists():
        print(f"ERROR: {submission_file} not found")
        exit(1)
    
    if not reference_file.exists():
        print(f"ERROR: {reference_file} not found")
        exit(1)
    
    print("=" * 75)
    print("RMSE Comparison: FloodLM predictions vs FloodModel predictions")
    print("=" * 75)
    
    rmse = calculate_rmse(str(submission_file), str(reference_file))
    
    print("\n" + "=" * 75)
    print(f"Final RMSE: {rmse:.6f}")
    print("=" * 75)
