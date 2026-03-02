#!/usr/bin/env python3
"""
Calculate Kaggle-aligned NRMSE between two prediction CSVs.

Implements the exact Kaggle metric:
  score = mean over model_ids of mean(NRMSE_1D, NRMSE_2D)
  NRMSE = RMSE / kaggle_sigma

Kaggle sigma values (fixed by competition):
  (model_id=1, node_type=1 [1D]): 16.878
  (model_id=1, node_type=2 [2D]): 14.379
  (model_id=2, node_type=1 [1D]):  3.192
  (model_id=2, node_type=2 [2D]):  2.727

Usage:
  python calculate_rmse.py                            # uses submission.csv vs submission_firsttry.csv
  python calculate_rmse.py my_submission.csv          # custom submission vs firsttry
  python calculate_rmse.py pred.csv ref.csv           # explicit files
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Kaggle metric normalization sigmas: {(model_id, node_type): sigma}
KAGGLE_SIGMA = {
    (1, 1): 16.878,
    (1, 2): 14.379,
    (2, 1):  3.192,
    (2, 2):  2.727,
}


def calculate_kaggle_nrmse(pred_file, ref_file):
    """
    Calculate the Kaggle-aligned NRMSE score between two prediction files.

    Returns:
        dict with overall score and per (model_id, node_type) breakdown
    """
    print(f"[INFO] Loading predictions: {pred_file}")
    df_pred = pd.read_csv(pred_file)
    print(f"       Shape: {df_pred.shape}")

    print(f"[INFO] Loading reference:   {ref_file}")
    df_ref = pd.read_csv(ref_file)
    print(f"       Shape: {df_ref.shape}")

    # Merge on all key columns
    key_cols = ['row_id', 'model_id', 'event_id', 'node_type', 'node_id']
    available_keys = [c for c in key_cols if c in df_pred.columns and c in df_ref.columns]
    print(f"\n[INFO] Merging on: {available_keys}")

    merged = pd.merge(
        df_pred[available_keys + ['water_level']],
        df_ref[available_keys + ['water_level']],
        on=available_keys,
        suffixes=('_pred', '_ref'),
    )
    print(f"       Merged rows: {len(merged):,}")

    if len(merged) == 0:
        raise ValueError("No matching rows after merge!")

    valid = merged['water_level_pred'].notna() & merged['water_level_ref'].notna()
    print(f"       Valid pairs: {valid.sum():,} / {len(merged):,}")
    merged = merged[valid].copy()

    merged['residual'] = merged['water_level_pred'] - merged['water_level_ref']
    merged['sq_error'] = merged['residual'] ** 2

    print(f"\n{'='*65}")
    print(f"  Kaggle NRMSE Breakdown")
    print(f"{'='*65}")
    print(f"  {'Model':>6}  {'NodeType':>8}  {'N':>8}  {'RMSE (m)':>10}  {'σ':>7}  {'NRMSE':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*8}")

    model_scores = {}
    for model_id in sorted(merged['model_id'].unique()):
        model_df = merged[merged['model_id'] == model_id]
        node_type_nrmse = {}

        for node_type in [1, 2]:
            nt_df = model_df[model_df['node_type'] == node_type]
            if len(nt_df) == 0:
                continue
            rmse = np.sqrt(nt_df['sq_error'].mean())
            sigma = KAGGLE_SIGMA.get((model_id, node_type), None)
            if sigma is None:
                print(f"  [WARN] No sigma for (model_id={model_id}, node_type={node_type}), skipping")
                continue
            nrmse = rmse / sigma
            node_type_nrmse[node_type] = nrmse
            label = '1D' if node_type == 1 else '2D'
            print(f"  {model_id:>6}  {label:>8}  {len(nt_df):>8,}  {rmse:>10.4f}  {sigma:>7.3f}  {nrmse:>8.4f}")

        if node_type_nrmse:
            model_score = np.mean(list(node_type_nrmse.values()))
            model_scores[model_id] = model_score
            print(f"  {'':>6}  {'→ avg':>8}  {'':>8}  {'':>10}  {'':>7}  {model_score:>8.4f}")
            print()

    overall_score = np.mean(list(model_scores.values())) if model_scores else float('nan')

    print(f"{'='*65}")
    print(f"  Overall Kaggle score (lower is better): {overall_score:.4f}")
    print(f"{'='*65}\n")

    # Plain RMSE across all rows for reference
    all_rmse = np.sqrt(merged['sq_error'].mean())
    mae = merged['residual'].abs().mean()
    print(f"[INFO] Plain RMSE (all rows, unweighted): {all_rmse:.4f} m")
    print(f"[INFO] MAE  (all rows, unweighted):       {mae:.4f} m")
    print(f"[INFO] Residual range: [{merged['residual'].min():.4f}, {merged['residual'].max():.4f}]")

    return {
        'kaggle_score': overall_score,
        'model_scores': model_scores,
        'plain_rmse': all_rmse,
        'mae': mae,
    }


def find_latest_submission(search_dir: Path) -> Path:
    """Find the most recently modified submission_*.csv in search_dir."""
    candidates = sorted(
        search_dir.glob('submission_*.csv'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # Fall back to submission.csv
    fallback = search_dir / 'submission.csv'
    if fallback.exists():
        return fallback
    return None


if __name__ == "__main__":
    floodlm_dir = Path(__file__).parent

    if len(sys.argv) == 3:
        submission_file = Path(sys.argv[1])
        reference_file = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        submission_file = Path(sys.argv[1])
        reference_file = floodlm_dir / "submission_firsttry.csv"
    else:
        # Auto-find: prefer most recently modified submission_*.csv
        latest = find_latest_submission(floodlm_dir)
        if latest is None:
            print(f"ERROR: No submission CSV found in {floodlm_dir}")
            sys.exit(1)
        submission_file = latest
        reference_file = floodlm_dir / "submission_firsttry.csv"

    print(f"\n[INFO] Submission: {submission_file}")
    print(f"[INFO] Reference:  {reference_file}\n")

    if not submission_file.exists():
        print(f"ERROR: {submission_file} not found")
        sys.exit(1)

    if not reference_file.exists():
        print(f"ERROR: {reference_file} not found")
        sys.exit(1)

    print("=" * 65)
    print("  FloodLM vs Reference — Kaggle NRMSE Evaluation")
    print("=" * 65)

    results = calculate_kaggle_nrmse(str(submission_file), str(reference_file))
