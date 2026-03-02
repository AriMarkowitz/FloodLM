#!/usr/bin/env python
"""
Autoregressive inference script for FloodLM competition submission.

This script:
1. Loads trained model and normalization statistics
2. Processes test events with autoregressive rollout
3. Denormalizes predictions to original scale
4. Formats output to match sample_submission.csv

Usage:
    # Generate predictions from project root
    python src/autoregressive_inference.py --checkpoint-dir checkpoints --output submission.csv
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup paths (works whether launched from project root or another cwd)
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import unnormalize_col, get_model_config, create_static_hetero_graph
from model import FloodAutoregressiveHeteroModel
from data_lazy import initialize_data
from data_config import SELECTED_MODEL, DATA_FOLDER, BASE_PATH


def load_checkpoint(checkpoint_path, device):
    """Load trained model from checkpoint."""
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


def load_test_data():
    """
    Load live competition test events from data/<SELECTED_MODEL>/test.
    Returns test event directories plus preprocessed training cache for static graph and normalizers.
    """
    print(f"\n[INFO] Loading test data...")

    # Initialize training-derived cache (normalizers + static graph inputs)
    data = initialize_data()
    required_keys = ['norm_stats']
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required data cache keys: {missing}")

    norm_stats = data['norm_stats']

    # Strict live test path requirement
    test_root = Path(BASE_PATH) / 'test'
    if not test_root.exists() or not test_root.is_dir():
        raise FileNotFoundError(f"Missing test directory: {test_root}")

    test_event_dirs = sorted(
        [p for p in test_root.glob('event_*') if p.is_dir()],
        key=lambda p: int(p.name.split('_')[-1])
    )

    if len(test_event_dirs) == 0:
        raise RuntimeError(f"No test events found under {test_root}")
    
    print(f"[INFO] Found {len(test_event_dirs)} live test events in {test_root}")
    
    return test_event_dirs, norm_stats, data


def load_model_normalizers(model_id, checkpoint_dir='checkpoints'):
    """
    Load model-specific normalizers from checkpoint directory.
    
    Args:
        model_id: Model identifier (1 or 2)
        checkpoint_dir: Directory containing saved normalizers
        
    Returns:
        dict: Dictionary with normalizer_1d and normalizer_2d objects
    """
    normalizer_path = os.path.join(checkpoint_dir, f'Model_{model_id}_normalizers.pkl')
    
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(f"Model-specific normalizers not found: {normalizer_path}")
    
    print(f"[INFO] Loading model-specific normalizers from {normalizer_path}")
    with open(normalizer_path, 'rb') as f:
        normalizers = pickle.load(f)
    
    return normalizers


def get_event_metadata(event_path):
    """Extract event information from event path."""
    path = Path(event_path)
    event_id = int(path.name.split('_')[-1])
    return event_id, str(path)


def load_event_data(event_path):
    """Load event data using strict expected file structure."""
    path = Path(event_path)

    # Strict required dynamic files
    node_1d_csv = path / '1d_nodes_dynamic_all.csv'
    node_2d_csv = path / '2d_nodes_dynamic_all.csv'

    if not node_1d_csv.exists() or not node_2d_csv.exists():
        raise FileNotFoundError(
            f"Missing required event files in {path}: "
            f"{node_1d_csv.name} and/or {node_2d_csv.name}"
        )

    node_1d = pd.read_csv(node_1d_csv)
    node_2d = pd.read_csv(node_2d_csv)

    return node_1d, node_2d


def build_static_graph_from_cache(data):
    """Build static hetero graph from initialize_data() cache payload."""
    required_keys = [
        'static_1d_sorted',
        'static_2d_sorted',
        'edges1d',
        'edges2d',
        'edges1d2d',
        'edges1dfeats',
        'edges2dfeats',
        'static_1d_cols',
        'static_2d_cols',
        'edge1_cols',
        'edge2_cols',
        'NODE_ID_COL',
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required cache keys for static graph construction: {missing}")

    # Use explicit copies to avoid negative-stride numpy views when converting to torch tensors.
    return create_static_hetero_graph(
        static_1d_norm=data['static_1d_sorted'].copy().reset_index(drop=True),
        static_2d_norm=data['static_2d_sorted'].copy().reset_index(drop=True),
        edges1d=data['edges1d'].copy().reset_index(drop=True),
        edges2d=data['edges2d'].copy().reset_index(drop=True),
        edges1d2d=data['edges1d2d'].copy().reset_index(drop=True),
        edges1dfeats_norm=data['edges1dfeats'].copy().reset_index(drop=True),
        edges2dfeats_norm=data['edges2dfeats'].copy().reset_index(drop=True),
        static_1d_cols=data['static_1d_cols'],
        static_2d_cols=data['static_2d_cols'],
        edge1_cols=data['edge1_cols'],
        edge2_cols=data['edge2_cols'],
        node_id_col=data['NODE_ID_COL'],
    )


def prepare_event_tensors(node_1d, node_2d, norm_stats, device):
    """
    Prepare normalized tensors for one event.
    Returns:
      - y1_all: [T, N1, 1]
      - y2_all: [T, N2, 1]
      - rain2_all: [T, N2, 1]
      - timesteps, node_ids_1d, node_ids_2d
    """
    required_norm = ['normalizer_1d', 'normalizer_2d']
    missing_norm = [k for k in required_norm if k not in norm_stats]
    if missing_norm:
        raise KeyError(f"Missing required key(s) in norm_stats: {missing_norm}")

    exclude_1d = norm_stats['exclude_1d'] if 'exclude_1d' in norm_stats else []
    exclude_2d = norm_stats['exclude_2d'] if 'exclude_2d' in norm_stats else []

    node_1d = node_1d.drop(columns=[c for c in exclude_1d if c in node_1d.columns])
    node_2d = node_2d.drop(columns=[c for c in exclude_2d if c in node_2d.columns])

    node_1d = norm_stats['normalizer_1d'].transform_dynamic(node_1d, exclude_cols=None)
    node_2d = norm_stats['normalizer_2d'].transform_dynamic(node_2d, exclude_cols=None)

    timesteps_1d = sorted(node_1d['timestep'].unique())
    timesteps_2d = sorted(node_2d['timestep'].unique())
    if timesteps_1d != timesteps_2d:
        raise RuntimeError("1D and 2D timestep grids differ; strict structure violation")

    timesteps = timesteps_1d
    node_ids_1d = sorted(node_1d['node_idx'].unique())
    node_ids_2d = sorted(node_2d['node_idx'].unique())

    T = len(timesteps)
    N1 = len(node_ids_1d)
    N2 = len(node_ids_2d)

    y1_all = np.zeros((T, N1, 1), dtype=np.float32)
    y2_all = np.zeros((T, N2, 1), dtype=np.float32)
    rain2_all = np.zeros((T, N2, 1), dtype=np.float32)

    for t_idx, t in enumerate(timesteps):
        t1 = node_1d[node_1d['timestep'] == t].sort_values('node_idx')
        t2 = node_2d[node_2d['timestep'] == t].sort_values('node_idx')

        if len(t1) != N1 or len(t2) != N2:
            raise RuntimeError(f"Incomplete event rows at timestep {t} (strict structure violation)")

        y1_all[t_idx, :, 0] = t1['water_level'].values
        y2_all[t_idx, :, 0] = t2['water_level'].values
        if 'rainfall' not in t2.columns:
            raise KeyError("Missing required column in 2D event data: rainfall")
        rain2_all[t_idx, :, 0] = t2['rainfall'].values

    return (
        torch.tensor(y1_all, dtype=torch.float32, device=device),
        torch.tensor(y2_all, dtype=torch.float32, device=device),
        torch.tensor(rain2_all, dtype=torch.float32, device=device),
        timesteps,
        node_ids_1d,
        node_ids_2d,
    )


def autoregressive_rollout_both(
    model, 
    static_graph,
    y1_hist,      # [H, N1, 1]
    y2_hist,      # [H, N2, 1]
    rain2_all,    # [T, N2, 1]
    device,
    history_len=10
):
    """
    Perform autoregressive rollout for entire event.
    
    Args:
        model: Trained FloodLM model
        static_graph: Static graph structure
        y1_hist: Initial 1D water levels [H, N1, 1]
        y2_hist: Initial 2D water levels [H, N2, 1]
        rain2_all: Full 2D rainfall [T, N2, 1]
        device: torch device
        history_len: History window size
        
    Returns:
        pred1: [T-H, N1, 1] and pred2: [T-H, N2, 1] predicted normalized water levels
    """
    model.eval()

    T_total = rain2_all.size(0)
    if T_total <= history_len:
        raise RuntimeError(f"Need >{history_len} timesteps for rollout, got {T_total}")

    h = model.init_hidden(static_graph, device)

    # Warm start with true history
    for t in range(history_len):
        x_dyn_t = {
            'oneD': y1_hist[t],
            'twoD': torch.cat([y2_hist[t], rain2_all[t]], dim=-1),
        }
        h = model.cell(static_graph, h, x_dyn_t)

    y1_prev = y1_hist[-1]
    y2_prev = y2_hist[-1]

    preds_1d = []
    preds_2d = []
    
    with torch.no_grad():
        for t in range(history_len, T_total):
            d1 = model.head(h['oneD'])
            d2 = model.head(h['twoD'])

            # Model predicts absolute water levels
            y1_next = d1
            y2_next = d2

            preds_1d.append(y1_next)
            preds_2d.append(y2_next)

            x_dyn_next = {
                'oneD': y1_next,
                'twoD': torch.cat([y2_next, rain2_all[t]], dim=-1),
            }
            h = model.cell(static_graph, h, x_dyn_next)

            y1_prev = y1_next
            y2_prev = y2_next

    return torch.stack(preds_1d, dim=0), torch.stack(preds_2d, dim=0)


def denormalize_predictions(predictions, norm_stats, node_type):
    """Denormalize water level predictions for a node type."""
    T, N, _ = predictions.shape
    
    # DEBUG: Check pred range before denorm
    pred_min, pred_max = predictions.min().item(), predictions.max().item()
    pred_mean = predictions.mean().item()
    pred_median = torch.median(predictions).item()
    print(f"[DEBUG] Normalized predictions ({node_type}): min={pred_min:.6f}, max={pred_max:.6f}, mean={pred_mean:.6f}, median={pred_median:.6f}, shape={predictions.shape}")
    
    # Check if predictions are outside [0,1] range (indicates denormalization will fail)
    outside_01 = ((predictions < -0.1) | (predictions > 1.1)).sum().item()
    if outside_01 > 0:
        pct = 100 * outside_01 / (T * N)
        print(f"[WARN] {pct:.1f}% of predictions outside [0,1] range - denormalization will produce garbage!")
    
    # DEBUG: Check normalizer params
    if node_type == 'oneD':
        norm = norm_stats['normalizer_1d']
        params_dict = 'dynamic_1d_params'
    else:
        norm = norm_stats['normalizer_2d']
        params_dict = 'dynamic_2d_params'
    
    if hasattr(norm, 'dynamic_params') and 'water_level' in norm.dynamic_params:
        wl_params = norm.dynamic_params['water_level']
        print(f"[DEBUG] {node_type} water_level normalizer params: {wl_params}")
    
    denorm_preds = []
    for t in range(T):
        pred_t = unnormalize_col(predictions[t], norm_stats, col=0, node_type=node_type)
        denorm_preds.append(pred_t.cpu().numpy())
    
    denorm_stack = np.stack(denorm_preds, axis=0)  # [T, N, 1]
    
    # Clamp to physical bounds: water level cannot be negative
    denorm_stack = np.maximum(denorm_stack, 0.0)
    
    # DEBUG: Check denorm range
    denorm_min, denorm_max = denorm_stack.min(), denorm_stack.max()
    print(f"[DEBUG] Denormalized predictions ({node_type}): min={denorm_min:.6f}, max={denorm_max:.6f}")
    
    return denorm_stack


def create_submission_rows(predictions, event_id, model_id, node_ids, node_type):
    """Create per-step submission rows for one event/node_type.

    predictions shape is [T_pred, N, 1] where step 0 corresponds to timestep 10.
    """
    rows = []
    T_pred, N, _ = predictions.shape

    for step_idx in range(T_pred):
        for n_idx, node_id in enumerate(node_ids):
            rows.append({
                'model_id': model_id,
                'event_id': event_id,
                'node_type': int(node_type),
                'node_id': int(node_id),
                'step_idx': int(step_idx),
                'water_level': float(predictions[step_idx, n_idx, 0]),
            })

    return rows


def process_all_events(
    model,
    test_events,
    norm_stats,
    data,
    device,
    model_id,
    max_events=None
):
    """
    Process all test events with autoregressive rollout.
    
    Returns DataFrame with predictions.
    """
    print(f"\n[INFO] Processing {len(test_events)} test events...")
    
    all_rows = []
    debug_printed = False
    
    # Build static graph once (same for all events)
    static_graph = build_static_graph_from_cache(data).to(device)
    
    for event_idx, event_path in enumerate(tqdm(test_events, desc="Processing events")):
        if max_events is not None and event_idx >= max_events:
            break
        
        try:
            # Get event metadata
            event_id, event_dir = get_event_metadata(event_path)
            
            # Load event data
            node_1d, node_2d = load_event_data(event_dir)
            
            # DEBUG: Print raw data from first event
            if event_idx == 0 and not debug_printed:
                print(f"\n[DEBUG] ===== RAW TEST DATA (Model {model_id}, Event {event_id}) =====")
                if 'water_level' in node_2d.columns:
                    raw_wl_2d = node_2d['water_level'].values
                    print(f"[DEBUG] 2D raw water_level: min={raw_wl_2d.min():.2f}, max={raw_wl_2d.max():.2f}, mean={raw_wl_2d.mean():.2f}")
                    print(f"[DEBUG]   Sample values: {raw_wl_2d[:5]}")
                if 'water_level' in node_1d.columns:
                    raw_wl_1d = node_1d['water_level'].values
                    print(f"[DEBUG] 1D raw water_level: min={raw_wl_1d.min():.2f}, max={raw_wl_1d.max():.2f}, mean={raw_wl_1d.mean():.2f}")
                    print(f"[DEBUG]   Sample values: {raw_wl_1d[:5]}")
                debug_printed = True
            
            # Prepare tensors
            y1_all, y2_all, rain2_all, timesteps, node_ids_1d, node_ids_2d = prepare_event_tensors(
                node_1d, node_2d, norm_stats, device
            )
            
            # DEBUG: Check normalized values after prepare_tensors
            if event_idx == 0 and debug_printed:
                print(f"\n[DEBUG] ===== NORMALIZED DATA TO MODEL =====")
                print(f"[DEBUG] y1_all (1D, normalized): min={y1_all.min():.6f}, max={y1_all.max():.6f}, shape={y1_all.shape}")
                print(f"[DEBUG]   Sample: {y1_all[:2, 0, 0]}")
                print(f"[DEBUG] y2_all (2D, normalized): min={y2_all.min():.6f}, max={y2_all.max():.6f}, shape={y2_all.shape}")
                print(f"[DEBUG]   Sample: {y2_all[:2, 0, 0]}")
            
            T_total = y2_all.size(0)
            
            if T_total < 11:  # Need at least 10 for history + 1 to predict
                print(f"[WARN] Skipping event {event_id}: only {T_total} timesteps")
                continue
            
            # First 10 timesteps are provided; rollout for remaining timesteps.
            pred1_norm, pred2_norm = autoregressive_rollout_both(
                model=model,
                static_graph=static_graph,
                y1_hist=y1_all[:10],
                y2_hist=y2_all[:10],
                rain2_all=rain2_all,
                device=device,
                history_len=10
            )
            
            # Denormalize
            pred1_denorm = denormalize_predictions(pred1_norm, norm_stats, node_type='oneD')
            pred2_denorm = denormalize_predictions(pred2_norm, norm_stats, node_type='twoD')

            event_rows = []
            event_rows.extend(create_submission_rows(pred1_denorm, event_id, model_id, node_ids_1d, node_type=1))
            event_rows.extend(create_submission_rows(pred2_denorm, event_id, model_id, node_ids_2d, node_type=2))

            all_rows.extend(event_rows)
            
        except Exception as e:
            print(f"[ERROR] Failed to process event {event_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_rows)
    
    print(f"\n[INFO] Generated {len(df)} prediction rows")
    print(f"[INFO] Events: {df['event_id'].nunique()}")
    print(f"[INFO] Rows per event: {len(df) // df['event_id'].nunique():.0f} avg")
    
    return df


def match_to_sample_submission(predictions_df, sample_path):
    """
    Match predictions to sample_submission.csv format and order.
    
    This ensures:
    1. All required rows are present (fills missing with NaN)
    2. Rows are in the exact order expected by Kaggle
    3. Extra rows are removed
    """
    print(f"\n[INFO] Matching to sample submission: {sample_path}")
    
    # Load sample submission - use chunking to avoid memory issues with 50M+ rows
    print(f"[INFO] Loading sample submission (this may take a moment for 50M+ rows)...")
    chunks = []
    for chunk in pd.read_csv(sample_path, chunksize=100000):
        chunks.append(chunk)
    sample = pd.concat(chunks, ignore_index=True)
    
    print(f"[INFO] Sample submission: {len(sample)} rows")
    print(f"[INFO] Predictions: {len(predictions_df)} rows")
    
    # Build per-key sequence index to disambiguate repeated rows across timesteps
    key_cols = ['model_id', 'event_id', 'node_type', 'node_id']
    sample = sample.copy()
    sample['step_idx'] = sample.groupby(key_cols).cumcount()

    required_pred_cols = key_cols + ['step_idx', 'water_level']
    missing_pred_cols = [c for c in required_pred_cols if c not in predictions_df.columns]
    if missing_pred_cols:
        raise KeyError(f"Predictions missing required columns: {missing_pred_cols}")

    result = sample.merge(
        predictions_df[required_pred_cols],
        on=key_cols + ['step_idx'],
        how='left',
        suffixes=('_orig', '_pred')
    )

    if 'water_level_pred' not in result.columns:
        print("[WARN] No predicted water levels found after merge - using NaN")
        result['water_level'] = np.nan
    else:
        result['water_level'] = result['water_level_pred']
    
    result = result[['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']]

    # Report completeness (but don't fail)
    nan_count = result['water_level'].isna().sum()
    coverage_pct = 100 * (1 - nan_count / len(result))
    print(f"[INFO] Coverage: {coverage_pct:.1f}% ({len(result) - nan_count}/{len(result)} rows with predictions)")
    
    if nan_count > 0:
        print(f"[WARN] {nan_count} rows missing predictions (will submit with NaN)")
    
    # Verify row count
    if len(result) != len(sample):
        print(f"[WARN] Result has {len(result)} rows but expected {len(sample)}")
    
    print(f"[INFO] Final submission: {len(result)} rows")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Autoregressive inference for FloodLM competition (automatically processes both Model_1 and Model_2)'
    )
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints (default: checkpoints/)')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV file path')
    parser.add_argument('--sample', type=str, default='../FloodModel/sample_submission.csv',
                        help='Path to sample_submission.csv')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Device to run inference on')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Maximum number of events to process (for testing)')
    
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
    print("FloodLM Autoregressive Inference (Both Models)")
    print(f"{'='*70}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint directory: {args.checkpoint_dir}")
    print(f"[INFO] Output: {args.output}")
    
    # Collect predictions from both models
    all_predictions = []
    
    for model_id in [1, 2]:
        print(f"\n[INFO] ========== Processing Model {model_id} ==========")
        
        # Import modules to allow SELECTED_MODEL to be reset
        import importlib
        import data_config as dc
        
        # Update SELECTED_MODEL dynamically
        dc.SELECTED_MODEL = f"Model_{model_id}"
        dc.BASE_PATH = f"data/{dc.SELECTED_MODEL}"
        
        # Reload modules to pick up new paths (critical for data)
        import data_lazy
        importlib.reload(data_lazy)
        
        # CRITICAL: Re-import after reload to get fresh function refs
        from data_lazy import initialize_data as initialize_data_fresh
        
        # Clear any stale data references
        print(f"[INFO] Initializing fresh data for Model {model_id}...")
        data_fresh = initialize_data_fresh()
        if 'norm_stats' not in data_fresh:
            raise KeyError(f"Missing norm_stats in Model {model_id} data cache")
        
        norm_stats = data_fresh['norm_stats']
        
        # Find test events for this model
        from pathlib import Path as PathlibPath
        test_root = PathlibPath(dc.BASE_PATH) / 'test'
        if not test_root.exists():
            print(f"[ERROR] No test directory for Model {model_id}!")
            continue
        
        test_events = sorted(
            [str(p) for p in test_root.glob('event_*') if p.is_dir()],
            key=lambda p: int(PathlibPath(p).name.split('_')[-1])
        )
        
        if len(test_events) == 0:
            print(f"[ERROR] No test events found for Model {model_id}!")
            continue
        
        print(f"[INFO] Found {len(test_events)} test events for Model {model_id}")
        
        # Determine model-specific checkpoint
        checkpoint_dir = args.checkpoint_dir
        
        # Try model-specific checkpoint patterns
        model_checkpoint_path = None
        candidates = [
            os.path.join(checkpoint_dir, f"Model_{model_id}_best.pt"),
            os.path.join(checkpoint_dir, f"Model_{model_id}_epoch_002.pt"),
            os.path.join(checkpoint_dir, f"Model_{model_id}_epoch_001.pt"),
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                model_checkpoint_path = candidate
                break
        
        if model_checkpoint_path is None:
            print(f"[ERROR] No checkpoint found for Model {model_id}!")
            print(f"[ERROR] Tried: {candidates}")
            print(f"[ERROR] Please train Model_{model_id} first")
            continue
        
        print(f"[INFO] Using checkpoint: {model_checkpoint_path}")
        
        # Load model for this specific model_id (architecture depends on graph size)
        model, checkpoint = load_checkpoint(model_checkpoint_path, device)
        
        # Use freshly loaded data (Model_1 and Model_2 have different graph structures)
        data = data_fresh
        
        # Load model-specific normalizers (trained on this model's data)
        try:
            model_normalizers = load_model_normalizers(model_id, checkpoint_dir)
            norm_stats['normalizer_1d'] = model_normalizers['normalizer_1d']
            norm_stats['normalizer_2d'] = model_normalizers['normalizer_2d']
            print(f"[INFO] Loaded model-specific normalizers for Model {model_id}")
            
            # DEBUG: Print normalizer params for water_level
            norm_1d = model_normalizers['normalizer_1d']
            norm_2d = model_normalizers['normalizer_2d']
            if 'water_level' in norm_2d.dynamic_params:
                wl_params = norm_2d.dynamic_params['water_level']
                print(f"[DEBUG] water_level (dynamic, 2D): min={wl_params['min']:.6f}, max={wl_params['max']:.6f}, log={wl_params['log']}")
            else:
                print(f"[DEBUG] water_level not found in 2D dynamic params. Available: {list(norm_2d.dynamic_params.keys())[:5]}")
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            print(f"[WARN] Using normalizers from cache (may not match training)")
        
        # Process events for this model
        predictions_df = process_all_events(
            model=model,
            test_events=test_events,
            norm_stats=norm_stats,
            data=data,
            device=device,
            model_id=model_id,
            max_events=args.max_events
        )
        
        all_predictions.append(predictions_df)
        print(f"[INFO] Generated {len(predictions_df)} rows for Model {model_id}")
    
    # Combine predictions from both models
    if len(all_predictions) > 0:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        print(f"\n[INFO] Combined predictions: {len(combined_predictions)} total rows")
        
        # Match to sample submission format
        if os.path.exists(args.sample):
            submission_df = match_to_sample_submission(combined_predictions, args.sample)
        else:
            print(f"[WARN] Sample submission not found: {args.sample}")
            print(f"[WARN] Using raw predictions (may not match Kaggle format)")
            submission_df = combined_predictions
            # Add row_id if not present
            if 'row_id' not in submission_df.columns:
                submission_df.insert(0, 'row_id', range(len(submission_df)))
        
        # Save submission
        submission_df.to_csv(args.output, index=False)
        print(f"\n[INFO] Saved submission to: {args.output}")
        
        # Show preview
        print(f"\n[INFO] First 10 rows:")
        print(submission_df.head(10))
        
        print(f"\n[INFO] Last 10 rows:")
        print(submission_df.tail(10))
        
        # Summary statistics
        print(f"\n[INFO] Summary:")
        print(f"  Total rows: {len(submission_df)}")
        print(f"  Models: {sorted(submission_df['model_id'].unique())}")
        print(f"  Events: {submission_df['event_id'].nunique()}")
        print(f"  Water level range: [{submission_df['water_level'].min():.6f}, {submission_df['water_level'].max():.6f}]")
        print(f"  Water level mean: {submission_df['water_level'].mean():.6f}")
        print(f"  Water level std: {submission_df['water_level'].std():.6f}")
        
        # Check for NaN values
        nan_count = submission_df['water_level'].isna().sum()
        if nan_count > 0:
            print(f"\n[ERROR] Submission has {nan_count} NaN values!")
        else:
            print(f"\n[SUCCESS] All rows have valid predictions!")
        
        print(f"\n[INFO] To submit to Kaggle, run:")
        print(f"  python submit_to_kaggle.py {args.output} --message \"FloodLM submission\"")
    else:
        print("[ERROR] Failed to generate predictions for any model")
    
    print(f"\n{'='*70}")
    print("Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
