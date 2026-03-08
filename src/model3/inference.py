#!/usr/bin/env python
"""
Model_3 inference — non-autoregressive encoder-decoder.

For each test event:
  1. Encode first history_len timesteps via GRU
  2. Decode all remaining timesteps in one parallel forward pass
  3. Denormalize predictions to original water level scale
  4. Format as submission rows

Usage:
    python src/model3/inference.py
    python src/model3/inference.py --checkpoint checkpoints/latest/Model_3_best.pt
    python src/model3/inference.py --output submission_model3.csv
"""

import os
import sys
import json
import pickle
import argparse
import glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model3.config import MODEL_ID, CONFIG, HIDDEN_DIMS, TEST_PATH
from model3.data import (
    initialize_data, build_static_graph, get_graph_config,
    make_x_dyn, preprocess_2d_nodes,
    NODE_ID_COL, EXCLUDE_1D_DYNAMIC, EXCLUDE_2D_DYNAMIC,
)
from model3.model import HeteroEncoderDecoderModel


def load_checkpoint(checkpoint_path, device):
    """Load Model_3 from checkpoint. Returns (ckpt, arch_config)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    arch = ckpt.get('model_arch_config', {})
    return ckpt, {
        'h_dim':              arch.get('h_dim',              CONFIG['h_dim']),
        'msg_dim':            arch.get('msg_dim',            CONFIG['msg_dim']),
        'hidden_dim':         arch.get('hidden_dim',         HIDDEN_DIMS),
        'T_max':              arch.get('T_max',              CONFIG['T_max']),
        'dec_d_model':        arch.get('dec_d_model',        CONFIG['dec_d_model']),
        'dec_nhead':          arch.get('dec_nhead',          CONFIG['dec_nhead']),
        'dec_num_layers':     arch.get('dec_num_layers',     CONFIG['dec_num_layers']),
        'dec_ffn_dim':        arch.get('dec_ffn_dim',        CONFIG['dec_ffn_dim']),
        'dec_dropout':        arch.get('dec_dropout',        CONFIG['dec_dropout']),
        'dec_node_chunk':     arch.get('dec_node_chunk',     CONFIG['dec_node_chunk']),
    }


def load_event_data(event_dir):
    """Load raw dynamic CSVs for one event."""
    n1 = pd.read_csv(event_dir + "/1d_nodes_dynamic_all.csv")
    n2 = pd.read_csv(event_dir + "/2d_nodes_dynamic_all.csv")
    return n1, n2


def prepare_event_tensors(n1_raw, n2_raw, norm_stats, device):
    """Normalize event data and return tensors [T, N, 1]."""
    n1 = n1_raw.copy()
    n2 = n2_raw.copy()

    n1 = n1.drop(columns=[c for c in EXCLUDE_1D_DYNAMIC if c in n1.columns])
    n2 = n2.drop(columns=[c for c in EXCLUDE_2D_DYNAMIC if c in n2.columns])

    normalizer_1d = norm_stats['normalizer_1d']
    normalizer_2d = norm_stats['normalizer_2d']
    n1 = normalizer_1d.transform_dynamic(n1)
    n2 = normalizer_2d.transform_dynamic(n2)

    tcol = "timestep_raw" if "timestep_raw" in n1.columns else "timestep"
    n1 = n1.sort_values([tcol, NODE_ID_COL]).reset_index(drop=True)
    n2 = n2.sort_values([tcol, NODE_ID_COL]).reset_index(drop=True)

    timesteps = sorted(n1[tcol].unique())
    T = len(timesteps)
    N_1d = len(n1[n1[tcol] == timesteps[0]])
    N_2d = len(n2[n2[tcol] == timesteps[0]])

    wl_1d_all   = []
    wl_2d_all   = []
    rain_2d_all = []
    for t in timesteps:
        rows1 = n1[n1[tcol] == t].sort_values(NODE_ID_COL)
        rows2 = n2[n2[tcol] == t].sort_values(NODE_ID_COL)
        wl_1d_all.append(torch.tensor(rows1['water_level'].values, dtype=torch.float32).unsqueeze(-1))
        wl_2d_all.append(torch.tensor(rows2['water_level'].values, dtype=torch.float32).unsqueeze(-1))
        r2 = rows2['rainfall'].values if 'rainfall' in rows2.columns else np.zeros(N_2d, dtype=np.float32)
        rain_2d_all.append(torch.tensor(r2, dtype=torch.float32).unsqueeze(-1))

    wl_1d   = torch.stack(wl_1d_all).to(device)    # [T, N_1d, 1]
    wl_2d   = torch.stack(wl_2d_all).to(device)    # [T, N_2d, 1]
    rain_2d = torch.stack(rain_2d_all).to(device)  # [T, N_2d, 1]
    return wl_1d, wl_2d, rain_2d


def unnormalize_water_level(y_norm, normalizer, node_type):
    """Inverse-normalize water level predictions."""
    params = normalizer.dynamic_params.get('water_level', {})
    if params.get('type', 'minmax') == 'meanstd':
        return y_norm * params['sigma'] + params['mean']
    vmin, vmax = params.get('min', 0.0), params.get('max', 1.0)
    return y_norm * (vmax - vmin) + vmin


def run_inference(checkpoint_path, output_path, sample_submission_path=None, device_str='cuda'):
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    ckpt, arch_config = load_checkpoint(checkpoint_path, device)

    # Initialize data (uses training cache for normalizers + static graph)
    print("[INFO] Initializing data...")
    data = initialize_data()
    norm_stats = data['norm_stats']

    # Build static graph from training data topology
    static_graph = build_static_graph(
        data['static_1d_sorted'], data['static_2d_sorted'],
        data['edges1d'], data['edges2d'], data['edges1d2d'],
        data['edges1dfeats'], data['edges2dfeats'],
        data['static_1d_cols'], data['static_2d_cols'],
        data['edge1_cols'], data['edge2_cols'],
    ).to(device)
    rain_1d_index = static_graph.rain_1d_index.to(device) if hasattr(static_graph, 'rain_1d_index') else None

    # Build model
    graph_config = get_graph_config(
        data['static_1d_cols'], data['static_2d_cols'],
        data['edge1_cols'], data['edge2_cols'],
    )
    graph_config.update(arch_config)
    model = HeteroEncoderDecoderModel(**graph_config).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[INFO] Model loaded (epoch {ckpt.get('epoch', '?')})")

    def _make_x_dyn(y1d, y2d, rain2):
        return make_x_dyn(y1d, y2d, rain2, rain_1d_index, device)

    # Find test events
    test_event_dirs = sorted(
        glob.glob(f"{TEST_PATH}/event_*"),
        key=lambda p: int(Path(p).name.split("_")[-1])
    )
    if not test_event_dirs:
        raise FileNotFoundError(f"No test events found at {TEST_PATH}")
    print(f"[INFO] Found {len(test_event_dirs)} test events")

    all_rows = []

    with torch.no_grad():
        for event_dir in test_event_dirs:
            event_id = int(Path(event_dir).name.split("_")[-1])
            n1_raw, n2_raw = load_event_data(event_dir)
            wl_1d, wl_2d, rain_2d = prepare_event_tensors(n1_raw, n2_raw, norm_stats, device)

            T = wl_1d.shape[0]
            H = CONFIG['history_len']
            if T <= H:
                print(f"[WARN] Event {event_id} too short ({T} timesteps), skipping")
                continue

            y_hist_1d    = wl_1d[:H]       # [H, N_1d, 1]
            y_hist_2d    = wl_2d[:H]       # [H, N_2d, 1]
            rain_hist_2d = rain_2d[:H]     # [H, N_2d, 1]
            rain_future  = rain_2d[H:]     # [T-H, N_2d, 1]

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(
                    data=static_graph,
                    y_hist_1d=y_hist_1d, y_hist_2d=y_hist_2d,
                    rain_hist_2d=rain_hist_2d, rain_future_2d=rain_future,
                    make_x_dyn=_make_x_dyn, rain_1d_index=rain_1d_index,
                    history_len=H, device=device,
                )

            # Denormalize predictions
            pred_1d = unnormalize_water_level(
                preds['oneD'].float().cpu(), norm_stats['normalizer_1d'], 'oneD')  # [T-H, N_1d, 1]
            pred_2d = unnormalize_water_level(
                preds['twoD'].float().cpu(), norm_stats['normalizer_2d'], 'twoD')  # [T-H, N_2d, 1]

            # Get node IDs
            n1_raw_sorted = n1_raw.sort_values(NODE_ID_COL).drop_duplicates(NODE_ID_COL)
            n2_raw_sorted = n2_raw.sort_values(NODE_ID_COL).drop_duplicates(NODE_ID_COL)
            node_ids_1d = n1_raw_sorted[NODE_ID_COL].values  # [N_1d]
            node_ids_2d = n2_raw_sorted[NODE_ID_COL].values  # [N_2d]

            T_pred = pred_1d.shape[0]
            # 1D predictions
            for t in range(T_pred):
                for j, nid in enumerate(node_ids_1d):
                    all_rows.append({
                        'event_id': event_id,
                        'node_id':  nid,
                        'node_type': '1d',
                        'timestep': H + t,
                        'water_level': pred_1d[t, j, 0].item(),
                    })
            # 2D predictions
            for t in range(T_pred):
                for j, nid in enumerate(node_ids_2d):
                    all_rows.append({
                        'event_id': event_id,
                        'node_id':  nid,
                        'node_type': '2d',
                        'timestep': H + t,
                        'water_level': pred_2d[t, j, 0].item(),
                    })

            print(f"  Event {event_id}: T={T}, predictions shape 1D={pred_1d.shape[:2]}, 2D={pred_2d.shape[:2]}")

    pred_df = pd.DataFrame(all_rows)

    if sample_submission_path and os.path.exists(sample_submission_path):
        sample = pd.read_csv(sample_submission_path)
        merged = sample.merge(
            pred_df[['event_id', 'node_id', 'node_type', 'timestep', 'water_level']],
            on=['event_id', 'node_id', 'node_type', 'timestep'],
            how='left',
        )
        merged['water_level'] = merged['water_level_y'].fillna(merged.get('water_level_x', 0.0))
        merged = merged.drop(columns=[c for c in ['water_level_x', 'water_level_y'] if c in merged.columns])
        merged.to_csv(output_path, index=False)
        print(f"[INFO] Submission saved (merged with sample): {output_path}")
    else:
        pred_df.to_csv(output_path, index=False)
        print(f"[INFO] Predictions saved: {output_path}")

    print(f"[INFO] Total prediction rows: {len(pred_df)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model_3 inference')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/model3/latest/Model_3_best.pt',
                        help='Path to Model_3 checkpoint .pt file')
    parser.add_argument('--output', type=str, default='submission_model3.csv')
    parser.add_argument('--sample-submission', type=str, default=None,
                        help='Path to Kaggle sample_submission.csv for row alignment')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        sample_submission_path=args.sample_submission,
        device_str=args.device,
    )
