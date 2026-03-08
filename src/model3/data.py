"""
Model_3 data module — fully self-contained.
Handles graph construction, normalization, and full-event dataset.

Key difference from Model_1/2: FullEventFloodDataset yields ONE sample per event
(10 warm-start timesteps + all remaining timesteps), with variable-length futures.
No sliding windows, no curriculum — each event is one training sample.
"""

import os
import glob
import pickle
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch_geometric as tg

from .normalization import FeatureNormalizer
from .config import (
    TRAIN_PATH, TEST_PATH, CACHE_PATH, MODEL_ID,
    VALIDATION_SPLIT, RANDOM_SEED, EXCLUDE_1D_DYNAMIC, EXCLUDE_2D_DYNAMIC,
    KAGGLE_SIGMA_1D, KAGGLE_SIGMA_2D, CONFIG,
)

NORMALIZATION_VERBOSE = False
NODE_ID_COL = "node_idx"


# ============================================================
# Static preprocessing helpers (copied from src/data.py)
# ============================================================

def preprocess_2d_nodes(nodes2d_df):
    """Encode aspect as (sin, cos); KNN-interpolate zero areas."""
    from sklearn.neighbors import NearestNeighbors
    df = nodes2d_df.copy()

    if "min_elevation" in df.columns and "elevation" in df.columns:
        mask = pd.isna(df["min_elevation"])
        df.loc[mask, "min_elevation"] = df.loc[mask, "elevation"]

    if "aspect" in df.columns:
        valid = df["aspect"] >= 0
        aspect_rad = np.where(valid, np.deg2rad(df["aspect"]), 0.0)
        df["aspect_sin"] = np.where(valid, np.sin(aspect_rad), 0.0)
        df["aspect_cos"] = np.where(valid, np.cos(aspect_rad), 0.0)
        df = df.drop(columns=["aspect"])

    if "area" in df.columns and ("position_x" in df.columns or "x" in df.columns):
        pos_x_col = "position_x" if "position_x" in df.columns else "x"
        pos_y_col = "position_y" if "position_y" in df.columns else "y"
        zero_mask = df["area"].abs() < 1e-6
        if zero_mask.any():
            positions = df[[pos_x_col, pos_y_col]].values
            nonzero_idx = (~zero_mask).values
            if nonzero_idx.sum() > 0:
                nonzero_positions = positions[nonzero_idx]
                nonzero_areas = df.loc[~zero_mask, "area"].values
                k = min(5, nonzero_idx.sum())
                if k > 0:
                    knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
                    knn.fit(nonzero_positions)
                    zero_positions = positions[zero_mask]
                    distances, indices = knn.kneighbors(zero_positions)
                    interpolated_areas = nonzero_areas[indices].mean(axis=1)
                    zero_rows = df.index[zero_mask].tolist()
                    for i, row_idx in enumerate(zero_rows):
                        df.loc[row_idx, "area"] = interpolated_areas[i]
    return df


def preprocess_1d_nodes(nodes1d_df, nodes2d_df, connections_df):
    """Add channel_2d_elev_diff = 2D_elev - 1D_invert_elev."""
    df = nodes1d_df.copy()
    if ("invert_elevation" in df.columns
            and "elevation" in nodes2d_df.columns
            and "node_1d" in connections_df.columns
            and "node_2d" in connections_df.columns):
        elev_2d = (connections_df
                   .merge(nodes2d_df[["node_idx", "elevation"]], left_on="node_2d", right_on="node_idx", how="left")
                   .set_index("node_1d")["elevation"])
        df["channel_2d_elev_diff"] = df["node_idx"].map(elev_2d) - df["invert_elevation"]
        median_val = df["channel_2d_elev_diff"].median()
        df["channel_2d_elev_diff"] = df["channel_2d_elev_diff"].fillna(median_val)
    return df


# ============================================================
# Graph construction (copied + adapted from src/data.py)
# ============================================================

def build_static_graph(
    static_1d_norm, static_2d_norm,
    edges1d, edges2d, edges1d2d,
    edges1dfeats_norm, edges2dfeats_norm,
    static_1d_cols, static_2d_cols,
    edge1_cols, edge2_cols,
):
    """
    Build a static heterogeneous graph for Model_3.
    Same topology as Model_2: oneD + twoD + global context node.
    Returns HeteroData with x_static, edge_index, edge_attr_static, num_nodes.
    """
    data = tg.data.HeteroData()

    def _extract_edge_index(df, src_candidates, dst_candidates, edge_name):
        src_col = next((c for c in src_candidates if c in df.columns), None)
        dst_col = next((c for c in dst_candidates if c in df.columns), None)
        if src_col is None or dst_col is None:
            raise KeyError(f"{edge_name}: could not find src/dst columns. Available: {list(df.columns)}")
        return torch.tensor(df.loc[:, [src_col, dst_col]].values.copy(), dtype=torch.long).T

    # Node static features
    data["oneD"].x_static = torch.tensor(
        static_1d_norm.loc[:, static_1d_cols].values, dtype=torch.float32)
    data["twoD"].x_static = torch.tensor(
        static_2d_norm.loc[:, static_2d_cols].values, dtype=torch.float32)
    data["oneD"].num_nodes = len(static_1d_norm)
    data["twoD"].num_nodes = len(static_2d_norm)
    data["oneD"].node_idx = torch.tensor(static_1d_norm[NODE_ID_COL].values, dtype=torch.long)
    data["twoD"].node_idx = torch.tensor(static_2d_norm[NODE_ID_COL].values, dtype=torch.long)

    _ANTISYMMETRIC = {"relative_position_x", "relative_position_y", "slope"}

    def _reverse_edge_attr(feat_tensor, col_names):
        t = feat_tensor.clone()
        for i, col in enumerate(col_names):
            if col in _ANTISYMMETRIC:
                t[:, i] = -t[:, i]
        return t

    # Homogeneous edges
    data["oneD", "oneDedge", "oneD"].edge_index = _extract_edge_index(
        edges1d, ["from_node", "source", "src", "from"], ["to_node", "target", "dst", "to"], "oneD->oneD")
    data["oneD", "oneDedgeRev", "oneD"].edge_index = _extract_edge_index(
        edges1d, ["to_node", "target", "dst", "to"], ["from_node", "source", "src", "from"], "oneD->oneD(rev)")
    data["twoD", "twoDedge", "twoD"].edge_index = _extract_edge_index(
        edges2d, ["from_node", "source", "src", "from"], ["to_node", "target", "dst", "to"], "twoD->twoD")
    data["twoD", "twoDedgeRev", "twoD"].edge_index = _extract_edge_index(
        edges2d, ["to_node", "target", "dst", "to"], ["from_node", "source", "src", "from"], "twoD->twoD(rev)")

    # Cross-type edges
    data["twoD", "twoDoneD", "oneD"].edge_index = _extract_edge_index(
        edges1d2d, ["node_2d", "from_node", "source", "src", "from"],
        ["node_1d", "to_node", "target", "dst", "to"], "twoD->oneD")
    data["oneD", "oneDtwoD", "twoD"].edge_index = _extract_edge_index(
        edges1d2d, ["node_1d", "to_node", "target", "dst", "to"],
        ["node_2d", "from_node", "source", "src", "from"], "oneD->twoD")

    # Static edge features
    e1_fwd = torch.tensor(edges1dfeats_norm.loc[:, edge1_cols].values, dtype=torch.float32)
    e2_fwd = torch.tensor(edges2dfeats_norm.loc[:, edge2_cols].values, dtype=torch.float32)
    data["oneD", "oneDedge", "oneD"].edge_attr_static = e1_fwd
    data["oneD", "oneDedgeRev", "oneD"].edge_attr_static = _reverse_edge_attr(e1_fwd, edge1_cols)
    data["twoD", "twoDedge", "twoD"].edge_attr_static = e2_fwd
    data["twoD", "twoDedgeRev", "twoD"].edge_attr_static = _reverse_edge_attr(e2_fwd, edge2_cols)

    # Cross-type edge features: [distance, elev_diff] (same as Model_2)
    n_cross = len(edges1d2d)
    node_1d_col_c = next((c for c in ["node_1d", "to_node", "target", "dst", "to"] if c in edges1d2d.columns), None)
    node_2d_col_c = next((c for c in ["node_2d", "from_node", "source", "src", "from"] if c in edges1d2d.columns), None)

    use_rich = (
        node_1d_col_c is not None and node_2d_col_c is not None
        and "invert_elevation" in static_1d_norm.columns
        and "elevation" in static_2d_norm.columns
    )

    if use_rich:
        idx_1d = static_1d_norm["node_idx"].values
        idx_2d = static_2d_norm["node_idx"].values
        x1d = static_1d_norm["position_x"].values
        y1d = static_1d_norm["position_y"].values
        inv_elev_1d = static_1d_norm["invert_elevation"].values
        x2d = static_2d_norm["position_x"].values
        y2d = static_2d_norm["position_y"].values
        elev_2d_arr = static_2d_norm["elevation"].values
        map_1d = {int(v): i for i, v in enumerate(idx_1d)}
        map_2d = {int(v): i for i, v in enumerate(idx_2d)}
        distances, elev_diffs = [], []
        for _, row in edges1d2d.iterrows():
            i1 = map_1d[int(row[node_1d_col_c])]
            i2 = map_2d[int(row[node_2d_col_c])]
            dist = float(((x1d[i1] - x2d[i2])**2 + (y1d[i1] - y2d[i2])**2) ** 0.5)
            distances.append(dist)
            elev_diffs.append(float(elev_2d_arr[i2]) - float(inv_elev_1d[i1]))
        dist_arr = np.array(distances, dtype=np.float32)
        diff_arr = np.array(elev_diffs, dtype=np.float32)
        dist_arr = (dist_arr - dist_arr.mean()) / (dist_arr.std() + 1e-8)
        diff_arr = (diff_arr - diff_arr.mean()) / (diff_arr.std() + 1e-8)
        cross_fwd = torch.stack([torch.from_numpy(dist_arr), torch.from_numpy(diff_arr)], dim=1)
        cross_rev = torch.stack([torch.from_numpy(dist_arr), -torch.from_numpy(diff_arr)], dim=1)
    else:
        cross_fwd = torch.zeros((n_cross, 1), dtype=torch.float32)
        cross_rev = torch.zeros((n_cross, 1), dtype=torch.float32)

    data["twoD", "twoDoneD", "oneD"].edge_attr_static = cross_fwd
    data["oneD", "oneDtwoD", "twoD"].edge_attr_static = cross_rev

    # rain_1d_index: [N_1d] — maps each 1D node to its connected 2D node index (for rainfall gathering)
    n_1d = len(static_1d_norm)
    node_1d_col = next((c for c in ["node_1d", "to_node", "target", "dst", "to"] if c in edges1d2d.columns), None)
    node_2d_col = next((c for c in ["node_2d", "from_node", "source", "src", "from"] if c in edges1d2d.columns), None)
    if node_1d_col is not None and node_2d_col is not None:
        rain_idx = torch.zeros(n_1d, dtype=torch.long)
        for _, row in edges1d2d.iterrows():
            rain_idx[int(row[node_1d_col])] = int(row[node_2d_col])
        data.rain_1d_index = rain_idx

    # Global context node (same as Model_2)
    n_2d = len(static_2d_norm)
    data["global"].x_static = torch.zeros(1, 1, dtype=torch.float32)
    data["global"].num_nodes = 1

    def _star_edge(n, src_to_ctx=True):
        real = torch.arange(n, dtype=torch.long)
        ctx  = torch.zeros(n, dtype=torch.long)
        return torch.stack([real, ctx] if src_to_ctx else [ctx, real], dim=0)

    data["oneD",   "oneDglobal",  "global"].edge_index      = _star_edge(n_1d, True)
    data["oneD",   "oneDglobal",  "global"].edge_attr_static = torch.zeros(n_1d, 1)
    data["global", "globaloneD",  "oneD"].edge_index         = _star_edge(n_1d, False)
    data["global", "globaloneD",  "oneD"].edge_attr_static   = torch.zeros(n_1d, 1)
    data["twoD",   "twoDglobal",  "global"].edge_index       = _star_edge(n_2d, True)
    data["twoD",   "twoDglobal",  "global"].edge_attr_static = torch.zeros(n_2d, 1)
    data["global", "globaltwoD",  "twoD"].edge_index         = _star_edge(n_2d, False)
    data["global", "globaltwoD",  "twoD"].edge_attr_static   = torch.zeros(n_2d, 1)

    data.validate()
    return data


def get_graph_config(static_1d_cols, static_2d_cols, edge1_cols, edge2_cols):
    """
    Return node_types, edge_types, and dimension dicts for HeteroEncoderDecoderModel.
    Same topology as Model_2 (with global node).
    """
    e_dim_cross = 2  # [distance, elev_diff]

    node_types = ["oneD", "twoD", "global"]
    edge_types = [
        ("oneD",   "oneDedge",    "oneD"),
        ("oneD",   "oneDedgeRev", "oneD"),
        ("twoD",   "twoDedge",    "twoD"),
        ("twoD",   "twoDedgeRev", "twoD"),
        ("twoD",   "twoDoneD",    "oneD"),
        ("oneD",   "oneDtwoD",    "twoD"),
        ("oneD",   "oneDglobal",  "global"),
        ("global", "globaloneD",  "oneD"),
        ("twoD",   "twoDglobal",  "global"),
        ("global", "globaltwoD",  "twoD"),
    ]
    node_static_dims = {
        "oneD":   len(static_1d_cols),
        "twoD":   len(static_2d_cols),
        "global": 1,
    }
    # Dynamic input dims for encoder GRU:
    #   oneD: [water_level, rainfall_from_2d, water_level_2d] = 3
    #   twoD: [water_level, rainfall] = 2
    #   global: dummy zero = 1
    node_dyn_input_dims = {
        "oneD":   3,
        "twoD":   2,
        "global": 1,
    }
    edge_static_dims = {
        ("oneD",   "oneDedge",    "oneD"):   len(edge1_cols),
        ("oneD",   "oneDedgeRev", "oneD"):   len(edge1_cols),
        ("twoD",   "twoDedge",    "twoD"):   len(edge2_cols),
        ("twoD",   "twoDedgeRev", "twoD"):   len(edge2_cols),
        ("twoD",   "twoDoneD",    "oneD"):   e_dim_cross,
        ("oneD",   "oneDtwoD",    "twoD"):   e_dim_cross,
        ("oneD",   "oneDglobal",  "global"): 1,
        ("global", "globaloneD",  "oneD"):   1,
        ("twoD",   "twoDglobal",  "global"): 1,
        ("global", "globaltwoD",  "twoD"):   1,
    }
    return {
        "node_types":          node_types,
        "edge_types":          edge_types,
        "node_static_dims":    node_static_dims,
        "node_dyn_input_dims": node_dyn_input_dims,
        "edge_static_dims":    edge_static_dims,
    }


# ============================================================
# make_x_dyn: dynamic input construction for encoder GRU
# ============================================================

def make_x_dyn(
    y_pred_1d: torch.Tensor,    # [N_1d, 1]   (B=1 always in Model_3)
    y_pred_2d: torch.Tensor,    # [N_2d, 1]
    rain_2d: torch.Tensor,      # [N_2d, 1]
    rain_1d_index: torch.Tensor,  # [N_1d] mapping 1D node -> 2D node
    device: torch.device,
) -> dict:
    """Construct dynamic input dict for one encoder timestep (B=1)."""
    x_dyn = {}
    x_dyn["twoD"] = torch.cat([y_pred_2d, rain_2d], dim=-1)  # [N_2d, 2]

    if rain_1d_index is not None:
        rain_1d = rain_2d[rain_1d_index]         # [N_1d, 1]
        wl_2d_for_1d = y_pred_2d[rain_1d_index]  # [N_1d, 1]
        x_dyn["oneD"] = torch.cat([y_pred_1d, rain_1d, wl_2d_for_1d], dim=-1)  # [N_1d, 3]
    else:
        x_dyn["oneD"] = y_pred_1d

    # Global context: dummy zero
    x_dyn["global"] = torch.zeros(1, 1, device=device, dtype=y_pred_1d.dtype)
    return x_dyn


# ============================================================
# Data initialization (lazy, cached)
# ============================================================

_cache = None


def initialize_data():
    """Load and preprocess data (lazy, disk-cached). Returns data dict."""
    global _cache
    if _cache is not None:
        return _cache

    # Try disk cache first
    if os.path.exists(CACHE_PATH):
        print(f"[INFO] Loading Model_3 cache from {CACHE_PATH}")
        try:
            with open(CACHE_PATH, 'rb') as f:
                _cache = pickle.load(f)
            return _cache
        except Exception as e:
            print(f"[WARN] Cache load failed: {e}, recomputing...")

    print("[INFO] Computing Model_3 preprocessing...")
    _cache = _compute_preprocessing()
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(_cache, f)
    print(f"[INFO] Saved Model_3 cache to {CACHE_PATH}")
    return _cache


def _split_events(event_dirs, val_split=0.2, random_seed=42):
    shuffled = event_dirs.copy()
    random.seed(random_seed)
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * val_split)
    n_train = len(shuffled) - n_val
    print(f"[INFO] Split: {n_train} train, {n_val} val events")
    return shuffled[:n_train], shuffled[n_train:]


def _compute_preprocessing():
    """Full preprocessing pipeline: load static data, fit normalizers, split events."""
    static_1d = pd.read_csv(f"{TRAIN_PATH}/1d_nodes_static.csv")
    static_2d = pd.read_csv(f"{TRAIN_PATH}/2d_nodes_static.csv")
    edges1dfeats = pd.read_csv(f"{TRAIN_PATH}/1d_edges_static.csv")
    edges2dfeats = pd.read_csv(f"{TRAIN_PATH}/2d_edges_static.csv")
    edges1d = pd.read_csv(f"{TRAIN_PATH}/1d_edge_index.csv")
    edges2d = pd.read_csv(f"{TRAIN_PATH}/2d_edge_index.csv")
    edges1d2d = pd.read_csv(f"{TRAIN_PATH}/1d2d_connections.csv")

    EDGE_ID_COL = "edge_idx" if "edge_idx" in edges1dfeats.columns else edges1dfeats.columns[0]

    # Normalize edge features
    norm_edge1d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
    norm_edge2d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
    norm_edge1d.fit_static(edges1dfeats.copy(), EDGE_ID_COL)
    norm_edge2d.fit_static(edges2dfeats.copy(), EDGE_ID_COL)
    edges1dfeats = norm_edge1d.transform_static(edges1dfeats, EDGE_ID_COL)
    edges2dfeats = norm_edge2d.transform_static(edges2dfeats, EDGE_ID_COL)
    edge1_cols = [c for c in edges1dfeats.columns if c != EDGE_ID_COL]
    edge2_cols = [c for c in edges2dfeats.columns if c != EDGE_ID_COL]

    # Get + split events
    event_dirs = sorted(
        glob.glob(f"{TRAIN_PATH}/event_*"),
        key=lambda p: int(Path(p).name.split("_")[-1])
    )
    train_events, val_events = _split_events(event_dirs, VALIDATION_SPLIT, RANDOM_SEED)

    # Preprocess static features
    print("[INFO] Preprocessing static nodes...")
    static_2d = preprocess_2d_nodes(static_2d)
    static_1d = preprocess_1d_nodes(static_1d, static_2d, edges1d2d)

    normalizer_1d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
    normalizer_2d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
    normalizer_1d.fit_static(static_1d.copy(), NODE_ID_COL)
    normalizer_2d.fit_static(static_2d.copy(), NODE_ID_COL)
    static_1d = normalizer_1d.transform_static(static_1d, NODE_ID_COL)
    static_2d = normalizer_2d.transform_static(static_2d, NODE_ID_COL)

    # Fit dynamic normalizers on training events only
    n1_temp = pd.read_csv(train_events[0] + "/1d_nodes_dynamic_all.csv")
    n2_temp = pd.read_csv(train_events[0] + "/2d_nodes_dynamic_all.csv")
    n1_temp = n1_temp.drop(columns=[c for c in EXCLUDE_1D_DYNAMIC if c in n1_temp.columns])
    n2_temp = n2_temp.drop(columns=[c for c in EXCLUDE_2D_DYNAMIC if c in n2_temp.columns])
    base_1d_feats = [c for c in n1_temp.columns if c not in [NODE_ID_COL, 'timestep']]
    base_2d_feats = [c for c in n2_temp.columns if c not in [NODE_ID_COL, 'timestep', 'rainfall']]
    dynamic_1d_cols = base_1d_feats
    dynamic_2d_cols = base_2d_feats + ['rainfall']

    normalizer_1d.init_dynamic_streaming(dynamic_1d_cols)
    normalizer_2d.init_dynamic_streaming(dynamic_2d_cols)

    print(f"[INFO] Streaming {len(train_events)} training events for dynamic normalization...")

    def _read_event(event_dir):
        n1 = pd.read_csv(event_dir + "/1d_nodes_dynamic_all.csv")
        n2 = pd.read_csv(event_dir + "/2d_nodes_dynamic_all.csv")
        n1 = n1.drop(columns=[c for c in EXCLUDE_1D_DYNAMIC if c in n1.columns])
        n2 = n2.drop(columns=[c for c in EXCLUDE_2D_DYNAMIC if c in n2.columns])
        return n1, n2

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_read_event, d): d for d in train_events}
        for fut in as_completed(futures):
            n1, n2 = fut.result()
            if n1 is not None:
                normalizer_1d.update_dynamic_streaming(n1)
                normalizer_2d.update_dynamic_streaming(n2)

    normalizer_1d.finalize_dynamic_streaming(
        skew_threshold=2.0,
        meanstd_overrides={'water_level': KAGGLE_SIGMA_1D},
    )
    normalizer_2d.finalize_dynamic_streaming(
        skew_threshold=2.0,
        meanstd_overrides={'water_level': KAGGLE_SIGMA_2D},
    )

    node1d_cols = base_1d_feats + list(normalizer_1d.static_features)
    node2d_cols = dynamic_2d_cols + list(normalizer_2d.static_features)
    exclude_node_cols = {NODE_ID_COL, 'timestep', 'timestep_raw'}
    node1d_cols = [c for c in node1d_cols if c not in exclude_node_cols]
    node2d_cols = [c for c in node2d_cols if c not in exclude_node_cols]

    static_1d_cols = [c for c in normalizer_1d.static_features if c != NODE_ID_COL]
    static_2d_cols = [c for c in normalizer_2d.static_features if c != NODE_ID_COL]

    static_1d_sorted = static_1d.sort_values(NODE_ID_COL).reset_index(drop=True)
    static_2d_sorted = static_2d.sort_values(NODE_ID_COL).reset_index(drop=True)

    norm_stats = {
        'normalizer_1d': normalizer_1d,
        'normalizer_2d': normalizer_2d,
        'static_1d_params': normalizer_1d.static_params,
        'static_2d_params': normalizer_2d.static_params,
        'dynamic_1d_params': normalizer_1d.dynamic_params,
        'dynamic_2d_params': normalizer_2d.dynamic_params,
        'node1d_cols': node1d_cols,
        'node2d_cols': node2d_cols,
        'exclude_1d': EXCLUDE_1D_DYNAMIC,
        'exclude_2d': EXCLUDE_2D_DYNAMIC,
    }

    print(f"[INFO] 1D static features: {len(static_1d_cols)}, dynamic: {len(base_1d_feats)}")
    print(f"[INFO] 2D static features: {len(static_2d_cols)}, dynamic: {len(dynamic_2d_cols)}")

    return {
        'train_events': [(i, d) for i, d in enumerate(train_events)],
        'val_events':   [(i, d) for i, d in enumerate(val_events)],
        'edges1d': edges1d,
        'edges2d': edges2d,
        'edges1d2d': edges1d2d,
        'edges1dfeats': edges1dfeats,
        'edges2dfeats': edges2dfeats,
        'static_1d_sorted': static_1d_sorted,
        'static_2d_sorted': static_2d_sorted,
        'static_1d_cols': static_1d_cols,
        'static_2d_cols': static_2d_cols,
        'edge1_cols': edge1_cols,
        'edge2_cols': edge2_cols,
        'norm_stats': norm_stats,
    }


# ============================================================
# FullEventFloodDataset
# ============================================================

class FullEventFloodDataset(IterableDataset):
    """
    Yields one sample per event: 10 warm-start timesteps + all remaining timesteps.
    No sliding windows — each event is one training sample.

    All events are pre-loaded into CPU memory at __init__ time (once per run).
    ~56 train events × ~400 timesteps × ~4500 nodes × float32 ≈ ~400MB — fits easily in RAM.
    Eliminates CSV I/O during training, greatly improving GPU utilization.

    Each sample:
        y_hist_1d:      [H, N_1d, 1]
        y_hist_2d:      [H, N_2d, 1]
        rain_hist_2d:   [H, N_2d, 1]
        y_future_1d:    [T_event, N_1d, 1]   T_event varies per event
        y_future_2d:    [T_event, N_2d, 1]
        rain_future_2d: [T_event, N_2d, 1]
    """

    def __init__(self, event_list, static_1d_norm, static_2d_norm, norm_stats,
                 static_1d_cols, static_2d_cols, history_len=10, shuffle=True):
        super().__init__()
        self.shuffle = shuffle
        self.history_len = history_len
        self._samples = []  # pre-built list of sample dicts (CPU tensors)

        normalizer_1d = norm_stats['normalizer_1d']
        normalizer_2d = norm_stats['normalizer_2d']
        exclude_1d = norm_stats.get('exclude_1d', [])
        exclude_2d = norm_stats.get('exclude_2d', [])

        print(f"[INFO] Pre-loading {len(event_list)} events into memory...")
        for event_idx, event_dir in event_list:
            try:
                n1 = pd.read_csv(event_dir + "/1d_nodes_dynamic_all.csv")
                n2 = pd.read_csv(event_dir + "/2d_nodes_dynamic_all.csv")
            except Exception as e:
                print(f"[WARN] Failed to load event {event_dir}: {e}")
                continue

            n1 = n1.drop(columns=[c for c in exclude_1d if c in n1.columns])
            n2 = n2.drop(columns=[c for c in exclude_2d if c in n2.columns])

            n1 = normalizer_1d.transform_dynamic(n1)
            n2 = normalizer_2d.transform_dynamic(n2)

            n1 = pd.merge(n1, static_1d_norm, on=NODE_ID_COL, how='left')
            n2 = pd.merge(n2, static_2d_norm, on=NODE_ID_COL, how='left')
            n2 = preprocess_2d_nodes(n2)

            tcol = "timestep_raw" if "timestep_raw" in n1.columns else "timestep"
            n1 = n1.sort_values([tcol, NODE_ID_COL]).reset_index(drop=True)
            n2 = n2.sort_values([tcol, NODE_ID_COL]).reset_index(drop=True)

            timesteps = sorted(n1[tcol].unique())
            T_total = len(timesteps)
            if T_total <= history_len:
                continue

            N_1d = len(n1[n1[tcol] == timesteps[0]])
            N_2d = len(n2[n2[tcol] == timesteps[0]])

            # Vectorised pivot: [T*N, features] → [T, N, 1] without per-timestep loops
            wl_1d_all   = torch.tensor(
                n1.sort_values([tcol, NODE_ID_COL])['water_level'].values,
                dtype=torch.float32).reshape(T_total, N_1d, 1)
            wl_2d_all   = torch.tensor(
                n2.sort_values([tcol, NODE_ID_COL])['water_level'].values,
                dtype=torch.float32).reshape(T_total, N_2d, 1)
            rain_vals = (n2.sort_values([tcol, NODE_ID_COL])['rainfall'].values
                         if 'rainfall' in n2.columns else np.zeros(T_total * N_2d, dtype=np.float32))
            rain_2d_all = torch.tensor(rain_vals, dtype=torch.float32).reshape(T_total, N_2d, 1)

            H = history_len
            self._samples.append({
                'y_hist_1d':      wl_1d_all[:H],
                'y_hist_2d':      wl_2d_all[:H],
                'rain_hist_2d':   rain_2d_all[:H],
                'y_future_1d':    wl_1d_all[H:],
                'y_future_2d':    wl_2d_all[H:],
                'rain_future_2d': rain_2d_all[H:],
            })

        print(f"[INFO] Pre-loaded {len(self._samples)} events.")

    def __iter__(self):
        indices = list(range(len(self._samples)))
        if self.shuffle:
            random.shuffle(indices)
        for i in indices:
            yield self._samples[i]


def get_full_event_dataloader(split='train', shuffle=True):
    """Get DataLoader yielding full-event samples for Model_3 training."""
    data = initialize_data()

    event_list = data['train_events'] if split == 'train' else data['val_events']

    dataset = FullEventFloodDataset(
        event_list=event_list,
        static_1d_norm=data['static_1d_sorted'],
        static_2d_norm=data['static_2d_sorted'],
        norm_stats=data['norm_stats'],
        static_1d_cols=data['static_1d_cols'],
        static_2d_cols=data['static_2d_cols'],
        history_len=CONFIG['history_len'],
        shuffle=shuffle,
    )

    # batch_size=1: events have variable T_future, can't stack across events
    # pin_memory speeds up CPU→GPU transfers; num_workers=0 is fine since all data is pre-loaded
    return DataLoader(dataset, batch_size=1, collate_fn=lambda b: b[0] if b else None,
                      num_workers=0, pin_memory=True)
