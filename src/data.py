import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as tud
from torch.utils.data import IterableDataset, ChainDataset
from torch_geometric.loader import DataLoader
import torch_geometric as tg
import glob
import random
from pathlib import Path
from normalization import FeatureNormalizer

# Wrapper to prevent PyG from batching HeteroData graphs
class NonBatchableGraph:
    """Wrapper that prevents PyGeometric from batching the enclosed graph."""
    def __init__(self, graph):
        self.graph = graph
    
    def __getattr__(self, name):
        return getattr(self.graph, name)
    
    def __repr__(self):
        return f"NonBatchableGraph({self.graph})"
from normalization import FeatureNormalizer

# Import data configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_config import SELECTED_MODEL, DATA_FOLDER, BASE_PATH, TRAIN_PATH, validate_data_paths

# Reduce log noise from per-feature normalization prints
NORMALIZATION_VERBOSE = False


def unnormalize_col(y_norm: torch.Tensor, norm_stats: dict, col: int, node_type: str):
    """
    Unnormalize a specific column using feature-aware params.
    
    Args:
        y_norm: Normalized values [0, 1] or tensor
        norm_stats: Dict containing all normalization params including normalizers
        col: Column index
        node_type: 'oneD' or 'twoD'
    
    Returns:
        Unnormalized values in original scale
    """
    # Get column name and normalizer
    if node_type == 'oneD':
        col_names = norm_stats['node1d_cols']
        normalizer = norm_stats['normalizer_1d']
    elif node_type == 'twoD':
        col_names = norm_stats['node2d_cols']
        normalizer = norm_stats['normalizer_2d']
    else:
        raise ValueError(f"Unknown node_type: {node_type}")
    
    col_name = col_names[col]
    
    # Use the normalizer's unnormalize method
    # Determine if static or dynamic
    if col_name in normalizer.static_features:
        feature_type = 'static'
    else:
        feature_type = 'dynamic'
    
    return normalizer.unnormalize(y_norm, col_name, feature_type)


# =========================
# Feature engineering
# =========================
def add_temporal_features(df: pd.DataFrame, has_rainfall: bool):
    """
    Expects df has at least: ['node_idx', 'timestep', 'water_level'] and optionally 'rainfall'.
    Adds features built strictly from history by shifting by 1 timestep:
      - cumulative sums: cum_water_level, (cum_rainfall)
      - rolling means over last 12/24/36 timesteps: mean_{k}_water_level, (mean_{k}_rainfall)
    Memory-efficient per-event (groupby node_idx).
    """
    # Use raw timestep for ordering if available
    tcol = "timestep_raw" if "timestep_raw" in df.columns else "timestep"
    df = df.sort_values(["node_idx", tcol]).copy()

    # Shift by 1 to avoid using current timestep in temporal features
    wl_hist = df.groupby("node_idx")["water_level"].shift(1)
    wl_hist = wl_hist.fillna(0.0)

    # Water level features (log-transform cumulative to prevent unbounded growth)
    cum_wl = wl_hist.groupby(df["node_idx"]).cumsum().values
    df["cum_water_level"] = np.sign(cum_wl) * np.log1p(np.abs(cum_wl))

    for k in (12, 24, 36):
        df[f"mean_{k}_water_level"] = (
            wl_hist.groupby(df["node_idx"])
                  .rolling(window=k, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
        )

    if has_rainfall and "rainfall" in df.columns:
        rf_hist = df.groupby("node_idx")["rainfall"].shift(1)
        rf_hist = rf_hist.fillna(0.0)

        cum_rf = rf_hist.groupby(df["node_idx"]).cumsum().values
        df["cum_rainfall"] = np.sign(cum_rf) * np.log1p(np.abs(cum_rf))

        for k in (12, 24, 36):
            df[f"mean_{k}_rainfall"] = (
                rf_hist.groupby(df["node_idx"])
                      .rolling(window=k, min_periods=1)
                      .mean()
                      .reset_index(level=0, drop=True)
            )

    return df


def preprocess_2d_nodes(nodes2d_df):
    """
    Preprocess 2D nodes:
    - Fill missing min_elevation with elevation values
    - Drop aspect and curvature columns
    - Use KNN to interpolate area for zero/near-zero values
    """
    from sklearn.neighbors import NearestNeighbors
    
    df = nodes2d_df.copy()
    
    # Fill min_elevation with elevation if missing
    if "min_elevation" in df.columns and "elevation" in df.columns:
        mask = pd.isna(df["min_elevation"])
        df.loc[mask, "min_elevation"] = df.loc[mask, "elevation"]
    
    # Drop aspect and curvature if they exist
    drop_cols = [c for c in ["aspect", "curvature"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    # KNN interpolation for zero/near-zero area
    if "area" in df.columns and ("position_x" in df.columns or "x" in df.columns):
        pos_x_col = "position_x" if "position_x" in df.columns else "x"
        pos_y_col = "position_y" if "position_y" in df.columns else "y"
        
        # Identify near-zero area rows (area < 1e-6)
        zero_mask = df["area"].abs() < 1e-6
        if zero_mask.any():
            # Get positions for all nodes
            positions = df[[pos_x_col, pos_y_col]].values
            
            # Get non-zero area nodes for interpolation
            nonzero_idx = (~zero_mask).values
            if nonzero_idx.sum() > 0:
                nonzero_positions = positions[nonzero_idx]
                nonzero_areas = df.loc[~zero_mask, "area"].values
                
                # KNN: find k nearest neighbors (cap at available)
                k = min(5, nonzero_idx.sum())
                if k > 0:
                    knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
                    knn.fit(nonzero_positions)
                    
                    # Find zero-area node positions
                    zero_positions = positions[zero_mask]
                    distances, indices = knn.kneighbors(zero_positions)
                    
                    # Average area of k nearest neighbors
                    interpolated_areas = nonzero_areas[indices].mean(axis=1)
                    
                    # Fill NaN/zero areas with interpolated values
                    zero_rows = df.index[zero_mask].tolist()
                    for i, row_idx in enumerate(zero_rows):
                        df.loc[row_idx, "area"] = interpolated_areas[i]
    
    return df


# =========================
# Graph builders
# =========================

def create_static_hetero_graph(
    static_1d_norm, static_2d_norm,
    edges1d, edges2d, edges1d2d,
    edges1dfeats_norm, edges2dfeats_norm,
    static_1d_cols, static_2d_cols,
    edge1_cols, edge2_cols,
    node_id_col: str = "node_idx",
):
    """
    Create a single static heterogeneous graph (no temporal unrolling).
    
    This graph contains ONLY static features and will be reused across all timesteps.
    Dynamic features (water level, rainfall) are passed separately via make_x_dyn.
    
    Args:
        static_1d_norm: normalized static 1D node features (already sorted by node_idx)
        static_2d_norm: normalized static 2D node features (already sorted by node_idx)
        edges1d/2d/1d2d: edge index dataframes with from_node, to_node
        edges1dfeats_norm/edges2dfeats_norm: normalized static edge features
        static_1d_cols/static_2d_cols: column names for static node features
        edge1_cols/edge2_cols: column names for static edge features
    
    Returns:
        HeteroData graph with:
          - data[nt].x_static: static node features
          - data[et].edge_index: edge connectivity
          - data[et].edge_attr_static: static edge features
          - data[nt].num_nodes: number of nodes for each type
    """
    data = tg.data.HeteroData()

    def _extract_edge_index(df, src_candidates, dst_candidates, edge_name):
        src_col = next((c for c in src_candidates if c in df.columns), None)
        dst_col = next((c for c in dst_candidates if c in df.columns), None)
        if src_col is None or dst_col is None:
            raise KeyError(
                f"{edge_name}: could not find source/destination columns. "
                f"Available columns: {list(df.columns)}"
            )
        # Use .copy() to avoid negative-stride numpy arrays when converting to torch
        return torch.tensor(df.loc[:, [src_col, dst_col]].values.copy(), dtype=torch.long).T
    
    # Static node features
    data["oneD"].x_static = torch.tensor(
        static_1d_norm.loc[:, static_1d_cols].values, 
        dtype=torch.float32
    )
    data["twoD"].x_static = torch.tensor(
        static_2d_norm.loc[:, static_2d_cols].values, 
        dtype=torch.float32
    )
    
    data["oneD"].num_nodes = len(static_1d_norm)
    data["twoD"].num_nodes = len(static_2d_norm)
    
    # Store node_idx for mapping dynamic features later
    data["oneD"].node_idx = torch.tensor(static_1d_norm[node_id_col].values, dtype=torch.long)
    data["twoD"].node_idx = torch.tensor(static_2d_norm[node_id_col].values, dtype=torch.long)
    
    # Edge connectivity (static spatial graph)
    data["oneD", "oneDedge", "oneD"].edge_index = _extract_edge_index(
        edges1d,
        src_candidates=["from_node", "source", "src", "from"],
        dst_candidates=["to_node", "target", "dst", "to"],
        edge_name="oneD->oneD",
    )
    
    data["twoD", "twoDedge", "twoD"].edge_index = _extract_edge_index(
        edges2d,
        src_candidates=["from_node", "source", "src", "from"],
        dst_candidates=["to_node", "target", "dst", "to"],
        edge_name="twoD->twoD",
    )
    
    data["twoD", "twoDoneD", "oneD"].edge_index = _extract_edge_index(
        edges1d2d,
        src_candidates=["node_2d", "from_node", "source", "src", "from"],
        dst_candidates=["node_1d", "to_node", "target", "dst", "to"],
        edge_name="twoD->oneD",
    )
    
    # Static edge features
    data["oneD", "oneDedge", "oneD"].edge_attr_static = torch.tensor(
        edges1dfeats_norm.loc[:, edge1_cols].values, dtype=torch.float32
    )
    
    data["twoD", "twoDedge", "twoD"].edge_attr_static = torch.tensor(
        edges2dfeats_norm.loc[:, edge2_cols].values, dtype=torch.float32
    )
    
    # Cross-type edges have no additional features (could add if needed)
    n_cross_edges = len(edges1d2d)
    data["twoD", "twoDoneD", "oneD"].edge_attr_static = torch.zeros(
        (n_cross_edges, 1), dtype=torch.float32
    )
    
    data.validate()
    return data


def make_x_dyn(
    y_pred_1d: torch.Tensor, 
    y_pred_2d: torch.Tensor,
    rain_2d: torch.Tensor, 
    data: tg.data.HeteroData,
    dynamic_1d_feats: torch.Tensor = None,
    dynamic_2d_feats: torch.Tensor = None,
) -> dict[str, torch.Tensor]:
    """
    Construct dynamic input dict for one timestep.
    
    This function is called by the model at each timestep to inject dynamic data.
    
    Args:
        y_pred_1d: [N_1d, 1] current water level estimate for 1D nodes
        y_pred_2d: [N_2d, 1] current water level estimate for 2D nodes
        rain_2d: [N_2d, R] rainfall features for 2D nodes
        data: HeteroData graph (contains num_nodes for sizing)
        dynamic_1d_feats: [N_1d, D1] optional additional dynamic features for 1D
        dynamic_2d_feats: [N_2d, D2] optional additional dynamic features for 2D
    
    Returns:
        x_dyn_t: dict[str, torch.Tensor] with dynamic inputs per node type
          - x_dyn_t["oneD"]: [N_1d, dyn_dim_1d]
          - x_dyn_t["twoD"]: [N_2d, dyn_dim_2d]
    """
    x_dyn = {}
    
    device = y_pred_1d.device
    
    # 1D nodes: water level + optional additional dynamics
    if dynamic_1d_feats is not None:
        x_dyn["oneD"] = torch.cat([y_pred_1d, dynamic_1d_feats], dim=-1)
    else:
        x_dyn["oneD"] = y_pred_1d
    
    # 2D nodes: water level + rainfall + optional additional dynamics
    components_2d = [y_pred_2d, rain_2d]
    if dynamic_2d_feats is not None:
        components_2d.append(dynamic_2d_feats)
    x_dyn["twoD"] = torch.cat(components_2d, dim=-1)
    
    return x_dyn


def idx_builder(timesteps, edgedata, nnodes):
    """
    Build temporal edges for time-unrolled graph.
    Assumes edgedata has 'from_node' and 'to_node' columns with 0-indexed node IDs.
    Maps edges across timesteps: (node_i at t) -> (node_j at t+1)
    
    Args:
        timesteps: number of timesteps in window
        edgedata: DataFrame with from_node, to_node (node IDs within a single timestep)
        nnodes: number of nodes per timestep
    
    Returns:
        edge_index [2, E] where E = len(edgedata) * (timesteps - 1)
    """
    stack = []
    for t in range(timesteps - 1):
        curedge = edgedata.copy()
        # Map from timestep t to timestep t+1
        # Row index = timestep_offset + node_id
        curedge['from_node'] = curedge['from_node'] + t * nnodes
        curedge['to_node'] = curedge['to_node'] + (t + 1) * nnodes
        stack.append(curedge)
    return torch.tensor(pd.concat(stack).loc[:, ["from_node", "to_node"]].values).int().T


def idx_builder_cross_node(timesteps, edgedata, nnodes1d, nnodes2d):
    stack = []
    for t in range(timesteps - 1):
        curedge = edgedata.copy()
        curedge["from_node"] = curedge.node_2d + t * nnodes2d
        curedge["to_node"] = curedge.node_1d + (t + 1) * nnodes1d
        stack.append(curedge)
    return torch.tensor(pd.concat(stack).loc[:, ["from_node", "to_node"]].values).int().T


def create_directed_temporal_graph(
    start_idx, end_idx,
    nodes1d, nodes2d,
    edges1d, edges2d, edges1d2d,
    edges1dfeats, edges2dfeats,
    node1d_cols, node2d_cols,
    edge1_cols, edge2_cols,
    norm_stats=None,
):
    data = tg.data.HeteroData()

    # Use raw timestep for slicing if available (normalized timestep is in [0,1])
    tcol1 = "timestep_raw" if "timestep_raw" in nodes1d.columns else "timestep"
    tcol2 = "timestep_raw" if "timestep_raw" in nodes2d.columns else "timestep"

    n1 = nodes1d.loc[(nodes1d[tcol1] >= start_idx) & (nodes1d[tcol1] <= end_idx), :]
    n2 = nodes2d.loc[(nodes2d[tcol2] >= start_idx) & (nodes2d[tcol2] <= end_idx), :]

    if n1.empty or n2.empty:
        return None

    # Sort by timestep, then node_idx to ensure consistent row ordering
    n1 = n1.sort_values([tcol1, 'node_idx']).reset_index(drop=True)
    n2 = n2.sort_values([tcol2, 'node_idx']).reset_index(drop=True)

    pred_mask_1d = torch.tensor(n1[tcol1].values == n1[tcol1].max(), dtype=torch.bool)
    pred_mask_2d = torch.tensor(n2[tcol2].values == n2[tcol2].max(), dtype=torch.bool)

    x1 = torch.tensor(n1.loc[:, node1d_cols].values, dtype=torch.float32)
    x2 = torch.tensor(n2.loc[:, node2d_cols].values, dtype=torch.float32)

    # CRITICAL: Store ground truth labels BEFORE masking inputs to prevent data leakage
    water_col_1d = node1d_cols.index('water_level') if 'water_level' in node1d_cols else None
    water_col_2d = node2d_cols.index('water_level') if 'water_level' in node2d_cols else None
    
    # Calculate nodes per timestep (needed for masking)
    nnodes1d = len(n1.node_idx.unique())
    nnodes2d = len(n2.node_idx.unique())
    
    # Extract ground truth water_level for prediction nodes (normalized)
    y1 = x1[pred_mask_1d][:, water_col_1d].clone() if water_col_1d is not None else None
    y2 = x2[pred_mask_2d][:, water_col_2d].clone() if water_col_2d is not None else None
    
    # Safety check: ensure labels are finite
    if y1 is not None and not torch.isfinite(y1).all():
        print(f"[WARN] Non-finite labels in 1D nodes for window [{start_idx}, {end_idx}]")
        return None
    if y2 is not None and not torch.isfinite(y2).all():
        print(f"[WARN] Non-finite labels in 2D nodes for window [{start_idx}, {end_idx}]")
        return None
    
    # Now mask out future information from inputs (VECTORIZED for speed)
    # For nodes at final timestep (pred_mask=True), replace current water_level with previous timestep's value
    if water_col_1d is not None:
        # Find indices of prediction nodes
        pred_indices = pred_mask_1d.nonzero(as_tuple=True)[0]
        # Calculate previous timestep indices (shift back by nnodes1d)
        prev_indices = pred_indices - nnodes1d
        # Only copy where previous timestep exists (prev_indices >= 0)
        valid_mask = prev_indices >= 0
        if valid_mask.any():
            x1[pred_indices[valid_mask], water_col_1d] = x1[prev_indices[valid_mask], water_col_1d]
        # Set to 0 where no previous timestep (shouldn't happen with 11 timestep windows)
        if (~valid_mask).any():
            x1[pred_indices[~valid_mask], water_col_1d] = 0.0
    
    if water_col_2d is not None:
        # Find indices of prediction nodes
        pred_indices = pred_mask_2d.nonzero(as_tuple=True)[0]
        # Calculate previous timestep indices (shift back by nnodes2d)
        prev_indices = pred_indices - nnodes2d
        # Only copy where previous timestep exists (prev_indices >= 0)
        valid_mask = prev_indices >= 0
        if valid_mask.any():
            x2[pred_indices[valid_mask], water_col_2d] = x2[prev_indices[valid_mask], water_col_2d]
        # Set to 0 where no previous timestep (shouldn't happen with 11 timestep windows)
        if (~valid_mask).any():
            x2[pred_indices[~valid_mask], water_col_2d] = 0.0
    
    data["oneD"].x = x1
    data["oneD"].y = y1  # Ground truth labels (normalized, only for pred nodes)
    data["oneD"].num_nodes = x1.size(0)
    data["oneD"].pred_mask = pred_mask_1d
    # Store base_area for flux computation (aligned with node rows)
    base_areas = torch.tensor(n1["base_area"].values, dtype=torch.float32)
    data["oneD"].base_area = base_areas  # [num_nodes_in_batch]

    data["twoD"].x = x2
    data["twoD"].y = y2  # Ground truth labels (normalized, only for pred nodes)
    data["twoD"].num_nodes = x2.size(0)
    data["twoD"].pred_mask = pred_mask_2d
    # Store cell_area for flux computation (aligned with node rows)
    cell_areas = torch.tensor(n2["area"].values, dtype=torch.float32)
    data["twoD"].cell_area = cell_areas  # [num_nodes_in_batch]

    # Use unique nodes in BATCH, not entire dataset (already calculated above)
    timesteps = end_idx - start_idx + 1

    if nnodes1d == 0 or nnodes2d == 0:
        return None
    
    # Check if data is complete (all nodes present in all timesteps)
    expected_1d_rows = nnodes1d * timesteps
    expected_2d_rows = nnodes2d * timesteps
    
    # If data incomplete, skip this window
    if len(n1) != expected_1d_rows or len(n2) != expected_2d_rows:
        print(f"Skipping incomplete window [{start_idx}, {end_idx}]: 1D {len(n1)}/{expected_1d_rows}, 2D {len(n2)}/{expected_2d_rows}")
        return None  # Signal to skip this window

    data["oneD", "oneDedge", "oneD"].edge_index = idx_builder(timesteps, edges1d, nnodes1d)
    e1 = torch.tensor(
        pd.concat([edges1dfeats.loc[:, edge1_cols] for _ in range(timesteps-1)]).values,
        dtype=torch.float32
    )
    if norm_stats is not None:
        e1 = normalize_tensor(e1, norm_stats["edge1_mu"], norm_stats["edge1_sigma"])
    data["oneD", "oneDedge", "oneD"].edge_attr = e1  # (you can rename to .edge_attr later)

    data["twoD", "twoDedge", "twoD"].edge_index = idx_builder(timesteps, edges2d, nnodes2d)
    e2 = torch.tensor(
        pd.concat([edges2dfeats.loc[:, edge2_cols] for _ in range(timesteps-1)]).values,
        dtype=torch.float32
    )
    if norm_stats is not None:
        e2 = normalize_tensor(e2, norm_stats["edge2_mu"], norm_stats["edge2_sigma"])
    data["twoD", "twoDedge", "twoD"].edge_attr = e2

    data["twoD", "twoDoneD", "oneD"].edge_index = idx_builder_cross_node(timesteps, edges1d2d, nnodes1d, nnodes2d)

    # Store num_timesteps on node types so it survives batching
    data["oneD"].num_timesteps = torch.tensor(timesteps, dtype=torch.long)
    data["twoD"].num_timesteps = torch.tensor(timesteps, dtype=torch.long)

    data.validate()
    return data


class TemporalGraphStream(IterableDataset):
    def __init__(
        self,
        idxs, nodes1d, nodes2d,
        edges1d, edges2d, edges1d2d,
        edges1dfeats, edges2dfeats,
        node1d_cols, node2d_cols, edge1_cols, edge2_cols,
        norm_stats,
        shuffle=False,
    ):
        super().__init__()
        self.idxs = idxs
        self.nodes1d = nodes1d
        self.nodes2d = nodes2d
        self.edges1d = edges1d
        self.edges2d = edges2d
        self.edges1d2d = edges1d2d
        self.edges1dfeats = edges1dfeats
        self.edges2dfeats = edges2dfeats
        self.node1d_cols = node1d_cols
        self.node2d_cols = node2d_cols
        self.edge1_cols = edge1_cols
        self.edge2_cols = edge2_cols
        self.norm_stats = norm_stats
        self.shuffle = shuffle

    def __iter__(self):
        idxs = list(self.idxs)
        if self.shuffle:
            random.shuffle(idxs)
        for idx_pair in idxs:
            start_idx, end_idx = idx_pair
            data = create_directed_temporal_graph(
                start_idx, end_idx,
                self.nodes1d, self.nodes2d,
                self.edges1d, self.edges2d, self.edges1d2d,
                self.edges1dfeats, self.edges2dfeats,
                self.node1d_cols, self.node2d_cols,
                self.edge1_cols, self.edge2_cols,
                norm_stats=self.norm_stats
            )
            
            # Skip incomplete windows
            if data is None:
                continue

            yield data


# =========================
# Recurrent dataset (static graph + dynamic time series)
# =========================

class RecurrentFloodDataset(IterableDataset):
    """
    Dataset for recurrent model with static graph and dynamic time series.

    Yields pre-formed batches of shape [B, ...] so that every batch contains
    windows from a single event only (temporal fidelity guaranteed).

    Shuffling behaviour:
      - shuffle=True : event order is randomised each epoch; window order
                       within each event is preserved (temporal order).
      - shuffle=False: events and windows are both iterated in original order.

    Returns batches as dicts with:
        - static_graph: HeteroData with static features only
        - y_hist_1d:      [B, H, N_1d, 1]
        - y_hist_2d:      [B, H, N_2d, 1]
        - rain_hist_2d:   [B, H, N_2d, 1]
        - y_future_1d:    [B, T, N_1d, 1]
        - y_future_2d:    [B, T, N_2d, 1]
        - rain_future_2d: [B, T, N_2d, 1]
    """
    def __init__(
        self,
        event_file_list,
        static_1d_norm, static_2d_norm,
        edges1d, edges2d, edges1d2d,
        edges1dfeats_norm, edges2dfeats_norm,
        static_1d_cols, static_2d_cols,
        edge1_cols, edge2_cols,
        norm_stats,
        history_len=10,
        forecast_len=1,
        batch_size=1,
        shuffle=True,
        node_id_col: str = "node_idx",
    ):
        super().__init__()
        self.event_file_list = event_file_list
        self.static_1d_norm = static_1d_norm
        self.static_2d_norm = static_2d_norm
        self.edges1d = edges1d
        self.edges2d = edges2d
        self.edges1d2d = edges1d2d
        self.edges1dfeats_norm = edges1dfeats_norm
        self.edges2dfeats_norm = edges2dfeats_norm
        self.static_1d_cols = static_1d_cols
        self.static_2d_cols = static_2d_cols
        self.edge1_cols = edge1_cols
        self.edge2_cols = edge2_cols
        self.norm_stats = norm_stats
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.node_id_col = node_id_col
        
        # Create static graph once (shared across all samples)
        self.static_graph = create_static_hetero_graph(
            static_1d_norm, static_2d_norm,
            edges1d, edges2d, edges1d2d,
            edges1dfeats_norm, edges2dfeats_norm,
            static_1d_cols, static_2d_cols,
            edge1_cols, edge2_cols,
            node_id_col=node_id_col,
        )
        
        # Get number of nodes for each type
        self.n_nodes_1d = len(static_1d_norm)
        self.n_nodes_2d = len(static_2d_norm)
        
    def _build_sample(self, n1, n2, tcol, t_start):
        """Extract and validate a single window sample. Returns a dict or None if invalid."""
        t_hist_end = t_start + self.history_len - 1
        t_pred_end = t_hist_end + self.forecast_len

        hist_1d = n1[(n1[tcol] >= t_start) & (n1[tcol] <= t_hist_end)]
        hist_2d = n2[(n2[tcol] >= t_start) & (n2[tcol] <= t_hist_end)]
        future_1d = n1[(n1[tcol] > t_hist_end) & (n1[tcol] <= t_pred_end)]
        future_2d = n2[(n2[tcol] > t_hist_end) & (n2[tcol] <= t_pred_end)]

        # Check windows are complete
        if (len(hist_1d) != self.n_nodes_1d * self.history_len or
                len(hist_2d) != self.n_nodes_2d * self.history_len or
                len(future_1d) != self.n_nodes_1d * self.forecast_len or
                len(future_2d) != self.n_nodes_2d * self.forecast_len):
            return None

        y_hist_1d = torch.tensor(
            hist_1d['water_level'].values.reshape(self.history_len, self.n_nodes_1d, 1),
            dtype=torch.float32)
        y_hist_2d = torch.tensor(
            hist_2d['water_level'].values.reshape(self.history_len, self.n_nodes_2d, 1),
            dtype=torch.float32)
        y_future_1d = torch.tensor(
            future_1d['water_level'].values.reshape(self.forecast_len, self.n_nodes_1d, 1),
            dtype=torch.float32)
        y_future_2d = torch.tensor(
            future_2d['water_level'].values.reshape(self.forecast_len, self.n_nodes_2d, 1),
            dtype=torch.float32)
        rain_hist_2d = torch.tensor(
            hist_2d['rainfall'].values.reshape(self.history_len, self.n_nodes_2d, 1),
            dtype=torch.float32)
        rain_future_2d = torch.tensor(
            future_2d['rainfall'].values.reshape(self.forecast_len, self.n_nodes_2d, 1),
            dtype=torch.float32)

        # Finite check
        if not (torch.isfinite(y_hist_1d).all() and torch.isfinite(y_hist_2d).all() and
                torch.isfinite(y_future_1d).all() and torch.isfinite(y_future_2d).all() and
                torch.isfinite(rain_hist_2d).all() and torch.isfinite(rain_future_2d).all()):
            return None

        # ================================================================
        # COMPREHENSIVE VALIDITY CHECKS (catch 90% of data leakage bugs)
        # ================================================================

        # (1) SHAPE & ALIGNMENT CHECKS
        assert y_hist_1d.shape == (self.history_len, self.n_nodes_1d, 1), \
            f"1D history shape mismatch: {y_hist_1d.shape} vs ({self.history_len}, {self.n_nodes_1d}, 1)"
        assert y_hist_2d.shape == (self.history_len, self.n_nodes_2d, 1), \
            f"2D history shape mismatch: {y_hist_2d.shape} vs ({self.history_len}, {self.n_nodes_2d}, 1)"
        assert y_future_1d.shape == (self.forecast_len, self.n_nodes_1d, 1), \
            f"1D future shape mismatch: {y_future_1d.shape} vs ({self.forecast_len}, {self.n_nodes_1d}, 1)"
        assert y_future_2d.shape == (self.forecast_len, self.n_nodes_2d, 1), \
            f"2D future shape mismatch: {y_future_2d.shape} vs ({self.forecast_len}, {self.n_nodes_2d}, 1)"
        assert rain_hist_2d.shape[0] == self.history_len and rain_hist_2d.shape[1] == self.n_nodes_2d, \
            f"Rain history shape mismatch: {rain_hist_2d.shape} vs ({self.history_len}, {self.n_nodes_2d}, *)"
        assert rain_future_2d.shape[0] == self.forecast_len and rain_future_2d.shape[1] == self.n_nodes_2d, \
            f"Rain future shape mismatch: {rain_future_2d.shape} vs ({self.forecast_len}, {self.n_nodes_2d}, *)"

        # (2) TIME ALIGNMENT CHECK
        t_hist_values = sorted(hist_2d[tcol].unique())
        t_future_values = sorted(future_2d[tcol].unique())
        assert len(t_hist_values) == self.history_len, \
            f"History timestep count mismatch: {len(t_hist_values)} vs {self.history_len}"
        assert len(t_future_values) == self.forecast_len, \
            f"Future timestep count mismatch: {len(t_future_values)} vs {self.forecast_len}"
        assert t_hist_values[-1] + 1 == t_future_values[0], \
            f"Timestep gap: history ends at {t_hist_values[-1]}, future starts at {t_future_values[0]}"

        # (3) EVENT BOUNDARY CHECK
        hist_events_2d = hist_2d.get('event_id', None)
        future_events_2d = future_2d.get('event_id', None)
        if hist_events_2d is not None:
            assert hist_events_2d.iloc[0] == future_events_2d.iloc[0], \
                f"Event boundary crossed: history={hist_events_2d.iloc[0]}, future={future_events_2d.iloc[0]}"

        # (4) NODE CONSISTENCY
        hist_2d_nodes_per_t = hist_2d.groupby(tcol)[self.node_id_col].nunique().values
        future_2d_nodes_per_t = future_2d.groupby(tcol)[self.node_id_col].nunique().values
        assert (hist_2d_nodes_per_t == self.n_nodes_2d).all(), \
            f"History missing nodes: {hist_2d_nodes_per_t}"
        assert (future_2d_nodes_per_t == self.n_nodes_2d).all(), \
            f"Future missing nodes: {future_2d_nodes_per_t}"

        # (5) NO-LEAKAGE CHECK
        assert t_hist_end < t_future_values[0], \
            f"History/future overlap: history ends at {t_hist_end}, future starts at {t_future_values[0]}"

        # (6) SANITY SLICE CHECK
        last_hist_val = hist_2d[hist_2d[tcol] == t_hist_values[-1]]['water_level'].iloc[0]
        assert abs(last_hist_val - y_hist_2d[-1, 0, 0].item()) < 1e-5, \
            f"History value mismatch at last timestep: raw={last_hist_val}, extracted={y_hist_2d[-1, 0, 0].item()}"
        first_fut_val = future_2d[future_2d[tcol] == t_future_values[0]]['water_level'].iloc[0]
        assert abs(first_fut_val - y_future_2d[0, 0, 0].item()) < 1e-5, \
            f"Future value mismatch at first timestep: raw={first_fut_val}, extracted={y_future_2d[0, 0, 0].item()}"

        return {
            'y_hist_1d': y_hist_1d,
            'y_hist_2d': y_hist_2d,
            'rain_hist_2d': rain_hist_2d,
            'y_future_1d': y_future_1d,
            'y_future_2d': y_future_2d,
            'rain_future_2d': rain_future_2d,
        }

    def __iter__(self):
        """
        Yield pre-formed batches of size self.batch_size.

        Events are shuffled each epoch (when shuffle=True).
        Windows within each event are iterated in temporal order — never shuffled.
        Tail windows that don't fill a complete batch are dropped (drop_last semantics).
        All windows in a batch are guaranteed to come from the same event.
        """
        event_list = list(self.event_file_list)
        if self.shuffle:
            random.shuffle(event_list)

        for item in event_list:
            if len(item) == 3:
                event_idx, event_dir, split_label = item
            else:
                event_idx, event_dir = item

            # Load event data
            n1 = pd.read_csv(event_dir + "/1d_nodes_dynamic_all.csv")
            n2 = pd.read_csv(event_dir + "/2d_nodes_dynamic_all.csv")

            # Drop excluded columns
            exclude_1d = self.norm_stats.get("exclude_1d", [])
            exclude_2d = self.norm_stats.get("exclude_2d", [])
            n1 = n1.drop(columns=[c for c in exclude_1d if c in n1.columns])
            n2 = n2.drop(columns=[c for c in exclude_2d if c in n2.columns])

            # Apply normalization
            normalizer_1d = self.norm_stats["normalizer_1d"]
            normalizer_2d = self.norm_stats["normalizer_2d"]
            n1 = normalizer_1d.transform_dynamic(n1, exclude_cols=None)
            n2 = normalizer_2d.transform_dynamic(n2, exclude_cols=None)

            # Merge with static features
            n1 = pd.merge(n1, self.static_1d_norm, on=self.node_id_col, how="left")
            n2 = pd.merge(n2, self.static_2d_norm, on=self.node_id_col, how="left")
            n2 = preprocess_2d_nodes(n2)

            # Add temporal features
            n1 = add_temporal_features(n1, has_rainfall=False)
            n2 = add_temporal_features(n2, has_rainfall=True)

            # Sort by timestep and node_idx (temporal order preserved)
            tcol = "timestep_raw" if "timestep_raw" in n1.columns else "timestep"
            n1 = n1.sort_values([tcol, self.node_id_col]).reset_index(drop=True)
            n2 = n2.sort_values([tcol, self.node_id_col]).reset_index(drop=True)

            min_timestep = int(n1[tcol].min())
            max_timestep = int(n1[tcol].max())

            # Collect all valid windows for this event in temporal order
            event_samples = []
            for t_start in range(min_timestep, max_timestep - self.history_len - self.forecast_len + 2):
                sample = self._build_sample(n1, n2, tcol, t_start)
                if sample is not None:
                    event_samples.append(sample)

            # Yield complete batches — all from this event, in temporal order
            # Tail remainder is dropped (drop_last semantics per event)
            for i in range(0, len(event_samples) - self.batch_size + 1, self.batch_size):
                batch_samples = event_samples[i: i + self.batch_size]
                yield {
                    'static_graph': NonBatchableGraph(self.static_graph),
                    'y_hist_1d':      torch.stack([s['y_hist_1d']      for s in batch_samples]),
                    'y_hist_2d':      torch.stack([s['y_hist_2d']      for s in batch_samples]),
                    'rain_hist_2d':   torch.stack([s['rain_hist_2d']   for s in batch_samples]),
                    'y_future_1d':    torch.stack([s['y_future_1d']    for s in batch_samples]),
                    'y_future_2d':    torch.stack([s['y_future_2d']    for s in batch_samples]),
                    'rain_future_2d': torch.stack([s['rain_future_2d'] for s in batch_samples]),
                }


# =========================
# Lazy initialization for expensive data loading
# =========================
_data_initialized = False
# Lazy initialization: defer expensive data loading until first use
from data_lazy import initialize_data


# =========================
# PASS 2: build datasets
# =========================

# Create single dataset with all events' windows that can shuffle across events
class MultiEventGraphStream(IterableDataset):
    def __init__(self, event_file_list, edges1d, edges2d, edges1d2d, edges1dfeats, edges2dfeats,
                 node1d_cols, node2d_cols, edge1_cols, edge2_cols, norm_stats, static_1d, static_2d,
                 shuffle=True):
        super().__init__()
        self.event_file_list = event_file_list  # List of (event_idx, file_path) tuples
        self.edges1d = edges1d
        self.edges2d = edges2d
        self.edges1d2d = edges1d2d
        self.edges1dfeats = edges1dfeats
        self.edges2dfeats = edges2dfeats
        self.node1d_cols = node1d_cols
        self.node2d_cols = node2d_cols
        self.edge1_cols = edge1_cols
        self.edge2_cols = edge2_cols
        self.norm_stats = norm_stats
        self.static_1d = static_1d
        self.static_2d = static_2d
        self.shuffle = shuffle
        
        # Build per-event window lists (avoid per-window file reads)
        self.event_windows = []
        for event_idx, f in event_file_list:
            # Read just to get timestep count, then discard
            nodes1d_temp = pd.read_csv(f + "/1d_nodes_dynamic_all.csv")
            num_timesteps = len(nodes1d_temp.timestep.unique())
            del nodes1d_temp  # Free memory immediately

            # Generate window indices (11 timesteps: from i to i+10 inclusive)
            idxs = [(i, i + 10) for i in range(num_timesteps - 10)]
            self.event_windows.append((event_idx, f, idxs))
    
    def __iter__(self):
        events = list(self.event_windows)
        if self.shuffle:
            random.shuffle(events)  # Shuffle event order

        for event_idx, file_path, windows in events:
            # Load event data once, reuse for all windows in this event
            nodes1d = pd.read_csv(file_path + "/1d_nodes_dynamic_all.csv")
            nodes2d = pd.read_csv(file_path + "/2d_nodes_dynamic_all.csv")

            # Preserve raw timestep for window slicing before normalization
            if "timestep" in nodes1d.columns:
                nodes1d["timestep_raw"] = nodes1d["timestep"]
            if "timestep" in nodes2d.columns:
                nodes2d["timestep_raw"] = nodes2d["timestep"]
            
            # Drop excluded columns
            exclude_1d = self.norm_stats.get('exclude_1d', [])
            exclude_2d = self.norm_stats.get('exclude_2d', [])
            nodes1d = nodes1d.drop(columns=[c for c in exclude_1d if c in nodes1d.columns])
            nodes2d = nodes2d.drop(columns=[c for c in exclude_2d if c in nodes2d.columns])
            
            # Apply dynamic normalization (before merge with static)
            normalizer_1d = self.norm_stats['normalizer_1d']
            normalizer_2d = self.norm_stats['normalizer_2d']
            nodes1d = normalizer_1d.transform_dynamic(nodes1d, exclude_cols=None)
            nodes2d = normalizer_2d.transform_dynamic(nodes2d, exclude_cols=None)
            
            # Merge with pre-normalized static features
            nodes1d = pd.merge(nodes1d, self.static_1d, on="node_idx", how="left")
            nodes2d = pd.merge(nodes2d, self.static_2d, on="node_idx", how="left")
            nodes2d = preprocess_2d_nodes(nodes2d)

            # Add temporal features
            nodes1d = add_temporal_features(nodes1d, has_rainfall=False)
            nodes2d = add_temporal_features(nodes2d, has_rainfall=True)
            
            # Normalize engineered temporal features
            engineered_1d = [c for c in nodes1d.columns if c.startswith('cum_') or c.startswith('mean_')]
            engineered_2d = [c for c in nodes2d.columns if c.startswith('cum_') or c.startswith('mean_')]
            
            for col in engineered_1d:
                if col in normalizer_1d.dynamic_params:
                    vals = nodes1d[col].astype(float).values
                    params = normalizer_1d.dynamic_params[col]
                    
                    if params['log']:
                        vals = np.log1p(np.abs(vals)) * np.sign(vals)
                    
                    if params['max'] > params['min']:
                        vals = (vals - params['min']) / (params['max'] - params['min'])
                    else:
                        vals = np.zeros_like(vals)
                    
                    # Convert column to float to avoid dtype errors
                    nodes1d[col] = vals.astype(np.float32)
            
            for col in engineered_2d:
                if col in normalizer_2d.dynamic_params:
                    vals = nodes2d[col].astype(float).values
                    params = normalizer_2d.dynamic_params[col]
                    
                    if params['log']:
                        vals = np.log1p(np.abs(vals)) * np.sign(vals)
                    
                    if params['max'] > params['min']:
                        vals = (vals - params['min']) / (params['max'] - params['min'])
                    else:
                        vals = np.zeros_like(vals)
                    
                    # Convert column to float to avoid dtype errors
                    nodes2d[col] = vals.astype(np.float32)
           
            windows_iter = list(windows)
            if self.shuffle:
                random.shuffle(windows_iter)

            for start_idx, end_idx in windows_iter:
                data = create_directed_temporal_graph(
                    start_idx, end_idx,
                    nodes1d, nodes2d,
                    self.edges1d, self.edges2d, self.edges1d2d,
                    self.edges1dfeats, self.edges2dfeats,
                    self.node1d_cols, self.node2d_cols,
                    self.edge1_cols, self.edge2_cols,
                    norm_stats=self.norm_stats
                )
                
                # Skip incomplete windows
                if data is None:
                    continue
                
                yield data

# =========================
# Cached Graph Dataset (fast loading from .pt files)
# =========================
class CachedGraphDataset(IterableDataset):
    """Load pre-cached graphs from disk (very fast)."""
    def __init__(self, cache_dir="cache/graphs", shuffle=True):
        super().__init__()
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        
        # Find all cached graph files
        graph_files = sorted(glob.glob(os.path.join(cache_dir, "event*.pt")))
        if not graph_files:
            raise FileNotFoundError(f"No cached graphs found in {cache_dir}. Run preprocess_cache.py first.")
        
        self.graph_files = graph_files
        print(f"[INFO] CachedGraphDataset: loaded {len(self.graph_files)} cached graphs")
    
    def __iter__(self):
        files = list(self.graph_files)
        if self.shuffle:
            random.shuffle(files)
        
        for fpath in files:
            data = torch.load(fpath)
            yield data


def cache_exists(cache_dir="cache/graphs"):
    """Check if cache directory has cached graphs."""
    return os.path.exists(cache_dir) and len(glob.glob(os.path.join(cache_dir, "event*.pt"))) > 0


def _running_in_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


DL_NUM_WORKERS = int(
    os.environ.get("FLOOD_FLUENT_WORKERS", "0" if _running_in_notebook() else "0")
)
DL_PIN_MEMORY = False
DL_PREFETCH = 2
DL_PERSISTENT = DL_NUM_WORKERS > 0

dl_kwargs = {
    "batch_size": 8,
    # shuffle is handled inside TemporalGraphStream for IterableDataset
    "drop_last": True,
    "num_workers": DL_NUM_WORKERS,
    "pin_memory": DL_PIN_MEMORY,
    "persistent_workers": DL_PERSISTENT,
}
if DL_NUM_WORKERS > 0:
    dl_kwargs["prefetch_factor"] = DL_PREFETCH


def get_dataloader(use_cache=True, cache_dir="cache/graphs"):
    """
    Get dataloader.
    
    Args:
        use_cache: If True, use cached graphs if available. If not available, build cache automatically.
        cache_dir: Path to cache directory.
    """
    if use_cache:
        if cache_exists(cache_dir):
            print(f"[INFO] Using cached graphs from {cache_dir}")
            dataset = CachedGraphDataset(cache_dir=cache_dir, shuffle=True)
        else:
            print(f"[INFO] Cache not found at {cache_dir}. Building cache...")
            # Import preprocess_cache here to avoid circular imports
            from preprocess_cache import build_cache
            build_cache(cache_dir=cache_dir)
            dataset = CachedGraphDataset(cache_dir=cache_dir, shuffle=True)
    else:
        print("[INFO] Using live data stream (not cached)")
        dataset = MultiEventGraphStream(
            event_file_list, edges1d, edges2d, edges1d2d, edges1dfeats, edges2dfeats,
            node1d_cols, node2d_cols, edge1_cols, edge2_cols, norm_stats, static_1d, static_2d,
            shuffle=True
        )
    
    return DataLoader(dataset, **dl_kwargs)


def get_dataloader_old():
    """Legacy: use live data stream without caching."""
    dataset = MultiEventGraphStream(
        event_file_list, edges1d, edges2d, edges1d2d, edges1dfeats, edges2dfeats,
        node1d_cols, node2d_cols, edge1_cols, edge2_cols, norm_stats, static_1d, static_2d,
        shuffle=True
    )
    return DataLoader(dataset, **dl_kwargs)


def get_recurrent_dataloader(history_len=10, forecast_len=1, batch_size=8, shuffle=True, split='train'):
    """
    Get dataloader for recurrent model with static graph + dynamic time series.
    
    Args:
        history_len: Number of historical timesteps for warm start
        forecast_len: Number of future timesteps to predict
        batch_size: Batch size
        shuffle: Whether to shuffle events
        split: Data split to use ('train', 'val', 'test', or 'all')
    
    Returns:
        DataLoader that yields dicts with:
          - static_graph: HeteroData with static features
          - y_hist_1d, y_hist_2d: [H, N, 1] historical water levels
          - rain_hist_2d: [H, N, R] historical rainfall
          - y_future_1d, y_future_2d: [T, N, 1] future water levels (labels)
          - rain_future_2d: [T, N, R] future rainfall
    """
    # Initialize data lazily on first call
    data = initialize_data()
    
    # Select event list based on split
    if split == 'train':
        event_file_list = data['train_event_file_list']
    elif split == 'val':
        event_file_list = data['val_event_file_list']
    elif split == 'test':
        event_file_list = data['test_event_file_list']
    elif split == 'all':
        event_file_list = data['event_file_list']
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', 'test', or 'all'.")
    
    static_1d_sorted = data['static_1d_sorted']
    static_2d_sorted = data['static_2d_sorted']
    edges1d = data['edges1d']
    edges2d = data['edges2d']
    edges1d2d = data['edges1d2d']
    edges1dfeats = data['edges1dfeats']
    edges2dfeats = data['edges2dfeats']
    static_1d_cols = data['static_1d_cols']
    static_2d_cols = data['static_2d_cols']
    edge1_cols = data['edge1_cols']
    edge2_cols = data['edge2_cols']
    norm_stats = data['norm_stats']
    node_id_col = data['NODE_ID_COL']
    
    dataset = RecurrentFloodDataset(
        event_file_list=event_file_list,
        static_1d_norm=static_1d_sorted,
        static_2d_norm=static_2d_sorted,
        edges1d=edges1d,
        edges2d=edges2d,
        edges1d2d=edges1d2d,
        edges1dfeats_norm=edges1dfeats,
        edges2dfeats_norm=edges2dfeats,
        static_1d_cols=static_1d_cols,
        static_2d_cols=static_2d_cols,
        edge1_cols=edge1_cols,
        edge2_cols=edge2_cols,
        norm_stats=norm_stats,
        history_len=history_len,
        forecast_len=forecast_len,
        batch_size=batch_size,
        shuffle=shuffle,
        node_id_col=node_id_col,
    )

    # Collate function: dataset already yields pre-formed batches, so the
    # DataLoader batch size is 1 — we just unwrap the outer list of length 1.
    def collate_fn(batch):
        if len(batch) == 0:
            return None
        item = batch[0]
        static_graph = item['static_graph']
        if isinstance(static_graph, NonBatchableGraph):
            static_graph = static_graph.graph
        return {
            'static_graph':   static_graph,
            'y_hist_1d':      item['y_hist_1d'],
            'y_hist_2d':      item['y_hist_2d'],
            'rain_hist_2d':   item['rain_hist_2d'],
            'y_future_1d':    item['y_future_1d'],
            'y_future_2d':    item['y_future_2d'],
            'rain_future_2d': item['rain_future_2d'],
        }

    # Use torch DataLoader (not PyG) to avoid automatic batching of HeteroData.
    # batch_size=1 because RecurrentFloodDataset already yields pre-formed batches.
    from torch.utils.data import DataLoader as TorchDataLoader
    return TorchDataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )


# =========================
# Model configuration helpers for new recurrent architecture
# =========================

def get_model_config():
    """
    Get configuration dict for initializing FloodAutoregressiveHeteroModel.
    
    Returns dict with:
        - node_types: list of node type names
        - edge_types: list of (src, rel, dst) tuples
        - node_static_dims: dict of static feature dimensions per node type
        - node_dyn_input_dims: dict of dynamic input dimensions per node type
        - edge_static_dims: dict of static edge feature dimensions per edge type
        - pred_node_type: which node type to predict (can be "oneD" or "twoD")
    """
    # Initialize data lazily on first call
    data = initialize_data()
    static_1d_cols = data['static_1d_cols']
    static_2d_cols = data['static_2d_cols']
    edge1_cols = data['edge1_cols']
    edge2_cols = data['edge2_cols']
    
    # Define node types
    node_types = ["oneD", "twoD"]
    
    # Define edge types
    edge_types = [
        ("oneD", "oneDedge", "oneD"),
        ("twoD", "twoDedge", "twoD"),
        ("twoD", "twoDoneD", "oneD"),
    ]
    
    # Static feature dimensions
    node_static_dims = {
        "oneD": len(static_1d_cols),
        "twoD": len(static_2d_cols),
    }
    
    # Dynamic input dimensions (what make_x_dyn will provide)
    # 1D: water_level only (1 feature)
    # 2D: water_level + rainfall (2 features)
    node_dyn_input_dims = {
        "oneD": 1,  # water_level
        "twoD": 2,  # water_level + rainfall
    }
    
    # Static edge feature dimensions
    edge_static_dims = {
        ("oneD", "oneDedge", "oneD"): len(edge1_cols),
        ("twoD", "twoDedge", "twoD"): len(edge2_cols),
        ("twoD", "twoDoneD", "oneD"): 1,  # Cross-type edges have placeholder features
    }
    
    return {
        "node_types": node_types,
        "edge_types": edge_types,
        "node_static_dims": node_static_dims,
        "node_dyn_input_dims": node_dyn_input_dims,
        "edge_static_dims": edge_static_dims,
        # pred_node_type removed: model now predicts both oneD and twoD nodes
    }


def get_make_x_dyn_fn():
    """
    Return the make_x_dyn function configured for this dataset.
    
    This function can be passed directly to model.forward_unroll().
    """
    def make_x_dyn_wrapper(y_pred_nodes, rain_pred_nodes, data):
        """
        Wrapper for make_x_dyn that matches the model's expected signature.
        
        Args:
            y_pred_nodes: [N_pred, 1] water level for prediction node type
            rain_pred_nodes: [N_pred, R] rainfall for prediction node type
            data: HeteroData static graph
        
        Returns:
            x_dyn_t dict for all node types
        """
        # Assumption: pred_node_type is "twoD"
        # For 1D nodes, use zeros or previous state (model will learn to propagate)
        n_1d = data["oneD"].num_nodes
        device = y_pred_nodes.device
        
        # For 1D nodes, use zero water level (or could propagate from 2D via graph)
        y_pred_1d = torch.zeros((n_1d, 1), device=device)
        
        # Call the main make_x_dyn function
        return make_x_dyn(
            y_pred_1d=y_pred_1d,
            y_pred_2d=y_pred_nodes,  # 2D is the pred node type
            rain_2d=rain_pred_nodes,
            data=data,
        )
    
    return make_x_dyn_wrapper
