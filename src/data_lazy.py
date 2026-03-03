"""
Lazy-loaded data initialization for FloodLM.
This module defers expensive data loading until first use.
Features parallelized event preprocessing.
"""

import os
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

# Global state for lazy initialization
_initialized = False
_cache = {}
_lock = threading.Lock()

CACHE_FILE = None  # Set based on SELECTED_MODEL in initialize_data()

def _get_cache_path():
    """Get cache file path for current model."""
    import os
    from data_config import SELECTED_MODEL, BASE_PATH
    cache_dir = os.path.join(BASE_PATH, '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{SELECTED_MODEL}_preprocessed.pkl')

def _load_cache_from_disk():
    """Try to load preprocessed cache from disk."""
    try:
        import pickle
        cache_path = _get_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"[INFO] Cache load failed (will recompute): {e}")
    return None

def _save_cache_to_disk(cache_data):
    """Save preprocessed cache to disk."""
    try:
        import pickle
        cache_path = _get_cache_path()
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"[INFO] Cache saved to {cache_path}")
    except Exception as e:
        print(f"[WARN] Failed to save cache: {e}")

def _split_events(event_dirs, val_split=0.1, test_split=0.1, random_seed=42):
    """Split events into train/val/test sets.
    
    Args:
        event_dirs: List of event directory paths
        val_split: Fraction for validation (default 0.1)
        test_split: Fraction for test (default 0.1)
        random_seed: Random seed for reproducibility
        
    Returns:
        dict with 'train', 'val', 'test' event lists
    """
    import random
    
    # Shuffle with fixed seed for reproducibility
    shuffled = event_dirs.copy()
    random.seed(random_seed)
    random.shuffle(shuffled)
    
    n_total = len(shuffled)
    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test
    
    train_events = shuffled[:n_train]
    val_events = shuffled[n_train:n_train+n_val]
    test_events = shuffled[n_train+n_val:]
    
    print(f"[INFO] Data split: {n_train} train, {n_val} val, {n_test} test events")
    print(f"[INFO] Split ratios: train={n_train/n_total:.1%}, val={n_val/n_total:.1%}, test={n_test/n_total:.1%}")
    
    return {
        'train': train_events,
        'val': val_events,
        'test': test_events,
        'random_seed': random_seed,
    }

def _process_event_for_dynamic_pass(args):
    """Process a single event for dynamic feature normalization (Pass 1).
    Returns dataframes needed for normalizer updates."""
    event_dir, NODE_ID_COL, EXCLUDE_1D_DYNAMIC, EXCLUDE_2D_DYNAMIC = args
    
    import pandas as pd
    import sys
    from pathlib import Path
    
    # Setup path for imports
    src_dir = str(Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    try:
        n1_dyn = pd.read_csv(event_dir + "/1d_nodes_dynamic_all.csv")
        n2_dyn = pd.read_csv(event_dir + "/2d_nodes_dynamic_all.csv")
        n1_dyn = n1_dyn.drop(columns=[c for c in EXCLUDE_1D_DYNAMIC if c in n1_dyn.columns])
        n2_dyn = n2_dyn.drop(columns=[c for c in EXCLUDE_2D_DYNAMIC if c in n2_dyn.columns])
        return (n1_dyn, n2_dyn)
    except Exception as e:
        print(f"[ERROR] Failed to process event {event_dir}: {e}")
        return (None, None)


def _parallel_process_events(event_dirs, worker_func, args_builder, max_workers=None, use_processes=False):
    """
    Process events in parallel using ThreadPoolExecutor or ProcessPoolExecutor.
    
    Args:
        event_dirs: List of event directory paths
        worker_func: Function to call for each event
        args_builder: Function(event_dir) -> args tuple for worker_func
        max_workers: Number of parallel workers (default: auto)
        use_processes: If True, use ProcessPoolExecutor (CPU-bound), else ThreadPoolExecutor (I/O-bound)
    
    Returns:
        List of results in same order as event_dirs
    """
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)  # Default to 8 or CPU count
    
    results = [None] * len(event_dirs)
    
    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with ExecutorClass(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_func, args_builder(event_dir, i)): i 
            for i, event_dir in enumerate(event_dirs)
        }
        
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
                completed += 1
                if (completed) % max(1, len(event_dirs) // 10) == 0 or completed == len(event_dirs):
                    print(f"[INFO]   {completed}/{len(event_dirs)} events processed...")
            except Exception as e:
                print(f"[ERROR] Event processing failed: {e}")
                results[idx] = (None, None)
    
    return results

def initialize_data():
    """Initialize data loading (called lazily on first dataloader request)."""
    global _initialized, _cache
    
    if _initialized:
        return _cache
    
    # Try loading from disk cache first
    print(f"\n[INFO] ========== INITIALIZING DATA FROM DISK ==========")
    disk_cache = _load_cache_from_disk()
    if disk_cache is not None:
        _cache = disk_cache
        _initialized = True
        print(f"[INFO] Loaded preprocessed data from cache!")
        return _cache
    
    print(f"[INFO] Computing preprocessing (this may take a minute)...")
    
    import pandas as pd
    import torch
    import numpy as np
    
    # Setup path for relative imports
    import sys
    from pathlib import Path
    src_dir = str(Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    from data_config import TRAIN_PATH, SELECTED_MODEL, validate_data_paths
    from normalization import FeatureNormalizer
    from data import preprocess_2d_nodes, NORMALIZATION_VERBOSE
    
    # Validate paths
    try:
        validate_data_paths()
        print(f"[INFO] Data validation successful for {SELECTED_MODEL}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise
    
    # Load static data
    static_1d = pd.read_csv(f"{TRAIN_PATH}/1d_nodes_static.csv")
    static_2d = pd.read_csv(f"{TRAIN_PATH}/2d_nodes_static.csv")
    edges1dfeats = pd.read_csv(f"{TRAIN_PATH}/1d_edges_static.csv")
    edges2dfeats = pd.read_csv(f"{TRAIN_PATH}/2d_edges_static.csv")
    edges1d = pd.read_csv(f"{TRAIN_PATH}/1d_edge_index.csv")
    edges2d = pd.read_csv(f"{TRAIN_PATH}/2d_edge_index.csv")
    edges1d2d = pd.read_csv(f"{TRAIN_PATH}/1d2d_connections.csv")
    
    NODE_ID_COL = "node_idx"
    EDGE_ID_COL = "edge_idx" if "edge_idx" in edges1dfeats.columns else edges1dfeats.columns[0]
    
    # Normalize edges (PARALLELIZED)
    print("[INFO] Fitting edge feature normalization (parallel)...")
    normalizer_edge1d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
    normalizer_edge2d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
    
    from concurrent.futures import ThreadPoolExecutor
    def fit_edge_1d():
        normalizer_edge1d.fit_static(edges1dfeats.copy(), EDGE_ID_COL, skew_threshold=2.0)
        return edges1dfeats, normalizer_edge1d
    
    def fit_edge_2d():
        normalizer_edge2d.fit_static(edges2dfeats.copy(), EDGE_ID_COL, skew_threshold=2.0)
        return edges2dfeats, normalizer_edge2d
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_1d = executor.submit(fit_edge_1d)
        future_2d = executor.submit(fit_edge_2d)
        edges1dfeats_tmp, normalizer_edge1d = future_1d.result()
        edges2dfeats_tmp, normalizer_edge2d = future_2d.result()
    
    edges1dfeats = normalizer_edge1d.transform_static(edges1dfeats, EDGE_ID_COL)
    edges2dfeats = normalizer_edge2d.transform_static(edges2dfeats, EDGE_ID_COL)
    
    edge1_cols = [c for c in edges1dfeats.columns if c != EDGE_ID_COL]
    edge2_cols = [c for c in edges2dfeats.columns if c != EDGE_ID_COL]
    
    # Get event directories
    event_dirs = sorted(
        glob.glob(f"{TRAIN_PATH}/event_*"),
        key=lambda p: int(Path(p).name.split("_")[-1])
    )
    
    # Allow limiting events via environment variable (for quick tests)
    max_events = int(os.environ.get("FLOOD_MAX_EVENTS", len(event_dirs)))
    if max_events < len(event_dirs):
        event_dirs = event_dirs[:max_events]
        print(f"[INFO] Limiting to {max_events} events (FLOOD_MAX_EVENTS set)")
    
    print(f"[INFO] Found {len(event_dirs)} events total")
    
    # CRITICAL: Split events BEFORE computing normalization to prevent data leakage
    from data_config import VALIDATION_SPLIT, RANDOM_SEED
    try:
        from data_config import TEST_SPLIT
    except ImportError:
        TEST_SPLIT = 0.0
    event_splits = _split_events(
        event_dirs,
        val_split=VALIDATION_SPLIT,
        test_split=TEST_SPLIT,
        random_seed=RANDOM_SEED
    )
    
    train_events = event_splits['train']
    val_events = event_splits['val']
    test_events = event_splits['test']
    
    # Define exclusions
    EXCLUDE_1D_DYNAMIC = ['inlet_flow']
    EXCLUDE_2D_DYNAMIC = ['water_volume']
    
    # Initialize normalizers
    normalizer_1d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
    normalizer_2d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
    
    # Preprocess and fit static
    print("[INFO] Preprocessing static 2D nodes...")
    static_2d = preprocess_2d_nodes(static_2d)
    print("[INFO] Fitting static feature normalization...")
    normalizer_1d.fit_static(static_1d.copy(), NODE_ID_COL, skew_threshold=2.0)
    normalizer_2d.fit_static(static_2d.copy(), NODE_ID_COL, skew_threshold=2.0)
    static_1d = normalizer_1d.transform_static(static_1d, NODE_ID_COL)
    static_2d = normalizer_2d.transform_static(static_2d, NODE_ID_COL)
    
    # Stream through events for dynamic normalization (Pass 1) - PARALLELIZED
    print(f"[INFO] Streaming through {len(train_events)} TRAINING events for dynamic normalization (parallel)...")
    n1_temp = pd.read_csv(train_events[0] + "/1d_nodes_dynamic_all.csv")
    n2_temp = pd.read_csv(train_events[0] + "/2d_nodes_dynamic_all.csv")
    n1_temp = n1_temp.drop(columns=[c for c in EXCLUDE_1D_DYNAMIC if c in n1_temp.columns])
    n2_temp = n2_temp.drop(columns=[c for c in EXCLUDE_2D_DYNAMIC if c in n2_temp.columns])
    
    base_1d_dynamic_feats = [c for c in n1_temp.columns if c not in [NODE_ID_COL, 'timestep']]
    base_2d_dynamic_feats = [c for c in n2_temp.columns if c not in [NODE_ID_COL, 'timestep', 'rainfall']]
    dynamic_1d_cols = base_1d_dynamic_feats
    dynamic_2d_cols = base_2d_dynamic_feats + ['rainfall']
    
    print(f"[INFO] 1D static features: {normalizer_1d.static_features}")
    print(f"[INFO] 2D static features: {normalizer_2d.static_features}")
    print(f"[INFO] Fitting dynamic feature normalization (streaming, parallel)...")
    print(f"[INFO] Using ONLY {len(train_events)} training events for normalization (avoiding data leakage)")
    normalizer_1d.init_dynamic_streaming(dynamic_1d_cols, exclude_cols=None)
    normalizer_2d.init_dynamic_streaming(dynamic_2d_cols, exclude_cols=None)
    
    # Parallel event reading for Pass 1 - USE ONLY TRAINING EVENTS
    def build_pass1_args(event_dir, idx):
        return (event_dir, NODE_ID_COL, EXCLUDE_1D_DYNAMIC, EXCLUDE_2D_DYNAMIC)
    
    results_pass1 = _parallel_process_events(
        train_events,  # CRITICAL: Only use training events for normalization
        _process_event_for_dynamic_pass,
        build_pass1_args,
        max_workers=8
    )
    
    # Sequential update (normalizer state is not thread-safe)
    for n1_dyn, n2_dyn in results_pass1:
        if n1_dyn is not None and n2_dyn is not None:
            normalizer_1d.update_dynamic_streaming(n1_dyn, exclude_cols=None)
            normalizer_2d.update_dynamic_streaming(n2_dyn, exclude_cols=None)
    
    normalizer_1d.finalize_dynamic_streaming(skew_threshold=2.0)
    normalizer_2d.finalize_dynamic_streaming(skew_threshold=2.0)

    # Derive column names from base dynamic + static features (no engineered cum/mean features)
    node1d_cols = base_1d_dynamic_feats + list(normalizer_1d.static_features)
    node2d_cols = dynamic_2d_cols + list(normalizer_2d.static_features)
    # Remove id/timestep cols that sneak in
    exclude_node_cols = {NODE_ID_COL, 'timestep', 'timestep_raw'}
    node1d_cols = [c for c in node1d_cols if c not in exclude_node_cols]
    node2d_cols = [c for c in node2d_cols if c not in exclude_node_cols]
    
    feature_type_1d = {col: ('static' if col in normalizer_1d.static_features else 'dynamic') for col in node1d_cols}
    feature_type_2d = {col: ('static' if col in normalizer_2d.static_features else 'dynamic') for col in node2d_cols}
    
    print(f"[INFO] Final 1D features: {len(node1d_cols)} ({sum(1 for v in feature_type_1d.values() if v=='static')} static, {sum(1 for v in feature_type_1d.values() if v=='dynamic')} dynamic)")
    print(f"[INFO] Final 2D features: {len(node2d_cols)} ({sum(1 for v in feature_type_2d.values() if v=='static')} static, {sum(1 for v in feature_type_2d.values() if v=='dynamic')} dynamic)")
    
    # Build norm_stats
    norm_stats = {
        "oneD_mu": torch.zeros(len(node1d_cols)),
        "oneD_sigma": torch.ones(len(node1d_cols)),
        "twoD_mu": torch.zeros(len(node2d_cols)),
        "twoD_sigma": torch.ones(len(node2d_cols)),
        "edge1_mu": torch.zeros(len(edge1_cols)),
        "edge1_sigma": torch.ones(len(edge1_cols)),
        "edge2_mu": torch.zeros(len(edge2_cols)),
        "edge2_sigma": torch.ones(len(edge2_cols)),
        "static_1d_params": normalizer_1d.static_params,
        "static_2d_params": normalizer_2d.static_params,
        "dynamic_1d_params": normalizer_1d.dynamic_params,
        "dynamic_2d_params": normalizer_2d.dynamic_params,
        "feature_type_1d": feature_type_1d,
        "feature_type_2d": feature_type_2d,
        "node1d_cols": node1d_cols,
        "node2d_cols": node2d_cols,
        "exclude_1d": EXCLUDE_1D_DYNAMIC,
        "exclude_2d": EXCLUDE_2D_DYNAMIC,
        "normalizer_1d": normalizer_1d,
        "normalizer_2d": normalizer_2d,
    }
    
    static_1d_cols = [c for c in normalizer_1d.static_features if c != NODE_ID_COL]
    static_2d_cols = [c for c in normalizer_2d.static_features if c != NODE_ID_COL]
    
    print(f"[INFO] Static-only 1D features: {len(static_1d_cols)}")
    print(f"[INFO] Static-only 2D features: {len(static_2d_cols)}")
    
    static_1d_sorted = static_1d.sort_values(NODE_ID_COL).reset_index(drop=True)
    static_2d_sorted = static_2d.sort_values(NODE_ID_COL).reset_index(drop=True)
    
    # Create event file lists with split labels
    train_event_file_list = [(event_idx, f, 'train') for event_idx, f in enumerate(train_events)]
    val_event_file_list = [(event_idx, f, 'val') for event_idx, f in enumerate(val_events)]
    test_event_file_list = [(event_idx, f, 'test') for event_idx, f in enumerate(test_events)]

    # Cache and return
    _cache = {
        'train_event_file_list': train_event_file_list,
        'val_event_file_list': val_event_file_list,
        'test_event_file_list': test_event_file_list,
        'event_splits': event_splits,  # Split metadata
        'edges1d': edges1d,
        'edges2d': edges2d,
        'edges1d2d': edges1d2d,
        'edges1dfeats': edges1dfeats,
        'edges2dfeats': edges2dfeats,
        'node1d_cols': node1d_cols,
        'node2d_cols': node2d_cols,
        'edge1_cols': edge1_cols,
        'edge2_cols': edge2_cols,
        'norm_stats': norm_stats,
        'static_1d': static_1d,
        'static_2d': static_2d,
        'static_1d_sorted': static_1d_sorted,
        'static_2d_sorted': static_2d_sorted,
        'static_1d_cols': static_1d_cols,
        'static_2d_cols': static_2d_cols,
        'NODE_ID_COL': NODE_ID_COL,
    }
    
    _initialized = True
    print(f"[INFO] Data initialization complete!")
    
    # Save to disk for next run
    _save_cache_to_disk(_cache)
    
    return _cache
