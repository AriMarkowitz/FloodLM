"""
Full-event data module for autoregressive training.

Reuses src/data_lazy.initialize_data() for normalization and preprocessing,
then wraps events into a FullEventDataset where each sample is one entire event.

Each sample:
    y_hist_1d:      [H, N_1d, 1]       warm-start ground truth
    y_hist_2d:      [H, N_2d, 1]
    rain_hist_2d:   [H, N_2d, 1]
    y_future_1d:    [T_future, N_1d, 1] autoregressive target (variable length)
    y_future_2d:    [T_future, N_2d, 1]
    rain_future_2d: [T_future, N_2d, 1] rainfall for rollout timesteps
"""

import os
import sys
import random
import hashlib
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader

# Make src/ importable
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
_ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
for _p in (_SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_lazy import initialize_data
from data import (
    create_static_hetero_graph, make_x_dyn,
    get_model_config, preprocess_2d_nodes, preprocess_1d_nodes,
)


NODE_ID_COL = "node_idx"


def _compute_cache_key(event_list, history_len):
    """Compute a deterministic cache key from event directories + history_len."""
    dirs = sorted(item[1] if len(item) >= 2 else str(item) for item in event_list)
    raw = f"h{history_len}|" + "|".join(dirs)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_dataset_cache_dir():
    """Return the dataset cache directory, creating it if needed."""
    cache_dir = os.path.join(_ROOT_DIR, 'data', '.dataset_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


class FullEventDataset(IterableDataset):
    """
    Yields one sample per flood event — the entire event as a single training item.

    History (first H timesteps) is used for warm-start (teacher-forced).
    Future (remaining T-H timesteps) is the autoregressive target.

    All events are pre-loaded into CPU memory at init (typically <500 MB).
    Processed tensors are cached to disk as .pt files for fast reloading.
    """

    def __init__(self, event_list, static_1d_norm, static_2d_norm, norm_stats,
                 history_len=10, shuffle=True, cache_tag=''):
        super().__init__()
        self.shuffle = shuffle
        self.history_len = history_len
        self._samples = []

        # Try loading from disk cache
        cache_key = _compute_cache_key(event_list, history_len)
        cache_file = os.path.join(
            _get_dataset_cache_dir(),
            f"fullevent_{cache_tag}_{cache_key}.pt" if cache_tag else f"fullevent_{cache_key}.pt"
        )
        if os.path.exists(cache_file):
            t0 = time.time()
            print(f"[INFO] Loading cached dataset from {cache_file}...")
            self._samples = torch.load(cache_file, weights_only=False)
            print(f"[INFO] Loaded {len(self._samples)} events from cache in {time.time()-t0:.1f}s "
                  f"(T_future range: {self._min_future()}-{self._max_future()} steps)")
            return

        normalizer_1d = norm_stats['normalizer_1d']
        normalizer_2d = norm_stats['normalizer_2d']
        exclude_1d = norm_stats.get('exclude_1d', [])
        exclude_2d = norm_stats.get('exclude_2d', [])

        t0 = time.time()
        print(f"[INFO] Pre-loading {len(event_list)} events into memory...")
        for item in event_list:
            # Handle both (idx, dir) and (idx, dir, split) tuples
            if len(item) == 3:
                event_idx, event_dir, _split = item
            else:
                event_idx, event_dir = item

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
                print(f"[WARN] Event {event_dir} has only {T_total} timesteps, skipping")
                continue

            N_1d = len(n1[n1[tcol] == timesteps[0]])
            N_2d = len(n2[n2[tcol] == timesteps[0]])

            # Vectorized reshape: [T*N, 1] → [T, N, 1]
            wl_1d = torch.tensor(
                n1.sort_values([tcol, NODE_ID_COL])['water_level'].values,
                dtype=torch.float32).reshape(T_total, N_1d, 1)
            wl_2d = torch.tensor(
                n2.sort_values([tcol, NODE_ID_COL])['water_level'].values,
                dtype=torch.float32).reshape(T_total, N_2d, 1)
            rain_vals = (
                n2.sort_values([tcol, NODE_ID_COL])['rainfall'].values
                if 'rainfall' in n2.columns
                else np.zeros(T_total * N_2d, dtype=np.float32)
            )
            rain_2d = torch.tensor(rain_vals, dtype=torch.float32).reshape(T_total, N_2d, 1)

            H = history_len
            self._samples.append({
                'y_hist_1d':      wl_1d[:H],        # [H, N_1d, 1]
                'y_hist_2d':      wl_2d[:H],        # [H, N_2d, 1]
                'rain_hist_2d':   rain_2d[:H],      # [H, N_2d, 1]
                'y_future_1d':    wl_1d[H:],        # [T_future, N_1d, 1]
                'y_future_2d':    wl_2d[H:],        # [T_future, N_2d, 1]
                'rain_future_2d': rain_2d[H:],      # [T_future, N_2d, 1]
            })

        elapsed = time.time() - t0
        print(f"[INFO] Pre-loaded {len(self._samples)} events in {elapsed:.1f}s "
              f"(T_future range: {self._min_future()}-{self._max_future()} steps)")

        # Save cache for next time
        try:
            torch.save(self._samples, cache_file)
            cache_mb = os.path.getsize(cache_file) / 1024 / 1024
            print(f"[INFO] Saved dataset cache to {cache_file} ({cache_mb:.1f} MB)")
        except Exception as e:
            print(f"[WARN] Failed to save dataset cache: {e}")

    def _min_future(self):
        if not self._samples:
            return 0
        return min(s['y_future_1d'].shape[0] for s in self._samples)

    def _max_future(self):
        if not self._samples:
            return 0
        return max(s['y_future_1d'].shape[0] for s in self._samples)

    def _t_future_counts(self):
        """Return dict mapping T_future → count of events with that length."""
        counts = defaultdict(int)
        for s in self._samples:
            counts[s['y_future_1d'].shape[0]] += 1
        return dict(sorted(counts.items()))

    def __len__(self):
        return len(self._samples)

    def __iter__(self):
        """Yield individual samples (B=1 mode)."""
        indices = list(range(len(self._samples)))
        if self.shuffle:
            random.shuffle(indices)
        for i in indices:
            yield self._samples[i]

    def iter_grouped(self, batch_size: int):
        """
        Yield batches of up to `batch_size` events with the **same** T_future.

        Events are grouped by their T_future value (no padding needed).
        Within each group, events are shuffled. Groups themselves are shuffled
        so the model doesn't always see the same T_future order each epoch.

        Each yielded batch is a dict with stacked tensors:
            y_hist_1d:      [B, H, N_1d, 1]
            y_hist_2d:      [B, H, N_2d, 1]
            rain_hist_2d:   [B, H, N_2d, 1]
            y_future_1d:    [B, T_future, N_1d, 1]
            y_future_2d:    [B, T_future, N_2d, 1]
            rain_future_2d: [B, T_future, N_2d, 1]

        B may be less than batch_size for the last batch of each group.
        """
        # Group indices by T_future
        groups = defaultdict(list)
        for idx, s in enumerate(self._samples):
            T = s['y_future_1d'].shape[0]
            groups[T].append(idx)

        # Build list of (T_future, batch_indices) tuples
        all_batches = []
        for T, indices in groups.items():
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                all_batches.append((T, batch_indices))

        # Shuffle batches across groups so training order is varied
        if self.shuffle:
            random.shuffle(all_batches)

        for T, batch_indices in all_batches:
            samples = [self._samples[i] for i in batch_indices]
            batch = {
                key: torch.stack([s[key] for s in samples], dim=0)
                for key in samples[0].keys()
            }
            yield batch


def get_full_event_dataloader(split='train', shuffle=True, history_len=10):
    """
    Get DataLoader yielding full-event samples.

    Returns DataLoader with batch_size=1 (events have variable T_future).
    Each item is a dict with [H, N, 1] / [T_future, N, 1] tensors.
    """
    data = initialize_data()

    if split == 'train':
        event_list = data['train_event_file_list']
    elif split == 'val':
        event_list = data['val_event_file_list']
    elif split == 'test':
        event_list = data['test_event_file_list']
    elif split == 'all':
        event_list = (
            data['train_event_file_list'] +
            data['val_event_file_list'] +
            data['test_event_file_list']
        )
    else:
        raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'val', 'test', 'all'.")

    dataset = FullEventDataset(
        event_list=event_list,
        static_1d_norm=data['static_1d_sorted'],
        static_2d_norm=data['static_2d_sorted'],
        norm_stats=data['norm_stats'],
        history_len=history_len,
        shuffle=shuffle,        cache_tag=split,    )

    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda b: b[0] if b else None,
        num_workers=0,
        pin_memory=True,
    )


def get_full_event_dataset(split='train', shuffle=True, history_len=10):
    """
    Get FullEventDataset directly (for grouped batching via iter_grouped).

    Use dataset.iter_grouped(batch_size) for B>1 same-length batches,
    or iterate normally for B=1.
    """
    data = initialize_data()

    if split == 'train':
        event_list = data['train_event_file_list']
    elif split == 'val':
        event_list = data['val_event_file_list']
    elif split == 'test':
        event_list = data['test_event_file_list']
    elif split == 'all':
        event_list = (
            data['train_event_file_list'] +
            data['val_event_file_list'] +
            data['test_event_file_list']
        )
    else:
        raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'val', 'test', 'all'.")

    return FullEventDataset(
        event_list=event_list,
        static_1d_norm=data['static_1d_sorted'],
        static_2d_norm=data['static_2d_sorted'],
        norm_stats=data['norm_stats'],
        history_len=history_len,
        shuffle=shuffle,
        cache_tag=split,
    )
