"""
Full-event data module for autoregressive training.

Reuses src/data_lazy.initialize_data() for normalization and preprocessing,
then wraps events into a FullEventDataset where each sample is one entire event.

Each sample:
    y_hist_1d:      [H, N_1d, 1]       warm-start ground truth (fixed split)
    y_hist_2d:      [H, N_2d, 1]
    rain_hist_2d:   [H, N_2d, 1]
    y_future_1d:    [T_future, N_1d, 1] autoregressive target (variable length)
    y_future_2d:    [T_future, N_2d, 1]
    rain_future_2d: [T_future, N_2d, 1] rainfall for rollout timesteps

    y_all_1d:       [T_total, N_1d, 1]  full raw sequence (for random-history mode)
    y_all_2d:       [T_total, N_2d, 1]
    rain_all_2d:    [T_total, N_2d, 1]

Random-history mode (iter_grouped_random_split):
    For each event, randomly sample split point h ~ Uniform(min_hist, T_total-1).
    Use event[:h] as warm-start (teacher-forced), predict event[h:] autoregressively.
    Trains the model to roll out from any point, not just t=H.
    Events are grouped by bucketed T_future = round(T_total-h, bucket_size) for
    memory-efficient same-length batching.
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
    """Compute a deterministic cache key from event directories.

    history_len is included for backward-compat with existing cache files,
    but since we now store full sequences (no fixed split), existing caches
    with a different history_len will simply regenerate (one-time cost).
    """
    dirs = sorted(item[1] if len(item) >= 2 else str(item) for item in event_list)
    # Use history_len=0 in key to indicate full-sequence cache (history-len-independent)
    raw = "h0|" + "|".join(dirs)
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
            T_range = (
                min(s['y_all_1d'].shape[0] for s in self._samples),
                max(s['y_all_1d'].shape[0] for s in self._samples),
            )
            print(f"[INFO] Loaded {len(self._samples)} events from cache in {time.time()-t0:.1f}s "
                  f"(T_total range: {T_range[0]}-{T_range[1]} steps, "
                  f"T_future range with H={history_len}: {self._min_future()}-{self._max_future()} steps)")
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

            # Store only the full raw sequence — no pre-split.
            # Slicing in iter_grouped / iter_grouped_random_split is O(1) (view).
            self._samples.append({
                'y_all_1d':    wl_1d,    # [T_total, N_1d, 1]
                'y_all_2d':    wl_2d,    # [T_total, N_2d, 1]
                'rain_all_2d': rain_2d,  # [T_total, N_2d, 1]
            })

        elapsed = time.time() - t0
        print(f"[INFO] Pre-loaded {len(self._samples)} events in {elapsed:.1f}s "
              f"(T_total range: "
              f"{min(s['y_all_1d'].shape[0] for s in self._samples)}-"
              f"{max(s['y_all_1d'].shape[0] for s in self._samples)} steps, "
              f"T_future with H={history_len}: {self._min_future()}-{self._max_future()} steps)")

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
        H = self.history_len
        return min(s['y_all_1d'].shape[0] - H for s in self._samples)

    def _max_future(self):
        if not self._samples:
            return 0
        H = self.history_len
        return max(s['y_all_1d'].shape[0] - H for s in self._samples)

    def _t_future_counts(self):
        """Return dict mapping T_future → count of events with that length."""
        H = self.history_len
        counts = defaultdict(int)
        for s in self._samples:
            counts[s['y_all_1d'].shape[0] - H] += 1
        return dict(sorted(counts.items()))

    def __len__(self):
        return len(self._samples)

    def _split_sample(self, s, h):
        """Slice a raw sample at split point h → (hist_dict, future_dict)."""
        return {
            'y_hist_1d':      s['y_all_1d'][:h],
            'y_hist_2d':      s['y_all_2d'][:h],
            'rain_hist_2d':   s['rain_all_2d'][:h],
            'y_future_1d':    s['y_all_1d'][h:],
            'y_future_2d':    s['y_all_2d'][h:],
            'rain_future_2d': s['rain_all_2d'][h:],
        }

    def __iter__(self):
        """Yield individual samples (B=1 mode) with fixed history_len split."""
        H = self.history_len
        indices = list(range(len(self._samples)))
        if self.shuffle:
            random.shuffle(indices)
        for i in indices:
            yield self._split_sample(self._samples[i], H)

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
        H = self.history_len

        # Group indices by T_future
        groups = defaultdict(list)
        for idx, s in enumerate(self._samples):
            T = s['y_all_1d'].shape[0] - H
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
            samples = [self._split_sample(self._samples[i], H) for i in batch_indices]
            batch = {
                key: torch.stack([s[key] for s in samples], dim=0)
                for key in samples[0].keys()
            }
            yield batch

    def iter_grouped_random_split(self, batch_size: int,
                                  eff_T: int,
                                  min_hist: int = 1,
                                  max_K: int = None):
        """
        Yield batches with curriculum-aware randomly-sampled warm-start splits.

        For each event of length T_total, the rollout target is exactly eff_T steps
        (the curriculum effective horizon). The split point h is sampled uniformly
        over all positions that leave at least eff_T future steps:

            h ~ Uniform(min_hist, T_total - eff_T)

        Number of splits per event (K) scales with available split positions:
            K = max(1, T_total - eff_T - min_hist + 1)
        i.e. proportional to how many distinct valid h values exist. This keeps
        the total gradient signal roughly constant across curriculum stages —
        early training (small eff_T) → many splits per event; late training
        (large eff_T) → few splits, approaching 1 at eff_T = T_total - min_hist.

        All batches within an epoch share T_future = eff_T, so no bucketing is
        needed — events pool together and batch directly, maximising GPU utilisation.

        Events shorter than (min_hist + eff_T) are skipped for this epoch.

        Each yielded batch:
            y_hist_1d:      [B, h_actual, N_1d, 1]  (h_actual = min h in batch)
            y_hist_2d:      [B, h_actual, N_2d, 1]
            rain_hist_2d:   [B, h_actual, N_2d, 1]
            y_future_1d:    [B, eff_T, N_1d, 1]
            y_future_2d:    [B, eff_T, N_2d, 1]
            rain_future_2d: [B, eff_T, N_2d, 1]
        """
        # Sample K independent splits per eligible event
        assignments = []  # (h, idx)
        for idx, s in enumerate(self._samples):
            T_total = s['y_all_1d'].shape[0]
            max_h = T_total - eff_T  # largest h that leaves eff_T future steps
            if max_h < min_hist:
                continue
            K = max(1, max_h - min_hist + 1)  # number of distinct valid split positions
            if max_K is not None:
                K = min(K, max_K)
            for _ in range(K):
                h = random.randint(min_hist, max_h)
                assignments.append((h, idx))

        # Sort by h so items batched together have similar warm-start lengths,
        # minimising history lost to min(h) truncation within each batch.
        assignments.sort(key=lambda x: x[0])

        # Build all batches, then shuffle batch order so the model doesn't always
        # see short-history batches before long-history batches.
        all_batches = [
            assignments[start:start + batch_size]
            for start in range(0, len(assignments), batch_size)
        ]
        if self.shuffle:
            random.shuffle(all_batches)

        for items in all_batches:

            # Within a batch, h values are now close → min(h) loses very few steps
            h_actual = min(h for h, _ in items)

            hist_1d_list, hist_2d_list, rain_hist_list = [], [], []
            fut_1d_list, fut_2d_list, rain_fut_list = [], [], []

            for h_i, idx in items:
                s = self._samples[idx]
                y_all  = s['y_all_1d']
                y2_all = s['y_all_2d']
                r2_all = s['rain_all_2d']

                hist_1d_list.append(y_all[:h_actual])
                hist_2d_list.append(y2_all[:h_actual])
                rain_hist_list.append(r2_all[:h_actual])
                fut_1d_list.append(y_all[h_i:h_i + eff_T])
                fut_2d_list.append(y2_all[h_i:h_i + eff_T])
                rain_fut_list.append(r2_all[h_i:h_i + eff_T])

            yield {
                'y_hist_1d':      torch.stack(hist_1d_list, dim=0),   # [B, h_actual, N_1d, 1]
                'y_hist_2d':      torch.stack(hist_2d_list, dim=0),
                'rain_hist_2d':   torch.stack(rain_hist_list, dim=0),
                'y_future_1d':    torch.stack(fut_1d_list, dim=0),    # [B, eff_T, N_1d, 1]
                'y_future_2d':    torch.stack(fut_2d_list, dim=0),
                'rain_future_2d': torch.stack(rain_fut_list, dim=0),
            }


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
