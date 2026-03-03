#!/usr/bin/env python3
"""
One-shot script to patch water_level normalization in cache pkl files
from min-max to mean/kaggle-sigma (meanstd).

Run once from the FloodLM root:
    conda run -n floodlm python scripts/patch_wl_normalization.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

BASE_PATH = '/users/admarkowitz/FloodModel'

KAGGLE_SIGMA = {
    ('Model_1', '1d'): 16.878,
    ('Model_1', '2d'): 14.379,
    ('Model_2', '1d'):  3.192,
    ('Model_2', '2d'):  2.727,
}

def compute_wl_means(train_event_file_list, base_path):
    """Stream through training events and compute mean water_level for 1D and 2D nodes."""
    sum_1d, count_1d = 0.0, 0
    sum_2d, count_2d = 0.0, 0

    for _, rel_path, split in train_event_file_list:
        if split != 'train':
            continue
        event_dir = os.path.join(base_path, rel_path)

        f1d = os.path.join(event_dir, '1d_nodes_dynamic_all.csv')
        if os.path.exists(f1d):
            df = pd.read_csv(f1d, usecols=['water_level'])
            vals = df['water_level'].dropna().values.astype(float)
            sum_1d += vals.sum()
            count_1d += len(vals)

        f2d = os.path.join(event_dir, '2d_nodes_dynamic_all.csv')
        if os.path.exists(f2d):
            df = pd.read_csv(f2d, usecols=['water_level'])
            vals = df['water_level'].dropna().values.astype(float)
            sum_2d += vals.sum()
            count_2d += len(vals)

    mean_1d = sum_1d / count_1d if count_1d > 0 else 0.0
    mean_2d = sum_2d / count_2d if count_2d > 0 else 0.0
    return mean_1d, mean_2d


def patch_model(model_name):
    cache_path = os.path.join(BASE_PATH, 'data', model_name, '.cache', f'{model_name}_preprocessed.pkl')
    print(f'\n=== Patching {model_name} ===')
    print(f'Cache: {cache_path}')

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    norm_stats = cache['norm_stats']
    train_events = cache['train_event_file_list']

    print(f'Training events: {sum(1 for _,_,s in train_events if s=="train")}')
    print('Computing water_level means from training data...')
    mean_1d, mean_2d = compute_wl_means(train_events, BASE_PATH)
    print(f'  1D mean: {mean_1d:.4f}')
    print(f'  2D mean: {mean_2d:.4f}')

    sigma_1d = KAGGLE_SIGMA[(model_name, '1d')]
    sigma_2d = KAGGLE_SIGMA[(model_name, '2d')]

    new_params_1d = {'type': 'meanstd', 'mean': float(mean_1d), 'sigma': float(sigma_1d), 'log': False}
    new_params_2d = {'type': 'meanstd', 'mean': float(mean_2d), 'sigma': float(sigma_2d), 'log': False}

    print(f'  New 1D params: {new_params_1d}')
    print(f'  New 2D params: {new_params_2d}')

    # Patch norm_stats dicts
    norm_stats['dynamic_1d_params']['water_level'] = new_params_1d
    norm_stats['dynamic_2d_params']['water_level'] = new_params_2d

    # Patch normalizer objects (these are the same objects used at runtime)
    norm_stats['normalizer_1d'].dynamic_params['water_level'] = new_params_1d
    norm_stats['normalizer_2d'].dynamic_params['water_level'] = new_params_2d

    # Save back
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f'Saved patched cache to {cache_path}')

    # Verify
    with open(cache_path, 'rb') as f:
        verify = pickle.load(f)
    assert verify['norm_stats']['dynamic_1d_params']['water_level']['type'] == 'meanstd'
    assert verify['norm_stats']['normalizer_1d'].dynamic_params['water_level']['type'] == 'meanstd'
    print('Verification passed.')


if __name__ == '__main__':
    patch_model('Model_1')
    patch_model('Model_2')
    print('\nDone. Both cache files patched.')
