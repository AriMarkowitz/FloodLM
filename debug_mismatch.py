#!/usr/bin/env python
"""Debug script to analyze prediction vs sample submission mismatch."""

import pandas as pd
from pathlib import Path

# Load sample submission
sample = pd.read_csv('../FloodModel/sample_submission.csv')
print(f"Sample submission shape: {sample.shape}")
print(f"\nModel ID value counts:")
print(sample['model_id'].value_counts().sort_index())
print(f"\nNode type value counts by model:")
for model_id in sorted(sample['model_id'].unique()):
    subset = sample[sample['model_id'] == model_id]
    print(f"  Model {model_id}: {subset['node_type'].value_counts().to_dict()}")

# Check Model_1 test events
model1_test = Path('data/Model_1/test')
events_m1 = sorted([p for p in model1_test.glob('event_*') if p.is_dir()])
print(f"\nModel_1 test events: {len(events_m1)}")

# Check first event structure
if events_m1:
    event0 = events_m1[0]
    df1d = pd.read_csv(event0 / '1d_nodes_dynamic_all.csv')
    df2d = pd.read_csv(event0 / '2d_nodes_dynamic_all.csv')
    print(f"\nEvent 0 (Model_1):")
    print(f"  1D nodes: {df1d['node_idx'].nunique()}, timesteps: {df1d['timestep'].nunique()}")
    print(f"  2D nodes: {df2d['node_idx'].nunique()}, timesteps: {df2d['timestep'].nunique()}")
    print(f"  2D water level coverage:")
    print(f"    NaN count: {df2d['water_level'].isna().sum()} / {len(df2d)}")
    print(f"    NaN%: {100*df2d['water_level'].isna().sum()/len(df2d):.1f}%")

# Compare with sample submission for Model_1
sample_m1 = sample[sample['model_id'] == 1]
print(f"\nSample submission (Model_1): {len(sample_m1)} rows")
print(f"  Unique events: {sample_m1['event_id'].nunique()}")
print(f"  Event ID range: {sample_m1['event_id'].min()} to {sample_m1['event_id'].max()}")
print(f"  Events in sample: {sorted(sample_m1['event_id'].unique())}")

# Check coverage
print(f"\n Analysis:")
print(f"  Generated predictions: ~18M")
print(f"  Sample submission Model_1: {len(sample_m1)} rows")
print(f"  Gap: {len(sample_m1) - 18612738} rows ({100*(len(sample_m1) - 18612738)/len(sample_m1):.1f}%)")
