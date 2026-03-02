import pandas as pd

# Load sample submission
sample_df = pd.read_csv("../FloodModel/sample_submission.csv")

print("=== SAMPLE SUBMISSION STRUCTURE ===")
print(f"Columns: {list(sample_df.columns)}")
print(f"Total rows: {len(sample_df)}\n")

print("First 20 rows:")
print(sample_df.head(20))

print(f"\n\nModel ID value counts:")
print(sample_df['model_id'].value_counts().sort_index())

print(f"\nNode type value counts:")
print(sample_df['node_type'].value_counts().sort_index())

# Check how many unique (model_id, event_id, node_type, node_id) combinations
print(f"\nUnique combinations:")
print(f"  Models: {sample_df['model_id'].nunique()}")
print(f"  Events: {sample_df['event_id'].nunique()}")
print(f"  Node types: {sample_df['node_type'].nunique()}")
print(f"  Nodes: {sample_df['node_id'].nunique()}")

# Group by model and count rows
for mid in sorted(sample_df['model_id'].unique()):
    subset = sample_df[sample_df['model_id'] == mid]
    print(f"\nModel {mid}: {len(subset)} rows")
    for ntype in sorted(subset['node_type'].unique()):
        ntype_subset = subset[subset['node_type'] == ntype]
        unique_nodes = ntype_subset['node_id'].nunique()
        rows_per_node = len(ntype_subset) / unique_nodes if unique_nodes > 0 else 0
        print(f"  Node type {ntype}: {len(ntype_subset)} rows, {unique_nodes} nodes, {rows_per_node:.1f} rows/node")
