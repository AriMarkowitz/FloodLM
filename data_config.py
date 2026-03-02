# FloodLM Data Configuration
# =======================
# Edit this file to change which model data is loaded
# Or set SELECTED_MODEL environment variable to override

import os

# Model selection - CHANGE THIS TO SWITCH BETWEEN MODELS
# Available options: "Model_1", "Model_2"
# Can also be overridden via environment variable: export SELECTED_MODEL=Model_2
SELECTED_MODEL = os.environ.get("SELECTED_MODEL", "Model_1")

# Data folder path - relative to FloodLM root directory
# Structure:
#   FloodLM/
#   └── data/
#       ├── Model_1/
#       │   └── train/
#       │       ├── 1d_nodes_static.csv
#       │       ├── 2d_nodes_static.csv
#       │       ├── 1d_edges_static.csv
#       │       ├── 2d_edges_static.csv
#       │       ├── 1d_edge_index.csv
#       │       ├── 2d_edge_index.csv
#       │       ├── 1d2d_connections.csv
#       │       ├── event_0/
#       │       │   ├── 1d_nodes_dynamic_all.csv
#       │       │   └── 2d_nodes_dynamic_all.csv
#       │       ├── event_1/
#       │       └── ...
#       └── Model_2/
#           └── train/
#               ├── ...
DATA_FOLDER = "data"

# Data subset size for testing (use all if <= 0)
# Examples: -1 (all), 10 (first 10 events), 5 (first 5 events)
MAX_EVENTS = -1  # -1 means use all available events

# Training configuration
VALIDATION_SPLIT = 0.2  # 20% validation split (randomly selected events)
TEST_SPLIT = 0.0        # No held-out test set — maximise training data
RANDOM_SEED = 42


# ===== Derived paths (do not edit) =====
BASE_PATH = f"{DATA_FOLDER}/{SELECTED_MODEL}"
TRAIN_PATH = f"{BASE_PATH}/train"


# ===== Path validation =====
import os

def validate_data_paths():
    """Validate that all required data files exist."""
    required_files = [
        f"{TRAIN_PATH}/1d_nodes_static.csv",
        f"{TRAIN_PATH}/2d_nodes_static.csv",
        f"{TRAIN_PATH}/1d_edges_static.csv",
        f"{TRAIN_PATH}/2d_edges_static.csv",
        f"{TRAIN_PATH}/1d_edge_index.csv",
        f"{TRAIN_PATH}/2d_edge_index.csv",
        f"{TRAIN_PATH}/1d2d_connections.csv",
    ]
    
    missing = []
    for fpath in required_files:
        if not os.path.exists(fpath):
            missing.append(fpath)
    
    if missing:
        print("\n[ERROR] Missing required data files:")
        for fpath in missing:
            print(f"  - {fpath}")
        print(f"\nExpected structure:")
        print(f"  FloodLM/data/{SELECTED_MODEL}/train/")
        raise FileNotFoundError(
            f"Data validation failed for {SELECTED_MODEL}. "
            f"See paths above and ensure data is in the correct location."
        )
    
    # Check for event directories
    event_dirs = [d for d in os.listdir(TRAIN_PATH) if d.startswith("event_")]
    if not event_dirs:
        print(f"\n[WARNING] No event directories found in {TRAIN_PATH}")
        return 0
    
    return len(event_dirs)


if __name__ == "__main__":
    print(f"Data configuration for {SELECTED_MODEL}")
    print(f"Data path: {BASE_PATH}")
    try:
        n_events = validate_data_paths()
        print(f"✓ Data validation successful ({n_events} events found)")
    except FileNotFoundError as e:
        print(f"✗ Data validation failed: {e}")
        exit(1)
