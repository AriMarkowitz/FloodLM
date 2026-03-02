"""
Example usage of the new recurrent flood model with static graph + dynamic time series.

This demonstrates how to:
1. Load data using the new RecurrentFloodDataset (for the configured model)
2. Initialize the FloodAutoregressiveHeteroModel
3. Train with the forward_unroll method (GPU optimized)
4. Run inference
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import data config FIRST to see which model is being used
from data_config import SELECTED_MODEL, BASE_PATH, TRAIN_PATH, validate_data_paths

from data import get_recurrent_dataloader, get_model_config, get_make_x_dyn_fn, make_x_dyn
from model import FloodAutoregressiveHeteroModel

# =========================
# Configuration
# =========================
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
USE_AMP = (DEVICE.type == 'cuda')  # Automatic Mixed Precision (CUDA only)
BATCH_SIZE = 4
HISTORY_LEN = 10
FORECAST_LEN = 1
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1  # Demo: just 1 epoch
ACCUMULATION_STEPS = 2  # Gradient accumulation

print("\n" + "="*70)
print(f"FloodLM: {SELECTED_MODEL}")
print("="*70)
print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Use AMP: {USE_AMP}")
print(f"[INFO] Data path: {BASE_PATH}")

if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Validate data for this model
try:
    n_events = validate_data_paths()
    print(f"[INFO] Total events: {n_events}")
except Exception as e:
    print(f"[ERROR] Data validation failed: {e}")
    sys.exit(1)


# =========================
# 1. Get dataloader
# =========================
print("\n[INFO] Creating recurrent dataloader...")
try:
    dataloader = get_recurrent_dataloader(
        history_len=HISTORY_LEN,
        forecast_len=FORECAST_LEN,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    print("[INFO] Dataloader created successfully")
except Exception as e:
    print(f"[ERROR] Failed to create dataloader: {e}")
    sys.exit(1)

# =========================
# 2. Get model configuration
# =========================
print("\n[INFO] Getting model configuration...")
config = get_model_config()

print(f"  Node types: {config['node_types']}")
print(f"  Static dims: {config['node_static_dims']}")
print(f"  Dynamic dims: {config['node_dyn_input_dims']}")

# =========================
# 3. Initialize model
# =========================
print("\n[INFO] Initializing model...")
model = FloodAutoregressiveHeteroModel(
    node_types=config['node_types'],
    edge_types=config['edge_types'],
    node_static_dims=config['node_static_dims'],
    node_dyn_input_dims=config['node_dyn_input_dims'],
    edge_static_dims=config['edge_static_dims'],
    pred_node_type=config['pred_node_type'],
    h_dim=64,
    msg_dim=64,
    hidden_dim=128,
    dropout=0.1,
    predict_delta=True,
)

model = model.to(DEVICE)

# GPU optimization: use tf32 for faster computation (less precision but usually fine)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print(f"[INFO] Model initialized")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# =========================
# 4. Setup optimizer and loss
# =========================
print("\n[INFO] Setting up optimizer and loss function...")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Mixed precision scaler
if USE_AMP:
    scaler = GradScaler()
    print("[INFO] Using Automatic Mixed Precision (AMP)")
else:
    scaler = None
    print("[INFO] Training in full precision")

# =========================
# 5. Get make_x_dyn function
# =========================
make_x_dyn_fn = get_make_x_dyn_fn()

# =========================
# Training loop (GPU optimized)
# =========================
print("\n[INFO] Starting training...")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Accumulation steps: {ACCUMULATION_STEPS}")

model.train()
epoch_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    n_batches = 0
    batch_start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        # Extract data from batch and move to GPU
        static_graph = batch['static_graph'].to(DEVICE)
        
        y_hist_1d = batch['y_hist_1d'].to(DEVICE, non_blocking=True)
        y_hist_2d = batch['y_hist_2d'].to(DEVICE, non_blocking=True)
        rain_hist_2d = batch['rain_hist_2d'].to(DEVICE, non_blocking=True)
        
        y_future_1d = batch['y_future_1d'].to(DEVICE, non_blocking=True)
        y_future_2d = batch['y_future_2d'].to(DEVICE, non_blocking=True)
        rain_future_2d = batch['rain_future_2d'].to(DEVICE, non_blocking=True)
        
        batch_size = y_hist_2d.size(0)
        history_len = y_hist_2d.size(1)
        forecast_len = y_future_2d.size(1)
        
        # Accumulate gradients
        batch_loss = 0.0
        
        for b in range(batch_size):
            y_hist_1d_sample = y_hist_1d[b]
            y_hist_2d_sample = y_hist_2d[b]
            rain_hist_sample = rain_hist_2d[b]
            
            rain_future_sample = rain_future_2d[b]
            y_future_2d_sample = y_future_2d[b]
            
            # Custom make_x_dyn
            def make_x_dyn_batch(y_2d, rain_2d, data):
                n_1d = data["oneD"].num_nodes
                device = y_2d.device
                y_1d = torch.zeros((n_1d, 1), device=device, dtype=y_2d.dtype)
                return make_x_dyn(y_pred_1d=y_1d, y_pred_2d=y_2d, rain_2d=rain_2d, data=data)
            
            # Forward pass with AMP
            if USE_AMP:
                with autocast(dtype=torch.float16):
                    predictions = model.forward_unroll(
                        data=static_graph,
                        y_hist_true=y_hist_2d_sample,
                        rain_hist=rain_hist_sample,
                        rain_future=rain_future_sample,
                        make_x_dyn=make_x_dyn_batch,
                        rollout_steps=forecast_len,
                        device=DEVICE,
                    )
                    loss = criterion(predictions, y_future_2d_sample)
            else:
                predictions = model.forward_unroll(
                    data=static_graph,
                    y_hist_true=y_hist_2d_sample,
                    rain_hist=rain_hist_sample,
                    rain_future=rain_future_sample,
                    make_x_dyn=make_x_dyn_batch,
                    rollout_steps=forecast_len,
                    device=DEVICE,
                )
                loss = criterion(predictions, y_future_2d_sample)
            
            batch_loss += loss / batch_size
        
        # Backward with gradient accumulation
        if USE_AMP:
            scaler.scale(batch_loss).backward()
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            batch_loss.backward()
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        epoch_loss += batch_loss.item()
        n_batches += 1
        
        # Periodic logging
        if batch_idx % 5 == 0:
            avg_loss = epoch_loss / n_batches
            elapsed = time.time() - batch_start_time
            print(f"  Batch {batch_idx:3d} | Loss: {batch_loss.item():.6f} | Avg: {avg_loss:.6f} | {elapsed:.1f}s")
        
        if batch_idx >= 30:  # Limit for demo
            break
    
    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
    epoch_time = time.time() - epoch_start_time
    print(f"\n[INFO] Epoch {epoch+1}/{NUM_EPOCHS}: Loss={avg_loss:.6f} | Time: {epoch_time:.1f}s")

# =========================
# Inference (GPU optimized)
# =========================
print("\n[INFO] Running inference...")

model.eval()
n_inferences = 0
inference_times = []

with torch.no_grad():
    if USE_AMP:
        with autocast(dtype=torch.float16):
            for batch in dataloader:
                if batch is None:
                    continue
                
                static_graph = batch['static_graph'].to(DEVICE)
                y_hist_2d = batch['y_hist_2d'][0].to(DEVICE, non_blocking=True)
                rain_hist_2d = batch['rain_hist_2d'][0].to(DEVICE, non_blocking=True)
                rain_future_2d = batch['rain_future_2d'][0].to(DEVICE, non_blocking=True)
                
                def make_x_dyn_batch(y_2d, rain_2d, data):
                    n_1d = data["oneD"].num_nodes
                    device = y_2d.device
                    y_1d = torch.zeros((n_1d, 1), device=device, dtype=y_2d.dtype)
                    return make_x_dyn(y_pred_1d=y_1d, y_pred_2d=y_2d, rain_2d=rain_2d, data=data)
                
                # Time inference
                inf_start = time.time()
                predictions = model.forward_unroll(
                    data=static_graph,
                    y_hist_true=y_hist_2d,
                    rain_hist=rain_hist_2d,
                    rain_future=rain_future_2d,
                    make_x_dyn=make_x_dyn_batch,
                    rollout_steps=1,
                    device=DEVICE,
                )
                inf_time = time.time() - inf_start
                inference_times.append(inf_time)
                
                print(f"  Inference {n_inferences+1}: {inf_time*1000:.2f}ms | Prediction shape: {predictions.shape}")
                print(f"    Sample predictions (first 5 nodes): {predictions[0, :5, 0].cpu()}")
                
                n_inferences += 1
                if n_inferences >= 3:
                    break
    else:
        for batch in dataloader:
            if batch is None:
                continue
            
            static_graph = batch['static_graph'].to(DEVICE)
            y_hist_2d = batch['y_hist_2d'][0].to(DEVICE, non_blocking=True)
            rain_hist_2d = batch['rain_hist_2d'][0].to(DEVICE, non_blocking=True)
            rain_future_2d = batch['rain_future_2d'][0].to(DEVICE, non_blocking=True)
            
            def make_x_dyn_batch(y_2d, rain_2d, data):
                n_1d = data["oneD"].num_nodes
                device = y_2d.device
                y_1d = torch.zeros((n_1d, 1), device=device, dtype=y_2d.dtype)
                return make_x_dyn(y_pred_1d=y_1d, y_pred_2d=y_2d, rain_2d=rain_2d, data=data)
            
            inf_start = time.time()
            predictions = model.forward_unroll(
                data=static_graph,
                y_hist_true=y_hist_2d,
                rain_hist=rain_hist_2d,
                rain_future=rain_future_2d,
                make_x_dyn=make_x_dyn_batch,
                rollout_steps=1,
                device=DEVICE,
            )
            inf_time = time.time() - inf_start
            inference_times.append(inf_time)
            
            print(f"  Inference {n_inferences+1}: {inf_time*1000:.2f}ms | Prediction shape: {predictions.shape}")
            print(f"    Sample predictions (first 5 nodes): {predictions[0, :5, 0].cpu()}")
            
            n_inferences += 1
            if n_inferences >= 3:
                break

if inference_times:
    avg_inf_time = sum(inference_times) / len(inference_times)
    print(f"\n[INFO] Average inference time: {avg_inf_time*1000:.2f}ms")

# =========================
# Summary
# =========================
print("\n" + "="*70)
print(f"SUMMARY - {SELECTED_MODEL}")
print("="*70)
print(f"Model:                 {SELECTED_MODEL}")
print(f"Data path:             {BASE_PATH}")
print(f"Device:                {DEVICE}")
print(f"Model parameters:      {total_params:,}")
print(f"Batch size:            {BATCH_SIZE}")
print(f"History length:        {HISTORY_LEN}")
print(f"Forecast length:       {FORECAST_LEN}")
print(f"Training epochs:       {NUM_EPOCHS}")
print(f"Final training loss:   {avg_loss:.6f}" if 'avg_loss' in locals() else "N/A")
print(f"Avg inference time:    {avg_inf_time*1000:.2f}ms" if inference_times else "N/A")

if torch.cuda.is_available():
    print(f"GPU Memory used:       {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory reserved:   {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

print("="*70)
print("[INFO] Test completed successfully!")


# =========================
# 1. Get dataloader
# =========================
print("[INFO] Creating recurrent dataloader...")
dataloader = get_recurrent_dataloader(
    history_len=10,      # Number of historical timesteps for warm start
    forecast_len=1,      # Number of future timesteps to predict
    batch_size=4,        # Batch size
    shuffle=True,        # Shuffle events
)

# =========================
# 2. Get model configuration
# =========================
print("[INFO] Getting model configuration...")
config = get_model_config()

print(f"Node types: {config['node_types']}")
print(f"Edge types: {config['edge_types']}")
print(f"Static dims: {config['node_static_dims']}")
print(f"Dynamic dims: {config['node_dyn_input_dims']}")
print(f"Edge dims: {config['edge_static_dims']}")
print(f"Prediction node type: {config['pred_node_type']}")

# =========================
# 3. Initialize model
# =========================
print("[INFO] Initializing model...")
model = FloodAutoregressiveHeteroModel(
    node_types=config['node_types'],
    edge_types=config['edge_types'],
    node_static_dims=config['node_static_dims'],
    node_dyn_input_dims=config['node_dyn_input_dims'],
    edge_static_dims=config['edge_static_dims'],
    pred_node_type=config['pred_node_type'],
    h_dim=64,
    msg_dim=64,
    hidden_dim=128,
    dropout=0.1,
    predict_delta=True,  # Predict delta water level
)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
model = model.to(device)

print(f"Model initialized on {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# =========================
# 4. Get make_x_dyn function
# =========================
make_x_dyn_fn = get_make_x_dyn_fn()

# =========================
# 5. Training loop example
# =========================
print("\n[INFO] Starting training example...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

model.train()

for epoch in range(1):  # Just 1 epoch for demonstration
    epoch_loss = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        # Extract data from batch
        static_graph = batch['static_graph'].to(device)
        
        # Batch dimensions: [B, H, N, 1] where B=batch, H=history, N=nodes
        y_hist_1d = batch['y_hist_1d'].to(device)     # [B, H, N_1d, 1]
        y_hist_2d = batch['y_hist_2d'].to(device)     # [B, H, N_2d, 1]
        rain_hist_2d = batch['rain_hist_2d'].to(device)  # [B, H, N_2d, 1]
        
        y_future_1d = batch['y_future_1d'].to(device)    # [B, T, N_1d, 1]
        y_future_2d = batch['y_future_2d'].to(device)    # [B, T, N_2d, 1]
        rain_future_2d = batch['rain_future_2d'].to(device)  # [B, T, N_2d, 1]
        
        optimizer.zero_grad()
        
        batch_size = y_hist_2d.size(0)
        history_len = y_hist_2d.size(1)
        forecast_len = y_future_2d.size(1)
        
        # Process each sample in batch (model expects time-first format)
        batch_loss = 0.0
        
        for b in range(batch_size):
            # Transpose from [B, H, N, C] to [H, N, C]
            y_hist_1d_sample = y_hist_1d[b]  # [H, N_1d, 1]
            y_hist_2d_sample = y_hist_2d[b]  # [H, N_2d, 1]
            rain_hist_sample = rain_hist_2d[b]  # [H, N_2d, 1]
            
            rain_future_sample = rain_future_2d[b]  # [T, N_2d, 1]
            y_future_2d_sample = y_future_2d[b]  # [T, N_2d, 1] (ground truth)
            
            # Custom make_x_dyn that handles both 1D and 2D nodes
            def make_x_dyn_batch(y_2d, rain_2d, data):
                """Build x_dyn for all node types."""
                n_1d = data["oneD"].num_nodes
                device = y_2d.device
                
                # For 1D nodes: just water level (zero or propagated)
                y_1d = torch.zeros((n_1d, 1), device=device)
                
                return make_x_dyn(
                    y_pred_1d=y_1d,
                    y_pred_2d=y_2d,
                    rain_2d=rain_2d,
                    data=data,
                )
            
            # Forward pass with autoregressive rollout
            predictions = model.forward_unroll(
                data=static_graph,
                y_hist_true=y_hist_2d_sample,     # [H, N_2d, 1]
                rain_hist=rain_hist_sample,       # [H, N_2d, 1]
                rain_future=rain_future_sample,   # [T, N_2d, 1]
                make_x_dyn=make_x_dyn_batch,      # Custom function
                rollout_steps=forecast_len,
                device=device,
            )
            
            # predictions: [T, N_2d, 1]
            # Compare with ground truth
            loss = criterion(predictions, y_future_2d_sample)
            batch_loss += loss
        
        # Average loss over batch
        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        optimizer.step()
        
        epoch_loss += batch_loss.item()
        n_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}, Loss: {batch_loss.item():.6f}")
        
        if batch_idx >= 50:  # Limit for demonstration
            break
    
    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
    print(f"\nEpoch {epoch+1}, Avg Loss: {avg_loss:.6f}")

print("\n[INFO] Training example completed!")

# =========================
# 6. Inference example
# =========================
print("\n[INFO] Running inference example...")

model.eval()

with torch.no_grad():
    for batch in dataloader:
        if batch is None:
            continue
        
        static_graph = batch['static_graph'].to(device)
        y_hist_2d = batch['y_hist_2d'][0].to(device)  # Take first sample
        rain_hist_2d = batch['rain_hist_2d'][0].to(device)
        rain_future_2d = batch['rain_future_2d'][0].to(device)
        
        # Custom make_x_dyn
        def make_x_dyn_batch(y_2d, rain_2d, data):
            n_1d = data["oneD"].num_nodes
            device = y_2d.device
            y_1d = torch.zeros((n_1d, 1), device=device)
            return make_x_dyn(y_pred_1d=y_1d, y_pred_2d=y_2d, rain_2d=rain_2d, data=data)
        
        # Run prediction
        predictions = model.forward_unroll(
            data=static_graph,
            y_hist_true=y_hist_2d,
            rain_hist=rain_hist_2d,
            rain_future=rain_future_2d,
            make_x_dyn=make_x_dyn_batch,
            rollout_steps=1,
            device=device,
        )
        
        print(f"Prediction shape: {predictions.shape}")
        print(f"Sample predictions (first 5 nodes): {predictions[0, :5, 0]}")
        
        break  # Just one example

print("\n[INFO] All done!")
