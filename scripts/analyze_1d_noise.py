#!/usr/bin/env python3
"""
Analyze noisiness of Model_2 static and dynamic 1D node data.

Metrics computed:
  Static:
    - Per-feature distribution stats (mean, std, skewness, kurtosis, min, max)
    - Coefficient of variation (std/|mean|) — relative spread across nodes
    - Fraction of near-zero values (potential sparse/inactive nodes)
    - Outlier count per feature (> 3 sigma from mean)

  Dynamic (water_level + inlet_flow):
    - Per-node temporal roughness: mean |Δy/Δt| across all events (avg step-to-step jump)
    - Per-node temporal SNR: signal range / roughness
    - Per-event temporal roughness distribution (are some events noisier?)
    - Cross-node correlation of water_level at each timestep (how coherent is the signal)
    - Autocorrelation at lag=1 per node (high = smooth, low = noisy)
    - Fraction of nodes with roughness > N × median (outlier node detector)
    - Distribution of peak water_level across events and nodes
    - inlet_flow: same roughness/SNR analysis

Outputs:
  - Console summary with ranked "noisiest" nodes and events
  - plots/1d_noise_analysis.png — multi-panel figure (saved to project root)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "Model_2" / "train"
PLOT_DIR = PROJECT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

MAX_EVENTS = int(os.environ.get("MAX_EVENTS", "0"))  # 0 = all

# ── helpers ──────────────────────────────────────────────────────────────────

def roughness(series: np.ndarray) -> float:
    """Mean absolute first difference — average step-to-step jump."""
    d = np.diff(series)
    return np.mean(np.abs(d)) if len(d) > 0 else 0.0

def snr(series: np.ndarray) -> float:
    """Signal-to-noise: peak-to-peak range / roughness."""
    r = roughness(series)
    return (series.max() - series.min()) / r if r > 0 else np.inf

def autocorr_lag1(series: np.ndarray) -> float:
    """Pearson autocorrelation at lag 1."""
    if len(series) < 3:
        return np.nan
    return np.corrcoef(series[:-1], series[1:])[0, 1]

# ── load data ─────────────────────────────────────────────────────────────────

print(f"[INFO] Data directory: {DATA_DIR}")
static_path = DATA_DIR / "1d_nodes_static.csv"
static = pd.read_csv(static_path)
print(f"[INFO] Static: {static.shape[0]} nodes, {static.shape[1]-1} features")

event_dirs = sorted(
    glob.glob(str(DATA_DIR / "event_*")),
    key=lambda p: int(Path(p).name.split("_")[-1])
)
if MAX_EVENTS > 0:
    event_dirs = event_dirs[:MAX_EVENTS]
print(f"[INFO] Events: {len(event_dirs)}")

# ── static analysis ───────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STATIC 1D NODE FEATURES")
print("="*70)

feat_cols = [c for c in static.columns if c != "node_idx"]
static_stats = []
for col in feat_cols:
    vals = static[col].dropna().values
    cv = np.std(vals) / np.abs(np.mean(vals)) if np.abs(np.mean(vals)) > 1e-9 else np.inf
    n_outliers = np.sum(np.abs(vals - vals.mean()) > 3 * vals.std()) if vals.std() > 0 else 0
    n_near_zero = np.sum(np.abs(vals) < 1e-3 * (np.abs(vals).max() + 1e-9))
    static_stats.append({
        "feature": col,
        "mean": vals.mean(),
        "std": vals.std(),
        "min": vals.min(),
        "max": vals.max(),
        "skewness": stats.skew(vals),
        "kurtosis": stats.kurtosis(vals),
        "cv": cv,
        "n_outliers_3sigma": n_outliers,
        "n_near_zero": n_near_zero,
        "n_unique": len(np.unique(vals)),
    })

static_df = pd.DataFrame(static_stats).set_index("feature")
print(static_df[["mean","std","min","max","skewness","cv","n_outliers_3sigma","n_near_zero"]].to_string(float_format="%.4f"))

# ── dynamic loading ───────────────────────────────────────────────────────────

print("\n[INFO] Loading dynamic data (this may take ~30s)...")

# node_roughness[node_idx][feature] = list of per-event roughness values
N_NODES = static.shape[0]
dyn_features = ["water_level", "inlet_flow"]

per_node_roughness  = {f: {n: [] for n in range(N_NODES)} for f in dyn_features}
per_node_snr        = {f: {n: [] for n in range(N_NODES)} for f in dyn_features}
per_node_autocorr   = {f: {n: [] for n in range(N_NODES)} for f in dyn_features}
per_event_roughness = {f: [] for f in dyn_features}  # (event_id, mean_roughness)

# cross-node correlation: collect water_level matrices per event
wl_corr_eigenratios = []  # ratio of first eigenvalue to sum — high = coherent signal

for ev_dir in event_dirs:
    ev_id = int(Path(ev_dir).name.split("_")[-1])
    dyn_path = Path(ev_dir) / "1d_nodes_dynamic_all.csv"
    if not dyn_path.exists():
        continue
    df = pd.read_csv(dyn_path)
    df = df.sort_values(["node_idx", "timestep"])

    for feat in dyn_features:
        if feat not in df.columns:
            continue
        ev_roughness_vals = []
        for node_id, grp in df.groupby("node_idx"):
            series = grp[feat].values
            r = roughness(series)
            s = snr(series)
            a = autocorr_lag1(series)
            per_node_roughness[feat][node_id].append(r)
            per_node_snr[feat][node_id].append(s)
            per_node_autocorr[feat][node_id].append(a)
            ev_roughness_vals.append(r)
        per_event_roughness[feat].append((ev_id, np.mean(ev_roughness_vals)))

    # Cross-node water_level coherence: pivot to [T, N], compute PCA variance ratio
    if "water_level" in df.columns:
        try:
            wl_mat = df.pivot(index="timestep", columns="node_idx", values="water_level").values
            if wl_mat.shape[0] > 2 and not np.any(np.isnan(wl_mat)):
                wl_centered = wl_mat - wl_mat.mean(axis=0)
                cov = np.cov(wl_centered.T)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = eigvals[eigvals > 0]
                ratio = eigvals[-1] / eigvals.sum() if eigvals.sum() > 0 else 0
                wl_corr_eigenratios.append((ev_id, ratio))
        except Exception:
            pass

print(f"[INFO] Done. Processed {len(event_dirs)} events.")

# ── aggregate per-node stats ──────────────────────────────────────────────────

def agg_node_stats(metric_dict):
    """Per-node mean across events."""
    return {n: np.mean(v) if v else np.nan for n, v in metric_dict.items()}

wl_roughness_mean  = agg_node_stats(per_node_roughness["water_level"])
wl_snr_mean        = agg_node_stats(per_node_snr["water_level"])
wl_autocorr_mean   = agg_node_stats(per_node_autocorr["water_level"])
if_roughness_mean  = agg_node_stats(per_node_roughness["inlet_flow"])
if_snr_mean        = agg_node_stats(per_node_snr["inlet_flow"])
if_autocorr_mean   = agg_node_stats(per_node_autocorr["inlet_flow"])

# ── console report ────────────────────────────────────────────────────────────

def report_noisiest(roughness_mean, feature_name, top_n=15):
    arr = [(n, v) for n, v in roughness_mean.items() if not np.isnan(v)]
    arr.sort(key=lambda x: -x[1])
    median_r = np.median([v for _, v in arr])
    print(f"\n{'─'*60}")
    print(f"Top {top_n} noisiest nodes — {feature_name} (by mean roughness)")
    print(f"  Median roughness across all nodes: {median_r:.4f}")
    print(f"  {'node_idx':>8}  {'mean_roughness':>14}  {'x median':>8}  {'mean_snr':>9}  {'mean_ac1':>9}")
    for node_id, r in arr[:top_n]:
        snr_val = wl_snr_mean[node_id] if feature_name == "water_level" else if_snr_mean[node_id]
        ac_val  = wl_autocorr_mean[node_id] if feature_name == "water_level" else if_autocorr_mean[node_id]
        mult = r / median_r if median_r > 0 else 0
        print(f"  {node_id:>8}  {r:>14.4f}  {mult:>8.1f}x  {snr_val:>9.2f}  {ac_val:>9.3f}")

print("\n" + "="*70)
print("DYNAMIC 1D NODE ANALYSIS")
print("="*70)

# water_level summary
wl_r_vals = np.array([v for v in wl_roughness_mean.values() if not np.isnan(v)])
wl_ac_vals = np.array([v for v in wl_autocorr_mean.values() if not np.isnan(v)])
wl_snr_vals = np.array([v for v in wl_snr_mean.values() if not np.isnan(v)])
print(f"\nwater_level roughness across nodes:")
print(f"  median={np.median(wl_r_vals):.4f}  mean={wl_r_vals.mean():.4f}  "
      f"p95={np.percentile(wl_r_vals,95):.4f}  max={wl_r_vals.max():.4f}")
print(f"  Nodes with roughness > 3×median: "
      f"{np.sum(wl_r_vals > 3*np.median(wl_r_vals))} / {len(wl_r_vals)}")
print(f"water_level autocorr@lag1 across nodes:")
print(f"  median={np.nanmedian(wl_ac_vals):.3f}  mean={np.nanmean(wl_ac_vals):.3f}  "
      f"p5={np.nanpercentile(wl_ac_vals,5):.3f}  min={np.nanmin(wl_ac_vals):.3f}")
print(f"  Nodes with autocorr < 0.5 (poorly predictable): "
      f"{np.sum(wl_ac_vals < 0.5)} / {len(wl_ac_vals)}")
print(f"water_level SNR across nodes:")
print(f"  median={np.nanmedian(wl_snr_vals):.2f}  p5={np.nanpercentile(wl_snr_vals,5):.2f}  "
      f"min={np.nanmin(wl_snr_vals):.2f}")

# inlet_flow summary
if_r_vals = np.array([v for v in if_roughness_mean.values() if not np.isnan(v)])
if_ac_vals = np.array([v for v in if_autocorr_mean.values() if not np.isnan(v)])
print(f"\ninlet_flow roughness across nodes:")
print(f"  median={np.median(if_r_vals):.4f}  mean={if_r_vals.mean():.4f}  "
      f"p95={np.percentile(if_r_vals,95):.4f}  max={if_r_vals.max():.4f}")
print(f"inlet_flow autocorr@lag1 across nodes:")
print(f"  median={np.nanmedian(if_ac_vals):.3f}  p5={np.nanpercentile(if_ac_vals,5):.3f}")

report_noisiest(wl_roughness_mean, "water_level")
report_noisiest(if_roughness_mean, "inlet_flow")

# Noisiest events
print(f"\n{'─'*60}")
print("Noisiest events by mean water_level roughness:")
ev_sorted = sorted(per_event_roughness["water_level"], key=lambda x: -x[1])
print(f"  {'event_id':>8}  {'mean_roughness':>14}")
for ev_id, r in ev_sorted[:10]:
    print(f"  {ev_id:>8}  {r:>14.4f}")

# Cross-node coherence
if wl_corr_eigenratios:
    ratios = np.array([r for _, r in wl_corr_eigenratios])
    print(f"\nCross-node water_level coherence (PCA 1st eigenvalue fraction):")
    print(f"  median={np.median(ratios):.3f}  mean={ratios.mean():.3f}  "
          f"min={ratios.min():.3f}  max={ratios.max():.3f}")
    print(f"  → 1.0 = all nodes move together; 0.0 = fully uncorrelated")
    low_coh = [(ev_id, r) for ev_id, r in wl_corr_eigenratios if r < np.percentile(ratios, 20)]
    print(f"  Least coherent events (bottom 20%): {[e for e, _ in sorted(low_coh, key=lambda x: x[1])[:5]]}")

# ── plots ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Model_2 — 1D Node Data Noise Analysis", fontsize=14, fontweight='bold')

# 1. Water level roughness distribution across nodes
ax = axes[0, 0]
ax.hist(wl_r_vals, bins=40, color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(np.median(wl_r_vals), color='red', linestyle='--', label=f'median={np.median(wl_r_vals):.3f}')
ax.axvline(3*np.median(wl_r_vals), color='orange', linestyle='--', label='3×median')
ax.set_xlabel("Mean roughness (|Δwl/Δt|)"); ax.set_ylabel("# nodes")
ax.set_title("water_level roughness dist. (per node)"); ax.legend(fontsize=7)

# 2. inlet_flow roughness distribution
ax = axes[0, 1]
ax.hist(if_r_vals, bins=40, color='coral', edgecolor='white', linewidth=0.4)
ax.axvline(np.median(if_r_vals), color='red', linestyle='--', label=f'median={np.median(if_r_vals):.3f}')
ax.set_xlabel("Mean roughness (|Δflow/Δt|)"); ax.set_ylabel("# nodes")
ax.set_title("inlet_flow roughness dist. (per node)"); ax.legend(fontsize=7)

# 3. Autocorrelation @ lag 1 — water_level
ax = axes[0, 2]
ax.hist(wl_ac_vals[~np.isnan(wl_ac_vals)], bins=40, color='seagreen', edgecolor='white', linewidth=0.4)
ax.axvline(0.9, color='orange', linestyle='--', label='ac=0.9')
ax.axvline(0.5, color='red', linestyle='--', label='ac=0.5 (problem)')
ax.set_xlabel("Autocorrelation lag-1"); ax.set_ylabel("# nodes")
ax.set_title("water_level autocorr@lag1 (per node)"); ax.legend(fontsize=7)

# 4. SNR — water_level
ax = axes[1, 0]
snr_clip = np.clip(wl_snr_vals[~np.isinf(wl_snr_vals)], 0, np.percentile(wl_snr_vals[~np.isinf(wl_snr_vals)], 99))
ax.hist(snr_clip, bins=40, color='mediumpurple', edgecolor='white', linewidth=0.4)
ax.set_xlabel("SNR (range / roughness)"); ax.set_ylabel("# nodes")
ax.set_title("water_level SNR (per node, clipped @p99)")

# 5. Per-node roughness bar chart (sorted, top 30 highlighted)
ax = axes[1, 1]
node_ids = np.array(sorted(wl_roughness_mean.keys()))
r_vals_sorted_idx = np.argsort([wl_roughness_mean[n] for n in node_ids])
r_sorted = np.array([wl_roughness_mean[node_ids[i]] for i in r_vals_sorted_idx])
colors = ['tomato' if r > 3*np.median(r_sorted) else 'steelblue' for r in r_sorted]
ax.bar(range(len(r_sorted)), r_sorted, color=colors, linewidth=0)
ax.axhline(np.median(r_sorted), color='black', linestyle='--', linewidth=0.8, label='median')
ax.axhline(3*np.median(r_sorted), color='orange', linestyle='--', linewidth=0.8, label='3×median')
ax.set_xlabel("Node rank (sorted by roughness)"); ax.set_ylabel("Mean roughness")
ax.set_title("water_level roughness per node (red=3×median)"); ax.legend(fontsize=7)

# 6. Per-event roughness — water_level
ax = axes[1, 2]
ev_ids_wl = [x[0] for x in per_event_roughness["water_level"]]
ev_r_wl   = [x[1] for x in per_event_roughness["water_level"]]
sort_idx = np.argsort(ev_r_wl)
ax.bar(range(len(ev_r_wl)), np.array(ev_r_wl)[sort_idx], color='steelblue', linewidth=0)
ax.axhline(np.median(ev_r_wl), color='red', linestyle='--', linewidth=0.8, label='median')
ax.set_xlabel("Event rank"); ax.set_ylabel("Mean roughness")
ax.set_title("Per-event mean water_level roughness"); ax.legend(fontsize=7)

# 7. Static feature distributions (CV)
ax = axes[2, 0]
cv_vals = static_df["cv"].replace([np.inf, -np.inf], np.nan).dropna()
ax.barh(cv_vals.index, cv_vals.values, color='teal')
ax.set_xlabel("Coefficient of variation (std/|mean|)")
ax.set_title("Static features — CV (spread across nodes)")

# 8. Cross-node coherence per event
ax = axes[2, 1]
if wl_corr_eigenratios:
    ev_coh_ids = [x[0] for x in wl_corr_eigenratios]
    ev_coh_r   = [x[1] for x in wl_corr_eigenratios]
    sort_idx = np.argsort(ev_coh_r)
    ax.bar(range(len(ev_coh_r)), np.array(ev_coh_r)[sort_idx], color='goldenrod', linewidth=0)
    ax.axhline(np.median(ev_coh_r), color='red', linestyle='--', linewidth=0.8, label='median')
    ax.set_xlabel("Event rank"); ax.set_ylabel("PCA 1st eigenvalue fraction")
    ax.set_title("Cross-node water_level coherence per event"); ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, "No coherence data", ha='center', va='center')

# 9. Roughness vs autocorr scatter (water_level) — align all three arrays by node_id
ax = axes[2, 2]
node_ids_list = sorted(wl_roughness_mean.keys())
_r   = np.array([wl_roughness_mean[n] for n in node_ids_list])
_ac  = np.array([wl_autocorr_mean[n]  for n in node_ids_list])
_snr = np.array([wl_snr_mean[n]       for n in node_ids_list])
valid = ~np.isnan(_r) & ~np.isnan(_ac) & ~np.isinf(_snr) & ~np.isnan(_snr)
sc = ax.scatter(_r[valid], _ac[valid], s=10, alpha=0.5,
                c=_snr[valid], cmap='RdYlGn',
                vmin=np.nanpercentile(_snr[valid],5), vmax=np.nanpercentile(_snr[valid],95))
plt.colorbar(sc, ax=ax, label='SNR')
ax.axvline(3*np.median(wl_r_vals), color='orange', linestyle='--', linewidth=0.8)
ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8)
ax.set_xlabel("Mean roughness"); ax.set_ylabel("Autocorr lag-1")
ax.set_title("water_level: roughness vs autocorr (color=SNR)")

plt.tight_layout()
out_path = PLOT_DIR / "1d_noise_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n[INFO] Plot saved to {out_path}")

# ── problem node summary ──────────────────────────────────────────────────────

print("\n" + "="*70)
print("PROBLEM NODE SUMMARY")
print("="*70)
median_r = np.median(wl_r_vals)
problem_high_roughness = [n for n, v in wl_roughness_mean.items()
                          if not np.isnan(v) and v > 3 * median_r]
problem_low_autocorr   = [n for n, v in wl_autocorr_mean.items()
                          if not np.isnan(v) and v < 0.5]
problem_low_snr        = [n for n, v in wl_snr_mean.items()
                          if not np.isinf(v) and not np.isnan(v) and v < 5]

print(f"Nodes with water_level roughness > 3×median ({3*median_r:.4f}): "
      f"{len(problem_high_roughness)} nodes → {sorted(problem_high_roughness)}")
print(f"Nodes with water_level autocorr < 0.5: "
      f"{len(problem_low_autocorr)} nodes → {sorted(problem_low_autocorr)}")
print(f"Nodes with water_level SNR < 5: "
      f"{len(problem_low_snr)} nodes → {sorted(problem_low_snr)}")

overlap = set(problem_high_roughness) & set(problem_low_autocorr)
print(f"\nNodes flagged by BOTH high-roughness AND low-autocorr: {len(overlap)} → {sorted(overlap)}")

# Cross-reference with static features
if overlap:
    print(f"\nStatic features for flagged nodes:")
    print(static[static["node_idx"].isin(overlap)].to_string(index=False))
