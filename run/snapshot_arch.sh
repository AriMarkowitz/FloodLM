#!/bin/bash
# snapshot_arch.sh — capture a reproducible architecture snapshot for a submission.
#
# Usage (source or call from another script):
#   bash run/snapshot_arch.sh <submission_file_basename> <project_dir>
#
# Creates: snapshots/<submission_basename>/
#   git_commit.txt   — full commit hash + message
#   git_diff.patch   — all uncommitted changes (empty if clean)
#   git_status.txt   — working tree status
#   model.py         — model architecture
#   train.py         — training config + loop
#   data.py          — data loading + graph construction
#   data_config.py   — model/split config
#   data_lazy.py     — normalization logic

set -euo pipefail

SUBMISSION_BASE="${1:?Usage: snapshot_arch.sh <submission_basename> <project_dir>}"
PROJECT_DIR="${2:?}"

SNAP_DIR="${PROJECT_DIR}/snapshots/${SUBMISSION_BASE}"
mkdir -p "${SNAP_DIR}"

cd "${PROJECT_DIR}"

# Git state
git log -1 --format="commit %H%nauthor %an <%ae>%ndate   %ai%n%n%s%n%b" \
    > "${SNAP_DIR}/git_commit.txt" 2>/dev/null || echo "(not a git repo or no commits)" > "${SNAP_DIR}/git_commit.txt"

git diff HEAD > "${SNAP_DIR}/git_diff.patch" 2>/dev/null || echo "(no diff)" > "${SNAP_DIR}/git_diff.patch"

git status --short > "${SNAP_DIR}/git_status.txt" 2>/dev/null || echo "(no status)" > "${SNAP_DIR}/git_status.txt"

# Key source files
for f in src/model.py src/train.py src/data.py src/data_config.py src/data_lazy.py; do
    [ -f "${PROJECT_DIR}/${f}" ] && cp "${PROJECT_DIR}/${f}" "${SNAP_DIR}/$(basename ${f})"
done

# Checkpoint metadata: epoch + loss for each model in latest/
for pt in "${PROJECT_DIR}/checkpoints/latest"/Model_*_best*.pt; do
    [ -f "${pt}" ] || continue
    python3 -c "
import torch, os, sys
ckpt = torch.load('${pt}', map_location='cpu', weights_only=False)
print(f\"$(basename ${pt}): epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?'):.6f}, model_id={ckpt.get('model_id','?')}\")
" >> "${SNAP_DIR}/checkpoint_metadata.txt" 2>/dev/null || true
done

echo "[snapshot] Saved to ${SNAP_DIR}"
echo "[snapshot] Contents: $(ls ${SNAP_DIR} | tr '\n' ' ')"
