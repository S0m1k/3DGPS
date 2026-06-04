#!/usr/bin/env bash
# Baseline pipeline: video -> COLMAP poses -> 3DGS training -> .ply export
# Usage: bash pipeline/run_baseline.sh <room_name>
# Expects: captures/<room>/video.mp4

set -euo pipefail

ROOM="${1:?usage: run_baseline.sh <room_name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VIDEO="$ROOT/captures/$ROOM/video.mp4"
WORK="$ROOT/results/$ROOM"
PROCESSED="$WORK/processed"
TRAIN_OUT="$WORK/train"
PLY_OUT="$WORK/baseline.ply"

# --- Reproducibility knobs (SPEC §7 / R6) ---
# Override via env: SEED=123 FRAMES_TARGET=200 MAX_ITERS=20000 bash run_baseline.sh <room>
SEED="${SEED:-42}"
FRAMES_TARGET="${FRAMES_TARGET:-300}"
MAX_ITERS="${MAX_ITERS:-30000}"
# NOTE (SPEC §7 determinism): a fixed seed reduces non-determinism but does NOT
# guarantee bit-exact results — gsplat relies on CUDA atomics whose ordering is
# hardware-scheduler-dependent. Expect reruns to land within ±0.5 PSNR, not identical.

if [ ! -f "$VIDEO" ]; then
  echo "ERROR: $VIDEO not found. Place captured video there first." >&2
  exit 1
fi

mkdir -p "$WORK"
START=$(date +%s)

echo "=== [1/3] Extract frames + COLMAP poses ==="
# --num-frames-target is nerfstudio's frame budget. Tune per video length.
# For 3-min walkthrough, 300 frames is a reasonable starting point.
ns-process-data video \
  --data "$VIDEO" \
  --output-dir "$PROCESSED" \
  --num-frames-target "$FRAMES_TARGET" \
  --verbose

echo "=== [2/3] 3DGS training (splatfacto) ==="
# Tune max-num-iterations down if 3070 Ti runs out of VRAM
# --machine.seed is the nerfstudio MachineConfig seed field; reviewer: confirm flag
# name against your installed nerfstudio version (added for SPEC §7 / R6 reproducibility).
ns-train splatfacto \
  --data "$PROCESSED" \
  --output-dir "$TRAIN_OUT" \
  --max-num-iterations "$MAX_ITERS" \
  --machine.seed "$SEED" \
  --viewer.quit-on-train-completion True

# Snapshot inputs/environment for reproducibility (SPEC §7 / R6).
# Best-effort: guarded with || true so no probe failure can abort the pipeline.
{
  echo "room=$ROOM"
  echo "method=baseline_colmap"
  echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ' || true)"
  echo "seed=$SEED"
  echo "num_frames_target=$FRAMES_TARGET"
  echo "max_num_iterations=$MAX_ITERS"
  echo "git_commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "nerfstudio_version=$(ns-train --version 2>/dev/null || pip show nerfstudio 2>/dev/null | grep -i version || echo unknown)"
  echo "elapsed_so_far=$(($(date +%s) - START))s"
} > "$WORK/run_info.txt" || true

echo "=== [3/3] Export gaussian splat .ply ==="
# Find the latest config.yml from training output
CONFIG=$(find "$TRAIN_OUT" -name config.yml | head -n 1)
if [ -z "$CONFIG" ]; then
  echo "ERROR: trained config.yml not found under $TRAIN_OUT" >&2
  exit 1
fi
ns-export gaussian-splat --load-config "$CONFIG" --output-dir "$WORK"
# nerfstudio writes splat.ply by default; rename for clarity
mv "$WORK/splat.ply" "$PLY_OUT" 2>/dev/null || true

END=$(date +%s)
ELAPSED=$((END - START))

# Quick stats
SIZE_MB=$(du -m "$PLY_OUT" 2>/dev/null | cut -f1 || echo "?")
echo ""
echo "==================== DONE ===================="
echo "Room:        $ROOM"
echo "Method:      baseline_colmap"
echo "Time:        ${ELAPSED}s"
echo "Output:      $PLY_OUT (${SIZE_MB} MB)"
echo ""
echo "Next steps:"
echo "  1. Drag-drop $PLY_OUT into https://playcanvas.com/supersplat/viewer"
echo "  2. Append a row to results/metrics.csv (see schema)"
echo "  3. Record subjective observations in results/notes.md"
echo "==============================================="
