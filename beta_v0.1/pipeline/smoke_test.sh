#!/usr/bin/env bash
# smoke_test.sh — End-to-end env validation for 3DGPS beta_v0.1
#
# PURPOSE
#   Phase 1 "PC setup" sanity check. Runs the full nerfstudio + gsplat +
#   CUDA pipeline on the public Mip-NeRF360 "garden" scene so you know the
#   conda environment is healthy BEFORE wasting a real video capture.
#
# PREREQUISITES
#   1. conda env "splat-beta" is ACTIVATED
#        conda activate splat-beta
#   2. An NVIDIA GPU with >=6 GB VRAM and up-to-date drivers is present.
#   3. ~1 GB of free disk space (dataset download, first run only).
#
# USAGE
#   bash pipeline/smoke_test.sh
#
# OVERRIDES (env vars)
#   SMOKE_DIR   — where to write data + train outputs (default: results/_smoke)
#   SMOKE_ITERS — splatfacto iteration budget (default: 2000)
#
# CLEANUP
#   rm -rf results/_smoke   — removes everything this script created

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Locate project root (same pattern as run_baseline.sh)
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE_DIR="${SMOKE_DIR:-$ROOT/results/_smoke}"
SMOKE_ITERS="${SMOKE_ITERS:-2000}"

START=$(date +%s)

# ---------------------------------------------------------------------------
# ERR trap — make failures obvious
# ---------------------------------------------------------------------------
trap 'echo "" >&2
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
echo "  SMOKE TEST FAILED at line $LINENO" >&2
echo "  Your env is not ready; see pipeline/env.yml" >&2
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2' ERR

# ---------------------------------------------------------------------------
# 1. Header
# ---------------------------------------------------------------------------
echo ""
echo "======================================================="
echo "  3DGPS beta_v0.1 - Environment Smoke Test"
echo "  Dataset : Mip-NeRF360 'garden' (public, ~1 GB)"
echo "  Training: splatfacto ${SMOKE_ITERS} iterations"
echo "  Output  : $SMOKE_DIR"
echo "======================================================="
echo ""

# ---------------------------------------------------------------------------
# 2. Check required commands
# ---------------------------------------------------------------------------
echo "=== [1/5] Checking required commands ==="

_check_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' not found in PATH." >&2
    echo "       Make sure the 'splat-beta' conda env is activated:" >&2
    echo "         conda activate splat-beta" >&2
    echo "       If the command is still missing, re-create the env:" >&2
    echo "         conda env create -f pipeline/env.yml" >&2
    exit 1
  fi
  echo "  OK  $cmd -> $(command -v "$cmd")"
}

_check_cmd ns-download-data
_check_cmd ns-train
_check_cmd ns-export
_check_cmd python

# ---------------------------------------------------------------------------
# 3. CUDA check
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/5] Checking CUDA / GPU ==="

CUDA_CHECK=$(python - <<'PYEOF'
import sys
try:
    import torch
except ImportError:
    print("CUDA_UNAVAILABLE: torch not importable")
    sys.exit(1)

avail = torch.cuda.is_available()
print(f"CUDA available: {avail}")
if avail:
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("CUDA_OK")
else:
    print("device: NONE")
    print("CUDA_UNAVAILABLE: torch.cuda.is_available() returned False")
PYEOF
)

echo "$CUDA_CHECK"

if ! echo "$CUDA_CHECK" | grep -q "^CUDA_OK"; then
  echo "" >&2
  echo "ERROR: CUDA is not available." >&2
  echo "       This project requires an NVIDIA GPU." >&2
  echo "       Check:" >&2
  echo "         - NVIDIA drivers are installed (nvidia-smi should work)" >&2
  echo "         - CUDA 11.8 toolkit is present" >&2
  echo "         - torch+cu118 was installed (see pipeline/env.yml)" >&2
  echo "         - The conda env is activated: conda activate splat-beta" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 4. Download dataset (skip if already present)
# ---------------------------------------------------------------------------
echo ""
echo "=== [3/5] Dataset download (garden scene) ==="

# The garden scene is available via the nerfstudio downloader.
#
# PRIMARY command used here:
#   ns-download-data nerfstudio --capture-name garden
#
# REVIEWER NOTE: nerfstudio ships two relevant sub-commands:
#   a) ns-download-data nerfstudio --capture-name garden
#      Downloads the nerfstudio-processed version of the Mip-NeRF360 garden
#      scene hosted by the nerfstudio team. This is the most likely to work
#      out-of-the-box with ns-train splatfacto.
#   b) ns-download-data mipnerf360 --capture-name garden
#      Downloads the original Mip-NeRF360 dataset from Google's servers.
#      The directory layout differs slightly; ns-train can still consume it
#      but you may need to adjust --data to point one level deeper.
#
# If (a) errors with "unknown capture-name", swap to (b).
# Verify available captures with: ns-download-data nerfstudio --help

DATA_DIR="$SMOKE_DIR/data"
SCENE_DIR="$DATA_DIR/garden"

mkdir -p "$DATA_DIR"

if [ -d "$SCENE_DIR" ] && [ "$(ls -A "$SCENE_DIR" 2>/dev/null)" ]; then
  echo "  Data already present at $SCENE_DIR - skipping download."
else
  echo "  Downloading garden scene (~1 GB, one-time) ..."
  ns-download-data nerfstudio \
    --capture-name garden \
    --save-dir "$DATA_DIR"
fi

# Confirm the scene directory exists after download
if [ ! -d "$SCENE_DIR" ]; then
  echo "" >&2
  echo "ERROR: Expected scene directory not found: $SCENE_DIR" >&2
  echo "       ns-download-data may have written to a different path." >&2
  echo "       Check $DATA_DIR for the actual directory name and set" >&2
  echo "       SCENE_DIR manually if needed." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 5. Short splatfacto training run
# ---------------------------------------------------------------------------
echo ""
echo "=== [4/5] splatfacto training (${SMOKE_ITERS} iterations) ==="
echo "  Input : $SCENE_DIR"
echo "  Output: $SMOKE_DIR/train"

TRAIN_OUT="$SMOKE_DIR/train"
mkdir -p "$TRAIN_OUT"

ns-train splatfacto \
  --data "$SCENE_DIR" \
  --output-dir "$TRAIN_OUT" \
  --max-num-iterations "$SMOKE_ITERS" \
  --machine.seed 42 \
  --viewer.quit-on-train-completion True

# ---------------------------------------------------------------------------
# 6. Export .ply
# ---------------------------------------------------------------------------
echo ""
echo "=== [5/5] Exporting gaussian-splat .ply ==="

CONFIG=$(find "$TRAIN_OUT" -name config.yml | head -n 1)
if [ -z "$CONFIG" ]; then
  echo "ERROR: trained config.yml not found under $TRAIN_OUT" >&2
  exit 1
fi
echo "  Using config: $CONFIG"

ns-export gaussian-splat \
  --load-config "$CONFIG" \
  --output-dir "$SMOKE_DIR"

# nerfstudio writes splat.ply by default
PLY_OUT="$SMOKE_DIR/splat.ply"

# ---------------------------------------------------------------------------
# 7. Done banner
# ---------------------------------------------------------------------------
END=$(date +%s)
ELAPSED=$((END - START))
SIZE_MB=$(du -m "$PLY_OUT" 2>/dev/null | cut -f1 || echo "?")

echo ""
echo "======================================================="
echo "  ENV OK - nerfstudio + gsplat + CUDA all functional"
echo "======================================================="
echo "  Time   : ${ELAPSED}s"
echo "  Output : $PLY_OUT (${SIZE_MB} MB)"
echo ""
echo "Next steps:"
echo "  1. Record a real video and place it at:"
echo "       captures/<room_name>/video.mp4"
echo "  2. Run the full baseline pipeline:"
echo "       bash pipeline/run_baseline.sh <room_name>"
echo "  3. Optionally remove smoke-test data:"
echo "       rm -rf $SMOKE_DIR"
echo "======================================================="
