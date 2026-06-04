#!/usr/bin/env bash
# Foundation-model pose pipeline (LingBot-Map) -> 3DGS training
# Usage: bash pipeline/run_lingbot.sh <room_name>
# Expects: captures/<room>/video.mp4
#
# STATUS: CLI is now wired to the real demo.py interface (github.com/Robbyant/lingbot-map).
#         Predictions NPZ parsing in lingbot_to_nerfstudio.py is UNVERIFIED against a real
#         run — KEY_* constants at the top of that file must be confirmed on first use.
#         See QUESTIONS.md item 5.
#
# ENV-VAR OVERRIDES (set in your shell or .env; do not edit this script):
#   LINGBOT_DEMO   — path to cloned lingbot-map repo's demo.py
#                    default: $HOME/lingbot-map/demo.py
#   LINGBOT_MODEL  — path to model weights (.pt)
#                    default: $HOME/models/lingbot-map-long.pt
#
# First-time setup:
#   git clone https://github.com/Robbyant/lingbot-map ~/lingbot-map
#   pip install -r ~/lingbot-map/requirements.txt
#   # Download weights from https://huggingface.co/robbyant/lingbot-map
#   mkdir -p ~/models && mv lingbot-map-long.pt ~/models/

set -euo pipefail

ROOM="${1:?usage: run_lingbot.sh <room_name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VIDEO="$ROOT/captures/$ROOM/video.mp4"
WORK="$ROOT/results/$ROOM"
FRAMES="$WORK/frames"
POSES_OUT="$WORK/lingbot_poses"
PROCESSED="$WORK/processed_lingbot"
TRAIN_OUT="$WORK/train_lingbot"
PLY_OUT="$WORK/lingbot.ply"

# Resolve LingBot-Map paths from env (with sensible defaults).
LINGBOT_DEMO="${LINGBOT_DEMO:-$HOME/lingbot-map/demo.py}"
LINGBOT_MODEL="${LINGBOT_MODEL:-$HOME/models/lingbot-map-long.pt}"

if [ ! -f "$VIDEO" ]; then
  echo "ERROR: $VIDEO not found. Place captured video there first." >&2
  exit 1
fi

# Guard: ensure lingbot-map is cloned and demo.py is reachable.
if [ ! -f "$LINGBOT_DEMO" ]; then
  echo "ERROR: LingBot-Map demo.py not found at '$LINGBOT_DEMO'." >&2
  echo "" >&2
  echo "  Fix options:" >&2
  echo "    1. Clone the repo:" >&2
  echo "         git clone https://github.com/Robbyant/lingbot-map ~/lingbot-map" >&2
  echo "    2. Or set the LINGBOT_DEMO env var to its actual location:" >&2
  echo "         export LINGBOT_DEMO=/path/to/lingbot-map/demo.py" >&2
  echo "" >&2
  echo "  Also ensure the model weights exist at '$LINGBOT_MODEL'." >&2
  echo "  Download from: https://huggingface.co/robbyant/lingbot-map" >&2
  echo "  Then: export LINGBOT_MODEL=/path/to/lingbot-map-long.pt" >&2
  exit 1
fi

mkdir -p "$WORK" "$FRAMES" "$POSES_OUT"
START=$(date +%s)

echo "=== [1/4] Extract frames from video ==="
# Smart frame extraction: blur filter + even subsampling.
python "$ROOT/pipeline/extract_frames.py" \
  --video "$VIDEO" \
  --output "$FRAMES" \
  --target-count 300 \
  --blur-threshold 100

echo ""
echo "=== [2/4] Run LingBot-Map pose estimation ==="
# Arg names verified against github.com/Robbyant/lingbot-map demo.py as of 2025-06.
# --mode windowed: processes the image sequence in overlapping temporal windows.
# --window_size 128 / --overlap_keyframes 16: tune for memory vs accuracy.
# --save_predictions: persists per-frame NPZ files into --output_folder.
#
# NOTE: the exact NPZ filenames and key layout written by --save_predictions have
# NOT been verified against a real run.  On first use, inspect $POSES_OUT/*.npz and
# confirm the KEY_* constants at the top of pipeline/lingbot_to_nerfstudio.py match.
# See QUESTIONS.md item 5.
python "$LINGBOT_DEMO" \
  --model_path   "$LINGBOT_MODEL" \
  --image_folder "$FRAMES" \
  --mode         windowed \
  --window_size  128 \
  --overlap_keyframes 16 \
  --save_predictions \
  --output_folder "$POSES_OUT"

echo ""
echo "=== [3/4] Convert predictions to nerfstudio format ==="
python "$ROOT/pipeline/lingbot_to_nerfstudio.py" \
  --predictions "$POSES_OUT" \
  --frames      "$FRAMES" \
  --output      "$PROCESSED"

echo ""
echo "=== [4/4] 3DGS training (splatfacto) ==="
ns-train splatfacto \
  --data "$PROCESSED" \
  --output-dir "$TRAIN_OUT" \
  --max-num-iterations 30000 \
  --machine.seed 42 \
  --viewer.quit-on-train-completion True

# Find the config written by the training run.
CONFIG=$(find "$TRAIN_OUT" -name config.yml | head -n 1)
if [ -z "$CONFIG" ]; then
  echo "ERROR: trained config.yml not found under $TRAIN_OUT" >&2
  exit 1
fi

echo ""
echo "=== Exporting gaussian splat ==="
ns-export gaussian-splat --load-config "$CONFIG" --output-dir "$WORK"
mv "$WORK/splat.ply" "$PLY_OUT" 2>/dev/null || true

END=$(date +%s)
ELAPSED=$((END - START))
SIZE_MB=$(du -m "$PLY_OUT" 2>/dev/null | cut -f1 || echo "?")

echo ""
echo "==================== DONE ===================="
echo "Room:        $ROOM"
echo "Method:      lingbot"
echo "Time:        ${ELAPSED}s"
echo "Output:      $PLY_OUT (${SIZE_MB} MB)"
echo ""
echo "Next steps:"
echo "  1. Drag-drop $PLY_OUT into https://playcanvas.com/supersplat/viewer"
echo "  2. Append a row to results/metrics.csv (see schema)"
echo "  3. Record subjective observations in results/notes.md"
echo "==============================================="
