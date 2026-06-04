#!/usr/bin/env bash
# Foundation-model pose pipeline (LingBot-Map / VGGT) -> 3DGS training
# Usage: bash pipeline/run_lingbot.sh <room_name>
# Expects: captures/<room>/video.mp4
#
# STATUS: SCAFFOLDED — NEEDS VERIFICATION OF ACTUAL LingBot-Map CLI/API
# Before running this, study:
#   - https://github.com/robbyant/lingbot-map  (README + inference script)
#   - https://huggingface.co/robbyant/lingbot-map  (model weights)
# Determine actual inference CLI/Python API, then replace the placeholder block below.

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

if [ ! -f "$VIDEO" ]; then
  echo "ERROR: $VIDEO not found." >&2
  exit 1
fi

mkdir -p "$WORK" "$FRAMES" "$POSES_OUT"
START=$(date +%s)

echo "=== [1/4] Extract frames from video ==="
# Smart frame extraction (blur filter + motion-based)
python "$ROOT/pipeline/extract_frames.py" \
  --video "$VIDEO" \
  --output "$FRAMES" \
  --target-count 300 \
  --blur-threshold 100

echo "=== [2/4] Run LingBot-Map for poses + sparse cloud ==="
# TODO: replace this block with actual LingBot-Map inference command.
# Expected interface (based on README we've seen — verify before running):
#   python -m lingbot_map.infer \
#     --frames-dir "$FRAMES" \
#     --output-dir "$POSES_OUT" \
#     --checkpoint <path-to-weights>
# Output expected: poses.json + sparse_points.ply
echo "PLACEHOLDER: implement LingBot-Map inference call here." >&2
echo "See https://github.com/robbyant/lingbot-map for actual CLI." >&2
exit 2

echo "=== [3/4] Convert to nerfstudio format ==="
# Convert LingBot-Map output (poses.json + sparse cloud) to nerfstudio's
# transforms.json + sparse_pc.ply layout. May need a small Python adapter.
python "$ROOT/pipeline/lingbot_to_nerfstudio.py" \
  --poses "$POSES_OUT/poses.json" \
  --sparse "$POSES_OUT/sparse_points.ply" \
  --frames "$FRAMES" \
  --output "$PROCESSED"

echo "=== [4/4] 3DGS training (splatfacto, same as baseline) ==="
ns-train splatfacto \
  --data "$PROCESSED" \
  --output-dir "$TRAIN_OUT" \
  --max-num-iterations 30000 \
  --viewer.quit-on-train-completion True

CONFIG=$(find "$TRAIN_OUT" -name config.yml | head -n 1)
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
echo "==============================================="
