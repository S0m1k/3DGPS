#!/usr/bin/env python3
"""Smart frame extraction from video for 3DGS training.

Strategy:
  1. Decode video frame by frame
  2. Compute Laplacian variance per frame (sharpness score)
  3. Drop frames below sharpness threshold (motion-blurred)
  4. From remaining candidates, select every Nth such that we hit
     --target-count frames roughly evenly spread in time

This is naive vs the ideal (pose-aware motion-based selection), but
we don't have poses yet at this stage — by definition this runs BEFORE
pose estimation. A v2 of this script can re-select after initial pose pass.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract(video_path: Path, out_dir: Path, target_count: int, blur_threshold: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"failed to open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"input: {total_frames} frames @ {fps:.1f} fps")

    # Pass 1: score every frame's sharpness
    scores: list[tuple[int, float]] = []
    for idx in tqdm(range(total_frames), desc="scoring"):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores.append((idx, laplacian_variance(gray)))
    cap.release()

    # Keep only sharp-enough frames
    sharp = [(i, s) for i, s in scores if s >= blur_threshold]
    print(f"sharp frames: {len(sharp)} / {len(scores)} (threshold={blur_threshold})")

    if len(sharp) == 0:
        raise SystemExit("no sharp frames — lower threshold or check video")

    # Evenly subsample to target_count
    if len(sharp) <= target_count:
        selected = sharp
    else:
        step = len(sharp) / target_count
        selected = [sharp[int(i * step)] for i in range(target_count)]
    print(f"selected: {len(selected)} frames")

    # Pass 2: write selected frames
    selected_set = {i for i, _ in selected}
    cap = cv2.VideoCapture(str(video_path))
    written = 0
    manifest: list[dict] = []
    for idx in tqdm(range(total_frames), desc="writing"):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in selected_set:
            name = f"frame_{written:05d}.jpg"
            cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({"orig_idx": idx, "file": name})
            written += 1
    cap.release()

    # Write manifest
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["orig_idx", "file"])
        w.writeheader()
        w.writerows(manifest)

    return {"input_frames": total_frames, "sharp": len(sharp), "written": written}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-count", type=int, default=300)
    p.add_argument("--blur-threshold", type=float, default=100.0,
                   help="Laplacian variance below this = considered blurry. "
                        "Typical sharp indoor frame: 200–800. Motion blur: <50.")
    args = p.parse_args()

    stats = extract(args.video, args.output, args.target_count, args.blur_threshold)
    print(f"done: {stats}")


if __name__ == "__main__":
    main()
