#!/usr/bin/env python3
"""Smart frame extraction from video for 3DGS training.

Strategies (--strategy):
  uniform
    Naive every-Nth baseline: no blur filter, just evenly subsample all
    decoded frames to --target-count.  Used as the comparison baseline
    described in Phase 5 of the README.

  blur  (DEFAULT — backward-compatible)
    1. Decode video frame-by-frame.
    2. Compute Laplacian variance per frame (sharpness score).
    3. Drop frames below --blur-threshold (motion-blurred / eyes-closed).
    4. Evenly subsample the survivors to --target-count frames spread in time.

  blur+motion
    Same sharpness scoring and blur-threshold filter as "blur", but instead
    of plain even-time subsampling the sharp survivors are selected to
    maximise CONTENT COVERAGE while avoiding near-duplicate frames caused
    by the camera pausing.

    Motion heuristic (appearance-based, no poses required):
      • During the scoring pass each sharp frame is also downscaled to a
        64×64 grayscale thumbnail (float32) and kept in memory (~1 MB total
        for 5 000 frames).
      • A first (cheap) linear scan over thumbnails computes the cumulative
        sum of mean-absolute-difference (MAD64) between consecutive sharp
        frames.  This gives a total-motion "budget".
      • An adaptive per-step threshold is derived:
            motion_step = total_motion / target_count
        so that a greedy forward walk will naturally terminate near
        target_count selected frames.
      • Greedy selection pass:
          - Always select the first sharp frame.
          - Maintain a "motion bucket" that accumulates MAD64 values.  When
            the bucket reaches motion_step, close the bucket: from all
            frames in it, keep the SHARPEST one.  Start a new bucket.
      • After the greedy pass the count may be slightly off due to integer
        rounding.  Apply safety valves:
          - Too many  → keep the sharpest target_count among the selected set.
          - Too few   → top up by even-subsampling the sharp frames not yet
                        selected (same logic as the "blur" strategy).

    This is still pre-pose; a future pass can re-select after initial SfM.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Side length of the tiny thumbnail used for motion estimation.
_THUMB_SIZE = 64


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _thumb(gray: np.ndarray) -> np.ndarray:
    """Return a 64×64 float32 thumbnail of *gray*."""
    return cv2.resize(gray, (_THUMB_SIZE, _THUMB_SIZE),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def _select_uniform(
    scores: list[tuple[int, float]],
    target_count: int,
) -> list[tuple[int, float]]:
    """Even-Nth over ALL frames (no blur filter)."""
    if len(scores) <= target_count:
        return list(scores)
    step = len(scores) / target_count
    return [scores[int(i * step)] for i in range(target_count)]


def _select_blur(
    sharp: list[tuple[int, float]],
    target_count: int,
) -> list[tuple[int, float]]:
    """Even-Nth over sharp frames."""
    if len(sharp) <= target_count:
        return list(sharp)
    step = len(sharp) / target_count
    return [sharp[int(i * step)] for i in range(target_count)]


def _select_blur_motion(
    sharp: list[tuple[int, float]],
    thumbs: dict[int, np.ndarray],
    target_count: int,
) -> list[tuple[int, float]]:
    """Greedy motion-budget selection over sharp frames.

    Parameters
    ----------
    sharp:
        List of (orig_frame_idx, sharpness_score) for frames that passed the
        blur threshold, in ascending frame-index order.
    thumbs:
        Mapping from orig_frame_idx → 64×64 float32 thumbnail.
    target_count:
        Desired number of output frames.
    """
    if len(sharp) <= target_count:
        return list(sharp)

    # --- Pass A: compute total motion budget over consecutive sharp frames ---
    total_motion = 0.0
    prev_thumb = thumbs[sharp[0][0]]
    for idx, _score in sharp[1:]:
        t = thumbs[idx]
        total_motion += float(np.mean(np.abs(t - prev_thumb)))
        prev_thumb = t

    # Derive per-step threshold; guard against near-static video.
    if total_motion < 1e-6:
        # Camera barely moved: fall back to plain even subsampling.
        return _select_blur(sharp, target_count)

    motion_step = total_motion / target_count

    # --- Pass B: greedy bucket accumulation ---
    selected: list[tuple[int, float]] = []
    # Each bucket collects (idx, score) candidates for one output frame.
    bucket: list[tuple[int, float]] = [sharp[0]]
    accumulated = 0.0
    prev_thumb = thumbs[sharp[0][0]]

    for idx, score in sharp[1:]:
        t = thumbs[idx]
        step_motion = float(np.mean(np.abs(t - prev_thumb)))
        accumulated += step_motion
        prev_thumb = t
        bucket.append((idx, score))

        if accumulated >= motion_step:
            # Close bucket: emit the sharpest frame in it.
            best = max(bucket, key=lambda x: x[1])
            selected.append(best)
            bucket = []
            accumulated = 0.0

    # Flush any leftover bucket.
    if bucket:
        best = max(bucket, key=lambda x: x[1])
        selected.append(best)

    # --- Safety valve: too many → keep sharpest target_count ---
    if len(selected) > target_count:
        selected.sort(key=lambda x: x[1], reverse=True)
        selected = selected[:target_count]
        selected.sort(key=lambda x: x[0])  # restore time order

    # --- Safety valve: too few → top up with even-subsampled leftovers ---
    if len(selected) < target_count:
        selected_set = {i for i, _ in selected}
        leftovers = [(i, s) for i, s in sharp if i not in selected_set]
        deficit = target_count - len(selected)
        if leftovers:
            step = max(1, len(leftovers) / deficit)
            extras = [leftovers[int(j * step)] for j in range(min(deficit, len(leftovers)))]
            selected.extend(extras)
            selected.sort(key=lambda x: x[0])

    return selected


def extract(
    video_path: Path,
    out_dir: Path,
    target_count: int,
    blur_threshold: float,
    strategy: str = "blur",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"failed to open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"input: {total_frames} frames @ {fps:.1f} fps  strategy={strategy}")

    # ------------------------------------------------------------------
    # Pass 1: score every frame's sharpness (+ collect thumbnails for
    #         blur+motion strategy to avoid a third decode pass).
    # ------------------------------------------------------------------
    scores: list[tuple[int, float]] = []
    thumbs: dict[int, np.ndarray] = {}   # only populated for blur+motion
    need_thumbs = (strategy == "blur+motion")

    for idx in tqdm(range(total_frames), desc="scoring"):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores.append((idx, laplacian_variance(gray)))
        if need_thumbs:
            thumbs[idx] = _thumb(gray)
    cap.release()

    # ------------------------------------------------------------------
    # Frame selection
    # ------------------------------------------------------------------
    if strategy == "uniform":
        selected = _select_uniform(scores, target_count)
        print(f"selected: {len(selected)} frames (no blur filter)")

    elif strategy == "blur":
        sharp = [(i, s) for i, s in scores if s >= blur_threshold]
        print(f"sharp frames: {len(sharp)} / {len(scores)} "
              f"(threshold={blur_threshold})")
        if len(sharp) == 0:
            raise SystemExit("no sharp frames - lower --blur-threshold or check video")
        selected = _select_blur(sharp, target_count)
        print(f"selected: {len(selected)} frames")

    elif strategy == "blur+motion":
        sharp = [(i, s) for i, s in scores if s >= blur_threshold]
        print(f"sharp frames: {len(sharp)} / {len(scores)} "
              f"(threshold={blur_threshold})")
        if len(sharp) == 0:
            raise SystemExit("no sharp frames - lower --blur-threshold or check video")
        # Restrict thumbs dict to sharp frames only (saves RAM for nothing).
        sharp_thumbs = {i: thumbs[i] for i, _ in sharp}
        selected = _select_blur_motion(sharp, sharp_thumbs, target_count)
        print(f"selected: {len(selected)} frames (motion-filtered)")

    else:
        raise SystemExit(f"unknown strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # Pass 2: write selected frames
    # ------------------------------------------------------------------
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

    sharp_count = len(scores) if strategy == "uniform" else sum(
        1 for _, s in scores if s >= blur_threshold
    )
    return {
        "input_frames": total_frames,
        "sharp": sharp_count if strategy != "uniform" else "n/a",
        "written": written,
        "strategy": strategy,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract training frames from a walkthrough video for 3DGS."
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-count", type=int, default=300)
    p.add_argument("--blur-threshold", type=float, default=100.0,
                   help="Laplacian variance below this = considered blurry. "
                        "Typical sharp indoor frame: 200-800. Motion blur: <50. "
                        "Ignored for --strategy uniform.")
    p.add_argument(
        "--strategy",
        choices=["blur", "blur+motion", "uniform"],
        default="blur",
        help=(
            "blur: blur-filter then even-time subsample (default, backward-compat). "
            "blur+motion: blur-filter then motion-budget greedy selection. "
            "uniform: no blur filter, just every-Nth baseline."
        ),
    )
    args = p.parse_args()

    stats = extract(
        args.video,
        args.output,
        args.target_count,
        args.blur_threshold,
        strategy=args.strategy,
    )
    print(f"done: {stats}")


if __name__ == "__main__":
    main()
