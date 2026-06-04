#!/usr/bin/env python3
"""Convert LingBot-Map saved predictions to a nerfstudio splatfacto dataset.

Usage:
    python pipeline/lingbot_to_nerfstudio.py \
        --predictions results/<room>/lingbot_poses \
        --frames     results/<room>/frames \
        --output     results/<room>/processed_lingbot

The output directory can then be passed directly to:
    ns-train splatfacto --data <output>

Coordinate convention
---------------------
LingBot-Map "extrinsic" is camera-to-world in OpenCV convention
(+Z forward, +Y down).  nerfstudio / Blender expect OpenGL convention
(camera looks down -Z, +Y up).  We flip by negating columns 1 and 2 of
the rotation part:

    c2w[:, 1] *= -1   # flip Y column
    c2w[:, 2] *= -1   # flip Z column

Disable with --no-opengl-flip for debugging mirrored output.

NPZ key names
-------------
The constants below MUST be verified against an actual LingBot-Map run.
If a key is missing the script will print the keys found in the NPZ and
exit with a helpful message.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS — verify these against a real LingBot-Map run.
# Source: github.com/Robbyant/lingbot-map demo.py --save_predictions logic.
# ---------------------------------------------------------------------------
KEY_EXTRINSIC = "extrinsic"       # camera-to-world 3×4 (already c2w, not w2c)
KEY_INTRINSIC = "intrinsic"       # camera intrinsics 3×3
KEY_POINTS = "world_points"       # (N, 3) float32 3D point positions
KEY_POINTS_CONF = "world_points_conf"  # (N,) float32 confidence per point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_keys(npz: np.lib.npyio.NpzFile, required: list[str], path: Path) -> None:
    """Raise SystemExit with a helpful message if any required key is absent."""
    found = list(npz.files)
    missing = [k for k in required if k not in found]
    if missing:
        raise SystemExit(
            f"ERROR: NPZ '{path}' is missing expected keys: {missing}\n"
            f"       Keys actually found: {found}\n"
            f"       Update the KEY_* constants at the top of {__file__}\n"
            f"       to match the keys produced by your LingBot-Map version."
        )


def _load_predictions(predictions_path: Path) -> list[dict]:
    """Load per-frame predictions from a directory of NPZ files or a single NPZ.

    Returns a list of dicts, one per frame, each with at least:
        extrinsic: np.ndarray shape (3, 4)  camera-to-world
        intrinsic: np.ndarray shape (3, 3)
        world_points: np.ndarray shape (N, 3)  or None
        world_points_conf: np.ndarray shape (N,)  or None
    Sorted by filename so frame order is deterministic.
    """
    required = [KEY_EXTRINSIC, KEY_INTRINSIC]
    optional = [KEY_POINTS, KEY_POINTS_CONF]

    # ---- Case 1: single .npz with a leading frame axis ----
    if predictions_path.is_file() and predictions_path.suffix == ".npz":
        npz = np.load(predictions_path, allow_pickle=False)
        _check_keys(npz, required, predictions_path)
        extrinsics = npz[KEY_EXTRINSIC]   # expected (F, 3, 4)
        intrinsics = npz[KEY_INTRINSIC]   # expected (F, 3, 3) or (3, 3)
        points_all = npz[KEY_POINTS] if KEY_POINTS in npz.files else None
        conf_all = npz[KEY_POINTS_CONF] if KEY_POINTS_CONF in npz.files else None

        if extrinsics.ndim == 2:
            # Single frame stored without a batch dimension — wrap it.
            extrinsics = extrinsics[np.newaxis]

        n_frames = extrinsics.shape[0]
        if intrinsics.ndim == 2:
            intrinsics = np.tile(intrinsics[np.newaxis], (n_frames, 1, 1))

        frames = []
        for i in range(n_frames):
            frames.append({
                KEY_EXTRINSIC: extrinsics[i],
                KEY_INTRINSIC: intrinsics[i],
                KEY_POINTS: points_all[i] if (points_all is not None and points_all.ndim == 3) else points_all,
                KEY_POINTS_CONF: conf_all[i] if (conf_all is not None and conf_all.ndim == 2) else conf_all,
            })
        return frames

    # ---- Case 2: directory of per-frame NPZ files ----
    if predictions_path.is_dir():
        npz_paths = sorted(predictions_path.glob("*.npz"))
        if not npz_paths:
            raise SystemExit(
                f"ERROR: No *.npz files found in '{predictions_path}'.\n"
                f"       Pass the directory that contains LingBot-Map's prediction NPZs,\n"
                f"       or point --predictions at a single combined .npz file."
            )
        frames = []
        for p in npz_paths:
            npz = np.load(p, allow_pickle=False)
            _check_keys(npz, required, p)
            entry: dict = {
                KEY_EXTRINSIC: npz[KEY_EXTRINSIC],
                KEY_INTRINSIC: npz[KEY_INTRINSIC],
                KEY_POINTS: npz[KEY_POINTS] if KEY_POINTS in npz.files else None,
                KEY_POINTS_CONF: npz[KEY_POINTS_CONF] if KEY_POINTS_CONF in npz.files else None,
            }
            # Squeeze a spurious leading batch dim from per-frame files.
            for key in [KEY_EXTRINSIC, KEY_INTRINSIC]:
                arr = entry[key]
                if arr.ndim == 3 and arr.shape[0] == 1:
                    entry[key] = arr[0]
            frames.append(entry)
        return frames

    raise SystemExit(
        f"ERROR: --predictions '{predictions_path}' is neither a .npz file nor a directory."
    )


def _opengl_flip(c2w_3x4: np.ndarray) -> np.ndarray:
    """Convert camera-to-world from OpenCV to OpenGL convention.

    Negate columns 1 (Y) and 2 (Z) of the rotation part.
    Translation column (3) is left unchanged.
    """
    mat = c2w_3x4.copy().astype(np.float64)
    mat[:, 1] *= -1.0
    mat[:, 2] *= -1.0
    return mat


def _pad_to_4x4(c2w_3x4: np.ndarray) -> list[list[float]]:
    """Return a 4×4 nested list with [0, 0, 0, 1] bottom row."""
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
    mat4 = np.vstack([c2w_3x4.astype(np.float64), bottom])
    return mat4.tolist()


def _image_wh(image_path: Path) -> tuple[int, int]:
    """Return (width, height) of an image file.  Tries PIL, falls back to cv2."""
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.width, im.height
    except ImportError:
        pass
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(image_path))
        if img is None:
            raise SystemExit(f"ERROR: cv2 could not read '{image_path}'")
        h, w = img.shape[:2]
        return w, h
    except ImportError:
        pass
    raise SystemExit(
        "ERROR: neither Pillow nor cv2 is installed.  Install one:\n"
        "    pip install Pillow\n"
        "    # or\n"
        "    pip install opencv-python-headless"
    )


# ---------------------------------------------------------------------------
# Point-cloud helpers
# ---------------------------------------------------------------------------

def _gather_points(
    frames: list[dict],
    conf_threshold: float,
    max_points: int,
) -> np.ndarray | None:
    """Collect and filter world_points from all frames into a single (N, 3) array."""
    all_pts: list[np.ndarray] = []
    for f in frames:
        pts = f.get(KEY_POINTS)
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            warnings.warn(
                f"world_points has unexpected shape {pts.shape}; skipping this frame.",
                stacklevel=2,
            )
            continue
        conf = f.get(KEY_POINTS_CONF)
        if conf is not None:
            conf = np.asarray(conf, dtype=np.float32).ravel()
            if conf.shape[0] == pts.shape[0]:
                pts = pts[conf >= conf_threshold]
        all_pts.append(pts)

    if not all_pts:
        return None

    combined = np.concatenate(all_pts, axis=0)
    if combined.shape[0] == 0:
        return None

    if combined.shape[0] > max_points:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(combined.shape[0], size=max_points, replace=False)
        combined = combined[idx]

    return combined


def _write_ply(path: Path, points: np.ndarray) -> None:
    """Write a binary-little-endian PLY with x, y, z float32 + r, g, b uchar.

    nerfstudio's seed-point loader (used when transforms.json sets
    ``ply_file_path``) reads per-point colours and breaks on a colour-less
    cloud.  LingBot-Map ``world_points`` carry no colour, so we write a neutral
    grey (128, 128, 128); splatfacto refines the actual colours during training.
    """
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    xyz = np.ascontiguousarray(points, dtype="<f4")          # 3 × float32
    rgb = np.full((n, 3), 128, dtype=np.uint8)               # neutral grey
    # Interleave per-vertex: struct {float x,y,z; uchar r,g,b;} — build a
    # structured array so the byte layout matches the header exactly.
    vertex = np.empty(
        n,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(vertex.tobytes())


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    predictions_path: Path,
    frames_dir: Path,
    output_dir: Path,
    conf_threshold: float,
    max_points: int,
    opengl_flip: bool,
) -> None:
    # ---- Validate inputs ----
    if not predictions_path.exists():
        raise SystemExit(f"ERROR: --predictions path does not exist: '{predictions_path}'")
    if not frames_dir.is_dir():
        raise SystemExit(f"ERROR: --frames directory does not exist: '{frames_dir}'")

    # ---- Load predictions ----
    print("Loading LingBot-Map predictions ...")
    frames_pred = _load_predictions(predictions_path)
    n_pred = len(frames_pred)
    print(f"  Loaded {n_pred} frame prediction(s)")

    # ---- Collect source images ----
    img_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    src_images = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix in img_exts and not p.name.startswith(".")
    )
    if not src_images:
        raise SystemExit(
            f"ERROR: No images found in --frames dir '{frames_dir}'.\n"
            f"       Expected .jpg / .png files written by extract_frames.py."
        )

    if len(src_images) != n_pred:
        warnings.warn(
            f"Number of source images ({len(src_images)}) != number of predictions "
            f"({n_pred}).  Using min({len(src_images)}, {n_pred}) pairs.",
            stacklevel=2,
        )

    n_frames = min(len(src_images), n_pred)
    src_images = src_images[:n_frames]
    frames_pred = frames_pred[:n_frames]

    # ---- Read image dimensions from the first frame ----
    w, h = _image_wh(src_images[0])
    print(f"  Image dimensions: {w}x{h}")

    # ---- Read intrinsics from first frame ----
    K = np.asarray(frames_pred[0][KEY_INTRINSIC], dtype=np.float64)
    if K.shape != (3, 3):
        raise SystemExit(
            f"ERROR: Expected intrinsic to be 3x3, got shape {K.shape}.\n"
            f"       Update KEY_INTRINSIC or the parsing logic."
        )
    fl_x = float(K[0, 0])
    fl_y = float(K[1, 1])
    cx   = float(K[0, 2])
    cy   = float(K[1, 2])

    # ---- Build output directory layout ----
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    images_out.mkdir(exist_ok=True)

    # ---- Copy frames + build frame list ----
    nerfstudio_frames: list[dict] = []
    print(f"Copying {n_frames} images to {images_out} ...")
    for i, (src, pred) in enumerate(zip(src_images, frames_pred)):
        dst_name = f"frame_{i:05d}.jpg"
        dst = images_out / dst_name

        # Copy (convert to JPEG if source is PNG, for consistency)
        if src.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(src, dst)
        else:
            try:
                from PIL import Image
                with Image.open(src) as im:
                    im.convert("RGB").save(dst, "JPEG", quality=95)
            except ImportError:
                import cv2  # type: ignore
                img = cv2.imread(str(src))
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Build transform_matrix
        extr = np.asarray(pred[KEY_EXTRINSIC], dtype=np.float64)
        if extr.shape != (3, 4):
            # Try to handle (4, 4) by taking the first 3 rows
            if extr.shape == (4, 4):
                extr = extr[:3, :]
            else:
                raise SystemExit(
                    f"ERROR: extrinsic for frame {i} has unexpected shape {extr.shape}.\n"
                    f"       Expected (3, 4) or (4, 4)."
                )

        if opengl_flip:
            extr = _opengl_flip(extr)

        nerfstudio_frames.append({
            "file_path": f"images/{dst_name}",
            "transform_matrix": _pad_to_4x4(extr),
        })

    # ---- Write sparse point cloud (best-effort) ----
    ply_written = False
    print("Building sparse point cloud ...")
    points = _gather_points(frames_pred, conf_threshold, max_points)
    if points is not None:
        ply_path = output_dir / "sparse_pc.ply"
        _write_ply(ply_path, points)
        ply_written = True
        print(f"  Wrote {points.shape[0]:,} points -> {ply_path}")
    else:
        warnings.warn(
            "No world_points found (or all filtered by confidence threshold). "
            "Skipping sparse_pc.ply.  splatfacto will initialise randomly.",
            stacklevel=2,
        )

    # ---- Write transforms.json ----
    transforms: dict = {
        "camera_model": "OPENCV",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "frames": nerfstudio_frames,
    }
    if ply_written:
        transforms["ply_file_path"] = "sparse_pc.ply"

    tf_path = output_dir / "transforms.json"
    with tf_path.open("w", encoding="utf-8") as fh:
        json.dump(transforms, fh, indent=2)
    print(f"  Wrote transforms.json ({n_frames} frames)")

    # ---- Summary ----
    print()
    print("==================== DONE ====================")
    print(f"Frames converted : {n_frames}")
    print(f"Points written   : {points.shape[0]:,}" if ply_written else "Points written   : 0 (skipped)")
    print(f"Output dir       : {output_dir.resolve()}")
    print(f"OpenGL flip      : {'yes' if opengl_flip else 'NO (--no-opengl-flip set)'}")
    print("Next step:")
    print(f"  ns-train splatfacto --data {output_dir.resolve()}")
    print("===============================================")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Convert LingBot-Map NPZ predictions to a nerfstudio splatfacto dataset.\n"
            "\n"
            "IMPORTANT: The NPZ key names (KEY_EXTRINSIC etc.) at the top of this\n"
            "file MUST be verified against an actual LingBot-Map run before use."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--predictions",
        type=Path,
        required=True,
        metavar="DIR_OR_NPZ",
        help=(
            "Directory containing per-frame *.npz files output by "
            "LingBot-Map --save_predictions, OR a single combined .npz."
        ),
    )
    p.add_argument(
        "--frames",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory of source frames written by extract_frames.py.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output nerfstudio dataset directory to create.",
    )
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.1,
        metavar="FLOAT",
        help=(
            "Minimum world_points_conf value to include a point in sparse_pc.ply. "
            "Default: 0.1.  Set to 0.0 to keep all points."
        ),
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        metavar="INT",
        help="Subsample sparse cloud to at most this many points. Default: 200000.",
    )
    p.add_argument(
        "--no-opengl-flip",
        action="store_true",
        default=False,
        help=(
            "Disable the OpenCV->OpenGL coordinate flip (negate Y and Z columns). "
            "Use for debugging only - output will appear mirrored in nerfstudio."
        ),
    )
    args = p.parse_args()

    convert(
        predictions_path=args.predictions,
        frames_dir=args.frames,
        output_dir=args.output,
        conf_threshold=args.conf_threshold,
        max_points=args.max_points,
        opengl_flip=not args.no_opengl_flip,
    )


if __name__ == "__main__":
    main()
