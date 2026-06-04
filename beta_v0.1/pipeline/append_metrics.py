#!/usr/bin/env python3
"""Append a schema-correct row to results/metrics.csv.

Usage (CLI):
    python pipeline/append_metrics.py --room bedroom --method baseline_colmap \\
        --num-frames-input 5400 --num-frames-used 300 --final-psnr 28.4

As a library:
    from pipeline.append_metrics import append_row, COLUMNS
    append_row({"room": "bedroom", "method": "lingbot"}, csv_path)
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

# Canonical column order — kept as module constant so UI can import it.
COLUMNS: list[str] = [
    "timestamp",
    "room",
    "method",
    "num_frames_input",
    "num_frames_used",
    "blur_threshold",
    "train_iterations",
    "train_time_sec",
    "vram_peak_mb",
    "final_psnr",
    "splat_count",
    "file_size_mb",
    "subjective_score_1to10",
    "wow_present_yes_no",
    "artifacts_observed",
    "notes",
]

# Default path — relative to this file's parent's parent (beta_v0.1/)
_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "results" / "metrics.csv"


def read_existing_header(csv_path: Path) -> list[str] | None:
    """Return the real header (first non-comment, non-blank line) or None."""
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Parse via csv so quoted fields survive
            row = next(csv.reader([line]))
            return row
    return None


def append_row(values: dict, csv_path: Path = _DEFAULT_CSV) -> None:
    """Append one row to *csv_path*, creating the file + header if needed.

    Args:
        values: mapping of column_name -> value.  Keys must be a subset of
                COLUMNS.  Missing columns are filled with "".
        csv_path: destination file (default: results/metrics.csv).

    Raises:
        ValueError: if *values* contains a key not in COLUMNS.
    """
    unknown = set(values) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Unknown columns: {unknown!r}. Allowed: {COLUMNS!r}")

    # Auto-fill timestamp if missing or empty
    ts = str(values.get("timestamp", "")).strip()
    if not ts:
        values = {**values, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # Build ordered row, filling blanks
    row = [str(values.get(col, "")) for col in COLUMNS]

    needs_header = read_existing_header(csv_path) is None
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if needs_header:
            writer.writerow(COLUMNS)
        writer.writerow(row)

    print("Appended row:")
    for col, val in zip(COLUMNS, row):
        if val:
            print(f"  {col}: {val}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Append one row to results/metrics.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline/append_metrics.py --room bedroom --method baseline_colmap\n"
            "      --num-frames-input 5400 --num-frames-used 300 --final-psnr 28.4\n\n"
            "Expected --method values: baseline_colmap | lingbot | vggt"
        ),
    )
    p.add_argument("--csv", type=Path, default=_DEFAULT_CSV,
                   help="Path to metrics CSV (default: results/metrics.csv)")
    p.add_argument("--timestamp",
                   help="ISO8601 UTC timestamp. Auto-filled with current time if omitted.")
    p.add_argument("--room",
                   help="Room name, e.g. bedroom / living-room / kitchen")
    p.add_argument("--method",
                   help="Reconstruction method. Expected: baseline_colmap | lingbot | vggt")
    p.add_argument("--num-frames-input", dest="num_frames_input",
                   help="Total frames decoded from video")
    p.add_argument("--num-frames-used", dest="num_frames_used",
                   help="Frames passed to COLMAP/training after blur filter")
    p.add_argument("--blur-threshold", dest="blur_threshold",
                   help="Laplacian variance threshold used during extraction")
    p.add_argument("--train-iterations", dest="train_iterations",
                   help="Gaussian Splatting training iterations")
    p.add_argument("--train-time-sec", dest="train_time_sec",
                   help="Wall-clock training time in seconds")
    p.add_argument("--vram-peak-mb", dest="vram_peak_mb",
                   help="Peak VRAM usage in MB")
    p.add_argument("--final-psnr", dest="final_psnr",
                   help="Final PSNR reported by trainer (dB)")
    p.add_argument("--splat-count", dest="splat_count",
                   help="Number of Gaussians in the trained scene")
    p.add_argument("--file-size-mb", dest="file_size_mb",
                   help="Output .ply file size in MB")
    p.add_argument("--subjective-score-1to10", dest="subjective_score_1to10",
                   help="Visual quality score, 1-10")
    p.add_argument("--wow-present-yes-no", dest="wow_present_yes_no",
                   help="Did the scene produce a 'wow' reaction? yes/no")
    p.add_argument("--artifacts-observed", dest="artifacts_observed",
                   help="Free-text description of visual artifacts (use quotes if contains commas)")
    p.add_argument("--notes",
                   help="Free-text run notes")

    args = p.parse_args()
    values: dict = {}
    for col in COLUMNS:
        val = getattr(args, col, None)
        if val is not None:
            values[col] = val

    append_row(values, args.csv)


if __name__ == "__main__":
    main()
