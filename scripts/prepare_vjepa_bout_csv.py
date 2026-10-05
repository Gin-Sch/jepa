#!/usr/bin/env python3
"""
Build a bout-scoped V-JEPA index CSV for pretraining / probing.

Expands a full-video index (``path label``) into one row per walking bout::

    "/abs/path/video.mp4" 0 415 695
    "/abs/path/video.mp4" 0 840 1150

using ``walking_bouts_frames.csv`` annotations
(``video_id,start_frame,end_frame[,bout_id]``).

Example::

  conda activate myenv
  python scripts/prepare_vjepa_bout_csv.py \\
    --index-csv trainCSV/human_cam2.csv \\
    --bout-csv /path/to/walking_bouts_frames.csv \\
    --output trainCSV/human_cam2_bouts.csv \\
    --min-bout-frames 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Allow running as ``python scripts/...`` without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.utils.video.bout_index import expand_index_with_bouts
from src.datasets.utils.video.vjepa_index import load_vjepa_index, write_vjepa_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        required=True,
        help="Base V-JEPA index CSV (path label per video).",
    )
    parser.add_argument(
        "--bout-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more walking_bouts_frames.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination bout-expanded index CSV.",
    )
    parser.add_argument(
        "--min-bout-frames",
        type=int,
        default=64,
        help=(
            "Drop bouts shorter than this many frames (inclusive length). "
            "Default 64 = num_frames(16) * sampling_rate(4)."
        ),
    )
    parser.add_argument(
        "--strip-camera-suffix",
        action="store_true",
        help=(
            "Match video stems after stripping a trailing -<digits> camera "
            "id (e.g. ...-2 ↔ ...-6). Off by default."
        ),
    )
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="Fail if any index video has no matching bout rows.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Drop base-index rows whose video path is missing on disk.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.index_csv.is_file():
        raise FileNotFoundError(f"Index CSV not found: {args.index_csv}")

    video_rows = load_vjepa_index(
        args.index_csv, skip_missing=args.skip_missing
    )
    expanded, stats = expand_index_with_bouts(
        video_rows,
        args.bout_csv,
        min_bout_frames=args.min_bout_frames,
        strip_camera_suffix=args.strip_camera_suffix,
        require_match=args.require_match,
    )
    out = write_vjepa_index(expanded, args.output)

    print(f"Wrote {out} ({stats.n_bouts_kept} bout samples)")
    print(
        "videos: "
        f"in={stats.n_videos_in} matched={stats.n_videos_matched} "
        f"unmatched={stats.n_videos_unmatched}"
    )
    print(
        "bouts: "
        f"in={stats.n_bouts_in} kept={stats.n_bouts_kept} "
        f"dropped_short={stats.n_bouts_dropped_short} "
        f"orphaned={stats.n_bouts_orphaned}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
