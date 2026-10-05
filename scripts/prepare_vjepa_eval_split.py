#!/usr/bin/env python3
"""
Build train/val V-JEPA index CSVs for the attentive probe.

Reads a space-delimited labeled index CSV (``\"path\" label`` or
``\"path\" label start_frame end_frame``) and writes stratified train/val
splits. Prefer subject-disjoint splits when a group id can be parsed from the
video basename (default: leading digits). Bout frame ranges are preserved.

Example::

  python scripts/prepare_vjepa_eval_split.py \\
    --index-csv trainCSV/human_cam2_bouts.csv \\
    --train-output trainCSV/human_cam2_eval_train.csv \\
    --val-output trainCSV/human_cam2_eval_val.csv
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Allow running as ``python scripts/...`` without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.utils.video.vjepa_index import (
    IndexRow,
    format_index_row,
    load_vjepa_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        required=True,
        help="Labeled V-JEPA index CSV used for pretrain / embedding extract.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        required=True,
        help="Destination train CSV path.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        required=True,
        help="Destination val CSV path.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Target fraction of groups held out per label stratum (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--group-regex",
        type=str,
        default=r"^(\d+)",
        help=(
            "Regex applied to the video basename; group(1) is the unit id used "
            "for a disjoint split. Empty string → split individual videos."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Drop rows whose video path does not exist on disk.",
    )
    parser.add_argument(
        "--num-classes-file",
        type=Path,
        default=None,
        help="Optional path to write a one-line integer (max_label + 1).",
    )
    return parser


def group_id_for_path(video_path: str, group_regex: str) -> str:
    """
    Return the split unit id for a video path.

    With an empty regex, each video is its own group. Otherwise the first
    capturing group of ``group_regex`` on the basename is used; if it does not
    match, the full basename stem is used.
    """
    stem = Path(video_path).name
    if not group_regex:
        return stem
    match = re.search(group_regex, stem)
    if match is None or match.lastindex is None or not match.group(1):
        return Path(stem).stem
    return str(match.group(1))


def majority_label(labels: Sequence[int]) -> int:
    """Majority label for a group; ties broken by smaller label id."""
    counts = Counter(int(x) for x in labels)
    max_count = max(counts.values())
    return min(lab for lab, n in counts.items() if n == max_count)


def split_groups_stratified(
    group_to_stratum: Dict[str, int],
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[Set[str], Set[str]]:
    """
    Group-disjoint stratified train/val assignment.

    Single-group strata stay in train. Larger strata keep ≥1 group in each split.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    by_stratum: Dict[int, List[str]] = defaultdict(list)
    for group, stratum in group_to_stratum.items():
        by_stratum[int(stratum)].append(str(group))

    rng = random.Random(seed)
    train: Set[str] = set()
    val: Set[str] = set()
    for stratum in sorted(by_stratum):
        groups = sorted(by_stratum[stratum])
        rng.shuffle(groups)
        if len(groups) == 1:
            train.add(groups[0])
            continue
        n_val = int(round(len(groups) * val_fraction))
        n_val = max(1, min(n_val, len(groups) - 1))
        val.update(groups[:n_val])
        train.update(groups[n_val:])
    return train, val


def write_eval_split(
    rows: Sequence[IndexRow],
    *,
    train_output: Path,
    val_output: Path,
    val_fraction: float,
    seed: int,
    group_regex: str,
) -> Tuple[int, int, int, int, int]:
    """
    Write train/val CSVs and return summary counts.

    Returns
    -------
    n_train, n_val, n_groups_train, n_groups_val, num_classes
    """
    group_labels: Dict[str, List[int]] = defaultdict(list)
    row_groups: List[str] = []
    for row in rows:
        gid = group_id_for_path(row.path, group_regex)
        group_labels[gid].append(row.label)
        row_groups.append(gid)

    group_to_stratum = {
        gid: majority_label(labs) for gid, labs in group_labels.items()
    }
    train_groups, val_groups = split_groups_stratified(
        group_to_stratum,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_lines: List[str] = []
    val_lines: List[str] = []
    for row, gid in zip(rows, row_groups):
        line = format_index_row(row)
        if gid in val_groups:
            val_lines.append(line)
        elif gid in train_groups:
            train_lines.append(line)
        else:
            raise RuntimeError(f"Group {gid!r} not assigned to train or val")

    if not train_lines or not val_lines:
        raise RuntimeError(
            f"Degenerate split: train={len(train_lines)} val={len(val_lines)}. "
            "Need ≥2 groups in at least one label stratum."
        )

    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_output.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    max_label = max(row.label for row in rows)
    num_classes = max_label + 1
    return (
        len(train_lines),
        len(val_lines),
        len(train_groups),
        len(val_groups),
        num_classes,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.index_csv.is_file():
        raise FileNotFoundError(f"Index CSV not found: {args.index_csv}")

    rows = load_vjepa_index(args.index_csv, skip_missing=args.skip_missing)
    n_train, n_val, n_g_train, n_g_val, num_classes = write_eval_split(
        rows,
        train_output=args.train_output,
        val_output=args.val_output,
        val_fraction=args.val_fraction,
        seed=args.seed,
        group_regex=args.group_regex,
    )

    label_counts = Counter(row.label for row in rows)
    n_ranged = sum(1 for row in rows if row.has_frame_range)
    print(f"Wrote {args.train_output} ({n_train} samples, {n_g_train} groups)")
    print(f"Wrote {args.val_output} ({n_val} samples, {n_g_val} groups)")
    print(
        f"num_classes={num_classes}  label_counts={dict(sorted(label_counts.items()))}  "
        f"with_frame_range={n_ranged}/{len(rows)}"
    )

    if args.num_classes_file is not None:
        args.num_classes_file.parent.mkdir(parents=True, exist_ok=True)
        args.num_classes_file.write_text(f"{num_classes}\n", encoding="utf-8")
        print(f"Wrote {args.num_classes_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
