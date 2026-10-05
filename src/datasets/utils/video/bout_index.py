# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Build bout-scoped V-JEPA index rows from walking-bout annotations.

Each full video in the base index is expanded into one row per walking bout::

    /path/video.mp4 0 415 695
    /path/video.mp4 0 840 1150
    ...

so :class:`~src.datasets.video_dataset.VideoDataset` can sample clips only
inside those inclusive frame ranges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.datasets.utils.video.vjepa_index import IndexRow


_CAMERA_SUFFIX_RE = re.compile(r"-\d+$")


@dataclass(frozen=True)
class BoutExpandStats:
    """Summary counters from :func:`expand_index_with_bouts`."""

    n_videos_in: int
    n_videos_matched: int
    n_videos_unmatched: int
    n_bouts_in: int
    n_bouts_kept: int
    n_bouts_dropped_short: int
    n_bouts_orphaned: int


def stem_key(path_or_id: Union[str, Path], *, strip_camera_suffix: bool = False) -> str:
    """Normalize a path or video_id to a lookup key (basename stem)."""
    stem = Path(path_or_id).stem
    if strip_camera_suffix:
        stem = _CAMERA_SUFFIX_RE.sub("", stem)
    return stem


def load_bout_table(
    bout_csv_paths: Sequence[Union[str, Path]],
) -> pd.DataFrame:
    """Load and concatenate walking-bout CSVs.

    Required columns: ``video_id``, ``start_frame``, ``end_frame``.
    Optional: ``bout_id`` (used only for stable sorting).
    """
    frames: List[pd.DataFrame] = []
    required = {"video_id", "start_frame", "end_frame"}
    for raw in bout_csv_paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"Bout CSV not found: {path}")
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        cols = ["video_id", "start_frame", "end_frame"]
        if "bout_id" in df.columns:
            cols = ["video_id", "bout_id", "start_frame", "end_frame"]
        frames.append(df[cols].copy())

    if not frames:
        raise ValueError("No bout CSV paths provided.")

    combined = pd.concat(frames, ignore_index=True)
    combined["video_id"] = combined["video_id"].astype(str)
    combined["start_frame"] = combined["start_frame"].astype(int)
    combined["end_frame"] = combined["end_frame"].astype(int)
    sort_cols = ["video_id"]
    if "bout_id" in combined.columns:
        sort_cols.append("bout_id")
    sort_cols.extend(["start_frame", "end_frame"])
    return combined.sort_values(sort_cols).reset_index(drop=True)


def _index_videos_by_stem(
    video_rows: Sequence[IndexRow],
    *,
    strip_camera_suffix: bool,
) -> Dict[str, IndexRow]:
    """Map stem key → index row. Duplicate keys keep the first occurrence."""
    mapping: Dict[str, IndexRow] = {}
    for row in video_rows:
        key = stem_key(row.path, strip_camera_suffix=strip_camera_suffix)
        mapping.setdefault(key, row)
    return mapping


def expand_index_with_bouts(
    video_rows: Sequence[IndexRow],
    bout_csv_paths: Sequence[Union[str, Path]],
    *,
    min_bout_frames: Optional[int] = None,
    strip_camera_suffix: bool = False,
    require_match: bool = False,
) -> Tuple[List[IndexRow], BoutExpandStats]:
    """Expand full-video index rows into one sample per walking bout.

    Parameters
    ----------
    video_rows:
        Base V-JEPA index (typically 2-column ``path label`` rows). Existing
        frame ranges on input rows are ignored; bout CSVs define the spans.
    bout_csv_paths:
        One or more ``walking_bouts_frames.csv``-style files.
    min_bout_frames:
        Drop bouts whose inclusive length is ``< min_bout_frames``. Use
        ``num_frames * sampling_rate`` (e.g. 16 * 4 = 64) so every bout can
        hold at least one V-JEPA clip. ``None`` keeps all bouts.
    strip_camera_suffix:
        If True, match ``11450_2_260129-2`` to ``11450_2_260129-6.avi`` by
        stripping a trailing ``-<digits>`` camera suffix. Prefer False when
        the index already lists the correct camera.
    require_match:
        If True, raise when any index video has no bout rows (or vice versa
        is only reported in stats).

    Returns
    -------
    rows, stats
        Expanded :class:`IndexRow` list (labels copied from the matched video)
        and aggregate counters.
    """
    bout_df = load_bout_table(bout_csv_paths)
    video_map = _index_videos_by_stem(
        video_rows, strip_camera_suffix=strip_camera_suffix
    )

    expanded: List[IndexRow] = []
    matched_video_keys = set()
    n_bouts_dropped_short = 0
    n_bouts_orphaned = 0

    for record in bout_df.itertuples(index=False):
        key = stem_key(record.video_id, strip_camera_suffix=strip_camera_suffix)
        video = video_map.get(key)
        if video is None:
            n_bouts_orphaned += 1
            continue

        start = int(record.start_frame)
        end = int(record.end_frame)
        if end < start:
            n_bouts_dropped_short += 1
            continue
        bout_len = end - start + 1
        if min_bout_frames is not None and bout_len < int(min_bout_frames):
            n_bouts_dropped_short += 1
            continue

        matched_video_keys.add(key)
        expanded.append(
            IndexRow(
                path=video.path,
                label=int(video.label),
                start_frame=start,
                end_frame=end,
            )
        )

    unmatched = set(video_map) - matched_video_keys
    if require_match and unmatched:
        examples = ", ".join(sorted(unmatched)[:8])
        raise ValueError(
            f"{len(unmatched)} index video(s) have no matching bouts "
            f"(examples: {examples})"
        )
    if not expanded:
        raise ValueError(
            "No bout-expanded rows produced. Check that video stems match "
            "bout video_id values (see strip_camera_suffix)."
        )

    stats = BoutExpandStats(
        n_videos_in=len(video_map),
        n_videos_matched=len(matched_video_keys),
        n_videos_unmatched=len(unmatched),
        n_bouts_in=len(bout_df),
        n_bouts_kept=len(expanded),
        n_bouts_dropped_short=n_bouts_dropped_short,
        n_bouts_orphaned=n_bouts_orphaned,
    )
    return expanded, stats


def iter_unmatched_video_stems(
    video_rows: Sequence[IndexRow],
    bout_csv_paths: Sequence[Union[str, Path]],
    *,
    strip_camera_suffix: bool = False,
) -> Iterable[str]:
    """Yield stem keys present in the video index but missing from bouts."""
    bout_df = load_bout_table(bout_csv_paths)
    bout_keys = {
        stem_key(vid, strip_camera_suffix=strip_camera_suffix)
        for vid in bout_df["video_id"].astype(str)
    }
    video_map = _index_videos_by_stem(
        video_rows, strip_camera_suffix=strip_camera_suffix
    )
    for key in sorted(set(video_map) - bout_keys):
        yield key
