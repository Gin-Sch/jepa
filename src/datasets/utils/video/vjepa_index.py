# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""V-JEPA video index CSV parsing and formatting.

Supported line formats (space-delimited; paths may be double-quoted)::

    /abs/path/video.mp4 0
    "/abs/path/with spaces.mp4" 0
    /abs/path/video.mp4 0 415 695

The optional trailing integers are inclusive ``start_frame`` / ``end_frame``
bounds used to restrict clip sampling to a walking bout (or any sub-span).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class IndexRow:
    """One training / eval sample from a V-JEPA index CSV."""

    path: str
    label: int
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    @property
    def has_frame_range(self) -> bool:
        """True when both bout endpoints are set."""
        return self.start_frame is not None and self.end_frame is not None

    @property
    def frame_range(self) -> Optional[Tuple[int, int]]:
        """Inclusive ``(start_frame, end_frame)`` or ``None``."""
        if not self.has_frame_range:
            return None
        return int(self.start_frame), int(self.end_frame)

    def bout_length(self) -> Optional[int]:
        """Number of frames in the bout (inclusive), or ``None`` if unbound."""
        if not self.has_frame_range:
            return None
        return int(self.end_frame) - int(self.start_frame) + 1


def parse_vjepa_csv_line(line: str) -> IndexRow:
    """Parse one index line into an :class:`IndexRow`.

    Parameters
    ----------
    line:
        Raw CSV line. Empty / whitespace-only lines raise ``ValueError``.

    Returns
    -------
    IndexRow
        Parsed sample. ``start_frame`` / ``end_frame`` are set only when both
        trailing integers are present.
    """
    stripped = line.strip()
    if not stripped:
        raise ValueError("Cannot parse an empty V-JEPA CSV line.")

    try:
        parts = shlex.split(stripped, posix=True)
    except ValueError as exc:
        raise ValueError(f"Malformed V-JEPA CSV line: {line!r}") from exc

    if len(parts) not in (2, 4):
        raise ValueError(
            "Expected 2 fields (path label) or 4 fields "
            f"(path label start_frame end_frame), got {len(parts)}: {line!r}"
        )

    path = parts[0]
    try:
        label = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Label must be an integer, got {parts[1]!r}") from exc

    if len(parts) == 2:
        return IndexRow(path=path, label=label)

    try:
        start_frame = int(parts[2])
        end_frame = int(parts[3])
    except ValueError as exc:
        raise ValueError(
            f"start_frame/end_frame must be integers, got {parts[2:]!r}"
        ) from exc

    if start_frame < 0 or end_frame < start_frame:
        raise ValueError(
            f"Invalid frame range [{start_frame}, {end_frame}] (inclusive) "
            f"in line: {line!r}"
        )
    return IndexRow(
        path=path,
        label=label,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def format_vjepa_csv_line(
    video_path: str,
    label: int,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> str:
    """Format one quoted index line.

    When both ``start_frame`` and ``end_frame`` are given, they are appended.
    """
    quoted = '"' + str(video_path).replace('"', '""') + '"'
    if start_frame is None and end_frame is None:
        return f"{quoted} {int(label)}"
    if start_frame is None or end_frame is None:
        raise ValueError("Provide both start_frame and end_frame, or neither.")
    return f"{quoted} {int(label)} {int(start_frame)} {int(end_frame)}"


def format_index_row(row: IndexRow) -> str:
    """Format an :class:`IndexRow` as a CSV line."""
    return format_vjepa_csv_line(
        row.path,
        row.label,
        start_frame=row.start_frame,
        end_frame=row.end_frame,
    )


def load_vjepa_index(
    index_csv: Union[str, Path],
    *,
    skip_missing: bool = False,
) -> List[IndexRow]:
    """Load all non-empty rows from a V-JEPA index CSV."""
    path = Path(index_csv)
    rows: List[IndexRow] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = parse_vjepa_csv_line(raw)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        if skip_missing and not Path(row.path).is_file():
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    return rows


def write_vjepa_index(
    rows: Sequence[IndexRow],
    output_csv: Union[str, Path],
) -> Path:
    """Write index rows to ``output_csv`` (one line per row)."""
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [format_index_row(row) for row in rows]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
