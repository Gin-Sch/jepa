#!/usr/bin/env python3
"""Unit tests for bout-scoped V-JEPA index helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.utils.video.bout_index import expand_index_with_bouts, stem_key
from src.datasets.utils.video.clip_sampling import sample_clip_indices
from src.datasets.utils.video.vjepa_index import (
    IndexRow,
    format_index_row,
    load_vjepa_index,
    parse_vjepa_csv_line,
)


def test_parse_two_and_four_field_lines():
    row2 = parse_vjepa_csv_line('/data/a.mp4 3')
    assert row2 == IndexRow(path='/data/a.mp4', label=3)

    row4 = parse_vjepa_csv_line('"/data/a b.mp4" 1 10 50')
    assert row4.path == '/data/a b.mp4'
    assert row4.label == 1
    assert row4.frame_range == (10, 50)
    assert row4.bout_length() == 41


def test_format_roundtrip(tmp_path: Path):
    rows = [
        IndexRow('/x.mp4', 0),
        IndexRow('/y.mp4', 2, start_frame=5, end_frame=20),
    ]
    out = tmp_path / 'idx.csv'
    out.write_text('\n'.join(format_index_row(r) for r in rows) + '\n')
    loaded = load_vjepa_index(out)
    assert loaded == rows


def test_expand_index_with_bouts(tmp_path: Path):
    index_csv = tmp_path / 'videos.csv'
    index_csv.write_text(
        '"/videos/11117_2_260223-2.avi" 0\n'
        '"/videos/missing_video.avi" 1\n'
    )
    bout_csv = tmp_path / 'bouts.csv'
    pd.DataFrame(
        [
            {
                'person_id': 11117,
                'video_id': '11117_2_260223-2',
                'bout_id': 0,
                'start_frame': 100,
                'end_frame': 200,
            },
            {
                'person_id': 11117,
                'video_id': '11117_2_260223-2',
                'bout_id': 1,
                'start_frame': 300,
                'end_frame': 330,  # length 31 → dropped when min=64
            },
            {
                'person_id': 11117,
                'video_id': '11117_2_260223-2',
                'bout_id': 2,
                'start_frame': 400,
                'end_frame': 500,
            },
        ]
    ).to_csv(bout_csv, index=False)

    video_rows = load_vjepa_index(index_csv)
    expanded, stats = expand_index_with_bouts(
        video_rows,
        [bout_csv],
        min_bout_frames=64,
    )
    assert len(expanded) == 2
    assert expanded[0].start_frame == 100
    assert expanded[1].end_frame == 500
    assert stats.n_videos_matched == 1
    assert stats.n_videos_unmatched == 1
    assert stats.n_bouts_dropped_short == 1
    assert stats.n_bouts_orphaned == 0


def test_stem_key_strip_camera():
    assert stem_key('11450_2_260129-2', strip_camera_suffix=True) == '11450_2_260129'
    assert stem_key('/a/11450_2_260129-6.avi', strip_camera_suffix=True) == '11450_2_260129'


def test_sample_clip_indices_respects_effective_len():
    all_idx, clips = sample_clip_indices(
        effective_len=128,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        random_clip_sampling=False,
        allow_clip_overlap=False,
        clip_len=64,
    )
    assert len(all_idx) == 16
    assert max(all_idx) < 128
    assert len(clips) == 1
