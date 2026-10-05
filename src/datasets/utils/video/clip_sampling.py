# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Frame-index sampling helpers for V-JEPA video clips."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def sample_clip_indices(
    *,
    effective_len: int,
    frames_per_clip: int,
    frame_step: int,
    num_clips: int,
    random_clip_sampling: bool,
    allow_clip_overlap: bool,
    clip_len: int,
) -> Tuple[List[int], List[np.ndarray]]:
    """Sample relative frame indices inside a span of length ``effective_len``.

    Indices are in ``[0, effective_len)``. Callers add a bout offset when needed.
    """
    partition_len = effective_len // num_clips

    all_indices: List[int] = []
    clip_indices: List[np.ndarray] = []
    for i in range(num_clips):
        if partition_len > clip_len:
            # Sample a random window of clip_len frames within the segment.
            end_indx = clip_len
            if random_clip_sampling:
                end_indx = np.random.randint(clip_len, partition_len)
            start_indx = end_indx - clip_len
            indices = np.linspace(start_indx, end_indx, num=frames_per_clip)
            indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
            indices = indices + i * partition_len
        else:
            if not allow_clip_overlap:
                # Pad by repeating the last frame in the segment.
                indices = np.linspace(0, partition_len, num=partition_len // frame_step)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(frames_per_clip - partition_len // frame_step)
                        * partition_len,
                    )
                )
                indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                # Allow adjacent clips to overlap when the span is short.
                sample_len = min(clip_len, effective_len) - 1
                indices = np.linspace(0, sample_len, num=sample_len // frame_step)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(frames_per_clip - sample_len // frame_step) * sample_len,
                    )
                )
                indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                clip_step = 0
                if effective_len > clip_len:
                    clip_step = (effective_len - clip_len) // (num_clips - 1)
                indices = indices + i * clip_step

        clip_indices.append(indices)
        all_indices.extend(list(indices))

    return all_indices, clip_indices
