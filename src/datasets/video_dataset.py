# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import warnings

from logging import getLogger

import numpy as np

from decord import VideoReader, cpu

import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from src.datasets.utils.video.clip_sampling import sample_clip_indices
from src.datasets.utils.video.vjepa_index import load_vjepa_index

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform)

    logger.info('VideoDataset dataset created')
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('VideoDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """ Video classification dataset. """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths, labels, and optional inclusive bout frame ranges.
        samples, labels, frame_ranges = [], [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

            if data_path[-4:] == '.csv':
                rows = load_vjepa_index(data_path)
                samples += [row.path for row in rows]
                labels += [row.label for row in rows]
                frame_ranges += [row.frame_range for row in rows]
                self.num_samples_per_dataset.append(len(rows))

            elif data_path[-4:] == '.npy':
                data = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], data))
                samples += data
                labels += [0] * len(data)
                frame_ranges += [None] * len(data)
                self.num_samples_per_dataset.append(len(data))

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels
        self.frame_ranges = frame_ranges

        n_ranged = sum(1 for fr in frame_ranges if fr is not None)
        if n_ranged:
            logger.info(
                'VideoDataset: %d/%d samples have bout frame ranges',
                n_ranged,
                len(frame_ranges),
            )

    def __getitem__(self, index):
        sample = self.samples[index]
        frame_range = self.frame_ranges[index]

        # Keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            buffer, clip_indices = self.loadvideo_decord(sample, frame_range=frame_range)
            loaded_video = len(buffer) > 0
            if not loaded_video:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]
                frame_range = self.frame_ranges[index]

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample, frame_range=None):
        """Load video content using Decord.

        Parameters
        ----------
        sample:
            Absolute path to the video file.
        frame_range:
            Optional inclusive ``(start_frame, end_frame)`` bout window. When
            set, clip sampling is restricted to that span (indices are still
            absolute in the source video).
        """

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fps = vr.get_avg_fps()
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)

        video_len = len(vr)
        frame_offset = 0
        effective_len = video_len

        if frame_range is not None:
            bout_start, bout_end = int(frame_range[0]), int(frame_range[1])
            bout_start = max(0, bout_start)
            bout_end = min(video_len - 1, bout_end)
            if bout_end < bout_start:
                warnings.warn(
                    f'invalid bout range [{frame_range[0]}, {frame_range[1]}] '
                    f'for video length {video_len}: {fname}'
                )
                return [], None
            frame_offset = bout_start
            effective_len = bout_end - bout_start + 1

        if self.filter_short_videos and effective_len < clip_len:
            warnings.warn(
                f'skipping span of length {effective_len} '
                f'(need >={clip_len}) in {fname}'
            )
            return [], None

        if effective_len <= 0:
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition the (possibly bout-restricted) span and sample clips.
        all_indices, clip_indices = sample_clip_indices(
            effective_len=effective_len,
            frames_per_clip=fpc,
            frame_step=fstp,
            num_clips=self.num_clips,
            random_clip_sampling=self.random_clip_sampling,
            allow_clip_overlap=self.allow_clip_overlap,
            clip_len=clip_len,
        )

        if frame_offset:
            all_indices = [int(i) + frame_offset for i in all_indices]
            clip_indices = [indices + frame_offset for indices in clip_indices]

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
