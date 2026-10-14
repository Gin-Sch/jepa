# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
from pathlib import Path

import torch


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class CSVLogger(object):
    """CSV metrics logger with a single header row.

    Parameters
    ----------
    fname :
        Output CSV path.
    *argv :
        ``(printf_fmt, column_name)`` pairs, e.g. ``('%d', 'epoch')``.
    append :
        If True and ``fname`` already has content, keep existing rows and
        do not rewrite the header (checkpoint resume). If False, truncate
        and write a fresh header (new run). Defaults to False so repeated
        jobs sharing an output folder cannot stack duplicate headers.
    """

    def __init__(self, fname, *argv, append=False):
        self.fname = fname
        self.types = [v[0] for v in argv]
        path = Path(fname)
        write_header = True
        if append and path.is_file() and path.stat().st_size > 0:
            mode = "a"
            write_header = False
        else:
            mode = "w"
        with open(self.fname, mode) as f:
            if write_header:
                print(",".join(v[1] for v in argv), file=f)

    def log(self, *argv):
        with open(self.fname, "a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = "," if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats


def adamw_logger(optimizer):
    """ logging magnitude of first and second momentum buffers in adamw """
    # TODO: assert that optimizer is instance of torch.optim.AdamW
    state = optimizer.state_dict().get('state')
    exp_avg_stats = AverageMeter()
    exp_avg_sq_stats = AverageMeter()
    for key in state:
        s = state.get(key)
        exp_avg_stats.update(float(s.get('exp_avg').abs().mean()))
        exp_avg_sq_stats.update(float(s.get('exp_avg_sq').abs().mean()))
    return {'exp_avg': exp_avg_stats, 'exp_avg_sq': exp_avg_sq_stats}
