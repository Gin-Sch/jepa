#!/usr/bin/env python3
"""
Plot V-JEPA pretraining metrics from the per-iteration CSVLogger output.

Reads ``{write_tag}_r{rank}.csv`` columns written by ``app/vjepa/train.py``:
  epoch, itr, loss, loss-jepa, reg-loss, enc-grad-norm, pred-grad-norm, ...

Aggregates mean loss per epoch and saves a PNG (plus an optional epoch-summary CSV).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-csv",
        type=Path,
        required=True,
        help="Training metrics CSV from CSVLogger (e.g. jepa_r0.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path for the loss curves.",
    )
    parser.add_argument(
        "--epoch-csv",
        type=Path,
        default=None,
        help="Optional path to write per-epoch mean metrics CSV.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="V-JEPA pretrain metrics",
        help="Plot title.",
    )
    return parser


def load_epoch_summary(log_csv: Path) -> pd.DataFrame:
    """Load the training CSV and aggregate mean metrics per epoch."""
    df = pd.read_csv(log_csv)
    required = {"epoch", "loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{log_csv} is missing columns {sorted(missing)}; "
            f"found {list(df.columns)}"
        )

    agg = {"loss": "mean"}
    if "loss-jepa" in df.columns:
        agg["loss-jepa"] = "mean"
    if "reg-loss" in df.columns:
        agg["reg-loss"] = "mean"

    summary = df.groupby("epoch", as_index=False).agg(agg)
    summary = summary.sort_values("epoch").reset_index(drop=True)
    return summary


def plot_metrics(
    summary: pd.DataFrame,
    output: Path,
    title: str,
) -> None:
    """Save a PNG with total / JEPA / reg loss curves vs epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["epoch"], summary["loss"], label="loss", linewidth=2)
    if "loss-jepa" in summary.columns:
        ax.plot(
            summary["epoch"],
            summary["loss-jepa"],
            label="loss-jepa",
            linewidth=1.5,
            alpha=0.9,
        )
    if "reg-loss" in summary.columns and summary["reg-loss"].abs().max() > 0:
        ax.plot(
            summary["epoch"],
            summary["reg-loss"],
            label="reg-loss",
            linewidth=1.5,
            alpha=0.9,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.log_csv.is_file():
        raise FileNotFoundError(f"Training log CSV not found: {args.log_csv}")

    summary = load_epoch_summary(args.log_csv)
    if summary.empty:
        raise RuntimeError(f"No rows found in {args.log_csv}")

    plot_metrics(summary, args.output, args.title)
    print(f"Wrote {args.output} ({len(summary)} epochs)")

    epoch_csv = args.epoch_csv
    if epoch_csv is None:
        epoch_csv = args.output.with_suffix(".csv")
    epoch_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(epoch_csv, index=False)
    print(f"Wrote {epoch_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
