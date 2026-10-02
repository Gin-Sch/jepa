#!/usr/bin/env python3
"""
Probe-free t-SNE of frozen V-JEPA encoder embeddings.

Loads the raw ``.npz`` written by ``export_embeddings_vjepa.py extract``
(``X``, optional ``row_index``) and colors points using labels from the
space-delimited index CSV (``video_path label``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-npz",
        type=Path,
        required=True,
        help="Raw embeddings npz from export_embeddings_vjepa.py extract.",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        required=True,
        help="Space-delimited video index CSV: video_path label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path for the t-SNE scatter plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="V-JEPA encoder t-SNE (mean-pooled tokens)",
        help="Plot title.",
    )
    parser.add_argument(
        "--tsne-pca-components",
        type=int,
        default=50,
        help="PCA dims before t-SNE (clamped to data size).",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional class names in label-id order (default: class_<id>).",
    )
    return parser


def load_index_labels(index_csv: Path) -> np.ndarray:
    """Return integer labels from a V-JEPA index CSV (column 1)."""
    df = pd.read_csv(index_csv, header=None, delimiter=" ", usecols=[0, 1])
    return df.iloc[:, 1].to_numpy(dtype=np.int64)


def resolve_labels(
    index_csv: Path,
    n_features: int,
    *,
    labels_from_npz: Optional[np.ndarray] = None,
    row_index: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Resolve per-embedding class labels for t-SNE coloring.

    Preference order:
    1. ``labels`` array stored in the raw npz (class ids from the loader)
    2. Unique ``row_index`` values used as positions into the index CSV
    3. ``row_index`` treated as class labels directly
    4. Index CSV column 1 in file order (requires matching length)
    """
    if labels_from_npz is not None:
        labels = np.asarray(labels_from_npz, dtype=np.int64)
        if labels.shape[0] != n_features:
            raise ValueError(
                f"labels length {labels.shape[0]} != X rows {n_features}"
            )
        return labels

    all_labels = load_index_labels(index_csv)
    if row_index is None:
        if len(all_labels) != n_features:
            raise ValueError(
                f"Index CSV has {len(all_labels)} labels but embeddings have "
                f"{n_features} rows and no labels/row_index were stored."
            )
        return all_labels

    row_index = np.asarray(row_index, dtype=np.int64)
    if row_index.shape[0] != n_features:
        raise ValueError(
            f"row_index length {row_index.shape[0]} != X rows {n_features}"
        )

    # Unique values that fit as CSV row positions → write-index style.
    if (
        row_index.size
        and row_index.size == np.unique(row_index).size
        and int(row_index.max()) < len(all_labels)
    ):
        return all_labels[row_index]

    # Otherwise the loader stored class labels in the label slot.
    return row_index


def class_names_for_labels(
    labels: np.ndarray,
    names: Optional[Sequence[str]],
) -> List[str]:
    max_lab = int(labels.max()) if labels.size else -1
    n = max(max_lab + 1, 0)
    if names is not None and len(names) >= n:
        return list(names)
    return [f"class_{i}" for i in range(n)]


def save_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
    title: str,
    *,
    pca_components: int,
    perplexity: float,
    seed: int,
) -> None:
    """PCA → t-SNE scatter colored by label."""
    if features.shape[0] < 2:
        raise RuntimeError("Need at least 2 embeddings to compute t-SNE.")

    n_pca = max(1, min(pca_components, features.shape[0] - 1, features.shape[1]))
    feats = features
    if feats.shape[1] > n_pca:
        feats = PCA(n_components=n_pca, random_state=seed).fit_transform(feats)

    n = feats.shape[0]
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    emb = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = sorted(int(u) for u in np.unique(labels))
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(uniq):
        mask = labels == lab
        name = class_names[lab] if 0 <= lab < len(class_names) else str(lab)
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=12,
            alpha=0.75,
            color=cmap(i % 10),
            label=name,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=1.5, fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.raw_npz.is_file():
        raise FileNotFoundError(f"Raw embeddings not found: {args.raw_npz}")
    if not args.index_csv.is_file():
        raise FileNotFoundError(f"Index CSV not found: {args.index_csv}")

    with np.load(args.raw_npz, allow_pickle=True) as payload:
        X = np.asarray(payload["X"], dtype=np.float32)
        labels_from_npz = (
            np.asarray(payload["labels"], dtype=np.int64)
            if "labels" in payload.files
            else None
        )
        row_index = (
            np.asarray(payload["row_index"], dtype=np.int64)
            if "row_index" in payload.files
            else None
        )

    labels = resolve_labels(
        args.index_csv,
        X.shape[0],
        labels_from_npz=labels_from_npz,
        row_index=row_index,
    )
    names = class_names_for_labels(labels, args.class_names)
    save_tsne(
        X,
        labels,
        names,
        args.output,
        args.title,
        pca_components=args.tsne_pca_components,
        perplexity=args.tsne_perplexity,
        seed=args.seed,
    )
    print(f"Wrote {args.output}: n={X.shape[0]}, dim={X.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
