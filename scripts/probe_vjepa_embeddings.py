#!/usr/bin/env python3
"""
Linear probe on frozen V-JEPA embeddings (``embeddings_raw.npz``).

Uses the same recipe as MouseBeamWalk's flat latent probes:

  StandardScaler → PCA → LogisticRegression(class_weight='balanced')

plus a Ridge regression probe when labels are continuous (e.g. age).

No videos / attentive probe checkpoint required — only the raw npz written by
``scripts/export_embeddings_vjepa.py extract``.

Example::

  conda activate myenv
  python scripts/probe_vjepa_embeddings.py \\
    --raw-npz outputs/vjepa_pretrain_human_cam2/embeddings_raw.npz \\
    --output-dir outputs/vjepa_pretrain_human_cam2/probe
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        required=True,
        help="Raw embeddings npz with arrays X and labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for metrics CSV/JSON and plots.",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=None,
        help=(
            "Optional V-JEPA index CSV (path label). Used only to derive "
            "subject groups from the basename (leading digits by default)."
        ),
    )
    parser.add_argument(
        "--group-regex",
        type=str,
        default=r"^(\d+)",
        help="Regex on the video basename; group(1) is the CV group id.",
    )
    parser.add_argument("--n-pca", type=int, default=50, help="PCA dims (clamped).")
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds.")
    parser.add_argument("--n-repeats", type=int, default=3, help="CV repeats.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--binary-median",
        action="store_true",
        default=True,
        help="Also run a below/above-median balanced classification probe.",
    )
    parser.add_argument(
        "--no-binary-median",
        action="store_false",
        dest="binary_median",
        help="Skip the median-split classification probe.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional names for multiclass labels (length >= max label + 1).",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        default=True,
        help="Write a t-SNE scatter colored by age with a continuous heatmap.",
    )
    parser.add_argument(
        "--no-tsne",
        action="store_false",
        dest="tsne",
        help="Skip the age-heatmap t-SNE plot.",
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
    parser.add_argument(
        "--tsne-cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for the continuous age heatmap (default: viridis).",
    )
    return parser


def load_embeddings(raw_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``X`` and ``labels`` from the export npz."""
    with np.load(raw_npz, allow_pickle=True) as payload:
        if "X" not in payload.files:
            raise KeyError(f"{raw_npz} has no 'X' array; found {payload.files}")
        X = np.asarray(payload["X"], dtype=np.float32)
        if "labels" in payload.files:
            y = np.asarray(payload["labels"], dtype=np.float64)
        elif "row_index" in payload.files:
            y = np.asarray(payload["row_index"], dtype=np.float64)
        else:
            raise KeyError(
                f"{raw_npz} has neither 'labels' nor 'row_index'; "
                f"found {payload.files}"
            )
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != labels {y.shape[0]}")
    return X, y


def parse_vjepa_csv_line(line: str) -> Tuple[str, int]:
    stripped = line.strip()
    path_part, _, label_part = stripped.rpartition(" ")
    if not path_part:
        raise ValueError(f"Malformed V-JEPA CSV line: {line!r}")
    path_part = path_part.strip()
    if path_part.startswith('"') and path_part.endswith('"'):
        path_part = path_part[1:-1].replace('""', '"')
    return path_part, int(label_part)


def groups_from_index_csv(
    index_csv: Path,
    labels: np.ndarray,
    group_regex: str,
) -> Optional[np.ndarray]:
    """
    Derive per-row group ids from an index CSV aligned by labels.

    Prefer exact length match (file order == embedding order). If that fails,
    fall back to None so the caller uses video-level CV.
    """
    rows: List[Tuple[str, int]] = []
    for raw in index_csv.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rows.append(parse_vjepa_csv_line(raw))
    if len(rows) != len(labels):
        print(
            f"Warning: index CSV has {len(rows)} rows but embeddings have "
            f"{len(labels)}; ignoring CSV for grouping."
        )
        return None

    groups: List[str] = []
    for path, _lab in rows:
        stem = Path(path).name
        if group_regex:
            match = re.search(group_regex, stem)
            if match is not None and match.lastindex and match.group(1):
                groups.append(str(match.group(1)))
                continue
        groups.append(Path(stem).stem)
    return np.asarray(groups)


def effective_pca(n_pca: int, n_samples: int, n_features: int) -> int:
    """Clamp PCA components to a valid rank for the train fold size."""
    return max(1, min(int(n_pca), n_samples - 1, n_features))


def ridge_pipeline(n_pca: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca, random_state=seed)),
            ("clf", Ridge(alpha=1.0, random_state=seed)),
        ]
    )


def logistic_pipeline(n_pca: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca, random_state=seed)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def repeated_group_indices(
    n: int,
    y_for_stratify: np.ndarray,
    groups: Optional[np.ndarray],
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build repeated CV splits, preferring group-aware fold generators."""
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    rng = np.random.RandomState(seed)

    for repeat in range(n_repeats):
        fold_seed = int(rng.randint(0, 10_000_000))
        if groups is not None:
            n_groups = len(np.unique(groups))
            n_fold = max(2, min(n_splits, n_groups))
            # StratifiedGroupKFold needs ≥ n_splits samples per class in groups;
            # fall back to GroupKFold when stratification is impossible.
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=n_fold,
                    shuffle=True,
                    random_state=fold_seed,
                )
                fold_iter = splitter.split(np.zeros(n), y_for_stratify, groups)
            except ValueError:
                splitter = GroupKFold(n_splits=n_fold)
                fold_iter = splitter.split(np.zeros(n), y_for_stratify, groups)
        else:
            n_fold = max(2, min(n_splits, n))
            classes, counts = np.unique(y_for_stratify, return_counts=True)
            if counts.min() >= n_fold:
                splitter = StratifiedKFold(
                    n_splits=n_fold,
                    shuffle=True,
                    random_state=fold_seed,
                )
                fold_iter = splitter.split(np.zeros(n), y_for_stratify)
            else:
                splitter = KFold(
                    n_splits=n_fold,
                    shuffle=True,
                    random_state=fold_seed,
                )
                fold_iter = splitter.split(np.zeros(n))
        for train_idx, test_idx in fold_iter:
            splits.append((train_idx, test_idx))
    return splits


def run_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    *,
    n_pca: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Dict[str, object]:
    """Cross-validated Ridge probe for continuous labels (e.g. age)."""
    n = X.shape[0]
    # Stratify bins only for fold construction when groups are absent.
    y_bins = pd.qcut(y, q=min(5, max(2, len(np.unique(y)) // 2)), duplicates="drop")
    y_strat = np.asarray(y_bins.codes)
    splits = repeated_group_indices(
        n,
        y_strat,
        groups,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
    )

    y_pred = np.full(n, np.nan, dtype=np.float64)
    fold_rows: List[Dict[str, float]] = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        n_pca_eff = effective_pca(n_pca, len(train_idx), X.shape[1])
        pipe = ridge_pipeline(n_pca_eff, seed + fold_i)
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        # Average predictions across repeats when a sample appears multiple times.
        for idx, value in zip(test_idx, pred):
            if np.isnan(y_pred[idx]):
                y_pred[idx] = float(value)
            else:
                y_pred[idx] = 0.5 * (float(y_pred[idx]) + float(value))
        fold_rows.append(
            {
                "fold": fold_i,
                "mae": float(mean_absolute_error(y[test_idx], pred)),
                "r2": float(r2_score(y[test_idx], pred)),
                "n_test": float(len(test_idx)),
                "n_pca": float(n_pca_eff),
            }
        )

    mask = ~np.isnan(y_pred)
    y_true = y[mask]
    y_hat = y_pred[mask]
    pearson = float(np.corrcoef(y_true, y_hat)[0, 1]) if y_true.size > 1 else float("nan")
    return {
        "task": "regression",
        "n": int(n),
        "n_scored": int(mask.sum()),
        "mae": float(mean_absolute_error(y_true, y_hat)),
        "rmse": float(np.sqrt(np.mean((y_true - y_hat) ** 2))),
        "r2": float(r2_score(y_true, y_hat)),
        "pearson_r": pearson,
        "chance_mae": float(np.mean(np.abs(y_true - np.mean(y_true)))),
        "y_true": y_true,
        "y_pred": y_hat,
        "folds": fold_rows,
    }


def run_classification_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    *,
    n_pca: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Cross-validated balanced logistic probe (MouseBeamWalk-style)."""
    y_int = y.astype(np.int64)
    classes = np.unique(y_int)
    n = X.shape[0]
    splits = repeated_group_indices(
        n,
        y_int,
        groups,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
    )

    # Collect soft votes across repeats.
    vote = np.zeros((n, len(classes)), dtype=np.float64)
    vote_count = np.zeros(n, dtype=np.int64)
    fold_rows: List[Dict[str, float]] = []
    class_to_col = {int(c): i for i, c in enumerate(classes)}

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_classes = np.unique(y_int[train_idx])
        if train_classes.size < 2:
            continue
        n_pca_eff = effective_pca(n_pca, len(train_idx), X.shape[1])
        pipe = logistic_pipeline(n_pca_eff, seed + fold_i)
        pipe.fit(X[train_idx], y_int[train_idx])
        proba = pipe.predict_proba(X[test_idx])
        pred = pipe.classes_[np.argmax(proba, axis=1)]
        for local_i, idx in enumerate(test_idx):
            vote_count[idx] += 1
            for cls_i, cls in enumerate(pipe.classes_):
                vote[idx, class_to_col[int(cls)]] += float(proba[local_i, cls_i])
        fold_rows.append(
            {
                "fold": fold_i,
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_int[test_idx], pred)
                ),
                "n_test": float(len(test_idx)),
                "n_pca": float(n_pca_eff),
            }
        )

    scored = vote_count > 0
    y_true = y_int[scored]
    y_pred = classes[np.argmax(vote[scored], axis=1)]
    names = (
        [class_names[int(c)] for c in classes]
        if class_names is not None and len(class_names) > int(classes.max())
        else [str(int(c)) for c in classes]
    )
    return {
        "task": "classification",
        "n": int(n),
        "n_scored": int(scored.sum()),
        "n_classes": int(classes.size),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "chance_balanced_accuracy": float(1.0 / classes.size),
        "classes": [int(c) for c in classes],
        "class_names": names,
        "y_true": y_true,
        "y_pred": y_pred,
        "folds": fold_rows,
    }


def save_regression_plot(result: Dict[str, object], out_path: Path, title: str) -> None:
    y_true = np.asarray(result["y_true"])
    y_pred = np.asarray(result["y_pred"])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=18, alpha=0.75)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    ax.set_title(
        f"{title}\nMAE={result['mae']:.2f}  R²={result['r2']:.3f}  "
        f"r={result['pearson_r']:.3f}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_tsne_age_heatmap(
    features: np.ndarray,
    ages: np.ndarray,
    out_path: Path,
    title: str,
    *,
    pca_components: int,
    perplexity: float,
    seed: int,
    cmap: str = "viridis",
) -> None:
    """PCA → t-SNE scatter with a continuous age colorbar."""
    if features.shape[0] < 2:
        raise RuntimeError("Need at least 2 embeddings to compute t-SNE.")

    n_pca = max(1, min(pca_components, features.shape[0] - 1, features.shape[1]))
    feats = features
    if feats.shape[1] > n_pca:
        feats = PCA(n_components=n_pca, random_state=seed).fit_transform(feats)

    n = feats.shape[0]
    perp = min(float(perplexity), max(5.0, (n - 1) / 3.0))
    emb = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=ages,
        cmap=cmap,
        s=28,
        alpha=0.9,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Age")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_plot(result: Dict[str, object], out_path: Path, title: str) -> None:
    names = list(result["class_names"])
    cm = confusion_matrix(
        result["y_true"],
        result["y_pred"],
        labels=result["classes"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=names,
        yticklabels=names,
        ylabel="True",
        xlabel="Predicted",
        title=(
            f"{title}\nbalanced acc="
            f"{result['balanced_accuracy']:.3f} "
            f"(chance={result['chance_balanced_accuracy']:.3f})"
        ),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.raw_npz.is_file():
        raise FileNotFoundError(f"Raw embeddings not found: {args.raw_npz}")

    X, y = load_embeddings(args.raw_npz)
    groups = None
    if args.index_csv is not None:
        if not args.index_csv.is_file():
            raise FileNotFoundError(f"Index CSV not found: {args.index_csv}")
        groups = groups_from_index_csv(args.index_csv, y, args.group_regex)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    details: Dict[str, object] = {
        "raw_npz": str(args.raw_npz),
        "n_videos": int(X.shape[0]),
        "embed_dim": int(X.shape[1]),
        "n_unique_labels": int(np.unique(y).size),
        "label_min": float(np.min(y)),
        "label_max": float(np.max(y)),
        "grouped_cv": bool(groups is not None),
    }

    print(
        f"Loaded {X.shape[0]} x {X.shape[1]} embeddings, "
        f"{np.unique(y).size} unique labels "
        f"(range {y.min():.0f}–{y.max():.0f})"
    )

    if args.tsne:
        tsne_path = out_dir / "embeddings_tsne_age_heatmap.png"
        save_tsne_age_heatmap(
            X,
            y,
            tsne_path,
            "V-JEPA encoder t-SNE (age heatmap)",
            pca_components=args.tsne_pca_components,
            perplexity=args.tsne_perplexity,
            seed=args.seed,
            cmap=args.tsne_cmap,
        )
        details["tsne"] = {
            "path": str(tsne_path),
            "pca_components": int(args.tsne_pca_components),
            "perplexity": float(args.tsne_perplexity),
            "cmap": str(args.tsne_cmap),
        }
        print(f"Wrote {tsne_path}")

    # Continuous probe — labels look like ages here.
    reg = run_regression_cv(
        X,
        y,
        groups,
        n_pca=args.n_pca,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )
    reg_plot = out_dir / "probe_age_regression.png"
    save_regression_plot(reg, reg_plot, "V-JEPA embedding Ridge probe")
    pd.DataFrame(reg["folds"]).to_csv(out_dir / "probe_regression_folds.csv", index=False)
    summary_rows.append(
        {
            "probe": "ridge_regression",
            "metric": "mae",
            "value": reg["mae"],
            "chance": reg["chance_mae"],
            "r2": reg["r2"],
            "pearson_r": reg["pearson_r"],
            "n": reg["n_scored"],
        }
    )
    details["regression"] = {
        k: v
        for k, v in reg.items()
        if k not in {"y_true", "y_pred", "folds"}
    }
    print(
        f"Regression: MAE={reg['mae']:.2f} (chance MAE={reg['chance_mae']:.2f})  "
        f"R²={reg['r2']:.3f}  r={reg['pearson_r']:.3f}"
    )
    print(f"Wrote {reg_plot}")

    if args.binary_median:
        median = float(np.median(y))
        y_bin = (y >= median).astype(np.int64)
        clf = run_classification_cv(
            X,
            y_bin.astype(np.float64),
            groups,
            n_pca=args.n_pca,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            seed=args.seed,
            class_names=[f"<{median:g}", f">={median:g}"],
        )
        # Override names for the two median bins.
        clf["class_names"] = [f"<{median:g}", f">={median:g}"]
        clf["classes"] = [0, 1]
        cm_path = out_dir / "probe_age_binary_confusion.png"
        save_confusion_plot(
            clf,
            cm_path,
            f"V-JEPA embedding LR probe (age median split @ {median:g})",
        )
        pd.DataFrame(clf["folds"]).to_csv(
            out_dir / "probe_binary_folds.csv", index=False
        )
        summary_rows.append(
            {
                "probe": "logistic_median_split",
                "metric": "balanced_accuracy",
                "value": clf["balanced_accuracy"],
                "chance": clf["chance_balanced_accuracy"],
                "median": median,
                "n": clf["n_scored"],
            }
        )
        details["binary_median"] = {
            k: v
            for k, v in clf.items()
            if k not in {"y_true", "y_pred", "folds"}
        }
        print(
            f"Binary (≥{median:g}): balanced acc="
            f"{clf['balanced_accuracy']:.3f} "
            f"(chance={clf['chance_balanced_accuracy']:.3f})"
        )
        print(f"Wrote {cm_path}")

    summary_path = out_dir / "probe_metrics.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    details_path = out_dir / "probe_metrics.json"
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
