#!/usr/bin/env python3
"""
Post-hoc diagnostics for the official V-JEPA attentive probe.

Run from this jepa checkout with the ``vjepa_pretrain`` env so ``src.*`` /
``evals.*`` imports resolve. Writes:

  - val confusion matrix PNG (official attentive probe) + balanced accuracy
  - val confusion matrix PNG from a class-reweighted sklearn LR on encoder features
  - val t-SNE of mean-pooled encoder tokens colored by true condition

Example (cluster)::

  cd /rhomes/gschum/jepa
  export PYTHONPATH=/rhomes/gschum/jepa:$PYTHONPATH
  python scripts/plot_vjepa_probe_diagnostics.py \\
    --eval-config configs/vjepa/eval_attentive_probe_vitl16.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

JEPA_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-config",
        type=Path,
        default=JEPA_ROOT / "configs/vjepa/eval_attentive_probe_vitl16.yaml",
        help="Attentive-probe YAML (same as evals.main --fname).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write PNGs (default: <pretrain.folder>/video_classification_frozen/<tag>/).",
    )
    p.add_argument(
        "--split",
        choices=("val", "train"),
        default="val",
        help="Which CSV split to score (default: val).",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on videos for faster plots (stratified not applied).",
    )
    p.add_argument(
        "--tsne-pca-components",
        type=int,
        default=50,
        help="PCA dims before t-SNE (clamped to data size).",
    )
    p.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for t-SNE / optional subsample.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device.",
    )
    p.add_argument(
        "--class-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional class names in label-id order (default: class_0..K-1).",
    )
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _resolve_output_dir(cfg: dict, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    pre = cfg["pretrain"]
    folder = Path(pre["folder"]) / "video_classification_frozen"
    tag = cfg.get("tag")
    if tag:
        folder = folder / str(tag)
    return folder


def _class_names(num_classes: int, names: Optional[Sequence[str]]) -> List[str]:
    if names is not None and len(names) == num_classes:
        return [str(n) for n in names]
    # Match sorted unique labels from write_vjepa_eval_csvs / pretrain CSV.
    default = [
        "2dss",
        "5xFAD",
        "5xFAD+AP2a_CKO",
        "AP2a_CKO",
        "CDZ_cKO",
        "V2aAc",
        "V2aInac",
        "ctrl",
    ]
    if num_classes == len(default):
        return default
    return [f"class_{i}" for i in range(num_classes)]


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def _fit_reweighted_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 42,
    max_iter: int = 2000,
) -> Pipeline:
    """Standardize features and fit LR with ``class_weight='balanced'``."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def _mirror_files(paths: Sequence[Path], mirror_dir: Path) -> None:
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, mirror_dir / path.name)
    print(f"Mirrored plots to {mirror_dir}")


def _save_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_tsne(
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
    n_pca = max(1, min(pca_components, features.shape[0], features.shape[1]))
    if features.shape[1] > n_pca:
        features = PCA(n_components=n_pca, random_state=seed).fit_transform(features)

    n = features.shape[0]
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    emb = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)

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


@torch.no_grad()
def _collect_features_and_preds(
    *,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    data_loader,
    device: torch.device,
    attend_across_segments: bool,
    max_videos: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (features [N,D], y_true [N], y_pred [N]).

    Features are mean-pooled encoder tokens (before the attentive probe head).
    Predictions match the official multi-view average softmax used in eval.
    """
    encoder.eval()
    classifier.eval()

    feats_list: List[np.ndarray] = []
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    n_seen = 0

    for data in data_loader:
        clips = [
            [dij.to(device, non_blocking=True) for dij in di]
            for di in data[0]
        ]
        clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
        labels = data[1].to(device)
        batch_size = int(labels.shape[0])

        outputs = encoder(clips, clip_indices)

        # Token features for t-SNE: average over views / tokens.
        if attend_across_segments:
            # list over spatial views: each [B, N, D]
            tok = torch.stack(outputs, dim=0).mean(dim=0)  # [B, N, D]
        else:
            # list[spatial][temporal] of [B, N, D]
            flat = [ost for os in outputs for ost in os]
            tok = torch.stack(flat, dim=0).mean(dim=0)
        feat = tok.mean(dim=1)  # [B, D]

        # Classifier logits (same aggregation as eval.py)
        if attend_across_segments:
            logits_list = [classifier(o) for o in outputs]
            probs = sum(F.softmax(o, dim=1) for o in logits_list) / len(logits_list)
        else:
            logits_nested = [[classifier(ost) for ost in os] for os in outputs]
            probs = sum(
                sum(F.softmax(ost, dim=1) for ost in os) for os in logits_nested
            ) / (len(logits_nested) * len(logits_nested[0]))
        preds = probs.argmax(dim=1)

        feats_list.append(feat.float().cpu().numpy())
        y_true_list.append(labels.cpu().numpy().astype(np.int64))
        y_pred_list.append(preds.cpu().numpy().astype(np.int64))
        n_seen += batch_size
        if max_videos is not None and n_seen >= max_videos:
            break

    if not feats_list:
        raise RuntimeError("No samples collected from dataloader.")

    features = np.concatenate(feats_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    if max_videos is not None and features.shape[0] > max_videos:
        features = features[:max_videos]
        y_true = y_true[:max_videos]
        y_pred = y_pred[:max_videos]
    return features, y_true, y_pred


def main() -> int:
    args = _parse_args()

    # JEPA imports must resolve against the facebookresearch/jepa checkout CWD.
    try:
        import src.models.vision_transformer as vit  # noqa: F401
        from src.models.attentive_pooler import AttentiveClassifier
        from evals.video_classification_frozen.eval import init_model, make_dataloader
        from evals.video_classification_frozen.utils import ClipAggregation
    except ImportError as exc:
        print(
            "Failed to import JEPA modules. Run this script from the jepa "
            f"checkout (e.g. cd {JEPA_ROOT}) with the vjepa_pretrain env.\n"
            f"ImportError: {exc}",
            file=sys.stderr,
        )
        return 1

    cfg = _load_yaml(args.eval_config)
    pre = cfg["pretrain"]
    data_cfg = cfg["data"]
    opt = cfg["optimization"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    attend_across_segments = bool(opt.get("attend_across_segments", True))
    num_classes = int(data_cfg["num_classes"])
    frames_per_clip = int(data_cfg.get("frames_per_clip", 16))
    num_segments = int(data_cfg.get("num_segments", 2))
    num_views = int(data_cfg.get("num_views_per_segment", 3))
    frame_step = int(data_cfg.get("frame_step", 4))
    resolution = int(opt.get("resolution", 224))
    batch_size = int(opt.get("batch_size", 4))

    pretrained_path = os.path.join(pre["folder"], pre["checkpoint"])
    probe_dir = _resolve_output_dir(cfg, None)
    probe_ckpt = probe_dir / f"{pre['write_tag']}-latest.pth.tar"
    out_dir = _resolve_output_dir(cfg, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(pretrained_path).is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {pretrained_path}")
    if not probe_ckpt.is_file():
        raise FileNotFoundError(
            f"Probe checkpoint not found: {probe_ckpt}\n"
            "Run the attentive probe eval first (scripts/eval_vjepa_attentive_probe.sh)."
        )

    print(f"Loading encoder from {pretrained_path}")
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=pre["model_name"],
        patch_size=int(pre.get("patch_size", 16)),
        tubelet_size=int(pre.get("tubelet_size", 2)),
        frames_per_clip=int(pre.get("frames_per_clip", frames_per_clip)),
        uniform_power=bool(pre.get("uniform_power", True)),
        checkpoint_key=pre.get("checkpoint_key", "target_encoder"),
        use_SiLU=bool(pre.get("use_silu", False)),
        tight_SiLU=bool(pre.get("tight_silu", False)),
        use_sdpa=bool(pre.get("use_sdpa", True)),
    )
    encoder = ClipAggregation(
        encoder,
        tubelet_size=int(pre.get("tubelet_size", 2)),
        attend_across_segments=attend_across_segments,
    ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print(f"Loading probe from {probe_ckpt}")
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(device)
    ckpt = torch.load(probe_ckpt, map_location="cpu")
    state = _strip_module_prefix(ckpt["classifier"])
    msg = classifier.load_state_dict(state, strict=False)
    print(f"Loaded classifier: {msg}")
    classifier.eval()

    csv_key = "dataset_val" if args.split == "val" else "dataset_train"
    root_path = [data_cfg[csv_key]]
    train_root = [data_cfg["dataset_train"]]
    num_workers = min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

    def _build_loader(roots: list[str]) -> object:
        print(f"Building loader from {roots[0]}")
        return make_dataloader(
            dataset_type=data_cfg.get("dataset_type", "VideoDataset"),
            root_path=roots,
            resolution=resolution,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            eval_duration=pre.get("clip_duration", None),
            num_segments=num_segments,
            num_views_per_segment=num_views,
            allow_segment_overlap=True,
            batch_size=batch_size,
            world_size=1,
            rank=0,
            training=False,
            num_workers=num_workers,
        )

    eval_loader = _build_loader(root_path)
    features, y_true, y_pred = _collect_features_and_preds(
        encoder=encoder,
        classifier=classifier,
        data_loader=eval_loader,
        device=device,
        attend_across_segments=attend_across_segments,
        max_videos=args.max_videos,
    )
    names = _class_names(num_classes, args.class_names)
    top1 = float((y_true == y_pred).mean()) if y_true.size else float("nan")
    bal = (
        float(balanced_accuracy_score(y_true, y_pred))
        if y_true.size
        else float("nan")
    )
    print(
        f"Attentive probe ({args.split}): n={len(y_true)}  "
        f"top1={top1 * 100:.2f}%  bal_acc={bal * 100:.2f}%"
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_path = out_dir / f"{args.split}_confusion_matrix.png"
    _save_confusion_matrix(
        cm,
        names,
        cm_path,
        title=(
            f"V-JEPA attentive probe ({args.split})  "
            f"top-1={top1 * 100:.1f}%  bal={bal * 100:.1f}%"
        ),
    )
    print(f"Wrote {cm_path}")

    # Class-reweighted sklearn LR on mean-pooled encoder features (fit on train).
    if args.split == "train":
        X_train, y_train = features, y_true
    else:
        train_loader = _build_loader(train_root)
        X_train, y_train, _ = _collect_features_and_preds(
            encoder=encoder,
            classifier=classifier,
            data_loader=train_loader,
            device=device,
            attend_across_segments=attend_across_segments,
            max_videos=args.max_videos,
        )
    print(f"Fitting reweighted LR on train features n={len(y_train)}")
    lr_pipe = _fit_reweighted_lr(X_train, y_train, seed=args.seed)
    y_pred_rw = lr_pipe.predict(features)
    top1_rw = float((y_true == y_pred_rw).mean()) if y_true.size else float("nan")
    bal_rw = (
        float(balanced_accuracy_score(y_true, y_pred_rw))
        if y_true.size
        else float("nan")
    )
    print(
        f"Reweighted LR ({args.split}): "
        f"top1={top1_rw * 100:.2f}%  bal_acc={bal_rw * 100:.2f}%"
    )
    cm_rw = confusion_matrix(y_true, y_pred_rw, labels=list(range(num_classes)))
    cm_rw_path = out_dir / f"{args.split}_confusion_matrix_reweighted_lr.png"
    _save_confusion_matrix(
        cm_rw,
        names,
        cm_rw_path,
        title=(
            f"Encoder + reweighted LR ({args.split})  "
            f"top-1={top1_rw * 100:.1f}%  bal={bal_rw * 100:.1f}%"
        ),
    )
    print(f"Wrote {cm_rw_path}")

    tsne_path = out_dir / f"{args.split}_tsne_encoder.png"
    _save_tsne(
        features,
        y_true,
        names,
        tsne_path,
        title=f"V-JEPA encoder t-SNE ({args.split}, mean-pooled tokens)",
        pca_components=args.tsne_pca_components,
        perplexity=args.tsne_perplexity,
        seed=args.seed,
    )
    print(f"Wrote {tsne_path}")

    mirror = JEPA_ROOT / "logs" / "vjepa_probe_diagnostics"
    try:
        _mirror_files([cm_path, cm_rw_path, tsne_path], mirror)
    except OSError as exc:
        print(f"Mirror skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
