#!/usr/bin/env python3
"""
Export frozen per-video embeddings from the self-pretrained V-JEPA ViT-L/16.

Lives in the facebookresearch/jepa checkout. The encoder is the
``target_encoder`` from the SSL checkpoint, used frozen. No attentive probe is
involved: unlike ``scripts/plot_vjepa_probe_diagnostics.py``, this script needs
no probe checkpoint, because it only reads out encoder tokens.

Three modes, because jepa's top-level ``src`` package shadows MouseBeamWalk's
``src``. ``extract`` imports only JEPA modules; ``write-index`` / ``finalize``
import MouseBeamWalk helpers via ``MBW_DIR`` and communicate through a raw
``.npz``.

1. ``write-index`` (myenv, needs MouseBeamWalk on ``MBW_DIR``) — build a V-JEPA
   index CSV labelled by manifest row index so features stay joinable.
2. ``extract`` (this checkout, vjepa_pretrain env, GPU) — run the frozen encoder
   and dump ``X`` plus labels. Accepts ``--eval-config`` or ``--pretrain-config``.
3. ``finalize`` (myenv, needs MouseBeamWalk on ``MBW_DIR``) — join manifest
   metadata onto the raw matrix and write the embedding table.

Example (pretrain-config path used by ``scripts/train_vjepa.sh``)::

  cd $JEPA_DIR
  export PYTHONPATH=$JEPA_DIR:$PYTHONPATH
  python scripts/export_embeddings_vjepa.py extract \\
    --pretrain-config configs/vjepa/pretrain_vitl16_h200.yaml \\
    --index-csv /path/to/videos.csv \\
    --raw-output /path/to/embeddings_raw.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

JEPA_ROOT = Path(__file__).resolve().parents[1]
MBW_ROOT = Path(
    os.environ.get("MBW_DIR", "/rhomes/gschum/MouseBeamWalk")
).expanduser().resolve()


def _add_mbw_to_path() -> None:
    """
    Put MouseBeamWalk on ``sys.path`` for write-index / finalize.

    Must not be called from ``extract``: jepa's ``src`` would be shadowed.
    """
    if not MBW_ROOT.is_dir():
        raise FileNotFoundError(
            f"MouseBeamWalk not found at {MBW_ROOT}. "
            "Set MBW_DIR to the MouseBeamWalk checkout."
        )
    if str(MBW_ROOT) not in sys.path:
        sys.path.insert(0, str(MBW_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    index_parser = subparsers.add_parser(
        "write-index", help="Write the row-index-labelled V-JEPA index CSV."
    )
    index_parser.add_argument(
        "--manifest-path", type=Path, default=Path("manifests/manifest.jsonl")
    )
    index_parser.add_argument(
        "--index-csv", type=Path, default=Path("manifests/vjepa_index_all.csv")
    )
    index_parser.add_argument(
        "--camera",
        type=int,
        default=6,
        help="Restrict to this camera id; -1 keeps all (default: 6).",
    )
    index_parser.add_argument(
        "--beam-width",
        default=None,
        help="Restrict to one beam width (e.g. 5mm). Default: keep all.",
    )
    index_parser.add_argument(
        "--conditions", nargs="*", default=None, help="Default: keep all conditions."
    )
    index_parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Omit rows whose video file is absent on disk.",
    )

    extract_parser = subparsers.add_parser(
        "extract", help="Run the frozen encoder (from a facebookresearch/jepa checkout)."
    )
    extract_cfg = extract_parser.add_mutually_exclusive_group(required=True)
    extract_cfg.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help=(
            "Attentive-probe YAML; only its 'pretrain' / 'data' / 'optimization' "
            "encoder settings are used."
        ),
    )
    extract_cfg.add_argument(
        "--pretrain-config",
        type=Path,
        default=None,
        help=(
            "Pretrain YAML (app: vjepa). Maps model/data/logging fields to the "
            "frozen encoder settings; checkpoint is "
            "{logging.folder}/{write_tag}-latest.pth.tar."
        ),
    )
    extract_parser.add_argument(
        "--index-csv",
        type=Path,
        required=True,
        help="Index CSV written by the write-index mode.",
    )
    extract_parser.add_argument("--raw-output", type=Path, required=True)
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from the config (eval optimization.batch_size or default 4).",
    )
    extract_parser.add_argument(
        "--num-segments",
        type=int,
        default=None,
        help="Temporal segments per video (default: 2, or data.num_segments from eval-config).",
    )
    extract_parser.add_argument(
        "--num-views-per-segment",
        type=int,
        default=None,
        help="Spatial views per segment (default: 3, or from eval-config).",
    )
    extract_parser.add_argument(
        "--attend-across-segments",
        type=lambda v: str(v).lower() in {"1", "true", "yes"},
        default=None,
        help="Average tokens across temporal segments before pooling (default: true).",
    )
    extract_parser.add_argument("--num-workers", type=int, default=4)
    extract_parser.add_argument(
        "--max-videos", type=int, default=None, help="Smoke-test cap."
    )

    finalize_parser = subparsers.add_parser(
        "finalize", help="Join manifest metadata and write the embedding table."
    )
    finalize_parser.add_argument("--raw-input", type=Path, required=True)
    finalize_parser.add_argument("--index-csv", type=Path, required=True)
    finalize_parser.add_argument(
        "--manifest-path", type=Path, default=Path("manifests/manifest.jsonl")
    )
    finalize_parser.add_argument("--output-prefix", type=Path, required=True)
    finalize_parser.add_argument("--model-name", default="V-JEPA ViT-L/16 (ours)")

    return parser


def build_filters(args: argparse.Namespace) -> dict:
    """Translate the CLI cohort flags into manifest filters."""
    filters: dict = {}
    if args.camera is not None and args.camera >= 0:
        filters["camera"] = [int(args.camera)]
    if args.beam_width is not None:
        filters["beam_width"] = [args.beam_width]
    if args.conditions:
        filters["condition"] = list(args.conditions)
    return filters


def run_write_index(args: argparse.Namespace) -> int:
    _add_mbw_to_path()
    from src.utils.vjepa_csv import write_vjepa_index_csv

    csv_path, video_paths = write_vjepa_index_csv(
        args.manifest_path,
        args.index_csv,
        filters=build_filters(args) or None,
        skip_missing=args.skip_missing,
    )
    print(f"Wrote {csv_path} with {len(video_paths)} videos (labels 0..{len(video_paths) - 1})")
    return 0


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def resolve_extract_settings(args: argparse.Namespace) -> dict:
    """
    Build encoder / loader settings from either an eval YAML or a pretrain YAML.

    Returns a dict with keys used by ``run_extract``.
    """
    # Defaults match current extract behavior when an eval config is used.
    defaults = {
        "attend_across_segments": True,
        "num_segments": 2,
        "num_views_per_segment": 3,
        "batch_size": 4,
        "checkpoint_key": "target_encoder",
        "use_silu": False,
        "tight_silu": False,
        "use_sdpa": True,
        "uniform_power": True,
        "clip_duration": None,
    }

    if args.eval_config is not None:
        cfg = _load_yaml(args.eval_config)
        pretrain_cfg = cfg["pretrain"]
        data_cfg = cfg["data"]
        opt_cfg = cfg.get("optimization", {})
        settings = {
            **defaults,
            "model_name": pretrain_cfg["model_name"],
            "checkpoint_path": Path(pretrain_cfg["folder"]) / pretrain_cfg["checkpoint"],
            "checkpoint_key": pretrain_cfg.get("checkpoint_key", defaults["checkpoint_key"]),
            "patch_size": int(pretrain_cfg.get("patch_size", 16)),
            "tubelet_size": int(pretrain_cfg.get("tubelet_size", 2)),
            "frames_per_clip": int(
                data_cfg.get(
                    "frames_per_clip",
                    pretrain_cfg.get("frames_per_clip", 16),
                )
            ),
            "pretrain_frames_per_clip": int(
                pretrain_cfg.get(
                    "frames_per_clip",
                    data_cfg.get("frames_per_clip", 16),
                )
            ),
            "frame_step": int(data_cfg.get("frame_step", 4)),
            "num_segments": int(data_cfg.get("num_segments", defaults["num_segments"])),
            "num_views_per_segment": int(
                data_cfg.get("num_views_per_segment", defaults["num_views_per_segment"])
            ),
            "resolution": int(opt_cfg.get("resolution", 224)),
            "batch_size": int(opt_cfg.get("batch_size", defaults["batch_size"])),
            "attend_across_segments": bool(
                opt_cfg.get("attend_across_segments", defaults["attend_across_segments"])
            ),
            "uniform_power": bool(pretrain_cfg.get("uniform_power", defaults["uniform_power"])),
            "use_silu": bool(pretrain_cfg.get("use_silu", defaults["use_silu"])),
            "tight_silu": bool(pretrain_cfg.get("tight_silu", defaults["tight_silu"])),
            "use_sdpa": bool(pretrain_cfg.get("use_sdpa", defaults["use_sdpa"])),
            "clip_duration": pretrain_cfg.get("clip_duration", defaults["clip_duration"]),
            "dataset_type": data_cfg.get("dataset_type", "VideoDataset"),
        }
    else:
        cfg = _load_yaml(args.pretrain_config)
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        meta_cfg = cfg.get("meta", {})
        log_cfg = cfg.get("logging", {})
        folder = Path(log_cfg["folder"])
        write_tag = log_cfg.get("write_tag", "jepa")
        settings = {
            **defaults,
            "model_name": model_cfg["model_name"],
            "checkpoint_path": folder / f"{write_tag}-latest.pth.tar",
            "checkpoint_key": defaults["checkpoint_key"],
            "patch_size": int(data_cfg.get("patch_size", 16)),
            "tubelet_size": int(data_cfg.get("tubelet_size", 2)),
            "frames_per_clip": int(data_cfg.get("num_frames", 16)),
            "pretrain_frames_per_clip": int(data_cfg.get("num_frames", 16)),
            "frame_step": int(data_cfg.get("sampling_rate", 4)),
            "resolution": int(data_cfg.get("crop_size", 224)),
            "uniform_power": bool(model_cfg.get("uniform_power", defaults["uniform_power"])),
            "use_sdpa": bool(meta_cfg.get("use_sdpa", defaults["use_sdpa"])),
            "clip_duration": data_cfg.get("clip_duration", defaults["clip_duration"]),
            "dataset_type": data_cfg.get("dataset_type", "VideoDataset"),
        }

    # CLI overrides (None means keep resolved value).
    if args.batch_size is not None:
        settings["batch_size"] = int(args.batch_size)
    if args.num_segments is not None:
        settings["num_segments"] = int(args.num_segments)
    if args.num_views_per_segment is not None:
        settings["num_views_per_segment"] = int(args.num_views_per_segment)
    if args.attend_across_segments is not None:
        settings["attend_across_segments"] = bool(args.attend_across_segments)

    return settings


def run_extract(args: argparse.Namespace) -> int:
    # JEPA imports must resolve against this checkout's CWD / PYTHONPATH;
    # never call _add_mbw_to_path() here.
    import torch

    try:
        from evals.video_classification_frozen.eval import init_model, make_dataloader
        from evals.video_classification_frozen.utils import ClipAggregation
    except ImportError as exc:
        print(
            "Failed to import JEPA modules. Run this mode from the jepa "
            f"checkout (e.g. cd {JEPA_ROOT}) with the vjepa_pretrain env "
            f"and PYTHONPATH={JEPA_ROOT}.\nImportError: {exc}",
            file=sys.stderr,
        )
        return 1

    settings = resolve_extract_settings(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    attend_across_segments = bool(settings["attend_across_segments"])
    frames_per_clip = int(settings["frames_per_clip"])
    num_segments = int(settings["num_segments"])
    num_views = int(settings["num_views_per_segment"])
    frame_step = int(settings["frame_step"])
    resolution = int(settings["resolution"])
    batch_size = int(settings["batch_size"])
    checkpoint_path = Path(settings["checkpoint_path"])

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")

    print(f"Loading frozen encoder from {checkpoint_path}", flush=True)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=str(checkpoint_path),
        model_name=settings["model_name"],
        patch_size=int(settings["patch_size"]),
        tubelet_size=int(settings["tubelet_size"]),
        frames_per_clip=int(settings["pretrain_frames_per_clip"]),
        uniform_power=bool(settings["uniform_power"]),
        checkpoint_key=settings["checkpoint_key"],
        use_SiLU=bool(settings["use_silu"]),
        tight_SiLU=bool(settings["tight_silu"]),
        use_sdpa=bool(settings["use_sdpa"]),
    )
    encoder = ClipAggregation(
        encoder,
        tubelet_size=int(settings["tubelet_size"]),
        attend_across_segments=attend_across_segments,
    ).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    print(f"Building loader from {args.index_csv}", flush=True)
    loader = make_dataloader(
        dataset_type=settings.get("dataset_type", "VideoDataset"),
        root_path=[str(args.index_csv)],
        resolution=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=settings.get("clip_duration", None),
        num_segments=num_segments,
        num_views_per_segment=num_views,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=1,
        rank=0,
        training=False,
        num_workers=args.num_workers,
    )

    feature_blocks: List[np.ndarray] = []
    row_index_blocks: List[np.ndarray] = []
    n_views_averaged = 0
    n_seen = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader, start=1):
            clips = [[view.to(device, non_blocking=True) for view in segment] for segment in data[0]]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            # Label slot from the index CSV (class id, or row index for write-index CSVs).
            batch_labels = data[1].to(torch.int64)

            outputs = encoder(clips, clip_indices)
            if attend_across_segments:
                # list over spatial views, each [B, N, D]
                tokens = torch.stack(outputs, dim=0).mean(dim=0)
                n_views_averaged = len(outputs)
            else:
                # list[spatial][temporal] of [B, N, D]
                flattened = [view for segment in outputs for view in segment]
                tokens = torch.stack(flattened, dim=0).mean(dim=0)
                n_views_averaged = len(flattened)
            features = tokens.mean(dim=1)  # [B, D], mean over tokens

            feature_blocks.append(features.float().cpu().numpy())
            row_index_blocks.append(batch_labels.cpu().numpy())
            n_seen += int(batch_labels.shape[0])
            if batch_idx % 20 == 0:
                print(f"  encoded {n_seen} videos", flush=True)
            if args.max_videos is not None and n_seen >= args.max_videos:
                break

    if not feature_blocks:
        raise RuntimeError("No videos were encoded; check the index CSV and video paths.")

    X = np.concatenate(feature_blocks, axis=0).astype(np.float32)
    labels = np.concatenate(row_index_blocks, axis=0).astype(np.int64)
    if args.max_videos is not None and X.shape[0] > args.max_videos:
        X = X[: args.max_videos]
        labels = labels[: args.max_videos]

    duplicates = labels.size - np.unique(labels).size
    if duplicates:
        # Class-labelled CSVs reuse label ids; that is fine for t-SNE coloring.
        # write-index CSVs (unique row ids) are still preferred for finalize joins.
        print(
            f"Note: {duplicates} duplicate label(s) in the loader output; "
            "treating CSV column 2 as class labels for embedding metadata "
            "(finalize row-join requires unique write-index labels).",
            flush=True,
        )

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.raw_output,
        X=X,
        labels=labels,
        # Keep row_index for backward compatibility with finalize (same values).
        row_index=labels,
        checkpoint=np.asarray(str(checkpoint_path)),
        model_name=np.asarray(str(settings["model_name"])),
        checkpoint_key=np.asarray(str(settings["checkpoint_key"])),
        frames_per_clip=np.asarray(frames_per_clip),
        frame_step=np.asarray(frame_step),
        num_segments=np.asarray(num_segments),
        num_views_per_segment=np.asarray(num_views),
        resolution=np.asarray(resolution),
        n_views_averaged=np.asarray(n_views_averaged),
    )
    print(
        f"Wrote {args.raw_output}: X={X.shape} over {labels.size} videos "
        f"({n_views_averaged} view vectors averaged per video)"
    )
    return 0


def run_finalize(args: argparse.Namespace) -> int:
    _add_mbw_to_path()

    from src.analysis.embedding_io import save_embedding_table
    from src.analysis.manifest_join import join_manifest_by_video_path, load_manifest_frame
    from src.utils.vjepa_csv import read_vjepa_index_csv

    with np.load(args.raw_input, allow_pickle=True) as payload:
        X = np.asarray(payload["X"], dtype=np.float32)
        row_index = np.asarray(payload["row_index"], dtype=np.int64)
        provenance = {
            key: str(payload[key].item() if payload[key].ndim == 0 else payload[key])
            for key in payload.files
            if key not in {"X", "row_index"}
        }

    video_paths = read_vjepa_index_csv(args.index_csv)
    if row_index.size and row_index.max() >= len(video_paths):
        raise ValueError(
            f"Raw npz references row index {row_index.max()} but the index CSV "
            f"has only {len(video_paths)} rows; the CSV and the extraction disagree."
        )
    encoded_paths = [video_paths[int(i)] for i in row_index]

    manifest = load_manifest_frame(args.manifest_path)
    metadata = join_manifest_by_video_path(manifest, encoded_paths)
    metadata["n_clips"] = int(provenance.get("n_views_averaged", 1) or 1)

    missing = len(video_paths) - len(encoded_paths)
    if missing:
        print(
            f"Note: {missing} of {len(video_paths)} indexed videos produced no "
            "features (skipped by the loader); they are absent from the table.",
            flush=True,
        )

    npz_path, parquet_path = save_embedding_table(
        args.output_prefix,
        X,
        metadata,
        model=args.model_name,
        provenance={
            "source": "vjepa_self_pretrained",
            "index_csv": str(args.index_csv),
            "manifest_path": str(args.manifest_path),
            "raw_input": str(args.raw_input),
            "n_indexed_videos": str(len(video_paths)),
            **provenance,
        },
    )
    print(
        f"Wrote {npz_path}\nWrote {parquet_path}\n"
        f"{X.shape[0]} videos, dim={X.shape[1]}, "
        f"{metadata['mouse_uid'].nunique()} mice, "
        f"{metadata['condition'].nunique()} conditions"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "write-index":
        return run_write_index(args)
    if args.mode == "extract":
        return run_extract(args)
    if args.mode == "finalize":
        return run_finalize(args)
    raise ValueError(f"Unknown mode {args.mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
