"""
Thin MLflow helpers for V-JEPA pretrain and frozen eval.

Logging is opt-in via a config dict (``logging.mlflow`` or top-level ``mlflow``).
All public APIs no-op when disabled or when ``rank != 0``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import yaml

logger = logging.getLogger(__name__)

# MLflow param value length limit
_MAX_PARAM_LEN = 500
_MAX_LIST_ITEMS = 8

_active = False
_rank = 0


def repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def default_mlflow_dir() -> Path:
    """
    Writable MLflow directory under ``<repo>/outputs/mlflow``.

    Kept under ``outputs/`` so SLURM jobs that can write training artifacts
    can also create the SQLite DB (repo root may be read-only on compute nodes).
    """
    return (repo_root() / "outputs" / "mlflow").resolve()


def default_tracking_uri() -> str:
    """SQLite tracking URI at ``<repo>/outputs/mlflow/mlflow.db``."""
    db_path = default_mlflow_dir() / "mlflow.db"
    # Absolute paths need sqlite:////... (four slashes).
    return f"sqlite:///{db_path}"


def default_artifact_root() -> str:
    """File URI for MLflow artifacts next to the SQLite DB."""
    art = default_mlflow_dir() / "artifacts"
    return art.as_uri()


def is_enabled(cfg: Optional[Mapping[str, Any]]) -> bool:
    """Return True if the mlflow config block requests logging."""
    if not cfg:
        return False
    return bool(cfg.get("enabled", True))


def _sqlite_path_from_uri(uri: str) -> Optional[Path]:
    """Extract a filesystem path from a ``sqlite:///...`` URI, if applicable."""
    if not uri.startswith("sqlite:"):
        return None
    # sqlite:///relative.db  or  sqlite:////abs/path.db
    rest = uri[len("sqlite:") :]
    if rest.startswith("////"):
        return Path("/" + rest.lstrip("/"))
    if rest.startswith("///"):
        # Absolute on Unix: sqlite:/// + /abs -> three slashes then abs path
        # or relative: sqlite:///rel.db
        path_part = rest[3:]
        path = Path(path_part)
        return path if path.is_absolute() else Path.cwd() / path
    return None


def setup_tracking(tracking_uri: Optional[str] = None) -> str:
    """
    Configure the MLflow tracking URI.

    Parameters
    ----------
    tracking_uri :
        Explicit URI, or ``None`` / empty to use
        ``sqlite:///<repo>/outputs/mlflow/mlflow.db``.

    Returns
    -------
    str
        The resolved tracking URI.
    """
    import mlflow

    if tracking_uri is None or tracking_uri == "" or tracking_uri == "null":
        uri = default_tracking_uri()
    else:
        tracking_uri = str(tracking_uri)
        if "://" not in tracking_uri:
            path = Path(tracking_uri).expanduser().resolve()
            # Bare ``*.db`` paths are treated as SQLite; other paths as file URIs.
            if path.suffix == ".db":
                uri = f"sqlite:///{path}"
            else:
                uri = path.as_uri()
        else:
            uri = tracking_uri

    # Ensure parent dirs exist before SQLite tries to create the DB file.
    db_path = _sqlite_path_from_uri(uri)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        default_mlflow_dir().mkdir(parents=True, exist_ok=True)

    art_dir = default_mlflow_dir() / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(uri)
    return uri


def flatten_params(
    obj: Any,
    prefix: str = "",
    out: Optional[MutableMapping[str, str]] = None,
) -> MutableMapping[str, str]:
    """
    Flatten nested dicts/lists into MLflow-safe scalar string params.

    Long lists are truncated; values longer than ``_MAX_PARAM_LEN`` are clipped.
    """
    if out is None:
        out = {}

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_str = str(key)
            nested = f"{prefix}.{key_str}" if prefix else key_str
            flatten_params(value, nested, out)
        return out

    if isinstance(obj, (list, tuple)):
        if len(obj) > _MAX_LIST_ITEMS:
            shown = list(obj[:_MAX_LIST_ITEMS]) + [f"...(+{len(obj) - _MAX_LIST_ITEMS})"]
        else:
            shown = list(obj)
        # Keep short homogeneous scalars as a single param; otherwise recurse.
        if all(not isinstance(x, (Mapping, list, tuple)) for x in obj):
            out[prefix] = _clip(str(shown))
        else:
            for i, value in enumerate(shown):
                if isinstance(value, str) and value.startswith("...(+"):
                    out[f"{prefix}.{value}"] = ""
                else:
                    flatten_params(value, f"{prefix}.{i}", out)
        return out

    out[prefix] = _clip("" if obj is None else str(obj))
    return out


def _clip(value: str) -> str:
    if len(value) <= _MAX_PARAM_LEN:
        return value
    return value[: _MAX_PARAM_LEN - 3] + "..."


def start_run(
    cfg: Optional[Mapping[str, Any]],
    *,
    rank: int,
    default_experiment_name: str,
    default_run_name: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, str]] = None,
    config_artifact: Optional[Mapping[str, Any]] = None,
) -> bool:
    """
    Start an MLflow run on rank 0 when ``cfg`` enables logging.

    Returns
    -------
    bool
        True if a run was started (caller should eventually ``end_run``).
    """
    global _active, _rank
    _rank = rank
    _active = False

    if rank != 0 or not is_enabled(cfg):
        return False

    import mlflow

    cfg = cfg or {}
    uri = setup_tracking(cfg.get("tracking_uri"))
    experiment_name = cfg.get("experiment_name") or default_experiment_name
    run_name = cfg.get("run_name") or default_run_name

    # Prefer artifacts under outputs/mlflow (not cwd-relative ./mlruns).
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(
            experiment_name,
            artifact_location=default_artifact_root(),
        )
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    _active = True

    if tags:
        mlflow.set_tags({str(k): str(v) for k, v in tags.items()})

    if params:
        flat = flatten_params(params)
        # MLflow rejects empty keys; drop them.
        flat = {k: v for k, v in flat.items() if k}
        # Batch in chunks to avoid oversized single calls.
        items = list(flat.items())
        chunk = 100
        for i in range(0, len(items), chunk):
            mlflow.log_params(dict(items[i : i + chunk]))

    if config_artifact is not None:
        _log_config_yaml(config_artifact)

    logger.info(
        "MLflow run started (experiment=%s, run_name=%s, tracking_uri=%s)",
        experiment_name,
        run_name,
        uri,
    )
    return True


def log_metrics(metrics: Mapping[str, float], step: Optional[int] = None) -> None:
    """Log numeric metrics to the active run (no-op if inactive)."""
    if not _active or _rank != 0:
        return
    import mlflow

    clean = {}
    for key, value in metrics.items():
        try:
            clean[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if clean:
        mlflow.log_metrics(clean, step=step)


def end_run() -> None:
    """End the active MLflow run if one was started by ``start_run``."""
    global _active
    if not _active:
        return
    import mlflow

    try:
        mlflow.end_run()
    finally:
        _active = False
        logger.info("MLflow run ended")


def _log_config_yaml(config: Mapping[str, Any]) -> None:
    """Write config to a temp YAML file and log it as an MLflow artifact."""
    import mlflow

    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="params-")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.safe_dump(dict(config), f, sort_keys=False)
        mlflow.log_artifact(path, artifact_path="config")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
