"""Shared numerical, reporting, and reproducibility utilities."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.linalg import hadamard

from . import BASE_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
ARTIFACT_DATE = "2026-07-23"


def stable_rng(label: str, seed: int = BASE_SEED) -> np.random.Generator:
    """Return a generator whose stream is stable under experiment reordering."""

    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    child_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(child_seed)


def normalized_hadamard(n: int) -> np.ndarray:
    """Return the normalized Sylvester Hadamard matrix of power-of-two order."""

    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard order must be a positive power of two, got {n}")
    return hadamard(n, dtype=np.float64) / math.sqrt(n)


def random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw a deterministic-seed Haar orthogonal matrix."""

    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def sym_psd_power(matrix: np.ndarray, exponent: float, floor: float = 0.0) -> np.ndarray:
    """Compute a symmetric PSD matrix power by an eigenvalue decomposition."""

    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(values))), 1.0)
    if float(np.min(values)) < -1.0e-10 * scale:
        raise ValueError("matrix is not positive semidefinite within tolerance")
    clipped = np.maximum(values, floor)
    if exponent < 0 and np.any(clipped <= 0):
        raise ValueError("negative PSD power requires a strictly positive floor")
    powered = np.power(clipped, exponent)
    return (vectors * powered) @ vectors.T


def nuclear_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.svd(matrix, compute_uv=False).sum())


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    label: str,
    confidence: float = 0.95,
    draws: int = 4000,
) -> tuple[float, float, float]:
    """Return the median and a deterministic percentile bootstrap interval."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return math.nan, math.nan, math.nan
    if array.size == 1:
        value = float(array[0])
        return value, value, value
    rng = stable_rng(f"bootstrap:{label}")
    indices = rng.integers(0, array.size, size=(draws, array.size))
    medians = np.median(array[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(medians, [tail, 1.0 - tail])
    return float(np.median(array)), float(low), float(high)


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {"q25": float(q25), "median": float(median), "q75": float(q75)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest(
    *,
    artifacts: Iterable[Path],
    dataset_hashes: Mapping[str, str],
    command: str,
) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "scikit-learn"):
        packages[name] = importlib.metadata.version(name)
    artifact_hashes = {}
    for path in sorted(set(Path(item).resolve() for item in artifacts)):
        if path.exists():
            artifact_hashes[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    return {
        "base_seed": BASE_SEED,
        "command": command,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "dataset_sha256": dict(dataset_hashes),
        "artifact_sha256": artifact_hashes,
        "thread_environment": {
            key: os.environ.get(key, "")
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }



def configure_matplotlib() -> None:
    """Set deterministic, legible manuscript plotting defaults."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "gauge-floors-20260723",
        }
    )


def save_figure(figure: Any, path: Path, **kwargs: Any) -> None:
    """Save a web-friendly SVG figure with stable descriptive metadata."""

    if path.suffix.lower() != ".svg":
        raise ValueError(f"expected an SVG figure path, got {path}")
    metadata = {
        "Title": path.stem.replace("_", " "),
        "Creator": "gauge-floors experiment pipeline",
        "Description": (
            "Reproducible chart for the gauge-floors computational artifacts. "
            "A text alternative and source data are linked from the project site."
        ),
        "Keywords": ["matrix quantization", "reproducible research"],
        "Date": ARTIFACT_DATE,
        "Language": "en-US",
    }
    figure.savefig(path, metadata=metadata, **kwargs)
