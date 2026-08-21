#!/usr/bin/env python3
"""Execute every tracked notebook in a fresh kernel without changing sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _git_notebooks(root: Path) -> list[Path] | None:
    """Return tracked notebooks, or ``None`` when root is not a Git checkout."""

    try:
        probe = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "--show-toplevel"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    if probe.returncode != 0:
        return None
    git_root = Path(os.fsdecode(probe.stdout).strip()).resolve()
    if git_root != root.resolve():
        return None

    result = subprocess.run(
        ["git", "-C", os.fspath(root), "ls-files", "-z", "--", "notebooks"],
        check=True,
        stdout=subprocess.PIPE,
    )
    notebooks = []
    for encoded in result.stdout.split(b"\0"):
        if not encoded:
            continue
        relative = PurePosixPath(os.fsdecode(encoded))
        if relative.suffix.casefold() == ".ipynb" and ".ipynb_checkpoints" not in relative.parts:
            notebooks.append(root / relative)
    return sorted(notebooks)


def discover_notebooks(root: Path) -> list[Path]:
    notebooks = _git_notebooks(root)
    if notebooks:
        return notebooks
    notebook_root = root / "notebooks"
    if not notebook_root.is_dir():
        return []
    return sorted(
        path
        for path in notebook_root.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts and path.is_file()
    )


def _read_notebook(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, [f"cannot read valid UTF-8 notebook JSON: {error}"]
    if not isinstance(payload, dict):
        return None, ["top-level notebook value must be an object"]
    if payload.get("nbformat") != 4:
        errors.append(f"unsupported nbformat {payload.get('nbformat')!r}; expected 4")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        errors.append("cells must be a list")
        return payload, errors
    for index, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            errors.append(f"cell {index} is not an object")
            continue
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        execution_count = cell.get("execution_count")
        if outputs:
            errors.append(f"code cell {index} has committed output")
        if execution_count is not None:
            errors.append(
                f"code cell {index} has execution_count={execution_count!r}; expected null"
            )
    return payload, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(root: Path, notebooks: list[Path]) -> tuple[dict[Path, str], list[str]]:
    hashes: dict[Path, str] = {}
    errors: list[str] = []
    for notebook in notebooks:
        try:
            relative = notebook.resolve().relative_to(root.resolve())
        except ValueError:
            errors.append(f"{notebook}: resolves outside the repository")
            continue
        if notebook.is_symlink():
            errors.append(f"{relative}: symlinked notebooks are not allowed")
            continue
        _, notebook_errors = _read_notebook(notebook)
        errors.extend(f"{relative}: {error}" for error in notebook_errors)
        try:
            hashes[notebook] = _sha256(notebook)
        except OSError as error:
            errors.append(f"{relative}: cannot hash source: {error}")
    return hashes, errors


def _execution_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.fspath(root)
        if not current_pythonpath
        else os.pathsep.join((os.fspath(root), current_pythonpath))
    )
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("PYTHONHASHSEED", "0")
    for variable in (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment.setdefault(variable, "1")
    return environment


def execute_notebooks(
    root: Path,
    notebooks: list[Path],
    output_root: Path,
    *,
    python: str,
    timeout: int,
    kernel_name: str,
) -> int:
    environment = _execution_environment(root)
    for index, notebook in enumerate(notebooks, start=1):
        relative = notebook.relative_to(root)
        destination = output_root / relative.relative_to("notebooks")
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(notebooks)}] executing {relative}", flush=True)
        command = [
            python,
            "-m",
            "nbconvert",
            "--to=notebook",
            "--execute",
            os.fspath(notebook),
            f"--output={destination.name}",
            f"--output-dir={destination.parent}",
            f"--ExecutePreprocessor.timeout={timeout}",
            f"--ExecutePreprocessor.kernel_name={kernel_name}",
        ]
        result = subprocess.run(command, cwd=root, env=environment, check=False)
        if result.returncode != 0:
            print(f"ERROR: execution failed for {relative}", file=sys.stderr)
            return result.returncode or 1
        if not destination.is_file():
            print(
                f"ERROR: notebook executor did not create {destination}",
                file=sys.stderr,
            )
            return 1
    return 0


def _sources_unchanged(root: Path, hashes: dict[Path, str]) -> list[str]:
    errors: list[str] = []
    for path, expected in hashes.items():
        relative = path.relative_to(root)
        try:
            actual = _sha256(path)
        except OSError as error:
            errors.append(f"{relative}: cannot re-read source after execution: {error}")
            continue
        if actual != expected:
            errors.append(f"{relative}: source notebook changed during execution")
    return errors


def _run(args: argparse.Namespace, root: Path, output_root: Path) -> int:
    notebooks = discover_notebooks(root)
    if not notebooks:
        print("ERROR: no notebooks found under notebooks/", file=sys.stderr)
        return 1

    source_hashes, errors = validate_sources(root, notebooks)
    if errors:
        print("notebook source validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    result = execute_notebooks(
        root,
        notebooks,
        output_root,
        python=args.python,
        timeout=args.timeout,
        kernel_name=args.kernel_name,
    )
    unchanged_errors = _sources_unchanged(root, source_hashes)
    if unchanged_errors:
        for error in unchanged_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if result != 0:
        return result

    print(
        f"executed {len(notebooks)} clean notebook(s); results are in {output_root}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter whose Jupyter installation should run kernels",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_integer,
        default=600,
        help="per-notebook execution timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help="Jupyter kernel name (default: python3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="persist executed copies outside the repository (default: temporary directory)",
    )
    args = parser.parse_args(argv)

    root = REPO_ROOT.resolve()
    if args.output_dir is not None:
        output_root = args.output_dir.expanduser().resolve()
        if output_root == root or root in output_root.parents:
            parser.error("--output-dir must be outside the repository")
        output_root.mkdir(parents=True, exist_ok=True)
        return _run(args, root, output_root)

    with tempfile.TemporaryDirectory(prefix="gauge-floors-notebooks-") as temporary:
        return _run(args, root, Path(temporary))


if __name__ == "__main__":
    raise SystemExit(main())
