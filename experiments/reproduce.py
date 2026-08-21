"""Command-line entry point for all generated experiment artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import time
from pathlib import Path
from typing import Any

from . import discrepancy, nuclear
from .common import REPO_ROOT, RESULTS_DIR, environment_manifest, write_json


def _validate_pinned_environment() -> None:
    """Refuse to regenerate artifacts outside the documented package set."""

    requirements_path = REPO_ROOT / "experiments" / "requirements.txt"
    mismatches: list[str] = []
    for line_number, raw_line in enumerate(
        requirements_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        requirement = raw_line.split("#", 1)[0].strip()
        if not requirement:
            continue
        if requirement.count("==") != 1:
            raise RuntimeError(
                f"{requirements_path}:{line_number}: expected an exact == pin"
            )
        name, expected = requirement.split("==", 1)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, found {actual}")

    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            "experiment environment does not match experiments/requirements.txt: "
            f"{details}"
        )


def reproduce(selection: str) -> dict[str, Any]:
    _validate_pinned_environment()
    started = time.perf_counter()
    artifacts: list[Path] = []
    dataset_hashes: dict[str, str] = {}
    outputs: dict[str, Any] = {}

    if selection in ("nuclear", "all"):
        stage = time.perf_counter()
        result = nuclear.run()
        outputs["nuclear"] = result["summary"]
        artifacts.extend(result["artifacts"])
        dataset_hashes.update(result["dataset_hashes"])
        print(f"nuclear experiments: {time.perf_counter() - stage:.2f} s", flush=True)

    if selection in ("discrepancy", "all"):
        stage = time.perf_counter()
        result = discrepancy.run()
        outputs["discrepancy"] = result["summary"]
        artifacts.extend(result["artifacts"])
        dataset_hashes.update(result["dataset_hashes"])
        print(
            f"discrepancy experiments: {time.perf_counter() - stage:.2f} s",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / "manifest.json"
    manifest = environment_manifest(
        artifacts=artifacts,
        dataset_hashes=dataset_hashes,
        command=f"python -m experiments.reproduce {selection}",
    )
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["selection"] = selection
    write_json(manifest_path, manifest)
    print(f"manifest: {manifest_path.relative_to(REPO_ROOT)}", flush=True)
    print(f"total: {manifest['elapsed_seconds']:.2f} s", flush=True)
    return {"outputs": outputs, "manifest": manifest, "manifest_path": manifest_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paper",
        choices=("nuclear", "discrepancy", "all"),
        help="artifact set to regenerate; every selection runs its full protocol",
    )
    args = parser.parse_args()
    reproduce(args.paper)


if __name__ == "__main__":
    main()
