"""Verify the SHA-256 inventory for the committed experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments" / "results" / "manifest.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> list[str]:
    """Return human-readable errors for a manifest, or an empty list."""

    manifest_path = manifest_path.resolve()
    try:
        payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read manifest {manifest_path}: {error}"]

    inventory = payload.get("artifact_sha256")
    if not isinstance(inventory, dict) or not inventory:
        return [f"{manifest_path}: artifact_sha256 must be a nonempty object"]

    root = REPO_ROOT.resolve()
    errors: list[str] = []
    for relative, expected in sorted(inventory.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append(f"{manifest_path}: artifact entries must map strings to strings")
            continue
        if SHA256_RE.fullmatch(expected) is None:
            errors.append(f"{relative}: invalid SHA-256 value {expected!r}")
            continue

        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root):
            errors.append(f"{relative}: path escapes repository root")
            continue
        if not artifact.is_file():
            errors.append(f"{relative}: artifact is missing or is not a file")
            continue

        actual = sha256_file(artifact)
        if actual != expected:
            errors.append(f"{relative}: expected {expected}, found {actual}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest to verify (default: experiments/results/manifest.json)",
    )
    args = parser.parse_args()
    errors = verify_manifest(args.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"artifact verification failed: {len(errors)} error(s)")
        return 1
    inventory = json.loads(args.manifest.read_text(encoding="utf-8"))["artifact_sha256"]
    print(f"verified {len(inventory)} artifact hashes from {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
