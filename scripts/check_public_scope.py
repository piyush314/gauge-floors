#!/usr/bin/env python3
"""Reject files and history that fall outside the public artifact boundary.

The public repository is intentionally limited to reproducibility code and
data, tutorials, notebooks, documentation, and project metadata.  The check
uses Git's index when available and also examines every path reachable from a
ref, so deleting a manuscript file in a later commit is not sufficient to make
that history publishable.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    ".python-version",
    ".zenodo.json",
    "CITATION.cff",
    "CITATION.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "LICENSE.md",
    "Makefile",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "environment.yml",
    "pyproject.toml",
    "requirements-lock.txt",
    "requirements.txt",
}

ALLOWED_ROOT_DIRECTORIES = {
    ".github",
    "LICENSES",
    "assets",
    "data",
    "docs",
    "experiments",
    "figures",
    "notebooks",
    "scripts",
    "tutorials",
}

ALLOWED_SUFFIXES = {
    ".bat",
    ".cff",
    ".cfg",
    ".csv",
    ".css",
    ".feather",
    ".gif",
    ".h5",
    ".hdf5",
    ".html",
    ".ico",
    ".ini",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".npy",
    ".npz",
    ".otf",
    ".parquet",
    ".png",
    ".ps1",
    ".py",
    ".pyi",
    ".qmd",
    ".scss",
    ".sh",
    ".svg",
    ".toml",
    ".tsv",
    ".ttf",
    ".txt",
    ".webp",
    ".woff",
    ".woff2",
    ".yaml",
    ".yml",
}

ALLOWED_EXTENSIONLESS_NAMES = {
    ".gitkeep",
    ".nojekyll",
    "CODEOWNERS",
    "LICENSE",
    "Makefile",
    "NOTICE",
}

FORBIDDEN_SUFFIXES = {
    # Manuscript and bibliography source.
    ".bbx",
    ".bib",
    ".bibtex",
    ".bst",
    ".cbx",
    ".cls",
    ".dtx",
    ".ins",
    ".lbx",
    ".ltx",
    ".sty",
    ".tex",
    # Submission bundles, office documents, and presentation sources.
    ".7z",
    ".bz2",
    ".doc",
    ".docx",
    ".eps",
    ".gz",
    ".key",
    ".odt",
    ".pages",
    ".pdf",
    ".ppt",
    ".pptx",
    ".ps",
    ".rar",
    ".rtf",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
}

FORBIDDEN_TOP_LEVEL_DIRECTORIES = {
    "arxiv",
    "manuscript",
    "manuscripts",
    "paper",
    "papers",
    "sec",
    "sections",
    "slides",
    "src",
    "style",
    "tex",
}

PRIVATE_PATH_TOKEN = re.compile(
    r"(?:^|[-_.])(?:"
    r"arxiv|camera[-_]?ready|deadline|manuscripts?|papers?|"
    r"private[-_]?review|rebuttals?|responses?[-_]?to[-_]?reviewers?|"
    r"reviews?|revisions?|submission|submissions|tracked[-_]?changes|"
    r"writing[-_]?audit"
    r")(?:$|[-_.])",
    re.IGNORECASE,
)

TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".ps1",
    ".py",
    ".pyi",
    ".qmd",
    ".scss",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

CONTENT_PATTERNS = (
    ("LaTeX document preamble", re.compile(r"\\documentclass(?:\[|\{)")),
    ("LaTeX document body", re.compile(r"\\begin\{document\}")),
    ("LaTeX bibliography source", re.compile(r"\\(?:addbibresource|bibliography)\{")),
    (
        "LaTeX manuscript/table body",
        re.compile(r"\\begin\{(?:abstract|tabular\*?|thebibliography)\}"),
    ),
    (
        "private manuscript input",
        re.compile(r"\\(?:include|input)\{(?:sec|sections|src|tex)/", re.IGNORECASE),
    ),
    ("TeX-generation command-line switch", re.compile(r"--with[-_]tex\b")),
    (
        "TeX-generation implementation",
        re.compile(r"\b(?:_write_tex|validate_tex_table_math)\b"),
    ),
    (
        "private generated-source path",
        re.compile(r"\btex/(?:discrepancy|nuclear)/generated(?:/|\b)"),
    ),
    (
        "private manuscript filename",
        re.compile(
            r"\b(?:main(?:_[a-z0-9-]+)?\.tex|references\.bib|cup-journal\.cls)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "source-only Git attribute",
        re.compile(
            r"(?m)^\s*(?:[^\s]*/)?\*?\.(?:bib|cls|docx|pptx|sty|tex)[ \t]+",
            re.IGNORECASE,
        ),
    ),
)

CONTENT_SCAN_EXEMPT = {"scripts/check_public_scope.py"}

FALLBACK_IGNORED_DIRECTORIES = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".quarto",
    ".ruff_cache",
    "__pycache__",
    "_site",
    "htmlcov",
}


@dataclass(frozen=True)
class Inventory:
    paths: tuple[str, ...]
    modes: dict[str, str]
    source: str
    git_repository: bool


class GitError(RuntimeError):
    """Raised when an expected read-only Git command fails."""


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise GitError("git executable was not found") from error
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise GitError(f"git {' '.join(arguments)} failed: {detail}")
    return result


def _is_repository_root(root: Path) -> bool:
    try:
        result = _git(root, "rev-parse", "--show-toplevel", check=False)
    except GitError:
        return False
    if result.returncode != 0:
        return False
    top_level = Path(os.fsdecode(result.stdout).strip()).resolve()
    return top_level == root.resolve()


def _fallback_paths(root: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if any(
            part in FALLBACK_IGNORED_DIRECTORIES or part.startswith(".venv")
            for part in relative.parts[:-1]
        ):
            continue
        if candidate.is_file() or candidate.is_symlink():
            paths.append(relative.as_posix())
    return tuple(sorted(set(paths)))


def inventory(root: Path) -> Inventory:
    if not _is_repository_root(root):
        return Inventory(_fallback_paths(root), {}, "source tree (Git unavailable)", False)

    raw = _git(root, "ls-files", "--stage", "-z").stdout
    modes: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode = metadata.split(b" ", 1)[0].decode("ascii")
        except (ValueError, UnicodeDecodeError) as error:
            raise GitError("could not parse git ls-files output") from error
        path = os.fsdecode(encoded_path)
        modes[path] = mode

    if modes:
        return Inventory(tuple(sorted(modes)), modes, "Git index", True)
    return Inventory(
        _fallback_paths(root),
        {},
        "source tree (Git index is empty)",
        True,
    )


def _path_problems(path: str, *, mode: str | None = None) -> list[str]:
    problems: list[str] = []
    if mode is not None and mode not in {"100644", "100755"}:
        problems.append(f"unsupported Git mode {mode}; symlinks and submodules are not public")

    if "\\" in path:
        problems.append("uses a non-portable backslash path separator")
        return problems

    pure = PurePosixPath(path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        problems.append("is not a safe repository-relative path")
        return problems

    top = pure.parts[0]
    approved_root_file = len(pure.parts) == 1 and path in ALLOWED_ROOT_FILES
    if len(pure.parts) == 1:
        if not approved_root_file:
            problems.append("is not an approved root metadata file")
    elif top not in ALLOWED_ROOT_DIRECTORIES:
        problems.append(f"top-level directory {top!r} is outside the public allowlist")

    if top.casefold() in FORBIDDEN_TOP_LEVEL_DIRECTORIES:
        problems.append("uses a manuscript-source top-level directory")

    suffixes = {suffix.casefold() for suffix in pure.suffixes}
    forbidden = sorted(suffixes & FORBIDDEN_SUFFIXES)
    if forbidden:
        problems.append(f"uses forbidden extension {forbidden[-1]}")
    elif (
        not approved_root_file
        and pure.suffix.casefold() not in ALLOWED_SUFFIXES
        and pure.name not in ALLOWED_EXTENSIONLESS_NAMES
    ):
        problems.append("uses an unapproved file extension")

    if pure.suffix.casefold() == ".pdf" and top != "figures":
        problems.append("PDFs are allowed only as computational figure artifacts")

    for part in pure.parts:
        if PRIVATE_PATH_TOKEN.search(part):
            problems.append(f"contains private/history path indicator {part!r}")
            break
    return problems


def check_paths(paths: Iterable[str], modes: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for path in sorted(set(paths)):
        for problem in _path_problems(path, mode=modes.get(path)):
            errors.append(f"{path}: {problem}")
    return errors


def check_content(root: Path, paths: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for relative in sorted(set(paths)):
        if relative in CONTENT_SCAN_EXEMPT:
            continue
        pure = PurePosixPath(relative)
        if pure.suffix.casefold() not in TEXT_SUFFIXES and pure.name not in {
            ".gitattributes",
            ".gitignore",
            "Makefile",
        }:
            continue
        path = root / pure
        if path.is_symlink() or not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError as error:
            errors.append(f"{relative}: cannot inspect content: {error}")
            continue
        if b"\0" in data:
            errors.append(f"{relative}: expected text content but found NUL bytes")
            continue
        text = data.decode("utf-8", "replace")
        for label, pattern in CONTENT_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            line = text.count("\n", 0, match.start()) + 1
            errors.append(f"{relative}:{line}: contains {label}")
    return errors


def history_paths(root: Path) -> tuple[tuple[str, ...], list[str]]:
    errors: list[str] = []
    shallow = _git(root, "rev-parse", "--is-shallow-repository").stdout.strip()
    if shallow == b"true":
        errors.append(
            "Git checkout is shallow; fetch the complete history before validating public scope"
        )

    raw_paths = _git(
        root,
        "log",
        "--all",
        "--format=",
        "--name-only",
        "-z",
        "--diff-filter=ACDMRTUXB",
    ).stdout
    paths = tuple(
        sorted(
            {
                os.fsdecode(record).lstrip("\n")
                for record in raw_paths.split(b"\0")
                if record.lstrip(b"\n")
            }
        )
    )

    raw_refs = _git(root, "for-each-ref", "--format=%(refname)").stdout
    for encoded_ref in raw_refs.splitlines():
        ref = os.fsdecode(encoded_ref)
        for component in PurePosixPath(ref).parts:
            if component in {"original", "replace"} or PRIVATE_PATH_TOKEN.search(component):
                errors.append(f"Git ref {ref!r} contains a private/history indicator")
                break
    return paths, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"repository root is not a directory: {root}")

    try:
        current = inventory(root)
        errors = check_paths(current.paths, current.modes)
        errors.extend(check_content(root, current.paths))

        historical: tuple[str, ...] = ()
        if current.git_repository:
            historical, history_errors = history_paths(root)
            errors.extend(history_errors)
            current_set = set(current.paths)
            historical_only = (path for path in historical if path not in current_set)
            errors.extend(
                f"historical path {error}"
                for error in check_paths(historical_only, {})
            )
    except GitError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if errors:
        print("public-scope check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            f"Found {len(errors)} violation(s). Create the public repository from a clean "
            "export; deleting private files in a later commit does not clean its history.",
            file=sys.stderr,
        )
        return 1

    history_note = (
        f" and {len(historical)} unique reachable historical paths"
        if current.git_repository
        else " (no Git history present)"
    )
    print(
        f"public-scope check passed for {len(current.paths)} files from "
        f"{current.source}{history_note}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
