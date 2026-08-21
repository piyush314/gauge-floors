#!/usr/bin/env python3
"""Check local Markdown and Quarto link targets without network access."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
TEXT_SUFFIXES = {".md", ".qmd"}


def tracked_documents() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = []
    for item in result.stdout.split(b"\0"):
        if not item:
            continue
        path = REPO_ROOT / item.decode("utf-8")
        if path.suffix.casefold() in TEXT_SUFFIXES:
            paths.append(path)
    return sorted(paths)


def local_target(raw: str) -> str | None:
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        raw = raw[1 : raw.index(">")]
    else:
        raw = raw.split(maxsplit=1)[0]
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or raw.startswith(("#", "/")):
        return None
    return unquote(parsed.path)


def main() -> int:
    errors: list[str] = []
    links = 0
    for document in tracked_documents():
        relative_document = document.relative_to(REPO_ROOT)
        text = document.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target_text = local_target(match.group(1))
            if not target_text:
                continue
            links += 1
            target = (document.parent / target_text).resolve()
            if not target.is_relative_to(REPO_ROOT):
                errors.append(f"{relative_document}: link escapes repository: {target_text}")
            elif not target.exists():
                errors.append(f"{relative_document}: missing link target: {target_text}")
    if errors:
        print("local link check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"local link check passed for {links} relative link(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
