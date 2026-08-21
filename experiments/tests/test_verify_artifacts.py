from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.verify_artifacts import verify_manifest


class VerifyArtifactsTests(unittest.TestCase):
    def test_manifest_accepts_matching_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("stable\n", encoding="utf-8")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"artifact_sha256": {"artifact.txt": digest}}),
                encoding="utf-8",
            )
            with patch("experiments.verify_artifacts.REPO_ROOT", root):
                self.assertEqual(verify_manifest(manifest), [])

    def test_manifest_rejects_mismatch_and_escape(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("changed\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "artifact_sha256": {
                            "artifact.txt": "0" * 64,
                            "../outside.txt": "1" * 64,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch("experiments.verify_artifacts.REPO_ROOT", root):
                errors = verify_manifest(manifest)
            self.assertEqual(len(errors), 2)
            self.assertTrue(any("expected" in error for error in errors))
            self.assertTrue(any("escapes" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
