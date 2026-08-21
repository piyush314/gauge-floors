from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.reproduce import _validate_pinned_environment


class ReproduceTests(unittest.TestCase):
    def test_pinned_environment_accepts_documented_versions(self) -> None:
        versions = {
            "numpy": "2.3.5",
            "scipy": "1.17.0",
            "matplotlib": "3.10.8",
            "scikit-learn": "1.8.0",
        }
        with patch(
            "experiments.reproduce.importlib.metadata.version",
            side_effect=versions.__getitem__,
        ):
            _validate_pinned_environment()

    def test_pinned_environment_rejects_mismatch(self) -> None:
        versions = {
            "numpy": "1.26.4",
            "scipy": "1.17.0",
            "matplotlib": "3.10.8",
            "scikit-learn": "1.5.2",
        }
        with patch(
            "experiments.reproduce.importlib.metadata.version",
            side_effect=versions.__getitem__,
        ):
            with self.assertRaisesRegex(RuntimeError, "numpy: expected 2.3.5"):
                _validate_pinned_environment()


if __name__ == "__main__":
    unittest.main()
