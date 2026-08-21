from __future__ import annotations

import unittest

import numpy as np

from experiments.common import (
    normalized_hadamard,
    stable_rng,
    sym_psd_power,
)


class CommonTests(unittest.TestCase):
    def test_stable_rng_streams(self) -> None:
        first = stable_rng("test-stream").standard_normal(8)
        second = stable_rng("test-stream").standard_normal(8)
        np.testing.assert_array_equal(first, second)

    def test_normalized_hadamard_is_orthogonal(self) -> None:
        matrix = normalized_hadamard(16)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(16), atol=1e-14)

    def test_symmetric_psd_power(self) -> None:
        rng = stable_rng("test-psd-power")
        factor = rng.standard_normal((8, 8))
        matrix = factor.T @ factor + 0.2 * np.eye(8)
        root = sym_psd_power(matrix, 0.5)
        inverse_root = sym_psd_power(matrix, -0.5)
        np.testing.assert_allclose(root @ root, matrix, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            inverse_root @ matrix @ inverse_root,
            np.eye(8),
            rtol=2e-12,
            atol=2e-12,
        )

if __name__ == "__main__":
    unittest.main()
