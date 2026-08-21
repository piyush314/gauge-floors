from __future__ import annotations

import unittest

import numpy as np

from experiments.common import nuclear_norm, random_orthogonal, stable_rng
from experiments.nuclear import (
    _method_metrics,
    balance_transform,
    deterministic_hadamard_rotation,
    diagonal_balance_transform,
    full_gauge_svd_hadamard,
    leading_metrics,
    randomized_hadamard_rotation,
    transformed_factors,
)


class NuclearTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = stable_rng("test-nuclear-factors")
        self.A = rng.standard_normal((12, 8))
        self.B = rng.standard_normal((8, 10))

    def test_balancing_attains_energy_factorization_identity(self) -> None:
        transform = balance_transform(self.A, self.B)
        X, Y = transformed_factors(self.A, self.B, T=transform)
        value = np.sum(X * X) * np.sum(Y * Y)
        target = nuclear_norm(self.A @ self.B) ** 2
        self.assertAlmostEqual(value / target, 1.0, places=9)

    def test_all_implemented_transforms_preserve_product(self) -> None:
        rng = stable_rng("test-nuclear-transforms")
        transforms = (
            np.eye(8),
            diagonal_balance_transform(self.A, self.B),
            balance_transform(self.A, self.B),
        )
        rotations = (
            np.eye(8),
            random_orthogonal(rng, 8),
            randomized_hadamard_rotation(rng, 8),
        )
        reference = self.A @ self.B
        for transform in transforms:
            for rotation in rotations:
                X, Y = transformed_factors(
                    self.A, self.B, T=transform, W=rotation
                )
                np.testing.assert_allclose(
                    X @ Y, reference, rtol=3e-11, atol=3e-11
                )

        X0, Y0 = transformed_factors(
            self.A, self.B, T=balance_transform(self.A, self.B)
        )
        rotation, _ = deterministic_hadamard_rotation(X0, Y0)
        X, Y = transformed_factors(X0, Y0, W=rotation)
        np.testing.assert_allclose(X @ Y, reference, rtol=3e-11, atol=3e-11)

    def test_equal_bit_ratio_is_bit_invariant(self) -> None:
        metrics4 = leading_metrics(self.A, self.B, bits_a=4, bits_b=4)
        metrics8 = leading_metrics(self.A, self.B, bits_a=8, bits_b=8)
        self.assertAlmostEqual(
            metrics4["objective_floor_ratio"],
            metrics8["objective_floor_ratio"],
            places=13,
        )

    def test_full_gauge_svd_hadamard_attains_floor_rectangular(self) -> None:
        rng = stable_rng("test-nuclear-full-gauge")
        left, _ = np.linalg.qr(rng.standard_normal((12, 8)), mode="reduced")
        right, _ = np.linalg.qr(rng.standard_normal((10, 8)), mode="reduced")
        singular_values = np.geomspace(1.0, 0.2, 8)
        root = np.sqrt(singular_values)
        A = left * root[None, :]
        B = root[:, None] * right.T

        X, Y, reconstructed = full_gauge_svd_hadamard(A, B)
        metrics = leading_metrics(X, Y)

        self.assertAlmostEqual(metrics["objective_floor_ratio"], 1.0, places=12)
        self.assertAlmostEqual(metrics["alpha_a"], 1.0, places=12)
        self.assertAlmostEqual(metrics["alpha_b"], 1.0, places=12)
        np.testing.assert_allclose(
            reconstructed,
            A @ B,
            rtol=3e-11,
            atol=3e-11,
        )

    def test_reported_transform_condition_uses_balancing_transform(self) -> None:
        scales = np.geomspace(0.05, 20.0, self.A.shape[1])
        A = self.A @ np.diag(scales)
        B = np.diag(1.0 / scales) @ self.B

        diagonal = _method_metrics(
            A,
            B,
            method="diagonal_hadamard",
            rng=stable_rng("test-nuclear-diagonal-condition"),
        )
        general = _method_metrics(
            A,
            B,
            method="balanced_haar",
            rng=stable_rng("test-nuclear-general-condition"),
        )

        self.assertGreater(diagonal["transform_condition"], 1.0)
        self.assertGreater(general["transform_condition"], 1.0)

    def test_conditional_expectation_meets_certificate(self) -> None:
        transform = balance_transform(self.A, self.B)
        X, Y = transformed_factors(self.A, self.B, T=transform)
        _, certificate = deterministic_hadamard_rotation(X, Y)
        self.assertLessEqual(
            certificate["coherence"], certificate["coherence_bound"] * (1 + 1e-10)
        )


if __name__ == "__main__":
    unittest.main()
