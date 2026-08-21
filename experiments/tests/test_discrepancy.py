from __future__ import annotations

import unittest

import numpy as np

from experiments.common import stable_rng
from experiments.discrepancy import (
    CALIBRATION_SIZE,
    CALIBRATION_SUBSETS,
    STATIC_SPLITS,
    _calibration_conditions,
    _digits_split,
    affine_metric_product_mse,
    arbitrary_rounding_finish,
    box_relaxation_start,
    conditional_expectation_finish,
    enumerate_binary_optimum,
    fixed_symmetric_grid,
    low_rank_block,
    metric_product_mse,
    null_space_reduce,
    null_space_walk,
    one_flip_metric_descent,
    optimal_output_bias,
    residual_face_finish,
    same_grid_bernoulli_energy,
)


class DiscrepancyTests(unittest.TestCase):
    def test_exact_enumeration(self) -> None:
        block = np.array([[1.0], [1.0], [1.0], [1.0]])
        theta = np.full(4, 0.5)
        chi, error = enumerate_binary_optimum(block, theta)
        self.assertEqual(error, 0.0)
        self.assertEqual(int(np.sum(chi)), 2)

    def test_null_walk_invariant_and_residual_size(self) -> None:
        rng = stable_rng("test-null-walk")
        block = rng.standard_normal((30, 4))
        theta = rng.uniform(size=30)
        chi, trace = null_space_walk(block, theta)
        self.assertLessEqual(trace["residual_size"], 4)
        self.assertLess(trace["invariant_error"], 1e-9)
        direct = float(np.linalg.norm((chi - theta) @ block) ** 2)
        self.assertAlmostEqual(direct, trace["squared_error"], places=12)

    def test_walk_finishes_are_separate_and_ce_meets_bound(self) -> None:
        rng = stable_rng("test-walk-finishes")
        block = rng.standard_normal((28, 4))
        theta = rng.uniform(size=28)
        x, walk = null_space_reduce(block, theta)
        self.assertLessEqual(walk["residual_size"], 4)
        arbitrary = arbitrary_rounding_finish(x)
        ce_chi, ce = conditional_expectation_finish(block, theta, x)
        face_chi, face_error = residual_face_finish(block, theta, x)
        arbitrary_error = float(np.linalg.norm((arbitrary - theta) @ block) ** 2)
        ce_error = float(np.linalg.norm((ce_chi - theta) @ block) ** 2)
        self.assertLessEqual(ce_error, ce["expected_bound"] + 1e-10)
        self.assertLessEqual(face_error, arbitrary_error + 1e-10)
        self.assertLessEqual(face_error, ce_error + 1e-10)
        self.assertTrue(np.all((face_chi == 0.0) | (face_chi == 1.0)))

    def test_same_grid_bernoulli_reference(self) -> None:
        block = np.ones((4, 1))
        theta = np.full(4, 0.5)
        self.assertAlmostEqual(
            same_grid_bernoulli_energy(block, theta), 1.0, places=14
        )

    def test_block_generator_controls_sensitivity_imbalance(self) -> None:
        balanced = low_rank_block(64, 4, label="test-balanced")
        imbalanced = low_rank_block(
            64, 4, label="test-imbalanced", imbalance_exponent=2.0
        )
        balanced_norms = np.linalg.norm(balanced, axis=1)
        imbalanced_norms = np.linalg.norm(imbalanced, axis=1)
        np.testing.assert_allclose(balanced_norms, np.ones(64), atol=1e-13)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(imbalanced_norms**2))), 1.0, places=13
        )
        self.assertLess(
            float(np.sqrt(np.mean(imbalanced_norms**2)) / np.max(imbalanced_norms)),
            0.2,
        )
        self.assertEqual(np.linalg.matrix_rank(imbalanced), 4)

    def test_affine_box_start_improves_relaxed_objective(self) -> None:
        rng = stable_rng("test-affine-box")
        block = rng.standard_normal((24, 3))
        theta = rng.uniform(size=24)
        fixed_offset = np.array([1.0, -0.7, 0.4])
        x, trace = box_relaxation_start(block, theta, fixed_offset)
        theta_objective = float(fixed_offset @ fixed_offset)
        self.assertLessEqual(trace["objective"], theta_objective + 1e-10)
        self.assertTrue(np.all(x >= 0.0))
        self.assertTrue(np.all(x <= 1.0))

    def test_analytic_metric_matches_direct_product_mse(self) -> None:
        rng = stable_rng("test-metric-identity")
        inputs = rng.standard_normal((41, 7))
        error = rng.standard_normal((7, 3))
        analytic, direct = metric_product_mse(inputs, error)
        self.assertAlmostEqual(analytic, direct, places=12)

    def test_affine_metric_and_optimal_bias_match_centered_metric(self) -> None:
        rng = stable_rng("test-centered-bias-identity")
        inputs = rng.standard_normal((43, 7)) + np.linspace(0.0, 2.0, 7)
        error = rng.standard_normal((7, 3))
        output_bias = optimal_output_bias(inputs, error)
        analytic, direct = affine_metric_product_mse(
            inputs, error, output_bias
        )
        centered = inputs - inputs.mean(axis=0, keepdims=True)
        centered_analytic, centered_direct = metric_product_mse(centered, error)
        self.assertAlmostEqual(analytic, direct, places=12)
        self.assertAlmostEqual(analytic, centered_analytic, places=12)
        self.assertAlmostEqual(direct, centered_direct, places=12)
        np.testing.assert_allclose(
            np.mean(inputs @ error + output_bias[None, :], axis=0),
            np.zeros(error.shape[1]),
            atol=2e-14,
        )

    def test_static_splits_and_calibration_conditions_are_deterministic(self) -> None:
        split_weights = []
        for split_index in range(STATIC_SPLITS):
            (
                weights,
                calibration,
                calibration_targets,
                test,
                _,
                dataset_hash,
                fit_size,
            ) = _digits_split(split_index)
            split_weights.append(weights)
            self.assertEqual(fit_size, 1078)
            self.assertEqual(calibration.shape, (359, 64))
            self.assertEqual(test.shape, (360, 64))
            self.assertEqual(len(dataset_hash), 64)

            conditions = _calibration_conditions(
                calibration_targets, split_index
            )
            self.assertEqual(len(conditions), CALIBRATION_SUBSETS + 1)
            for condition, subset_index, indices, sampling in conditions[:-1]:
                self.assertEqual(condition, "n128")
                self.assertGreaterEqual(subset_index, 0)
                self.assertEqual(sampling, "stratified_without_replacement")
                self.assertEqual(indices.size, CALIBRATION_SIZE)
                self.assertEqual(np.unique(indices).size, CALIBRATION_SIZE)
                class_counts = np.bincount(
                    calibration_targets[indices], minlength=10
                )
                self.assertLessEqual(
                    int(class_counts.max() - class_counts.min()), 1
                )
            condition, subset_index, indices, sampling = conditions[-1]
            self.assertEqual(condition, "full")
            self.assertEqual(subset_index, -1)
            self.assertEqual(sampling, "full_calibration_pool")
            np.testing.assert_array_equal(
                indices, np.arange(calibration.shape[0])
            )

        repeat = _digits_split(0)
        np.testing.assert_array_equal(split_weights[0], repeat[0])
        self.assertTrue(
            any(
                not np.array_equal(split_weights[0], weights)
                for weights in split_weights[1:]
            )
        )

    def test_one_flip_descent_reaches_local_optimum(self) -> None:
        rng = stable_rng("test-one-flip")
        target = rng.uniform(-1.0, 1.0, size=(9, 2))
        lower, upper, theta, variable = fixed_symmetric_grid(target, bits=3)
        factor = rng.standard_normal((20, 9))
        metric = factor.T @ factor / factor.shape[0]
        rounded, _ = one_flip_metric_descent(
            target, lower, upper, theta, variable, metric
        )
        for column in range(target.shape[1]):
            error = rounded[:, column] - target[:, column]
            objective = float(error @ metric @ error)
            for coordinate in np.flatnonzero(variable[:, column]):
                alternative = (
                    upper[coordinate, column]
                    if np.isclose(
                        rounded[coordinate, column],
                        lower[coordinate, column],
                        rtol=0.0,
                        atol=1e-14,
                    )
                    else lower[coordinate, column]
                )
                candidate = error.copy()
                candidate[coordinate] += alternative - rounded[coordinate, column]
                self.assertGreaterEqual(
                    float(candidate @ metric @ candidate),
                    objective - 2e-11,
                )


if __name__ == "__main__":
    unittest.main()
