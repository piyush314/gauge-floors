"""Experiments for the nuclear-norm gauge-floor study."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .common import (
    BASE_SEED,
    REPO_ROOT,
    RESULTS_DIR,
    bootstrap_median_ci,
    configure_matplotlib,
    environment_manifest,
    normalized_hadamard,
    nuclear_norm,
    quantiles,
    random_orthogonal,
    save_figure,
    stable_rng,
    sym_psd_power,
    write_csv,
    write_json,
)

DIMENSION = 64
TRIALS = 50
FLATTENING_DRAWS = 100
BITS = (4, 8)
SINGULAR_RANK = 32
SINGULAR_EPSILONS = np.logspace(-1, -10, 10)
DIMENSION_SCALING_DIMENSIONS = (16, 32, 64, 128, 256, 1024)
DIMENSION_SCALING_TRIALS = 8


def bit_noise_constant(bits: int) -> float:
    q = 2 ** (bits - 1) - 1
    return 1.0 / (12.0 * q * q)


def balance_transform(A: np.ndarray, B: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """Return the symmetric Gramian-balancing contraction transform."""

    k = A.shape[1]
    P = A.T @ A + float(epsilon) * np.eye(k)
    Q = B @ B.T + float(epsilon) * np.eye(k)
    scale = max(float(np.trace(P) / k), 1.0)
    eigen_floor = 2.0e-14 * scale
    P_half = sym_psd_power(P, 0.5, floor=eigen_floor)
    P_inv_half = sym_psd_power(P, -0.5, floor=eigen_floor)
    middle = P_half @ Q @ P_half
    middle_half = sym_psd_power(middle, 0.5, floor=0.0)
    S = P_inv_half @ middle_half @ P_inv_half
    return sym_psd_power(S, 0.5, floor=eigen_floor)


def diagonal_balance_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Minimize the energy surrogate over positive diagonal gauges."""

    p = np.sum(A * A, axis=0)
    q = np.sum(B * B, axis=1)
    tiny = np.finfo(np.float64).tiny
    scales = np.power(np.maximum(q, tiny) / np.maximum(p, tiny), 0.25)
    scales /= math.exp(float(np.mean(np.log(scales))))
    return np.diag(scales)


def transformed_factors(
    A: np.ndarray,
    B: np.ndarray,
    T: np.ndarray | None = None,
    W: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a product-preserving contraction gauge and orthogonal rotation."""

    k = A.shape[1]
    if T is None:
        T = np.eye(k)
    X = A @ T
    Y = np.linalg.solve(T, B)
    if W is not None:
        X = X @ W
        Y = W.T @ Y
    return X, Y


def _full_gauge_flat_factors(
    singular_values: np.ndarray,
    *,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the SVD-aligned, Hadamard-flat factors in outer coordinates."""

    singular_values = np.asarray(singular_values, dtype=np.float64)
    k = singular_values.size
    if rows < k or columns < k:
        raise ValueError("full-rank full-gauge construction requires m,n >= K")
    if np.any(singular_values <= 0.0):
        raise ValueError("full-gauge baseline requires a full-rank product")
    H = normalized_hadamard(k)
    root = np.sqrt(singular_values)
    X = np.zeros((rows, k), dtype=np.float64)
    Y = np.zeros((k, columns), dtype=np.float64)
    X[:k, :] = root[:, None] * H
    Y[:, :k] = H.T * root[None, :]
    return X, Y


def full_gauge_svd_hadamard(
    A: np.ndarray, B: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the exact full-gauge baseline and undo its outer rotations.

    The returned ``X`` and ``Y`` are the factors in the SVD-aligned outer
    coordinates in which they are quantized. ``reconstructed`` applies the
    inverse outer rotations to ``X @ Y`` and therefore lives in the original
    coordinates of ``A @ B``.
    """

    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("A and B must be compatible matrices")
    k = A.shape[1]
    product = A @ B
    U, singular_values, Vt = np.linalg.svd(product, full_matrices=False)
    if singular_values.size < k:
        raise ValueError("full-rank full-gauge construction requires m,n >= K")
    singular_values = singular_values[:k]
    tolerance = (
        max(product.shape)
        * np.finfo(np.float64).eps
        * max(float(singular_values[0]), 1.0)
    )
    if float(singular_values[-1]) <= tolerance:
        raise ValueError("full-gauge baseline requires rank(A @ B) = K")
    X, Y = _full_gauge_flat_factors(
        singular_values,
        rows=A.shape[0],
        columns=B.shape[1],
    )
    reconstructed = U[:, :k] @ (X[:k, :] @ Y[:, :k]) @ Vt[:k, :]
    return X, Y, reconstructed


def leading_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    bits_a: int = 4,
    bits_b: int = 4,
    output_nuclear: float | None = None,
) -> dict[str, float]:
    """Compute the range objective, output floor, and coherence factors."""

    c_a = bit_noise_constant(bits_a)
    c_b = bit_noise_constant(bits_b)
    k = X.shape[1]
    a_ranges_sq = np.max(np.abs(X), axis=1) ** 2
    b_ranges_sq = np.max(np.abs(Y), axis=0) ** 2
    x_energy = float(np.sum(X * X))
    y_energy = float(np.sum(Y * Y))
    objective = c_a * y_energy * float(a_ranges_sq.sum())
    objective += c_b * x_energy * float(b_ranges_sq.sum())
    product_nuclear = (
        nuclear_norm(X @ Y)
        if output_nuclear is None
        else float(output_nuclear)
    )
    floor = (c_a + c_b) * product_nuclear**2 / k
    alpha_a = k * float(a_ranges_sq.sum()) / x_energy
    alpha_b = k * float(b_ranges_sq.sum()) / y_energy
    alpha_effective = (c_a * alpha_a + c_b * alpha_b) / (c_a + c_b)
    energy_ratio = x_energy * y_energy / (product_nuclear**2)
    uncertainty_ratio = alpha_effective * energy_ratio
    return {
        "objective": objective,
        "floor": floor,
        "objective_floor_ratio": objective / floor,
        "alpha_a": alpha_a,
        "alpha_b": alpha_b,
        "alpha_effective": alpha_effective,
        "energy_nuclear_ratio": energy_ratio,
        "uncertainty_identity_residual": abs(
            objective / floor - uncertainty_ratio
        ),
        "product_nuclear": product_nuclear,
    }


def finite_set_coherence(X: np.ndarray, Y: np.ndarray) -> float:
    vectors = np.vstack((X, Y.T))
    norms_sq = np.sum(vectors * vectors, axis=1)
    nonzero = norms_sq > 0.0
    ratios = X.shape[1] * np.max(np.abs(vectors[nonzero]), axis=1) ** 2
    ratios /= norms_sq[nonzero]
    return float(np.max(ratios))


def deterministic_hadamard_rotation(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Derandomize one shared Hadamard sign vector by conditional expectation."""

    k = X.shape[1]
    H = normalized_hadamard(k)
    vectors = np.vstack((X, Y.T))
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 0.0]
    norms = norms[norms > 0.0]
    normalized = vectors / norms[:, None]
    count = normalized.shape[0]
    tau = math.sqrt(2.0 * math.log(4.0 * count * k) / k)
    lam = k * tau

    # coeff[j, output_coordinate, sign_coordinate]
    coeff = normalized[:, None, :] * H[None, :, :]
    scaled = lam * coeff
    log_cosh = np.logaddexp(scaled, -scaled) - math.log(2.0)
    remaining = np.sum(log_cosh, axis=2)
    fixed = np.zeros((count, k), dtype=np.float64)
    signs = np.ones(k, dtype=np.float64)

    for coordinate in range(k):
        remaining -= log_cosh[:, :, coordinate]
        increment = scaled[:, :, coordinate]
        candidate_logs = []
        for sign in (1.0, -1.0):
            signed_fixed = fixed + sign * increment
            terms = np.concatenate(
                (
                    (-lam * tau + remaining + signed_fixed).ravel(),
                    (-lam * tau + remaining - signed_fixed).ravel(),
                )
            )
            candidate_logs.append(float(logsumexp(terms)))
        signs[coordinate] = 1.0 if candidate_logs[0] <= candidate_logs[1] else -1.0
        fixed += signs[coordinate] * increment

    W = np.diag(signs) @ H.T
    X_flat, Y_flat = transformed_factors(X, Y, W=W)
    achieved = finite_set_coherence(X_flat, Y_flat)
    theorem_bound = k * tau * tau
    if achieved > theorem_bound * (1.0 + 5.0e-11):
        raise RuntimeError(
            f"conditional-expectation certificate failed: {achieved} > {theorem_bound}"
        )
    return W, {
        "coherence": achieved,
        "coherence_bound": theorem_bound,
        "tau": tau,
    }


def randomized_hadamard_rotation(rng: np.random.Generator, k: int) -> np.ndarray:
    signs = rng.choice(np.array([-1.0, 1.0]), size=k)
    return np.diag(signs) @ normalized_hadamard(k).T


def _base_factors(
    rng: np.random.Generator, family: str
) -> tuple[np.ndarray, np.ndarray]:
    k = DIMENSION
    singular_values = np.geomspace(1.0, 0.15, k)
    root = np.sqrt(singular_values)

    if family == "spiky":
        A0 = np.diag(root)
        B0 = np.diag(root)
        # The output A0 B0 is fixed across trials. Each factorization receives
        # an independently sampled diagonal gauge with eight deliberately
        # amplified log-scale coordinates. Centering precedes clipping, so no
        # determinant normalization is claimed or needed.
        log_scales = rng.normal(0.0, 1.35, size=k)
        spike_coordinates = rng.choice(k, size=8, replace=False)
        log_scales[spike_coordinates] += rng.choice(
            np.array([-2.25, 2.25]), size=spike_coordinates.size
        )
        log_scales -= float(np.mean(log_scales))
        log_scales = np.clip(log_scales, -math.log(25.0), math.log(25.0))
        scales = np.exp(log_scales)
        G = np.diag(scales)
    else:
        U = random_orthogonal(rng, k)
        V = random_orthogonal(rng, k)
        R = random_orthogonal(rng, k)
        A0 = U @ np.diag(root) @ R.T
        B0 = R @ np.diag(root) @ V.T
        if family == "dense_diagonal":
            scales = np.exp(rng.normal(0.0, 1.35, size=k))
            G = np.diag(scales)
        elif family == "general_gauge":
            Q = random_orthogonal(rng, k)
            scales = np.exp(np.linspace(-2.0, 2.0, k))
            G = Q @ np.diag(scales) @ Q.T
        else:
            raise ValueError(f"unknown family: {family}")
    return A0 @ G, np.linalg.solve(G, B0)


def _method_metrics(
    A: np.ndarray,
    B: np.ndarray,
    *,
    method: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    reference = A @ B
    reference_scale = max(float(np.linalg.norm(reference, ord="fro")), 1.0)
    if method == "full_gauge_svd_hadamard":
        X, Y, reconstructed = full_gauge_svd_hadamard(A, B)
        metrics = leading_metrics(X, Y)
        metrics_bits8 = leading_metrics(X, Y, bits_a=8, bits_b=8)
        metrics.update(
            {
                "finite_set_coherence": finite_set_coherence(X, Y),
                "coherence_bound": math.nan,
                "transform_condition": math.nan,
                "objective_floor_ratio_bits8": metrics_bits8[
                    "objective_floor_ratio"
                ],
                "equal_bit_ratio_difference": abs(
                    metrics["objective_floor_ratio"]
                    - metrics_bits8["objective_floor_ratio"]
                ),
                "product_residual": float(
                    np.linalg.norm(reconstructed - reference, ord="fro")
                ),
                "relative_product_residual": float(
                    np.linalg.norm(reconstructed - reference, ord="fro")
                    / reference_scale
                ),
            }
        )
        return metrics
    if method == "identity":
        X, Y = transformed_factors(A, B)
        transform_condition = 1.0
    elif method == "diagonal_hadamard":
        T = diagonal_balance_transform(A, B)
        X0, Y0 = transformed_factors(A, B, T=T)
        W = randomized_hadamard_rotation(rng, A.shape[1])
        X, Y = transformed_factors(X0, Y0, W=W)
        transform_condition = float(np.linalg.cond(T))
    elif method == "balanced_haar":
        T = balance_transform(A, B)
        X0, Y0 = transformed_factors(A, B, T=T)
        W = random_orthogonal(rng, A.shape[1])
        X, Y = transformed_factors(X0, Y0, W=W)
        transform_condition = float(np.linalg.cond(T))
    elif method == "balanced_deterministic":
        T = balance_transform(A, B)
        X0, Y0 = transformed_factors(A, B, T=T)
        W, certificate = deterministic_hadamard_rotation(X0, Y0)
        X, Y = transformed_factors(X0, Y0, W=W)
        metrics = leading_metrics(X, Y)
        metrics_bits8 = leading_metrics(X, Y, bits_a=8, bits_b=8)
        metrics.update(
            {
                "finite_set_coherence": certificate["coherence"],
                "coherence_bound": certificate["coherence_bound"],
                "transform_condition": float(np.linalg.cond(T)),
                "objective_floor_ratio_bits8": metrics_bits8[
                    "objective_floor_ratio"
                ],
                "equal_bit_ratio_difference": abs(
                    metrics["objective_floor_ratio"]
                    - metrics_bits8["objective_floor_ratio"]
                ),
                "product_residual": float(
                    np.linalg.norm(X @ Y - reference, ord="fro")
                ),
                "relative_product_residual": float(
                    np.linalg.norm(X @ Y - reference, ord="fro")
                    / reference_scale
                ),
            }
        )
        return metrics
    else:
        raise ValueError(method)
    metrics = leading_metrics(X, Y)
    metrics_bits8 = leading_metrics(X, Y, bits_a=8, bits_b=8)
    metrics.update(
        {
            "finite_set_coherence": finite_set_coherence(X, Y),
            "coherence_bound": math.nan,
            "transform_condition": transform_condition,
            "objective_floor_ratio_bits8": metrics_bits8["objective_floor_ratio"],
            "equal_bit_ratio_difference": abs(
                metrics["objective_floor_ratio"]
                - metrics_bits8["objective_floor_ratio"]
            ),
            "product_residual": float(np.linalg.norm(X @ Y - reference, ord="fro")),
            "relative_product_residual": float(
                np.linalg.norm(X @ Y - reference, ord="fro") / reference_scale
            ),
        }
    )
    return metrics


def _run_family_trials() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    methods = (
        "full_gauge_svd_hadamard",
        "identity",
        "diagonal_hadamard",
        "balanced_haar",
        "balanced_deterministic",
    )
    for family in ("spiky", "dense_diagonal", "general_gauge"):
        for trial in range(TRIALS):
            stream = f"nuclear:{family}:trial:{trial}"
            A, B = _base_factors(stable_rng(stream), family)
            for method in methods:
                metrics = _method_metrics(
                    A,
                    B,
                    method=method,
                    rng=stable_rng(f"{stream}:method:{method}"),
                )
                records.append(
                    {
                        "experiment": "family",
                        "family": family,
                        "trial": trial,
                        "method": method,
                        "K": DIMENSION,
                        "bits_a": 4,
                        "bits_b": 4,
                        **metrics,
                    }
                )
    return records


def _same_output_trials() -> tuple[list[dict[str, Any]], dict[str, float]]:
    records: list[dict[str, Any]] = []
    maximum_floor_spread = 0.0
    for trial in range(TRIALS):
        rng = stable_rng(f"nuclear:same-output:{trial}")
        k = DIMENSION
        U = random_orthogonal(rng, k)
        V = random_orthogonal(rng, k)
        R = random_orthogonal(rng, k)
        root = np.sqrt(np.geomspace(1.0, 0.15, k))
        A0 = U @ np.diag(root) @ R.T
        B0 = R @ np.diag(root) @ V.T
        gauges = {
            "identity": np.eye(k),
            "diagonal": np.diag(np.exp(rng.normal(0.0, 1.4, size=k))),
            "orthogonal": random_orthogonal(rng, k),
        }
        Q = random_orthogonal(rng, k)
        gauges["dense_spd"] = Q @ np.diag(np.exp(np.linspace(-2.0, 2.0, k))) @ Q.T
        floors = []
        for gauge_name, G in gauges.items():
            A = A0 @ G
            B = np.linalg.solve(G, B0)
            for method in ("identity", "balanced_deterministic"):
                metrics = _method_metrics(A, B, method=method, rng=rng)
                floors.append(metrics["floor"])
                records.append(
                    {
                        "experiment": "same_output",
                        "family": "same_output",
                        "trial": trial,
                        "factorization": gauge_name,
                        "method": method,
                        "K": k,
                        "bits_a": 4,
                        "bits_b": 4,
                        **metrics,
                    }
                )
        spread = (max(floors) - min(floors)) / np.mean(floors)
        maximum_floor_spread = max(maximum_floor_spread, float(spread))
    return records, {"maximum_relative_floor_spread": maximum_floor_spread}


def _run_singular_sweep() -> list[dict[str, Any]]:
    rng = stable_rng("nuclear:singular")
    k = DIMENSION
    r = SINGULAR_RANK
    U = random_orthogonal(rng, k)[:, :r]
    V = random_orthogonal(rng, k)[:, :r]
    values = np.sqrt(np.geomspace(1.0, 0.2, r))
    R_a = np.eye(k)[:, :r]
    R_b = np.eye(k)[:, 16 : 16 + r]
    A = U @ np.diag(values) @ R_a.T
    B = R_b @ np.diag(values) @ V.T
    reference = A @ B
    reference_scale = max(float(np.linalg.norm(reference, ord="fro")), 1.0)
    target = nuclear_norm(reference) ** 2
    rows = []
    for epsilon in SINGULAR_EPSILONS:
        T = balance_transform(A, B, epsilon=float(epsilon))
        X, Y = transformed_factors(A, B, T=T)
        energy_product = float(np.sum(X * X) * np.sum(Y * Y))
        rows.append(
            {
                "experiment": "singular_sweep",
                "epsilon": float(epsilon),
                "condition_T": float(np.linalg.cond(T)),
                "energy_product": energy_product,
                "nuclear_squared": target,
                "relative_excess": energy_product / target - 1.0,
                "product_residual": float(
                    np.linalg.norm(X @ Y - reference, ord="fro")
                ),
                "relative_product_residual": float(
                    np.linalg.norm(X @ Y - reference, ord="fro")
                    / reference_scale
                ),
            }
        )
    return rows


def _random_stiefel(
    rng: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    """Sample an orthonormal ``rows``-by-``columns`` frame by Gaussian QR."""

    if rows < columns:
        raise ValueError("a Stiefel frame requires rows >= columns")
    frame, upper = np.linalg.qr(
        rng.standard_normal((rows, columns)),
        mode="reduced",
    )
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    return frame * signs


def _run_dimension_scaling() -> list[dict[str, Any]]:
    """Compare inner flattening with the constant full-gauge baseline."""

    rows: list[dict[str, Any]] = []
    for k in DIMENSION_SCALING_DIMENSIONS:
        m = 3 * k // 2
        n = 5 * k // 4
        singular_values = np.geomspace(1.0, 0.15, k)
        root = np.sqrt(singular_values)
        product_nuclear = float(np.sum(singular_values))
        for trial in range(DIMENSION_SCALING_TRIALS):
            stream = f"nuclear:dimension-scaling:K={k}:trial={trial}"
            factor_rng = stable_rng(stream)
            U = _random_stiefel(factor_rng, m, k)
            V = _random_stiefel(factor_rng, n, k)
            X0 = U * root[None, :]
            Y0 = root[:, None] * V.T

            for method in (
                "full_gauge_svd_hadamard",
                "balanced_haar",
                "balanced_randomized_hadamard",
            ):
                method_rng = stable_rng(f"{stream}:method:{method}")
                if method == "full_gauge_svd_hadamard":
                    X, Y = _full_gauge_flat_factors(
                        singular_values,
                        rows=m,
                        columns=n,
                    )
                elif method == "balanced_haar":
                    W = random_orthogonal(method_rng, k)
                    X, Y = transformed_factors(X0, Y0, W=W)
                else:
                    W = randomized_hadamard_rotation(method_rng, k)
                    X, Y = transformed_factors(X0, Y0, W=W)
                metrics = leading_metrics(
                    X,
                    Y,
                    output_nuclear=product_nuclear,
                )
                rows.append(
                    {
                        "experiment": "dimension_scaling",
                        "trial": trial,
                        "method": method,
                        "m": m,
                        "n": n,
                        "K": k,
                        "objective_floor_ratio": metrics[
                            "objective_floor_ratio"
                        ],
                        "alpha_effective": metrics["alpha_effective"],
                        "energy_nuclear_ratio": metrics[
                            "energy_nuclear_ratio"
                        ],
                        "finite_set_coherence": finite_set_coherence(X, Y),
                    }
                )
    return rows


def _validate_run(
    family_rows: list[dict[str, Any]],
    same_rows: list[dict[str, Any]],
    same_diagnostics: dict[str, float],
    singular_rows: list[dict[str, Any]],
    flatten_rows: list[dict[str, Any]],
    dimension_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Fail fast when a generated artifact violates a theorem-linked invariant."""

    factor_rows = family_rows + same_rows
    minimum_ratio = float(
        min(row["objective_floor_ratio"] for row in factor_rows)
    )
    maximum_product_residual = float(
        max(
            [row["relative_product_residual"] for row in factor_rows]
            + [row["relative_product_residual"] for row in singular_rows]
        )
    )
    maximum_bit_difference = float(
        max(
            row["equal_bit_ratio_difference"]
            / max(abs(row["objective_floor_ratio"]), 1.0)
            for row in factor_rows
        )
    )
    maximum_uncertainty_residual = float(
        max(
            row["uncertainty_identity_residual"]
            / max(abs(row["objective_floor_ratio"]), 1.0)
            for row in factor_rows
        )
    )
    certified_rows = [
        row
        for row in factor_rows + flatten_rows
        if "coherence_bound" in row
        and math.isfinite(float(row["coherence_bound"]))
    ]
    maximum_coherence_ratio = float(
        max(
            row["finite_set_coherence"] / row["coherence_bound"]
            for row in certified_rows
        )
    )
    maximum_floor_spread = float(
        same_diagnostics["maximum_relative_floor_spread"]
    )
    full_gauge_rows = [
        row
        for row in family_rows + flatten_rows + dimension_rows
        if row["method"] == "full_gauge_svd_hadamard"
    ]
    maximum_full_gauge_floor_deviation = float(
        max(abs(row["objective_floor_ratio"] - 1.0) for row in full_gauge_rows)
    )

    tolerance = 5.0e-10
    if minimum_ratio < 1.0 - tolerance:
        raise RuntimeError(
            f"nuclear-floor invariant failed: minimum ratio {minimum_ratio}"
        )
    if maximum_product_residual > tolerance:
        raise RuntimeError(
            "product-preservation invariant failed: "
            f"maximum relative residual {maximum_product_residual}"
        )
    if maximum_bit_difference > tolerance:
        raise RuntimeError(
            "equal-bit ratio cancellation failed: "
            f"maximum relative difference {maximum_bit_difference}"
        )
    if maximum_uncertainty_residual > tolerance:
        raise RuntimeError(
            "uncertainty identity failed: "
            f"maximum relative residual {maximum_uncertainty_residual}"
        )
    if maximum_coherence_ratio > 1.0 + tolerance:
        raise RuntimeError(
            "deterministic coherence certificate failed: "
            f"maximum achieved/bound ratio {maximum_coherence_ratio}"
        )
    if maximum_floor_spread > tolerance:
        raise RuntimeError(
            "same-output floor invariant failed: "
            f"maximum relative spread {maximum_floor_spread}"
        )
    if maximum_full_gauge_floor_deviation > tolerance:
        raise RuntimeError(
            "full-gauge Hadamard baseline failed: "
            f"maximum floor-ratio deviation {maximum_full_gauge_floor_deviation}"
        )

    return {
        "minimum_objective_floor_ratio": minimum_ratio,
        "maximum_relative_product_residual": maximum_product_residual,
        "maximum_relative_4bit_8bit_ratio_difference": maximum_bit_difference,
        "maximum_relative_uncertainty_identity_residual": (
            maximum_uncertainty_residual
        ),
        "maximum_deterministic_coherence_to_bound_ratio": (
            maximum_coherence_ratio
        ),
        "maximum_same_output_relative_floor_spread": maximum_floor_spread,
        "maximum_full_gauge_floor_ratio_deviation": (
            maximum_full_gauge_floor_deviation
        ),
        "validation_tolerance": tolerance,
    }


def _run_flattening_comparison() -> list[dict[str, Any]]:
    rng_instance = stable_rng("nuclear:flattening:instance")
    A, B = _base_factors(rng_instance, "general_gauge")
    T = balance_transform(A, B)
    X0, Y0 = transformed_factors(A, B, T=T)
    records: list[dict[str, Any]] = []

    X, Y, _ = full_gauge_svd_hadamard(A, B)
    metrics = leading_metrics(X, Y)
    records.append(
        {
            "experiment": "flattening",
            "method": "full_gauge_svd_hadamard",
            "draw": 0,
            "objective_floor_ratio": metrics["objective_floor_ratio"],
            "alpha_effective": metrics["alpha_effective"],
            "finite_set_coherence": finite_set_coherence(X, Y),
        }
    )

    for method in ("haar", "randomized_hadamard"):
        for draw in range(FLATTENING_DRAWS):
            rng = stable_rng(f"nuclear:flattening:{method}:{draw}")
            if method == "haar":
                W = random_orthogonal(rng, DIMENSION)
            else:
                W = randomized_hadamard_rotation(rng, DIMENSION)
            X, Y = transformed_factors(X0, Y0, W=W)
            metrics = leading_metrics(X, Y)
            records.append(
                {
                    "experiment": "flattening",
                    "method": method,
                    "draw": draw,
                    "objective_floor_ratio": metrics["objective_floor_ratio"],
                    "alpha_effective": metrics["alpha_effective"],
                    "finite_set_coherence": finite_set_coherence(X, Y),
                }
            )

    W, certificate = deterministic_hadamard_rotation(X0, Y0)
    X, Y = transformed_factors(X0, Y0, W=W)
    metrics = leading_metrics(X, Y)
    records.append(
        {
            "experiment": "flattening",
            "method": "deterministic_hadamard",
            "draw": 0,
            "objective_floor_ratio": metrics["objective_floor_ratio"],
            "alpha_effective": metrics["alpha_effective"],
            "finite_set_coherence": certificate["coherence"],
            "coherence_bound": certificate["coherence_bound"],
        }
    )
    return records


def _summarize(
    family_rows: list[dict[str, Any]],
    same_rows: list[dict[str, Any]],
    same_diagnostics: dict[str, float],
    singular_rows: list[dict[str, Any]],
    flatten_rows: list[dict[str, Any]],
    dimension_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    family_summary: dict[str, Any] = {}
    for family in ("spiky", "dense_diagonal", "general_gauge"):
        family_summary[family] = {}
        for method in (
            "full_gauge_svd_hadamard",
            "identity",
            "diagonal_hadamard",
            "balanced_haar",
            "balanced_deterministic",
        ):
            values = [
                row["objective_floor_ratio"]
                for row in family_rows
                if row["family"] == family and row["method"] == method
            ]
            median, low, high = bootstrap_median_ci(
                values, label=f"nuclear:{family}:{method}"
            )
            family_summary[family][method] = {
                **quantiles(values),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "count": len(values),
            }

    same_summary: dict[str, Any] = {"diagnostics": same_diagnostics}
    for method in ("identity", "balanced_deterministic"):
        values = [
            row["objective_floor_ratio"]
            for row in same_rows
            if row["method"] == method
        ]
        same_summary[method] = quantiles(values)

    flatten_summary = {}
    for method in (
        "full_gauge_svd_hadamard",
        "haar",
        "randomized_hadamard",
        "deterministic_hadamard",
    ):
        selected = [row for row in flatten_rows if row["method"] == method]
        flatten_summary[method] = {
            "objective_floor_ratio": quantiles(
                [row["objective_floor_ratio"] for row in selected]
            ),
            "finite_set_coherence": quantiles(
                [row["finite_set_coherence"] for row in selected]
            ),
            "count": len(selected),
        }

    dimension_summary: dict[str, Any] = {}
    for k in DIMENSION_SCALING_DIMENSIONS:
        m = 3 * k // 2
        n = 5 * k // 4
        dimension_summary[str(k)] = {
            "dimensions": {"m": m, "n": n, "K": k},
            "methods": {},
        }
        for method in (
            "full_gauge_svd_hadamard",
            "balanced_haar",
            "balanced_randomized_hadamard",
        ):
            values = [
                row["objective_floor_ratio"]
                for row in dimension_rows
                if row["K"] == k and row["method"] == method
            ]
            dimension_summary[str(k)]["methods"][method] = {
                **quantiles(values),
                "count": len(values),
            }

    return {
        "base_seed": BASE_SEED,
        "dimensions": {"m": DIMENSION, "n": DIMENSION, "K": DIMENSION},
        "trials_per_family": TRIALS,
        "bit_widths": list(BITS),
        "noise_constants": {str(bits): bit_noise_constant(bits) for bits in BITS},
        "equal_bit_ratio_invariance": {
            "explanation": (
                "When b_A=b_B, the common bit-dependent constant multiplies "
                "both numerator and floor and cancels from their ratio."
            ),
            "maximum_absolute_4bit_8bit_ratio_difference": float(
                max(
                    row["equal_bit_ratio_difference"]
                    for row in family_rows + same_rows
                )
            ),
            "maximum_relative_4bit_8bit_ratio_difference": float(
                max(
                    row["equal_bit_ratio_difference"]
                    / row["objective_floor_ratio"]
                    for row in family_rows + same_rows
                )
            ),
        },
        "estimand": (
            "leading range objective divided by "
            "(c_A+c_B)||AB||_*^2/K; equal bit widths"
        ),
        "families": family_summary,
        "family_generators": {
            "common": (
                "K=64; singular values geomspace(1,0.15,K); all Gaussian-QR "
                "orthogonal draws use independent stable streams; every "
                "gauge is invertible, so all products have rank K"
            ),
            "spiky": (
                "A0=B0=diag(sqrt(sigma)); 64 iid N(0,1.35^2) log-scales; "
                "eight uniformly selected coordinates shifted independently "
                "by +/-2.25; mean-center, clip to [-log(25),log(25)], exponentiate"
            ),
            "dense_diagonal": (
                "A0=U sqrt(Sigma) R^T, B0=R sqrt(Sigma) V^T for independent "
                "Haar U,V,R; gauge diag(exp(z)) with iid z~N(0,1.35^2)"
            ),
            "general_gauge": (
                "same Haar-oriented A0,B0; gauge "
                "Q diag(exp(linspace(-2,2,K))) Q^T for independent Haar Q"
            ),
        },
        "spiky_generator": {
            "fixed_output": "diag(geomspace(1, 0.15, 64))",
            "gauge": (
                "64 iid N(0,1.35^2) log-scales, eight uniformly selected "
                "coordinates shifted independently by +/-2.25, log-scales "
                "mean-normalized then clipped to [-log(25),log(25)]"
            ),
            "independent_trials": TRIALS,
        },
        "same_output": same_summary,
        "singular_sweep": {
            "rank_A": SINGULAR_RANK,
            "rank_B": SINGULAR_RANK,
            "product_rank": int(
                np.linalg.matrix_rank(
                    np.eye(DIMENSION)[:, :SINGULAR_RANK].T
                    @ np.eye(DIMENSION)[:, 16 : 16 + SINGULAR_RANK]
                )
            ),
            "rows": singular_rows,
        },
        "flattening": flatten_summary,
        "dimension_scaling": {
            "dimensions": list(DIMENSION_SCALING_DIMENSIONS),
            "trials_per_dimension": DIMENSION_SCALING_TRIALS,
            "shape_rule": "m=3K/2, n=5K/4",
            "generator": (
                "independent Gaussian-QR Stiefel frames U and V; singular "
                "values geomspace(1,0.15,K); balanced factors "
                "U sqrt(Sigma) and sqrt(Sigma) V^T"
            ),
            "rows": dimension_summary,
        },
    }


def _write_figures(
    summary: dict[str, Any],
    singular_rows: list[dict[str, Any]],
    flatten_rows: list[dict[str, Any]],
) -> list[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir = REPO_ROOT / "figures" / "nuclear"
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    colors = {
        "full_gauge_svd_hadamard": "#CC79A7",
        "identity": "#666666",
        "diagonal_hadamard": "#0072B2",
        "balanced_haar": "#D55E00",
        "balanced_deterministic": "#009E73",
    }
    labels = {
        "full_gauge_svd_hadamard": "Full gauge (SVD + H)",
        "identity": "Original gauge",
        "diagonal_hadamard": "Diagonal + random H",
        "balanced_haar": "General inner + Haar",
        "balanced_deterministic": "General inner + deterministic H",
    }
    hatches = {
        "full_gauge_svd_hadamard": "...",
        "identity": "////",
        "diagonal_hadamard": "\\\\",
        "balanced_haar": "xx",
        "balanced_deterministic": "++",
    }

    families = ("spiky", "dense_diagonal", "general_gauge")
    methods = tuple(colors)
    x = np.arange(len(families), dtype=float)
    width = 0.16
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    for index, method in enumerate(methods):
        medians = [summary["families"][family][method]["median"] for family in families]
        lows = [summary["families"][family][method]["q25"] for family in families]
        highs = [summary["families"][family][method]["q75"] for family in families]
        yerr = np.vstack((np.asarray(medians) - lows, np.asarray(highs) - medians))
        ax.bar(
            x + (index - 2.0) * width,
            medians,
            width,
            color=colors[method],
            edgecolor="#222222",
            hatch=hatches[method],
            label=labels[method],
            yerr=yerr,
            capsize=2,
            linewidth=0.45,
        )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("leading objective / nuclear floor")
    ax.set_xticks(x, ["spiky", "dense diagonal", "dense general"])
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.grid(axis="y", which="both", alpha=0.18)
    fig.tight_layout()
    path = figure_dir / "objective_ratios.svg"
    save_figure(fig, path)
    plt.close(fig)
    artifacts.append(path)

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    eps = np.array([row["epsilon"] for row in singular_rows])
    cond = np.array([row["condition_T"] for row in singular_rows])
    excess = np.array([row["relative_excess"] for row in singular_rows])
    ax.plot(cond, excess, "o-", color="#0072B2", linewidth=1.2, markersize=4.0)
    for index in (0, len(eps) - 1):
        ax.annotate(
            rf"$\varepsilon=10^{{{int(round(math.log10(eps[index])))}}}$",
            (cond[index], excess[index]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa_2(T_\varepsilon)$")
    ax.set_ylabel(r"energy objective excess")
    ax.grid(which="both", alpha=0.18)
    fig.tight_layout()
    path = figure_dir / "singular_frontier.svg"
    save_figure(fig, path)
    plt.close(fig)
    artifacts.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    flatten_methods = (
        "full_gauge_svd_hadamard",
        "haar",
        "randomized_hadamard",
        "deterministic_hadamard",
    )
    plot_labels = (
        "full gauge\nSVD + H",
        "inner\nHaar",
        "inner\nrandomized H",
        "inner\ndeterministic H",
    )
    objective_data = [
        [
            row["objective_floor_ratio"]
            for row in flatten_rows
            if row["method"] == method
        ]
        for method in flatten_methods
    ]
    coherence_data = [
        [
            row["finite_set_coherence"]
            for row in flatten_rows
            if row["method"] == method
        ]
        for method in flatten_methods
    ]
    for ax, data, ylabel in (
        (axes[0], objective_data, "objective / floor"),
        (axes[1], coherence_data, "maximum finite-set flatness"),
    ):
        box = ax.boxplot(data, tick_labels=plot_labels, patch_artist=True, widths=0.55)
        for patch, color, hatch in zip(
            box["boxes"],
            ("#CC79A7", "#D55E00", "#0072B2", "#009E73"),
            ("...", "////", "\\\\", "xx"),
            strict=True,
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor("#222222")
            patch.set_hatch(hatch)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    path = figure_dir / "flattening_comparison.svg"
    save_figure(fig, path)
    plt.close(fig)
    artifacts.append(path)

    scaling = summary["dimension_scaling"]["rows"]
    dimensions = np.asarray(DIMENSION_SCALING_DIMENSIONS)
    scaling_methods = (
        "full_gauge_svd_hadamard",
        "balanced_haar",
        "balanced_randomized_hadamard",
    )
    scaling_labels = (
        "Full gauge (SVD + H)",
        "Balanced inner + Haar",
        "Balanced inner + randomized H",
    )
    scaling_colors = ("#CC79A7", "#D55E00", "#0072B2")
    scaling_markers = ("o", "s", "^")
    scaling_linestyles = ("-", "--", "-.")
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for method, label, color, marker, linestyle in zip(
        scaling_methods,
        scaling_labels,
        scaling_colors,
        scaling_markers,
        scaling_linestyles,
        strict=True,
    ):
        scaling_medians = np.asarray(
            [scaling[str(k)]["methods"][method]["median"] for k in dimensions]
        )
        scaling_lows = np.asarray(
            [scaling[str(k)]["methods"][method]["q25"] for k in dimensions]
        )
        scaling_highs = np.asarray(
            [scaling[str(k)]["methods"][method]["q75"] for k in dimensions]
        )
        ax.plot(
            dimensions,
            scaling_medians,
            color=color,
            label=label,
            marker=marker,
            linestyle=linestyle,
            markersize=4.0,
        )
        ax.fill_between(
            dimensions, scaling_lows, scaling_highs, color=color, alpha=0.14
        )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dimensions, [str(k) for k in dimensions])
    ax.set_xlabel(r"inner dimension $K$")
    ax.set_ylabel("leading objective / nuclear floor")
    ax.grid(which="both", alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "dimension_scaling.svg"
    save_figure(fig, path)
    plt.close(fig)
    artifacts.append(path)
    return artifacts



def _write_nuclear_manifest(artifacts: list[Path]) -> Path:
    """Write provenance that remains valid after another selector is run."""

    path = RESULTS_DIR / "nuclear_manifest.json"
    manifest = environment_manifest(
        artifacts=artifacts,
        dataset_hashes={},
        command="python -m experiments.reproduce nuclear",
    )
    manifest["protocol"] = {
        "dimension": DIMENSION,
        "trials_per_family": TRIALS,
        "flattening_draws": FLATTENING_DRAWS,
        "dimension_scaling_dimensions": list(DIMENSION_SCALING_DIMENSIONS),
        "dimension_scaling_trials": DIMENSION_SCALING_TRIALS,
        "singular_rank": SINGULAR_RANK,
        "singular_regularizers": [float(value) for value in SINGULAR_EPSILONS],
    }
    write_json(path, manifest)
    return path


def run() -> dict[str, Any]:
    """Run the nuclear-floor experiments and write reproducibility artifacts."""

    family_rows = _run_family_trials()
    same_rows, same_diagnostics = _same_output_trials()
    singular_rows = _run_singular_sweep()
    flatten_rows = _run_flattening_comparison()
    dimension_rows = _run_dimension_scaling()
    validation = _validate_run(
        family_rows,
        same_rows,
        same_diagnostics,
        singular_rows,
        flatten_rows,
        dimension_rows,
    )
    summary = _summarize(
        family_rows,
        same_rows,
        same_diagnostics,
        singular_rows,
        flatten_rows,
        dimension_rows,
    )
    summary["validation"] = validation

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    family_path = RESULTS_DIR / "nuclear_trials.csv"
    singular_path = RESULTS_DIR / "nuclear_singular.csv"
    flatten_path = RESULTS_DIR / "nuclear_flattening.csv"
    dimension_path = RESULTS_DIR / "nuclear_dimension_scaling.csv"
    summary_path = RESULTS_DIR / "nuclear_summary.json"
    write_csv(family_path, family_rows + same_rows)
    write_csv(singular_path, singular_rows)
    write_csv(flatten_path, flatten_rows)
    write_csv(dimension_path, dimension_rows)
    write_json(summary_path, summary)
    figure_paths = _write_figures(summary, singular_rows, flatten_rows)
    artifacts = [
        family_path,
        singular_path,
        flatten_path,
        dimension_path,
        summary_path,
        *figure_paths,
    ]
    nuclear_manifest = _write_nuclear_manifest(artifacts)
    artifacts.append(nuclear_manifest)
    return {
        "summary": summary,
        "artifacts": artifacts,
        "dataset_hashes": {},
    }
