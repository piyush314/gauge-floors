"""Experiments for product-aware deterministic metric rounding."""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.optimize import lsq_linear

from .common import (
    BASE_SEED,
    REPO_ROOT,
    RESULTS_DIR,
    bootstrap_median_ci,
    configure_matplotlib,
    environment_manifest,
    quantiles,
    save_figure,
    stable_rng,
    write_csv,
    write_json,
)

EXACT_K = 16
EXACT_RANKS = (1, 2, 4, 8)
SCALABLE_K = (64, 128, 256, 512, 1024)
SCALABLE_RANKS = (1, 2, 4, 8, 16)
DYNAMIC_TRIALS = 25
DYNAMIC_BLOCKS = 10
IMBALANCE_K = 256
IMBALANCE_RANKS = (1, 4)
IMBALANCE_EXPONENTS = (0.0, 0.5, 1.0, 2.0)
AFFINE_K = 256
AFFINE_RANK = 4
AFFINE_CLIP_FRACTIONS = (0.0, 0.05, 0.10)
STATIC_BITS = (3, 4)
STATIC_SPLITS = 5
CALIBRATION_SUBSETS = 20
CALIBRATION_SIZE = 128
RIDGE = 1.0e-3
STATIC_METHODS = ("rtn", "centered", "centered_bias", "uncentered")
STATIC_CONDITIONS = ("n128", "full")


def low_rank_block(
    k: int,
    rank: int,
    *,
    label: str,
    imbalance_exponent: float = 0.0,
) -> np.ndarray:
    """Generate a rank-r block with a controlled row-sensitivity profile.

    The output dimension is ``p=rank``. Row directions are independent
    normalized Gaussian vectors. Before a seeded random permutation, row
    sensitivities are proportional to ``j**(-imbalance_exponent)`` for
    ``j=1,...,K``; they are finally rescaled to unit RMS. Thus exponent zero
    gives exactly balanced unit-norm rows and every configuration has the same
    analytic dither energy ``K/12`` when the common gap is one.
    """

    if not 1 <= rank <= k:
        raise ValueError((k, rank))
    if imbalance_exponent < 0.0:
        raise ValueError("imbalance exponent must be nonnegative")
    rng = stable_rng(
        f"discrepancy:block:{label}:K{k}:r{rank}:alpha{imbalance_exponent:g}"
    )
    directions = rng.standard_normal((k, rank))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sensitivities = np.arange(1, k + 1, dtype=np.float64) ** (
        -float(imbalance_exponent)
    )
    sensitivities = sensitivities[rng.permutation(k)]
    sensitivities /= float(np.sqrt(np.mean(sensitivities * sensitivities)))
    block = directions * sensitivities[:, None]
    numerical_rank = int(np.linalg.matrix_rank(block))
    if numerical_rank != rank:
        raise RuntimeError(f"unexpected generated rank {numerical_rank}, wanted {rank}")
    return block


def equalized_low_rank_block(k: int, rank: int, *, label: str) -> np.ndarray:
    """Backward-compatible balanced-block constructor used by unit tests."""

    return low_rank_block(k, rank, label=label, imbalance_exponent=0.0)


@lru_cache(maxsize=None)
def binary_assignments(size: int) -> np.ndarray:
    """Return all binary rows in lexicographic integer order."""

    if size < 0 or size > 20:
        raise ValueError("enumeration is restricted to at most 20 variables")
    integers = np.arange(1 << size, dtype=np.uint32)[:, None]
    shifts = np.arange(size, dtype=np.uint32)[None, :]
    return ((integers >> shifts) & 1).astype(np.float64)


def enumerate_binary_optimum(
    block: np.ndarray,
    theta: np.ndarray,
    *,
    precomputed_products: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Exactly minimize ||(chi-theta)^T block||_2^2 by enumeration."""

    size = theta.size
    assignments = binary_assignments(size)
    products = assignments @ block if precomputed_products is None else precomputed_products
    target = theta @ block
    delta = products - target
    squared = np.einsum("ij,ij->i", delta, delta, optimize=True)
    index = int(np.argmin(squared))
    return assignments[index].copy(), float(squared[index])


def _step_to_boundary(x: np.ndarray, h: np.ndarray, direction: float) -> float:
    signed = direction * h
    movable = np.abs(signed) > 1.0e-15
    limits = np.full_like(signed, np.inf)
    positive = movable & (signed > 0.0)
    negative = movable & (signed < 0.0)
    limits[positive] = (1.0 - x[positive]) / signed[positive]
    limits[negative] = -x[negative] / signed[negative]
    limits = limits[np.isfinite(limits) & (limits >= 0.0)]
    if limits.size == 0:
        raise RuntimeError("null direction cannot move to a boundary")
    return float(np.min(limits))


def null_space_reduce(
    block: np.ndarray,
    start: np.ndarray,
    *,
    tolerance: float = 2.0e-11,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Preserve ``B.T @ x`` until at most ``rank(B)`` entries are fractional.

    Every iteration takes exactly rank+1 active coordinates and computes a null
    direction with a small SVD. This is the polynomial null-space reduction;
    finishing rules are deliberately separate below.
    """

    block = np.asarray(block, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(block))
    x = start.copy()
    steps = 0

    while True:
        x[np.abs(x) <= tolerance] = 0.0
        x[np.abs(1.0 - x) <= tolerance] = 1.0
        fractional = np.flatnonzero((x > tolerance) & (x < 1.0 - tolerance))
        if fractional.size <= rank:
            break
        active = fractional[: rank + 1]
        _, _, vh = np.linalg.svd(block[active, :].T, full_matrices=True)
        h = vh[-1, :]
        residual = float(np.linalg.norm(block[active, :].T @ h))
        scale = max(float(np.linalg.norm(block[active, :])), 1.0)
        if residual > 2.0e-10 * scale:
            raise RuntimeError(f"small-SVD null residual is too large: {residual}")

        alpha_plus = _step_to_boundary(x[active], h, 1.0)
        alpha_minus = _step_to_boundary(x[active], h, -1.0)
        candidates = (
            x[active] + alpha_plus * h,
            x[active] - alpha_minus * h,
        )
        scores = [
            float(np.sum(np.minimum(np.clip(candidate, 0.0, 1.0), 1.0 - np.clip(candidate, 0.0, 1.0))))
            for candidate in candidates
        ]
        if scores[0] <= scores[1]:
            x[active] = candidates[0]
        else:
            x[active] = candidates[1]
        x[active] = np.clip(x[active], 0.0, 1.0)
        steps += 1
        if steps > start.size + 2:
            raise RuntimeError("null-space walk failed to reduce the fractional set")

    invariant_error = float(np.linalg.norm(block.T @ (x - start)))
    fractional = np.flatnonzero((x > tolerance) & (x < 1.0 - tolerance))
    return x, {
        "rank": rank,
        "steps": steps,
        "residual_size": int(fractional.size),
        "invariant_error": invariant_error,
    }


def arbitrary_rounding_finish(x: np.ndarray) -> np.ndarray:
    """Finish the walk in polynomial time by deterministic nearest rounding."""

    return (np.asarray(x) >= 0.5).astype(np.float64)


def conditional_expectation_finish(
    block: np.ndarray,
    theta: np.ndarray,
    x: np.ndarray,
    *,
    fixed_offset: np.ndarray | None = None,
    tolerance: float = 2.0e-11,
) -> tuple[np.ndarray, dict[str, float]]:
    """Derandomize same-grid Bernoulli rounding on the reached residual face."""

    block = np.asarray(block, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if fixed_offset is None:
        fixed_offset = np.zeros(block.shape[1], dtype=np.float64)
    else:
        fixed_offset = np.asarray(fixed_offset, dtype=np.float64)

    fractional = np.flatnonzero((x > tolerance) & (x < 1.0 - tolerance))
    chi = np.rint(x)
    # At x, all not-yet-fixed Bernoulli increments have mean zero.
    running = fixed_offset + (x - theta) @ block
    initial_objective = float(running @ running)
    variance = float(
        np.sum(
            x[fractional]
            * (1.0 - x[fractional])
            * np.sum(block[fractional, :] ** 2, axis=1)
        )
    )
    expected_bound = initial_objective + variance
    for coordinate in fractional:
        row = block[coordinate, :]
        candidate_zero = running - x[coordinate] * row
        candidate_one = running + (1.0 - x[coordinate]) * row
        if float(candidate_zero @ candidate_zero) <= float(
            candidate_one @ candidate_one
        ):
            chi[coordinate] = 0.0
            running = candidate_zero
        else:
            chi[coordinate] = 1.0
            running = candidate_one
    squared_error = float(running @ running)
    tolerance_scale = 5.0e-11 * max(1.0, expected_bound)
    if squared_error > expected_bound + tolerance_scale:
        raise RuntimeError(
            "conditional-expectation finish exceeded its expectation bound: "
            f"{squared_error} > {expected_bound}"
        )
    return chi, {
        "initial_objective": initial_objective,
        "variance_bound": variance,
        "expected_bound": expected_bound,
        "squared_error": squared_error,
        "bound_violation": max(squared_error - expected_bound, 0.0),
    }


def residual_face_finish(
    block: np.ndarray,
    theta: np.ndarray,
    x: np.ndarray,
    *,
    fixed_offset: np.ndarray | None = None,
    tolerance: float = 2.0e-11,
) -> tuple[np.ndarray, float]:
    """Exactly optimize the reached face at exponential cost in its dimension."""

    block = np.asarray(block, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if fixed_offset is None:
        fixed_offset = np.zeros(block.shape[1], dtype=np.float64)
    fractional = np.flatnonzero((x > tolerance) & (x < 1.0 - tolerance))
    chi = np.rint(x)
    if not fractional.size:
        error = fixed_offset + (chi - theta) @ block
        return chi, float(error @ error)
    chi[fractional] = 0.0
    base = fixed_offset + (chi - theta) @ block
    assignments = binary_assignments(fractional.size)
    candidates = base[None, :] + assignments @ block[fractional, :]
    squared = np.einsum("ij,ij->i", candidates, candidates, optimize=True)
    index = int(np.argmin(squared))
    chi[fractional] = assignments[index]
    return chi, float(squared[index])


def same_grid_bernoulli_energy(block: np.ndarray, theta: np.ndarray) -> float:
    """Analytic expected squared error for independent same-grid rounding."""

    row_energy = np.sum(np.asarray(block, dtype=np.float64) ** 2, axis=1)
    theta = np.asarray(theta, dtype=np.float64)
    return float(np.sum(theta * (1.0 - theta) * row_energy))


def null_space_walk(
    block: np.ndarray,
    theta: np.ndarray,
    *,
    tolerance: float = 2.0e-11,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Backward-compatible walk plus residual-face oracle.

    New experiments call the reduction and all finishing rules separately so
    this exponential oracle is never mislabeled as the polynomial walk.
    """

    x, trace = null_space_reduce(block, theta, tolerance=tolerance)
    chi, squared_error = residual_face_finish(
        block, theta, x, tolerance=tolerance
    )
    return chi, {**trace, "squared_error": squared_error}


def _dynamic_method_row(
    *,
    group: str,
    k: int,
    rank: int,
    block_id: int,
    trial: int,
    method: str,
    block: np.ndarray,
    theta: np.ndarray,
    chi: np.ndarray | None,
    runtime_seconds: float,
    exact_error: float | None,
    walk: dict[str, float | int] | None = None,
    squared_error_override: float | None = None,
    walk_runtime_seconds: float = 0.0,
    finish_runtime_seconds: float = 0.0,
    imbalance_exponent: float = 0.0,
    ce: dict[str, float] | None = None,
) -> dict[str, Any]:
    if squared_error_override is None:
        if chi is None:
            raise ValueError("chi is required unless squared_error_override is set")
        error = float(np.linalg.norm((chi - theta) @ block) ** 2)
    else:
        error = float(squared_error_override)
    row_norms = np.linalg.norm(block, axis=1)
    baseline = float(np.sum(row_norms * row_norms) / 12.0)
    bernoulli = same_grid_bernoulli_energy(block, theta)
    beta_rms = float(np.sqrt(np.mean(row_norms * row_norms)))
    beta_max = float(np.max(row_norms))
    return {
        "experiment": f"dynamic_{group}",
        "group": group,
        "K": k,
        "rank": rank,
        "output_dimension_p": block.shape[1],
        "block_id": block_id,
        "trial": trial,
        "method": method,
        "imbalance_exponent": imbalance_exponent,
        "squared_error": error,
        "dither_expected_squared_error": baseline,
        "same_grid_bernoulli_expected_squared_error": bernoulli,
        "ratio_to_dither": error / baseline,
        "ratio_to_same_grid_bernoulli": error / bernoulli,
        "normalized_additive_gap": (
            max(error - exact_error, 0.0) / baseline
            if exact_error is not None
            else ""
        ),
        "exact_cancellation": int(error <= 1.0e-20 * max(baseline, 1.0)),
        "runtime_seconds": runtime_seconds,
        "walk_runtime_seconds": walk_runtime_seconds,
        "finish_runtime_seconds": finish_runtime_seconds,
        "beta_rms": beta_rms,
        "beta_max": beta_max,
        "beta_rms_over_beta_max": beta_rms / beta_max,
        "ce_dither_certificate_ratio": 3.0
        * rank
        / k
        * (beta_max / beta_rms) ** 2,
        "numerical_rank": int(np.linalg.matrix_rank(block)),
        "residual_size": walk["residual_size"] if walk else 0,
        "walk_steps": walk["steps"] if walk else 0,
        "invariant_error": walk["invariant_error"] if walk else 0.0,
        "ce_expected_bound": ce["expected_bound"] if ce else "",
        "ce_bound_violation": ce["bound_violation"] if ce else 0.0,
    }


def _append_dynamic_finishes(
    rows: list[dict[str, Any]],
    *,
    group: str,
    k: int,
    rank: int,
    block_id: int,
    trial: int,
    block: np.ndarray,
    theta: np.ndarray,
    exact_error: float | None,
    imbalance_exponent: float,
) -> None:
    """Run the three distinct walk finishes and the two simple references."""

    start_time = time.perf_counter()
    rtn_chi = (theta >= 0.5).astype(np.float64)
    rtn_runtime = time.perf_counter() - start_time
    rows.append(
        _dynamic_method_row(
            group=group,
            k=k,
            rank=rank,
            block_id=block_id,
            trial=trial,
            method="rtn",
            block=block,
            theta=theta,
            chi=rtn_chi,
            runtime_seconds=rtn_runtime,
            exact_error=exact_error,
            imbalance_exponent=imbalance_exponent,
        )
    )

    start_time = time.perf_counter()
    bernoulli = same_grid_bernoulli_energy(block, theta)
    bernoulli_runtime = time.perf_counter() - start_time
    rows.append(
        _dynamic_method_row(
            group=group,
            k=k,
            rank=rank,
            block_id=block_id,
            trial=trial,
            method="bernoulli_reference",
            block=block,
            theta=theta,
            chi=None,
            squared_error_override=bernoulli,
            runtime_seconds=bernoulli_runtime,
            exact_error=exact_error,
            imbalance_exponent=imbalance_exponent,
        )
    )

    walk_start = time.perf_counter()
    x, walk = null_space_reduce(block, theta)
    walk_runtime = time.perf_counter() - walk_start

    finish_start = time.perf_counter()
    arbitrary_chi = arbitrary_rounding_finish(x)
    arbitrary_finish_runtime = time.perf_counter() - finish_start
    rows.append(
        _dynamic_method_row(
            group=group,
            k=k,
            rank=rank,
            block_id=block_id,
            trial=trial,
            method="walk_arbitrary",
            block=block,
            theta=theta,
            chi=arbitrary_chi,
            runtime_seconds=walk_runtime + arbitrary_finish_runtime,
            walk_runtime_seconds=walk_runtime,
            finish_runtime_seconds=arbitrary_finish_runtime,
            exact_error=exact_error,
            walk=walk,
            imbalance_exponent=imbalance_exponent,
        )
    )

    finish_start = time.perf_counter()
    ce_chi, ce = conditional_expectation_finish(block, theta, x)
    ce_finish_runtime = time.perf_counter() - finish_start
    rows.append(
        _dynamic_method_row(
            group=group,
            k=k,
            rank=rank,
            block_id=block_id,
            trial=trial,
            method="walk_ce",
            block=block,
            theta=theta,
            chi=ce_chi,
            runtime_seconds=walk_runtime + ce_finish_runtime,
            walk_runtime_seconds=walk_runtime,
            finish_runtime_seconds=ce_finish_runtime,
            exact_error=exact_error,
            walk=walk,
            imbalance_exponent=imbalance_exponent,
            ce=ce,
        )
    )

    finish_start = time.perf_counter()
    face_chi, _ = residual_face_finish(block, theta, x)
    face_finish_runtime = time.perf_counter() - finish_start
    rows.append(
        _dynamic_method_row(
            group=group,
            k=k,
            rank=rank,
            block_id=block_id,
            trial=trial,
            method="null_face",
            block=block,
            theta=theta,
            chi=face_chi,
            runtime_seconds=walk_runtime + face_finish_runtime,
            walk_runtime_seconds=walk_runtime,
            finish_runtime_seconds=face_finish_runtime,
            exact_error=exact_error,
            walk=walk,
            imbalance_exponent=imbalance_exponent,
        )
    )


def _run_dynamic() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for rank in EXACT_RANKS:
        for block_id in range(DYNAMIC_BLOCKS):
            block = low_rank_block(
                EXACT_K, rank, label=f"exact:block{block_id}"
            )
            products = binary_assignments(EXACT_K) @ block
            for trial in range(DYNAMIC_TRIALS):
                rng = stable_rng(
                    "discrepancy:dynamic:exact:"
                    f"r{rank}:block{block_id}:trial{trial}"
                )
                theta = rng.uniform(0.0, 1.0, size=EXACT_K)
                start_time = time.perf_counter()
                exact_chi, exact_error = enumerate_binary_optimum(
                    block, theta, precomputed_products=products
                )
                exact_runtime = time.perf_counter() - start_time
                rows.append(
                    _dynamic_method_row(
                        group="exact",
                        k=EXACT_K,
                        rank=rank,
                        block_id=block_id,
                        trial=trial,
                        method="exact",
                        block=block,
                        theta=theta,
                        chi=exact_chi,
                        runtime_seconds=exact_runtime,
                        exact_error=exact_error,
                    )
                )
                _append_dynamic_finishes(
                    rows,
                    group="exact",
                    k=EXACT_K,
                    rank=rank,
                    block_id=block_id,
                    trial=trial,
                    block=block,
                    theta=theta,
                    exact_error=exact_error,
                    imbalance_exponent=0.0,
                )

    for k in SCALABLE_K:
        for rank in SCALABLE_RANKS:
            if rank > k:
                continue
            for block_id in range(DYNAMIC_BLOCKS):
                block = low_rank_block(
                    k, rank, label=f"scalable:block{block_id}"
                )
                for trial in range(DYNAMIC_TRIALS):
                    rng = stable_rng(
                        "discrepancy:dynamic:scalable:"
                        f"K{k}:r{rank}:block{block_id}:trial{trial}"
                    )
                    theta = rng.uniform(0.0, 1.0, size=k)
                    _append_dynamic_finishes(
                        rows,
                        group="scalable",
                        k=k,
                        rank=rank,
                        block_id=block_id,
                        trial=trial,
                        block=block,
                        theta=theta,
                        exact_error=None,
                        imbalance_exponent=0.0,
                    )

    for rank in IMBALANCE_RANKS:
        for exponent in IMBALANCE_EXPONENTS:
            for block_id in range(DYNAMIC_BLOCKS):
                block = low_rank_block(
                    IMBALANCE_K,
                    rank,
                    label=f"imbalance:block{block_id}",
                    imbalance_exponent=exponent,
                )
                for trial in range(DYNAMIC_TRIALS):
                    rng = stable_rng(
                        "discrepancy:dynamic:imbalance:"
                        f"r{rank}:alpha{exponent:g}:"
                        f"block{block_id}:trial{trial}"
                    )
                    theta = rng.uniform(0.0, 1.0, size=IMBALANCE_K)
                    _append_dynamic_finishes(
                        rows,
                        group="imbalance",
                        k=IMBALANCE_K,
                        rank=rank,
                        block_id=block_id,
                        trial=trial,
                        block=block,
                        theta=theta,
                        exact_error=None,
                        imbalance_exponent=exponent,
                    )
    return rows


def box_relaxation_start(
    block: np.ndarray,
    theta: np.ndarray,
    fixed_offset: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve the affine box-constrained least-squares relaxation."""

    target = block.T @ theta - fixed_offset
    result = lsq_linear(
        block.T,
        target,
        bounds=(0.0, 1.0),
        method="trf",
        tol=1.0e-11,
        lsmr_tol=1.0e-11,
        max_iter=500,
    )
    if not result.success:
        raise RuntimeError(f"box relaxation failed: {result.message}")
    x = np.clip(result.x, 0.0, 1.0)
    residual = fixed_offset + (x - theta) @ block
    return x, {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(residual @ residual),
        "optimality": float(result.optimality),
    }


def _run_affine() -> list[dict[str, Any]]:
    """Exercise fixed saturation offsets with residue and QP starts."""

    rows: list[dict[str, Any]] = []
    for clip_fraction in AFFINE_CLIP_FRACTIONS:
        for block_id in range(DYNAMIC_BLOCKS):
            full_block = low_rank_block(
                AFFINE_K,
                AFFINE_RANK,
                label=f"affine:rho{clip_fraction:g}:block{block_id}",
            )
            for trial in range(DYNAMIC_TRIALS):
                rng = stable_rng(
                    "discrepancy:affine:"
                    f"rho{clip_fraction:g}:block{block_id}:trial{trial}"
                )
                fixed_count = int(round(clip_fraction * AFFINE_K))
                permutation = rng.permutation(AFFINE_K)
                fixed_indices = permutation[:fixed_count]
                active_indices = permutation[fixed_count:]
                block = full_block[active_indices, :]
                theta = rng.uniform(0.0, 1.0, size=active_indices.size)
                if fixed_count:
                    # A clipped value lies beyond one endpoint. Its forced
                    # reconstruction error points back toward the interval.
                    outside_sign = rng.choice(
                        np.array([-1.0, 1.0]), size=fixed_count
                    )
                    overshoot = rng.uniform(0.25, 1.25, size=fixed_count)
                    fixed_errors = -outside_sign * overshoot
                    fixed_offset = fixed_errors @ full_block[fixed_indices, :]
                else:
                    fixed_offset = np.zeros(AFFINE_RANK, dtype=np.float64)

                bernoulli_reference = float(
                    fixed_offset @ fixed_offset
                    + same_grid_bernoulli_energy(block, theta)
                )
                for start_method in ("theta_start", "box_start"):
                    qp_runtime = 0.0
                    qp = {
                        "iterations": 0,
                        "objective": float(fixed_offset @ fixed_offset),
                        "optimality": 0.0,
                    }
                    if start_method == "theta_start":
                        x0 = theta
                    else:
                        start_time = time.perf_counter()
                        x0, qp = box_relaxation_start(
                            block, theta, fixed_offset
                        )
                        qp_runtime = time.perf_counter() - start_time
                    walk_start = time.perf_counter()
                    x, walk = null_space_reduce(block, x0)
                    walk_runtime = time.perf_counter() - walk_start
                    finish_start = time.perf_counter()
                    chi, ce = conditional_expectation_finish(
                        block,
                        theta,
                        x,
                        fixed_offset=fixed_offset,
                    )
                    finish_runtime = time.perf_counter() - finish_start
                    direct = fixed_offset + (chi - theta) @ block
                    squared_error = float(direct @ direct)
                    rows.append(
                        {
                            "experiment": "dynamic_affine",
                            "K": AFFINE_K,
                            "rank": AFFINE_RANK,
                            "output_dimension_p": AFFINE_RANK,
                            "clip_fraction": clip_fraction,
                            "fixed_count": fixed_count,
                            "active_count": active_indices.size,
                            "block_id": block_id,
                            "trial": trial,
                            "method": start_method,
                            "squared_error": squared_error,
                            "affine_same_grid_bernoulli_reference": (
                                bernoulli_reference
                            ),
                            "ratio_to_affine_bernoulli": (
                                squared_error / bernoulli_reference
                            ),
                            "fixed_offset_squared_norm": float(
                                fixed_offset @ fixed_offset
                            ),
                            "box_objective": qp["objective"],
                            "box_iterations": qp["iterations"],
                            "box_optimality": qp["optimality"],
                            "runtime_seconds": (
                                qp_runtime + walk_runtime + finish_runtime
                            ),
                            "qp_runtime_seconds": qp_runtime,
                            "walk_runtime_seconds": walk_runtime,
                            "finish_runtime_seconds": finish_runtime,
                            "residual_size": walk["residual_size"],
                            "numerical_rank": int(np.linalg.matrix_rank(block)),
                            "invariant_error": walk["invariant_error"],
                            "ce_expected_bound": ce["expected_bound"],
                            "ce_bound_violation": ce["bound_violation"],
                        }
                    )
    return rows


def fixed_symmetric_grid(
    weights: np.ndarray, bits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return adjacent fixed-grid levels and the fractional residues."""

    q = 2 ** (bits - 1) - 1
    scales = np.max(np.abs(weights), axis=0) / q
    scales = np.where(scales > 0.0, scales, 1.0)
    scaled = weights / scales[None, :]
    nearest = np.rint(scaled)
    exact = np.abs(scaled - nearest) <= 5.0e-13
    lower_index = np.clip(np.floor(scaled), -q, q)
    upper_index = np.clip(lower_index + 1.0, -q, q)
    lower_index[exact] = np.clip(nearest[exact], -q, q)
    upper_index[exact] = lower_index[exact]
    lower = lower_index * scales[None, :]
    upper = upper_index * scales[None, :]
    widths = upper - lower
    variable = widths > 0.0
    theta = np.zeros_like(weights)
    theta[variable] = (weights[variable] - lower[variable]) / widths[variable]
    return lower, upper, theta, variable


def round_to_nearest(
    lower: np.ndarray, upper: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    return np.where(theta >= 0.5, upper, lower)


def one_flip_metric_descent(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    theta: np.ndarray,
    variable: np.ndarray,
    metric: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Best-improvement one-flip descent, independently for each column."""

    rounded = round_to_nearest(lower, upper, theta)
    total_flips = 0
    worst_final_improvement = 0.0
    for column in range(target.shape[1]):
        active = np.flatnonzero(variable[:, column])
        if not active.size:
            continue
        current = rounded[:, column].copy()
        error = current - target[:, column]
        gradient = metric @ error
        objective = float(error @ gradient)
        iterations = 0
        while True:
            alternatives = np.where(
                np.isclose(current[active], lower[active, column], rtol=0.0, atol=1e-14),
                upper[active, column],
                lower[active, column],
            )
            changes = alternatives - current[active]
            deltas = (
                2.0 * changes * gradient[active]
                + changes * changes * np.diag(metric)[active]
            )
            best_local = int(np.argmin(deltas))
            best_delta = float(deltas[best_local])
            tolerance = 2.0e-13 * max(1.0, abs(objective))
            if best_delta >= -tolerance:
                worst_final_improvement = min(worst_final_improvement, best_delta)
                break
            coordinate = int(active[best_local])
            change = float(changes[best_local])
            current[coordinate] += change
            error[coordinate] += change
            gradient += change * metric[:, coordinate]
            objective += best_delta
            total_flips += 1
            iterations += 1
            if iterations > 10000:
                raise RuntimeError("one-flip descent exceeded the finite-state guard")
        rounded[:, column] = current
    return rounded, {
        "flip_count": total_flips,
        "worst_final_improvement": worst_final_improvement,
    }


def metric_product_mse(inputs: np.ndarray, error: np.ndarray) -> tuple[float, float]:
    """Return analytically and directly evaluated per-output product MSE."""

    metric = inputs.T @ inputs / inputs.shape[0]
    analytic = float(
        np.einsum("ic,jc,ij->", error, error, metric, optimize=True)
        / error.shape[1]
    )
    direct = float(np.mean((inputs @ error) ** 2))
    return analytic, direct


def affine_metric_product_mse(
    inputs: np.ndarray, error: np.ndarray, output_bias: np.ndarray
) -> tuple[float, float]:
    """Evaluate product MSE after adding a fixed per-output correction."""

    inputs = np.asarray(inputs, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    output_bias = np.asarray(output_bias, dtype=np.float64)
    if output_bias.shape != (error.shape[1],):
        raise ValueError(
            f"output bias has shape {output_bias.shape}, expected {(error.shape[1],)}"
        )
    metric = inputs.T @ inputs / inputs.shape[0]
    mean = inputs.mean(axis=0)
    mean_error = mean @ error
    analytic = float(
        (
            np.einsum("ic,jc,ij->", error, error, metric, optimize=True)
            + 2.0 * (output_bias @ mean_error)
            + output_bias @ output_bias
        )
        / error.shape[1]
    )
    direct = float(np.mean((inputs @ error + output_bias[None, :]) ** 2))
    return analytic, direct


def optimal_output_bias(inputs: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Return ``c=-mu @ error``, the optimal constant output correction."""

    inputs = np.asarray(inputs, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    return -(inputs.mean(axis=0) @ error)


def _digits_split(
    split_index: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    int,
]:
    """Construct one deterministic stratified fit/calibration/test split."""

    if not 0 <= split_index < STATIC_SPLITS:
        raise ValueError(
            f"split index {split_index} is outside [0, {STATIC_SPLITS})"
        )
    digits = load_digits()
    inputs = digits.data.astype(np.float64) / 16.0
    targets = digits.target.astype(np.int64)
    all_indices = np.arange(inputs.shape[0])
    fit_indices, remainder = train_test_split(
        all_indices,
        train_size=0.60,
        random_state=BASE_SEED + 2 * split_index,
        stratify=targets,
    )
    calibration_indices, test_indices = train_test_split(
        remainder,
        train_size=0.50,
        random_state=BASE_SEED + 2 * split_index + 1,
        stratify=targets[remainder],
    )
    y = np.eye(10, dtype=np.float64)[targets[fit_indices]]
    X_fit = inputs[fit_indices]
    weights = np.linalg.solve(
        X_fit.T @ X_fit + RIDGE * np.eye(X_fit.shape[1]), X_fit.T @ y
    )
    digest = hashlib.sha256()
    digest.update(digits.data.tobytes(order="C"))
    digest.update(targets.tobytes(order="C"))
    return (
        weights,
        inputs[calibration_indices],
        targets[calibration_indices],
        inputs[test_indices],
        targets[test_indices],
        digest.hexdigest(),
        fit_indices.size,
    )


def _digits_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return the original split-zero problem for backward compatibility."""

    (
        weights,
        calibration_pool,
        _,
        test_inputs,
        test_targets,
        dataset_hash,
        _,
    ) = _digits_split(0)
    return (
        weights,
        calibration_pool,
        test_inputs,
        test_targets,
        dataset_hash,
    )


def _calibration_conditions(
    calibration_targets: np.ndarray, split_index: int
) -> list[tuple[str, int, np.ndarray, str]]:
    """Return stratified 128-sample subsets and the complete pool."""

    all_indices = np.arange(calibration_targets.size)
    conditions: list[tuple[str, int, np.ndarray, str]] = []
    for subset_index in range(CALIBRATION_SUBSETS):
        subset, _ = train_test_split(
            all_indices,
            train_size=CALIBRATION_SIZE,
            random_state=(
                BASE_SEED
                + 1000
                + split_index * CALIBRATION_SUBSETS
                + subset_index
            ),
            stratify=calibration_targets,
        )
        conditions.append(
            (
                "n128",
                subset_index,
                np.sort(subset),
                "stratified_without_replacement",
            )
        )
    conditions.append(("full", -1, all_indices, "full_calibration_pool"))
    return conditions


def _run_static() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_metadata: list[dict[str, int]] = []
    dataset_hashes: set[str] = set()

    for split_index in range(STATIC_SPLITS):
        (
            weights,
            calibration_pool,
            calibration_targets,
            test_inputs,
            test_targets,
            dataset_hash,
            fit_size,
        ) = _digits_split(split_index)
        del test_targets  # Labels are used only for the stratified split.
        dataset_hashes.add(dataset_hash)
        split_metadata.append(
            {
                "split_index": split_index,
                "fit_size": int(fit_size),
                "calibration_pool_size": int(calibration_pool.shape[0]),
                "test_size": int(test_inputs.shape[0]),
            }
        )

        for bits in STATIC_BITS:
            lower, upper, theta, variable = fixed_symmetric_grid(weights, bits)
            rtn = round_to_nearest(lower, upper, theta)
            rtn_error = rtn - weights
            rtn_test_analytic, rtn_test_direct = metric_product_mse(
                test_inputs, rtn_error
            )
            if not np.isclose(
                rtn_test_analytic,
                rtn_test_direct,
                rtol=2e-12,
                atol=2e-15,
            ):
                raise RuntimeError(
                    "held-out analytic/direct identity failed for RTN"
                )

            for (
                calibration_condition,
                subset_index,
                indices,
                sampling,
            ) in _calibration_conditions(calibration_targets, split_index):
                calibration = calibration_pool[indices]
                calibration_mean = calibration.mean(axis=0)
                uncentered = calibration.T @ calibration / calibration.shape[0]
                centered_inputs = calibration - calibration_mean[None, :]
                centered = (
                    centered_inputs.T @ centered_inputs / calibration.shape[0]
                )
                rtn_cal_analytic, rtn_cal_direct = metric_product_mse(
                    calibration, rtn_error
                )
                if not np.isclose(
                    rtn_cal_analytic,
                    rtn_cal_direct,
                    rtol=2e-12,
                    atol=2e-15,
                ):
                    raise RuntimeError(
                        "calibration analytic/direct identity failed for RTN"
                    )

                rounded_by_method: dict[
                    str, tuple[np.ndarray, dict[str, float | int], float]
                ] = {}
                start = time.perf_counter()
                rounded_by_method["rtn"] = (
                    rtn.copy(),
                    {"flip_count": 0, "worst_final_improvement": 0.0},
                    time.perf_counter() - start,
                )

                start = time.perf_counter()
                centered_rounded, centered_descent = one_flip_metric_descent(
                    weights, lower, upper, theta, variable, centered
                )
                centered_runtime = time.perf_counter() - start
                rounded_by_method["centered"] = (
                    centered_rounded,
                    centered_descent,
                    centered_runtime,
                )
                rounded_by_method["centered_bias"] = (
                    centered_rounded.copy(),
                    centered_descent.copy(),
                    centered_runtime,
                )

                start = time.perf_counter()
                uncentered_rounded, uncentered_descent = (
                    one_flip_metric_descent(
                        weights, lower, upper, theta, variable, uncentered
                    )
                )
                rounded_by_method["uncentered"] = (
                    uncentered_rounded,
                    uncentered_descent,
                    time.perf_counter() - start,
                )

                for method in STATIC_METHODS:
                    rounded, descent, runtime = rounded_by_method[method]
                    error = rounded - weights
                    raw_cal_analytic, raw_cal_direct = metric_product_mse(
                        calibration, error
                    )
                    if method == "centered_bias":
                        output_bias = optimal_output_bias(calibration, error)
                    else:
                        output_bias = np.zeros(error.shape[1], dtype=np.float64)
                    cal_analytic, cal_direct = affine_metric_product_mse(
                        calibration, error, output_bias
                    )
                    test_analytic, test_direct = affine_metric_product_mse(
                        test_inputs, error, output_bias
                    )
                    optimization_metric = (
                        centered
                        if method in ("centered", "centered_bias")
                        else uncentered
                    )
                    optimization_objective = float(
                        np.einsum(
                            "ic,jc,ij->",
                            error,
                            error,
                            optimization_metric,
                            optimize=True,
                        )
                        / error.shape[1]
                    )
                    rows.append(
                        {
                            "experiment": "static_digits",
                            "split_index": split_index,
                            "bits": bits,
                            "calibration_condition": calibration_condition,
                            "calibration_subset": subset_index,
                            "calibration_size": calibration.shape[0],
                            "calibration_pool_size": calibration_pool.shape[0],
                            "calibration_sampling": sampling,
                            "test_size": test_inputs.shape[0],
                            "method": method,
                            "bias_correction_applied": (
                                method == "centered_bias"
                            ),
                            "output_bias_l2": float(
                                np.linalg.norm(output_bias)
                            ),
                            "calibration_uncentered_mse": raw_cal_analytic,
                            "calibration_evaluation_mse": cal_analytic,
                            "calibration_direct_mse": cal_direct,
                            "calibration_identity_error": abs(
                                cal_analytic - cal_direct
                            ),
                            "calibration_identity_relative_error": (
                                abs(cal_analytic - cal_direct)
                                / max(abs(cal_direct), 1.0e-15)
                            ),
                            "calibration_ratio_to_rtn": (
                                cal_direct / rtn_cal_direct
                            ),
                            "optimization_metric_objective": (
                                optimization_objective
                            ),
                            "centered_bias_objective_gap": (
                                abs(
                                    cal_direct - optimization_objective
                                )
                                if method == "centered_bias"
                                else 0.0
                            ),
                            "heldout_analytic_mse": test_analytic,
                            "heldout_direct_mse": test_direct,
                            "heldout_identity_error": abs(
                                test_analytic - test_direct
                            ),
                            "heldout_identity_relative_error": (
                                abs(test_analytic - test_direct)
                                / max(abs(test_direct), 1.0e-15)
                            ),
                            "heldout_ratio_to_rtn": (
                                test_direct / rtn_test_direct
                            ),
                            "rtn_calibration_mse": rtn_cal_direct,
                            "rtn_heldout_mse": rtn_test_direct,
                            "flip_count": descent["flip_count"],
                            "worst_final_improvement": descent[
                                "worst_final_improvement"
                            ],
                            "runtime_seconds": runtime,
                            "variable_entries": int(np.sum(variable)),
                            "total_entries": int(variable.size),
                        }
                    )
    if len(dataset_hashes) != 1:
        raise RuntimeError(f"inconsistent Digits dataset hashes: {dataset_hashes}")
    fit_sizes = {item["fit_size"] for item in split_metadata}
    calibration_pool_sizes = {
        item["calibration_pool_size"] for item in split_metadata
    }
    test_sizes = {item["test_size"] for item in split_metadata}
    if not (
        len(fit_sizes) == len(calibration_pool_sizes) == len(test_sizes) == 1
    ):
        raise RuntimeError(f"split sizes are not stable: {split_metadata}")
    metadata = {
        "dataset_sha256": dataset_hashes.pop(),
        "splits": STATIC_SPLITS,
        "split_details": split_metadata,
        "split_random_states": [
            [BASE_SEED + 2 * index, BASE_SEED + 2 * index + 1]
            for index in range(STATIC_SPLITS)
        ],
        "fit_size": fit_sizes.pop(),
        "calibration_pool_size": calibration_pool_sizes.pop(),
        "test_size": test_sizes.pop(),
        "input_dimension": 64,
        "output_dimension": 10,
        "ridge": RIDGE,
        "intercept": False,
        "calibration_subset_sampling": (
            "stratified without replacement within each subset"
        ),
        "calibration_conditions": {
            "n128": {
                "size": CALIBRATION_SIZE,
                "subsets_per_split": CALIBRATION_SUBSETS,
            },
            "full": {
                "size": split_metadata[0]["calibration_pool_size"],
                "subsets_per_split": 1,
            },
        },
    }
    return rows, metadata


def _validate_run(
    dynamic_rows: list[dict[str, Any]],
    static_rows: list[dict[str, Any]],
    affine_rows: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Fail fast when walk or product-metric invariants drift."""

    walk_rows = [
        row
        for row in dynamic_rows
        if row["method"] in ("walk_arbitrary", "walk_ce", "null_face")
    ]
    affine_walk_rows = affine_rows
    maximum_walk_invariant_error = float(
        max(row["invariant_error"] for row in walk_rows)
    )
    maximum_residual_rank_excess = int(
        max(
            0,
            max(
                row["residual_size"] - row["numerical_rank"]
                for row in walk_rows
            ),
        )
    )
    maximum_affine_walk_invariant_error = float(
        max(row["invariant_error"] for row in affine_walk_rows)
    )
    maximum_affine_residual_rank_excess = int(
        max(
            0,
            max(
                row["residual_size"] - row["numerical_rank"]
                for row in affine_walk_rows
            ),
        )
    )
    maximum_ce_bound_violation = float(
        max(
            [
                row["ce_bound_violation"]
                for row in dynamic_rows
                if row["method"] == "walk_ce"
            ]
            + [row["ce_bound_violation"] for row in affine_rows]
        )
    )
    maximum_static_identity_error = float(
        max(
            max(row["calibration_identity_error"], row["heldout_identity_error"])
            for row in static_rows
        )
    )
    maximum_static_relative_identity_error = float(
        max(
            max(
                row["calibration_identity_relative_error"],
                row["heldout_identity_relative_error"],
            )
            for row in static_rows
        )
    )
    maximum_centered_bias_objective_gap = float(
        max(
            row["centered_bias_objective_gap"]
            for row in static_rows
            if row["method"] == "centered_bias"
        )
    )
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in dynamic_rows:
        key = (
            row["group"],
            row["K"],
            row["rank"],
            row["imbalance_exponent"],
            row["block_id"],
            row["trial"],
        )
        grouped[key][row["method"]] = row

    maximum_face_oracle_violation = 0.0
    maximum_exact_optimum_violation = 0.0
    for methods in grouped.values():
        face = methods["null_face"]["squared_error"]
        maximum_face_oracle_violation = max(
            maximum_face_oracle_violation,
            face - methods["walk_arbitrary"]["squared_error"],
            face - methods["walk_ce"]["squared_error"],
        )
        if "exact" in methods:
            exact = methods["exact"]["squared_error"]
            maximum_exact_optimum_violation = max(
                maximum_exact_optimum_violation,
                max(
                    exact - row["squared_error"]
                    for name, row in methods.items()
                    if name != "exact"
                ),
            )

    maximum_ce_certificate_violation = float(
        max(
            max(
                row["ratio_to_dither"] - row["ce_dither_certificate_ratio"],
                0.0,
            )
            for row in dynamic_rows
            if row["method"] == "walk_ce"
        )
    )
    maximum_bernoulli_identity_error = float(
        max(
            abs(
                row["squared_error"]
                - row["same_grid_bernoulli_expected_squared_error"]
            )
            for row in dynamic_rows
            if row["method"] == "bernoulli_reference"
        )
    )
    expected_dynamic_rows = (
        len(EXACT_RANKS) * DYNAMIC_BLOCKS * DYNAMIC_TRIALS * 6
        + len(SCALABLE_K)
        * len(SCALABLE_RANKS)
        * DYNAMIC_BLOCKS
        * DYNAMIC_TRIALS
        * 5
        + len(IMBALANCE_RANKS)
        * len(IMBALANCE_EXPONENTS)
        * DYNAMIC_BLOCKS
        * DYNAMIC_TRIALS
        * 5
    )
    expected_affine_rows = (
        len(AFFINE_CLIP_FRACTIONS) * DYNAMIC_BLOCKS * DYNAMIC_TRIALS * 2
    )
    expected_static_rows = (
        STATIC_SPLITS
        * len(STATIC_BITS)
        * len(STATIC_METHODS)
        * (CALIBRATION_SUBSETS + 1)
    )

    walk_tolerance = 1.0e-8
    identity_tolerance = 5.0e-10
    if maximum_walk_invariant_error > walk_tolerance:
        raise RuntimeError(
            "null-space invariant failed: "
            f"maximum absolute error {maximum_walk_invariant_error}"
        )
    if maximum_residual_rank_excess > 0:
        raise RuntimeError(
            "null-space residual rank bound failed: "
            f"maximum excess {maximum_residual_rank_excess}"
        )
    if maximum_affine_walk_invariant_error > walk_tolerance:
        raise RuntimeError(
            "affine null-space invariant failed: "
            f"maximum absolute error {maximum_affine_walk_invariant_error}"
        )
    if maximum_affine_residual_rank_excess > 0:
        raise RuntimeError(
            "affine null-space residual rank bound failed: "
            f"maximum excess {maximum_affine_residual_rank_excess}"
        )
    if maximum_ce_bound_violation > 5.0e-10:
        raise RuntimeError(
            "conditional-expectation guarantee failed: "
            f"maximum violation {maximum_ce_bound_violation}"
        )
    if maximum_ce_certificate_violation > 5.0e-9:
        raise RuntimeError(
            "rank/sensitivity CE certificate failed: "
            f"maximum normalized violation {maximum_ce_certificate_violation}"
        )
    if maximum_face_oracle_violation > 5.0e-10:
        raise RuntimeError(
            "reached-face oracle is worse than a feasible finish: "
            f"maximum violation {maximum_face_oracle_violation}"
        )
    if maximum_exact_optimum_violation > 5.0e-10:
        raise RuntimeError(
            "global exact optimum is worse than a comparison: "
            f"maximum violation {maximum_exact_optimum_violation}"
        )
    if maximum_bernoulli_identity_error > 1.0e-12:
        raise RuntimeError(
            "same-grid Bernoulli analytic-row mismatch: "
            f"maximum error {maximum_bernoulli_identity_error}"
        )
    if len(dynamic_rows) != expected_dynamic_rows:
        raise RuntimeError(
            f"dynamic configuration incomplete: {len(dynamic_rows)} "
            f"rows, expected {expected_dynamic_rows}"
        )
    if len(affine_rows) != expected_affine_rows:
        raise RuntimeError(
            f"affine configuration incomplete: {len(affine_rows)} "
            f"rows, expected {expected_affine_rows}"
        )
    if len(static_rows) != expected_static_rows:
        raise RuntimeError(
            f"static configuration incomplete: {len(static_rows)} "
            f"rows, expected {expected_static_rows}"
        )
    if maximum_static_relative_identity_error > identity_tolerance:
        raise RuntimeError(
            "static metric identity failed: "
            f"maximum relative error {maximum_static_relative_identity_error}"
        )
    if maximum_centered_bias_objective_gap > identity_tolerance:
        raise RuntimeError(
            "centered-plus-bias objective identity failed: "
            f"maximum absolute error {maximum_centered_bias_objective_gap}"
        )

    return {
        "maximum_walk_invariant_error": maximum_walk_invariant_error,
        "maximum_residual_rank_excess": maximum_residual_rank_excess,
        "maximum_affine_walk_invariant_error": (
            maximum_affine_walk_invariant_error
        ),
        "maximum_affine_residual_rank_excess": (
            maximum_affine_residual_rank_excess
        ),
        "maximum_conditional_expectation_bound_violation": (
            maximum_ce_bound_violation
        ),
        "maximum_rank_sensitivity_certificate_violation": (
            maximum_ce_certificate_violation
        ),
        "maximum_reached_face_oracle_violation": max(
            maximum_face_oracle_violation, 0.0
        ),
        "maximum_global_exact_optimum_violation": max(
            maximum_exact_optimum_violation, 0.0
        ),
        "maximum_same_grid_bernoulli_identity_error": (
            maximum_bernoulli_identity_error
        ),
        "dynamic_rows_observed": len(dynamic_rows),
        "dynamic_rows_expected": expected_dynamic_rows,
        "affine_rows_observed": len(affine_rows),
        "affine_rows_expected": expected_affine_rows,
        "static_rows_observed": len(static_rows),
        "static_rows_expected": expected_static_rows,
        "maximum_static_analytic_direct_error": maximum_static_identity_error,
        "maximum_static_relative_analytic_direct_error": (
            maximum_static_relative_identity_error
        ),
        "maximum_centered_bias_objective_gap": (
            maximum_centered_bias_objective_gap
        ),
        "walk_validation_tolerance": walk_tolerance,
        "identity_validation_tolerance": identity_tolerance,
    }


def _correlation(x: list[float], y: list[float]) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _dynamic_stats(
    selected: list[dict[str, Any]],
    *,
    label: str,
    include_gap: bool,
) -> dict[str, Any]:
    ratios = [row["ratio_to_dither"] for row in selected]
    median, low, high = bootstrap_median_ci(ratios, label=label)
    block_medians = [
        float(
            np.median(
                [
                    row["ratio_to_dither"]
                    for row in selected
                    if row["block_id"] == block_id
                ]
            )
        )
        for block_id in sorted({int(row["block_id"]) for row in selected})
    ]
    result: dict[str, Any] = {
        **quantiles(ratios),
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
        "per_block_median": quantiles(block_medians),
        "runtime_seconds": quantiles(
            [row["runtime_seconds"] for row in selected]
        ),
        "same_grid_bernoulli_ratio_to_dither": quantiles(
            [
                row["same_grid_bernoulli_expected_squared_error"]
                / row["dither_expected_squared_error"]
                for row in selected
            ]
        ),
        "exact_cancellation_frequency": float(
            np.mean([row["exact_cancellation"] for row in selected])
        ),
        "blocks": len(block_medians),
        "residues_per_block": len(selected) // len(block_medians),
        "count": len(selected),
    }
    if include_gap:
        result["normalized_additive_gap"] = quantiles(
            [float(row["normalized_additive_gap"]) for row in selected]
        )
    return result


def _summarize_dynamic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[str, Any] = {}
    exact_methods = (
        "exact",
        "rtn",
        "bernoulli_reference",
        "walk_arbitrary",
        "walk_ce",
        "null_face",
    )
    for rank in EXACT_RANKS:
        exact[str(rank)] = {}
        for method in exact_methods:
            selected = [
                row
                for row in rows
                if row["group"] == "exact"
                and row["rank"] == rank
                and row["method"] == method
            ]
            exact[str(rank)][method] = _dynamic_stats(
                selected,
                label=f"discrepancy:exact:r{rank}:{method}",
                include_gap=True,
            )

    scalable: dict[str, Any] = {}
    scalable_methods = (
        "rtn",
        "bernoulli_reference",
        "walk_arbitrary",
        "walk_ce",
        "null_face",
    )
    for k in SCALABLE_K:
        scalable[str(k)] = {}
        for rank in SCALABLE_RANKS:
            if rank > k:
                continue
            scalable[str(k)][str(rank)] = {}
            for method in scalable_methods:
                selected = [
                    row
                    for row in rows
                    if row["group"] == "scalable"
                    and row["K"] == k
                    and row["rank"] == rank
                    and row["method"] == method
                ]
                scalable[str(k)][str(rank)][method] = _dynamic_stats(
                    selected,
                    label=f"discrepancy:scale:K{k}:r{rank}:{method}",
                    include_gap=False,
                )

    imbalance: dict[str, Any] = {}
    for rank in IMBALANCE_RANKS:
        imbalance[str(rank)] = {}
        for exponent in IMBALANCE_EXPONENTS:
            key = f"{exponent:g}"
            imbalance[str(rank)][key] = {}
            for method in scalable_methods:
                selected = [
                    row
                    for row in rows
                    if row["group"] == "imbalance"
                    and row["rank"] == rank
                    and row["imbalance_exponent"] == exponent
                    and row["method"] == method
                ]
                stats = _dynamic_stats(
                    selected,
                    label=(
                        f"discrepancy:imbalance:r{rank}:"
                        f"alpha{exponent:g}:{method}"
                    ),
                    include_gap=False,
                )
                stats["beta_rms_over_beta_max"] = quantiles(
                    [row["beta_rms_over_beta_max"] for row in selected]
                )
                stats["ce_dither_certificate_ratio"] = quantiles(
                    [row["ce_dither_certificate_ratio"] for row in selected]
                )
                imbalance[str(rank)][key][method] = stats

    runtime: dict[str, Any] = {}
    for group in ("exact", "scalable", "imbalance"):
        runtime[group] = {}
        for method in sorted(
            {row["method"] for row in rows if row["group"] == group}
        ):
            runtime_values = [
                float(row["runtime_seconds"])
                for row in rows
                if row["group"] == group and row["method"] == method
            ]
            runtime[group][method] = quantiles(runtime_values)
    return {
        "exact_K": EXACT_K,
        "trials_per_configuration": DYNAMIC_TRIALS,
        "independent_blocks_per_configuration": DYNAMIC_BLOCKS,
        "dither_estimand": "Delta^2 sum_k ||B_k||_2^2 / 12 with Delta=1",
        "same_grid_bernoulli_estimand": (
            "Delta^2 sum_k theta_k(1-theta_k)||B_k||_2^2 with Delta=1"
        ),
        "generator": {
            "output_dimension": "p=r",
            "directions": (
                "iid standard-Gaussian row directions, normalized rowwise"
            ),
            "balanced_profile": "all row sensitivities exactly one",
            "imbalance_profile": (
                "randomly permuted j^{-alpha}, j=1,...,K, rescaled to unit RMS"
            ),
            "imbalance_K": IMBALANCE_K,
            "imbalance_ranks": list(IMBALANCE_RANKS),
            "imbalance_exponents": list(IMBALANCE_EXPONENTS),
            "numerical_rank": "numpy.linalg.matrix_rank default SVD tolerance",
            "fractional_residues": "iid Uniform[0,1]",
            "common_gap": 1.0,
        },
        "methods": {
            "walk_arbitrary": (
                "small-SVD null-space reduction plus nearest residual rounding; "
                "polynomial"
            ),
            "walk_ce": (
                "same reduction plus deterministic conditional-expectation "
                "residual rounding; polynomial"
            ),
            "null_face": (
                "same reduction plus exhaustive reached-face enumeration; "
                "cost 2^|S| and not the polynomial walk"
            ),
            "bernoulli_reference": (
                "analytic independent same-grid stochastic-rounding expectation"
            ),
        },
        "exact": exact,
        "scalable": scalable,
        "imbalance": imbalance,
        "runtime_seconds": runtime,
    }


def _summarize_affine(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_fraction: dict[str, Any] = {}
    for fraction in AFFINE_CLIP_FRACTIONS:
        key = f"{fraction:g}"
        by_fraction[key] = {}
        for method in ("theta_start", "box_start"):
            selected = [
                row
                for row in rows
                if row["clip_fraction"] == fraction and row["method"] == method
            ]
            ratios = [row["ratio_to_affine_bernoulli"] for row in selected]
            median, low, high = bootstrap_median_ci(
                ratios,
                label=f"discrepancy:affine:rho{fraction:g}:{method}",
            )
            block_medians = [
                float(
                    np.median(
                        [
                            row["ratio_to_affine_bernoulli"]
                            for row in selected
                            if row["block_id"] == block_id
                        ]
                    )
                )
                for block_id in range(DYNAMIC_BLOCKS)
            ]
            by_fraction[key][method] = {
                **quantiles(ratios),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "per_block_median": quantiles(block_medians),
                "runtime_seconds": quantiles(
                    [row["runtime_seconds"] for row in selected]
                ),
                "box_objective": quantiles(
                    [row["box_objective"] for row in selected]
                ),
                "count": len(selected),
            }
    return {
        "K": AFFINE_K,
        "rank": AFFINE_RANK,
        "clip_fractions": list(AFFINE_CLIP_FRACTIONS),
        "blocks_per_fraction": DYNAMIC_BLOCKS,
        "residues_per_block": DYNAMIC_TRIALS,
        "fixed_error_generator": (
            "round(rho*K) coordinates forced to a clipping endpoint; "
            "outside-side signs are iid Rademacher and overshoots iid "
            "Uniform[0.25,1.25] grid steps"
        ),
        "reference": (
            "||g_fix||^2 plus the active-coordinate same-grid Bernoulli "
            "expected energy"
        ),
        "methods": by_fraction,
    }


def _summarize_static(
    rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition in STATIC_CONDITIONS:
        condition_methods: dict[str, Any] = {}
        for bits in STATIC_BITS:
            condition_methods[str(bits)] = {}
            for method in STATIC_METHODS:
                selected = [
                    row
                    for row in rows
                    if row["bits"] == bits
                    and row["method"] == method
                    and row["calibration_condition"] == condition
                ]
                split_indices = sorted(
                    {int(row["split_index"]) for row in selected}
                )
                split_ratios = [
                    float(
                        np.median(
                            [
                                row["heldout_ratio_to_rtn"]
                                for row in selected
                                if row["split_index"] == split_index
                            ]
                        )
                    )
                    for split_index in split_indices
                ]
                split_calibration_mse = [
                    float(
                        np.median(
                            [
                                row["calibration_evaluation_mse"]
                                for row in selected
                                if row["split_index"] == split_index
                            ]
                        )
                    )
                    for split_index in split_indices
                ]
                split_heldout_mse = [
                    float(
                        np.median(
                            [
                                row["heldout_direct_mse"]
                                for row in selected
                                if row["split_index"] == split_index
                            ]
                        )
                    )
                    for split_index in split_indices
                ]
                median, low, high = bootstrap_median_ci(
                    split_ratios,
                    label=(
                        f"discrepancy:static:{condition}:"
                        f"b{bits}:{method}:split-medians"
                    ),
                )
                split_correlations = []
                for split_index in split_indices:
                    split_rows = [
                        row
                        for row in selected
                        if row["split_index"] == split_index
                    ]
                    if len(split_rows) < 2:
                        continue
                    correlation = _correlation(
                        [
                            row["calibration_evaluation_mse"]
                            for row in split_rows
                        ],
                        [row["heldout_direct_mse"] for row in split_rows],
                    )
                    if math.isfinite(correlation):
                        split_correlations.append(correlation)
                if split_correlations:
                    correlation_by_split: dict[str, float | int] = {
                        **quantiles(split_correlations),
                        "count": len(split_correlations),
                    }
                else:
                    correlation_by_split = {
                        "q25": math.nan,
                        "median": math.nan,
                        "q75": math.nan,
                        "count": 0,
                    }
                ratios = [row["heldout_ratio_to_rtn"] for row in selected]
                condition_methods[str(bits)][method] = {
                    **quantiles(split_ratios),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "per_split_median": {
                        str(split_index): split_ratio
                        for split_index, split_ratio in zip(
                            split_indices, split_ratios, strict=True
                        )
                    },
                    "raw_ratio": quantiles(ratios),
                    "calibration_to_heldout_correlation": _correlation(
                        [
                            row["calibration_evaluation_mse"]
                            for row in selected
                        ],
                        [row["heldout_direct_mse"] for row in selected],
                    ),
                    "calibration_ratio_to_heldout_ratio_correlation": (
                        _correlation(
                            [
                                row["calibration_ratio_to_rtn"]
                                for row in selected
                            ],
                            [
                                row["heldout_ratio_to_rtn"]
                                for row in selected
                            ],
                        )
                    ),
                    "split_median_calibration_to_heldout_correlation": (
                        _correlation(split_calibration_mse, split_heldout_mse)
                    ),
                    "calibration_to_heldout_correlation_by_split": (
                        correlation_by_split
                    ),
                    "heldout_mse": quantiles(
                        [row["heldout_direct_mse"] for row in selected]
                    ),
                    "flip_count": quantiles(
                        [row["flip_count"] for row in selected]
                    ),
                    "runtime_seconds": quantiles(
                        [row["runtime_seconds"] for row in selected]
                    ),
                    "output_bias_l2": quantiles(
                        [row["output_bias_l2"] for row in selected]
                    ),
                    "split_count": len(split_indices),
                    "count": len(selected),
                }
        conditions[condition] = {
            "calibration_size": (
                CALIBRATION_SIZE
                if condition == "n128"
                else metadata["calibration_pool_size"]
            ),
            "subsets_per_split": (
                CALIBRATION_SUBSETS if condition == "n128" else 1
            ),
            "methods": condition_methods,
        }
    return {
        "dataset": "scikit-learn Digits, raw pixels divided by 16",
        "split": (
            "five deterministic stratified 60% fit / 20% calibration pool / "
            "20% test splits"
        ),
        "splits": STATIC_SPLITS,
        "calibration_subsets_per_split": CALIBRATION_SUBSETS,
        "calibration_subsets": CALIBRATION_SUBSETS,
        "calibration_size": CALIBRATION_SIZE,
        "calibration_conditions": list(STATIC_CONDITIONS),
        "bits": list(STATIC_BITS),
        "metadata": metadata,
        "maximum_analytic_direct_error": float(
            max(
                max(row["calibration_identity_error"], row["heldout_identity_error"])
                for row in rows
            )
        ),
        "maximum_centered_bias_objective_gap": float(
            max(
                row["centered_bias_objective_gap"]
                for row in rows
                if row["method"] == "centered_bias"
            )
        ),
        "conditions": conditions,
        # Preserve the original access path for the 128-sample condition.
        "methods": conditions["n128"]["methods"],
    }


def _write_figures(
    dynamic_rows: list[dict[str, Any]],
    static_rows: list[dict[str, Any]],
    affine_rows: list[dict[str, Any]],
) -> list[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir = REPO_ROOT / "figures" / "discrepancy"
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    colors = {
        1: "#0072B2",
        2: "#D55E00",
        4: "#009E73",
        8: "#CC79A7",
        16: "#666666",
    }
    markers = {1: "o", 2: "s", 4: "^", 8: "D", 16: "v"}
    rtn_x = []
    rtn_y = []
    for k in SCALABLE_K:
        for rank in SCALABLE_RANKS:
            selected = [
                row["ratio_to_dither"]
                for row in dynamic_rows
                if row["group"] == "scalable"
                and row["method"] == "rtn"
                and row["K"] == k
                and row["rank"] == rank
            ]
            rtn_x.append(k / rank)
            rtn_y.append(float(np.median(selected)))
    ax.scatter(
        rtn_x,
        rtn_y,
        s=24,
        facecolors="none",
        edgecolors="#666666",
        linewidths=0.8,
        label="RTN median",
        zorder=2,
    )
    for rank in SCALABLE_RANKS:
        x_values = []
        medians = []
        q25s = []
        q75s = []
        for k in SCALABLE_K:
            selected = [
                row["ratio_to_dither"]
                for row in dynamic_rows
                if row["group"] == "scalable"
                and row["method"] == "walk_ce"
                and row["K"] == k
                and row["rank"] == rank
            ]
            stats = quantiles(selected)
            x_values.append(k / rank)
            medians.append(stats["median"])
            q25s.append(stats["q25"])
            q75s.append(stats["q75"])
        x_array = np.asarray(x_values)
        ax.plot(
            x_array,
            medians,
            marker=markers[rank],
            linestyle="-",
            color=colors[rank],
            label=rf"$r={rank}$",
            linewidth=1.1,
            markersize=3.2,
            zorder=3,
        )
        ax.fill_between(x_array, q25s, q75s, color=colors[rank], alpha=0.12)
    bernoulli_median = float(
        np.median(
            [
                row["ratio_to_dither"]
                for row in dynamic_rows
                if row["group"] == "scalable"
                and row["method"] == "bernoulli_reference"
            ]
        )
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="dither")
    ax.axhline(
        bernoulli_median,
        color="#777777",
        linestyle=":",
        linewidth=0.9,
        label="same-grid Bernoulli median",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"dimension-to-rank ratio $K/r$")
    ax.set_ylabel("squared error / dither expectation")
    ax.set_title("CE completion")
    ax.legend(ncol=2, frameon=False)
    ax.grid(which="both", alpha=0.18)

    ax = axes[1]
    ratio_values = sorted(
        {
            row["K"] / row["rank"]
            for row in dynamic_rows
            if row["group"] == "scalable"
        }
    )
    method_style = {
        "rtn": ("RTN", "#777777", "o", "--"),
        "bernoulli_reference": ("same-grid Bernoulli", "#CC79A7", "P", ":"),
        "walk_arbitrary": ("nearest completion", "#D55E00", "s", "-."),
        "walk_ce": ("CE completion", "#0072B2", "^", "-"),
        "null_face": ("reached-face oracle", "#009E73", "D", "-"),
    }
    for method, (label, color, marker, linestyle) in method_style.items():
        medians = []
        for ratio in ratio_values:
            values = [
                row["ratio_to_dither"]
                for row in dynamic_rows
                if row["group"] == "scalable"
                and row["method"] == method
                and row["K"] / row["rank"] == ratio
            ]
            medians.append(float(np.median(values)))
        ax.plot(
            ratio_values,
            medians,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.0,
            markersize=3.2,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"dimension-to-rank ratio $K/r$")
    ax.set_title("finishing-rule comparison")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(which="both", alpha=0.18)
    fig.tight_layout()
    path = figure_dir / "dynamic_scaling.svg"
    save_figure(fig, path)
    plt.close(fig)
    artifacts.append(path)

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    for rank, color, marker in (
        (1, "#0072B2", "o"),
        (4, "#D55E00", "s"),
    ):
        measured_y = []
        certified_y = []
        for exponent in IMBALANCE_EXPONENTS:
            selected = [
                row
                for row in dynamic_rows
                if row["group"] == "imbalance"
                and row["rank"] == rank
                and row["imbalance_exponent"] == exponent
                and row["method"] == "walk_ce"
            ]
            measured_y.append(
                float(np.median([row["ratio_to_dither"] for row in selected]))
            )
            certified_y.append(
                float(
                    np.median(
                        [row["ce_dither_certificate_ratio"] for row in selected]
                    )
                )
            )
        x = np.asarray(IMBALANCE_EXPONENTS)
        ax.plot(
            x,
            measured_y,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=1.35,
            markersize=4.2,
            label=rf"$r={rank}$: measured",
            zorder=3,
        )
        ax.plot(
            x,
            certified_y,
            color=color,
            linestyle="--",
            linewidth=1.15,
            label=rf"$r={rank}$: certificate",
            zorder=2,
        )
        ax.text(
            2.08,
            measured_y[-1],
            rf"$r={rank}$ measured",
            color=color,
            fontsize=8.0,
            va="center",
        )
        ax.text(
            2.08,
            certified_y[-1],
            rf"$r={rank}$ certificate",
            color=color,
            fontsize=8.0,
            va="center",
        )
    ax.axhline(1.0, color="#555555", linestyle=":", linewidth=0.85, zorder=1)
    ax.text(
        2.58,
        1.12,
        "dither",
        color="#555555",
        fontsize=8.0,
        ha="right",
        va="bottom",
    )
    ax.set_yscale("log")
    ax.set_xlim(-0.06, 2.65)
    ax.set_ylim(1.0e-4, 2.0e1)
    ax.set_xticks(IMBALANCE_EXPONENTS)
    ax.set_xticklabels(("0", "0.5", "1", "2"))
    ax.set_xlabel(r"row-norm exponent $\alpha$ (more imbalance $\rightarrow$)")
    ax.set_ylabel("error / dither expectation")
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.55)
    ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.35)
    fig.tight_layout(pad=0.25)
    path = figure_dir / "sensitivity_imbalance.svg"
    save_figure(fig, path, pad_inches=0.02)
    plt.close(fig)
    artifacts.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=False)
    method_styles = {
        "centered": ("centered, fixed bias", "#D55E00", "o"),
        "centered_bias": ("centered + bias", "#009E73", "s"),
        "uncentered": ("uncentered", "#0072B2", "^"),
    }
    condition_styles = {
        "n128": ("128 samples", 0.65, 22, True),
        "full": ("full pool", 1.0, 34, False),
    }
    for ax, bits in zip(axes, STATIC_BITS, strict=True):
        for method, (method_label, color, marker) in method_styles.items():
            for condition, (
                condition_label,
                alpha,
                size,
                filled,
            ) in condition_styles.items():
                selected = [
                    row
                    for row in static_rows
                    if row["bits"] == bits
                    and row["method"] == method
                    and row["calibration_condition"] == condition
                ]
                ax.scatter(
                    [row["calibration_ratio_to_rtn"] for row in selected],
                    [row["heldout_ratio_to_rtn"] for row in selected],
                    s=size,
                    alpha=alpha,
                    marker=marker,
                    label=f"{method_label}, {condition_label}",
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    linewidths=0.8,
                )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(1.0, color="black", linestyle=":", linewidth=0.6)
        ax.set_title(f"{bits}-bit grid")
        ax.set_xlabel("calibration MSE / split-matched RTN")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("held-out product MSE / RTN")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=9,
        ncol=3,
        loc="lower center",
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    path = figure_dir / "static_transfer.svg"
    save_figure(fig, path)
    plt.close(fig)
    artifacts.append(path)

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    for method, label, color, marker, marker_face in (
        ("theta_start", "residue start", "#D55E00", "o", "#D55E00"),
        ("box_start", "QP start", "#0072B2", "s", "white"),
    ):
        medians = []
        q25s = []
        q75s = []
        for fraction in AFFINE_CLIP_FRACTIONS:
            values = [
                row["ratio_to_affine_bernoulli"]
                for row in affine_rows
                if row["clip_fraction"] == fraction and row["method"] == method
            ]
            stats = quantiles(values)
            medians.append(stats["median"])
            q25s.append(stats["q25"])
            q75s.append(stats["q75"])
        x = 100.0 * np.asarray(AFFINE_CLIP_FRACTIONS)
        medians_array = np.asarray(medians)
        interval = np.vstack(
            (medians_array - np.asarray(q25s), np.asarray(q75s) - medians_array)
        )
        ax.errorbar(
            x,
            medians_array,
            yerr=interval,
            color=color,
            marker=marker,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=1.0,
            linewidth=1.45,
            markersize=4.4,
            elinewidth=0.8,
            capsize=2.2,
            label=label,
            zorder=3,
        )
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-0.008, 0.255)
    ax.set_xticks((0, 5, 10))
    ax.set_xlabel(r"clipped coordinates $\rho$ (%)")
    ax.set_ylabel("error / affine Bernoulli reference")
    ax.legend(
        frameon=False,
        fontsize=7.0,
        handlelength=2.2,
        handletextpad=0.5,
        loc="upper left",
    )
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.55)
    fig.tight_layout(pad=0.25)
    path = figure_dir / "affine_saturation.svg"
    save_figure(fig, path, pad_inches=0.02)
    plt.close(fig)
    artifacts.append(path)
    return artifacts



def _runtime_rows(
    dynamic_rows: list[dict[str, Any]],
    static_rows: list[dict[str, Any]],
    affine_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in dynamic_rows:
        key = (
            f"dynamic_{row['group']}",
            row["K"],
            row["rank"],
            row["imbalance_exponent"],
            row["method"],
        )
        grouped[key].append(row["runtime_seconds"])
    for row in affine_rows:
        key = (
            "dynamic_affine",
            row["K"],
            row["rank"],
            row["clip_fraction"],
            row["method"],
        )
        grouped[key].append(row["runtime_seconds"])
    for row in static_rows:
        key = (
            "static_digits",
            64,
            10,
            f"{row['bits']}bit_{row['calibration_condition']}",
            row["method"],
        )
        grouped[key].append(row["runtime_seconds"])

    records = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        stats = quantiles(values)
        records.append(
            {
                "experiment": key[0],
                "dimension": key[1],
                "rank_or_output_dimension": key[2],
                "configuration_parameter": key[3],
                "method": key[4],
                "median_seconds": stats["median"],
                "q25_seconds": stats["q25"],
                "q75_seconds": stats["q75"],
                "count": len(values),
            }
        )
    return records


def _write_discrepancy_manifest(
    artifacts: list[Path],
    *,
    dataset_hash: str,
) -> Path:
    path = RESULTS_DIR / "discrepancy_manifest.json"
    manifest = environment_manifest(
        artifacts=artifacts,
        dataset_hashes={"sklearn_digits_data_and_targets": dataset_hash},
        command="python -m experiments.reproduce discrepancy",
    )
    manifest["protocol"] = {
        "dynamic_blocks_per_configuration": DYNAMIC_BLOCKS,
        "residues_per_block": DYNAMIC_TRIALS,
        "static_splits": STATIC_SPLITS,
        "static_128_sample_subsets_per_split": CALIBRATION_SUBSETS,
        "static_calibration_conditions": list(STATIC_CONDITIONS),
        "static_methods": list(STATIC_METHODS),
    }
    write_json(path, manifest)
    return path


def run() -> dict[str, Any]:
    """Run the discrepancy experiments and write reproducibility artifacts."""

    dynamic_rows = _run_dynamic()
    affine_rows = _run_affine()
    static_rows, static_metadata = _run_static()
    validation = _validate_run(dynamic_rows, static_rows, affine_rows)
    summary = {
        "base_seed": BASE_SEED,
        "dynamic": _summarize_dynamic(dynamic_rows),
        "affine_saturation": _summarize_affine(affine_rows),
        "static": _summarize_static(static_rows, static_metadata),
        "validation": validation,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dynamic_path = RESULTS_DIR / "discrepancy_dynamic.csv"
    affine_path = RESULTS_DIR / "discrepancy_affine.csv"
    static_path = RESULTS_DIR / "discrepancy_static.csv"
    runtime_path = RESULTS_DIR / "discrepancy_runtime.csv"
    summary_path = RESULTS_DIR / "discrepancy_summary.json"
    write_csv(dynamic_path, dynamic_rows)
    write_csv(affine_path, affine_rows)
    write_csv(static_path, static_rows)
    write_csv(
        runtime_path, _runtime_rows(dynamic_rows, static_rows, affine_rows)
    )
    write_json(summary_path, summary)
    figures = _write_figures(dynamic_rows, static_rows, affine_rows)
    artifacts = [
        dynamic_path,
        affine_path,
        static_path,
        runtime_path,
        summary_path,
        *figures,
    ]
    discrepancy_manifest = _write_discrepancy_manifest(
        artifacts,
        dataset_hash=static_metadata["dataset_sha256"],
    )
    artifacts.append(discrepancy_manifest)
    return {
        "summary": summary,
        "artifacts": artifacts,
        "dataset_hashes": {
            "sklearn_digits_data_and_targets": static_metadata["dataset_sha256"]
        },
    }
