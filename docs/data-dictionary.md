# Experiment data dictionary

All CSV files are UTF-8 with a header row. Indices are zero-based. Times use
seconds, errors are squared Euclidean product errors or mean-squared errors as
named, and ratios are dimensionless.
Blank CSV cells mean “not applicable”; JSON converts non-finite floating-point
values to `null`.

## Common concepts

| Field or suffix | Meaning |
| --- | --- |
| `K`, `m`, `n`, `output_dimension_p` | Matrix dimensions defined in the corresponding experiment protocol |
| `trial`, `draw`, `block_id`, `split_index` | Deterministic replicate identifiers derived from base seed `20260723` |
| `method` | Algorithm or comparison baseline; use the protocol guides for exact definitions |
| `*_ratio`, `ratio_to_*` | Numerator divided by the named reference; values below one favor the numerator |
| `q25`, `median`, `q75` | 25th percentile, median, and 75th percentile |
| `runtime_seconds`, `*_runtime_seconds` | Wall-clock diagnostic; not a reproducibility invariant |
| `invariant_error`, `product_residual`, `*_identity_error` | Numerical residual that should be near floating-point zero |

## Nuclear-norm files

### `nuclear_trials.csv`

One row per factor family, trial, method, and same-output diagnostic. Dimensions,
bit widths, and `factorization` identify the instance. `objective`, `floor`, and
`objective_floor_ratio` are the main estimand. `alpha_a`, `alpha_b`, and
`alpha_effective` record range-to-energy factors. `energy_nuclear_ratio` isolates
the factorization-energy gap. Coherence and transform-conditioning fields
describe the construction; residual fields validate product preservation and
bit-width invariance.

### `nuclear_singular.csv`

| Columns | Meaning |
| --- | --- |
| `epsilon`, `condition_T` | Regularization and resulting transform condition number |
| `energy_product`, `nuclear_squared`, `relative_excess` | Energy objective, limiting nuclear-norm value, and relative gap |
| `product_residual`, `relative_product_residual` | Absolute and relative preservation checks |

### `nuclear_flattening.csv`

`method` and `draw` identify a full-gauge, Haar, randomized-Hadamard, or
deterministic-Hadamard construction. `objective_floor_ratio` measures loss above
the floor; `finite_set_coherence` measures realized flatness; `coherence_bound`
is present when a deterministic certificate is available.

### `nuclear_dimension_scaling.csv`

Each row records `m`, `n`, `K`, trial, and method for the dimension sweep. The
reported metrics are `objective_floor_ratio`, `alpha_effective`,
`energy_nuclear_ratio`, and `finite_set_coherence`.

## Discrepancy files

### `discrepancy_dynamic.csv`

| Column group | Fields | Meaning |
| --- | --- | --- |
| Instance | `group`, `K`, `rank`, `output_dimension_p`, `block_id`, `trial`, `imbalance_exponent` | Exact, scalable, or imbalance configuration |
| Error | `squared_error`, `dither_expected_squared_error`, `same_grid_bernoulli_expected_squared_error` | Observed deterministic loss and two analytic references |
| Ratios | `ratio_to_dither`, `ratio_to_same_grid_bernoulli`, `normalized_additive_gap` | Normalized comparisons used in tables and figures |
| Sensitivity | `beta_rms`, `beta_max`, `beta_rms_over_beta_max`, `ce_dither_certificate_ratio` | Row-energy concentration and deterministic certificate |
| Walk | `numerical_rank`, `residual_size`, `walk_steps`, `invariant_error` | Null-space reduction diagnostics |
| Finish | `ce_expected_bound`, `ce_bound_violation`, `exact_cancellation` | Conditional-expectation certificate and exact-zero indicator |
| Timing | `runtime_seconds`, `walk_runtime_seconds`, `finish_runtime_seconds` | Non-portable wall-clock measurements |

### `discrepancy_affine.csv`

`clip_fraction`, `fixed_count`, and `active_count` define the fixed-offset
instance. `squared_error`, `affine_same_grid_bernoulli_reference`, and
`ratio_to_affine_bernoulli` report performance. `fixed_offset_squared_norm` and
the `box_*` fields describe the continuous start. The remaining rank, walk,
conditional-expectation, and timing fields have the meanings above.

### `discrepancy_static.csv`

| Column group | Fields | Meaning |
| --- | --- | --- |
| Split and grid | `split_index`, `bits`, `calibration_condition`, `calibration_subset`, `calibration_size`, `calibration_pool_size`, `calibration_sampling`, `test_size` | Deterministic Digits split and calibration condition |
| Method | `method`, `bias_correction_applied`, `output_bias_l2` | Rounding metric and optional post-rounding bias |
| Calibration | `calibration_uncentered_mse`, `calibration_evaluation_mse`, `calibration_direct_mse`, `calibration_ratio_to_rtn` | Calibration objectives and RTN-normalized result |
| Held out | `heldout_analytic_mse`, `heldout_direct_mse`, `heldout_ratio_to_rtn`, `rtn_heldout_mse` | Analytic/direct test error and RTN reference |
| Identities | `calibration_identity_error`, `calibration_identity_relative_error`, `heldout_identity_error`, `heldout_identity_relative_error`, `centered_bias_objective_gap` | Floating-point validation of derived identities |
| Optimization | `optimization_metric_objective`, `flip_count`, `worst_final_improvement`, `variable_entries`, `total_entries` | One-flip descent objective and stopping diagnostics |
| Timing/reference | `runtime_seconds`, `rtn_calibration_mse` | Wall-clock time and calibration RTN baseline |

### `discrepancy_runtime.csv`

This derived timing table has one row per `experiment`, `dimension`,
`rank_or_output_dimension`, `configuration_parameter`, and `method`. It reports
the median, quartiles, and sample `count`. These values describe one run; they
are not cross-machine benchmarks.

## JSON and provenance

`nuclear_summary.json` and `discrepancy_summary.json` provide nested protocol
metadata, distribution summaries, reporting values, and maximum validation
residuals. Family-specific `*_manifest.json` files record the seed,
environment, protocol, dataset hash where applicable, and SHA-256 values for
their artifacts. `manifest.json` inventories the complete run, including the
family-specific manifests.

Validate the committed inventory with:

```sh
python -m experiments.verify_artifacts
```

For the precise generator definitions and sample counts, see
[`experiments/README.md`](../experiments/README.md) and
[`experiments/DISCREPANCY.md`](../experiments/DISCREPANCY.md).
