# Discrepancy experiment protocol

Run the discrepancy evidence:

```sh
python -m experiments.reproduce discrepancy
```

This entry point validates the pinned scientific packages before starting.

Run its focused tests with:

```sh
python -m unittest experiments.tests.test_discrepancy -v
```

All streams derive from base seed `20260723`. There is no reduced or quick
configuration hidden behind either command.

## Dynamic evidence

For each balanced configuration, the output block has shape `K x r`. Its row
directions are independent standard-Gaussian draws normalized rowwise, so every
row has sensitivity one; `p=r`, the common adjacent-level gap is one, numerical
rank uses `numpy.linalg.matrix_rank`, and residues are iid `Uniform[0,1]`.
Every `(K,r)` uses 10 independently generated blocks and 25 independent residue
vectors per block.

- Exact study: `K=16`, `r in {1,2,4,8}`.
- Scalable study: `K in {64,128,256,512,1024}` and
  `r in {1,2,4,8,16}`.
- Imbalance study: `K=256`, `r in {1,4}`, and row sensitivities proportional to
  `j**(-alpha)` for `alpha in {0,0.5,1,2}`, randomly permuted and rescaled to
  unit RMS.

The methods are intentionally separate:

- `walk_arbitrary`: polynomial small-SVD null-space reduction followed by
  deterministic nearest rounding of the residual coordinates;
- `walk_ce`: the same reduction followed by deterministic
  conditional-expectation rounding;
- `null_face`: exhaustive optimization over the reached residual face, with
  cost `2**|S|`; this is an oracle, not the polynomial walk;
- `bernoulli_reference`: the analytic same-grid independent stochastic
  rounding expectation
  `sum(theta*(1-theta)*row_norm_squared)`;
- `rtn`: ordinary round-to-nearest;
- `exact`: full `2**16` enumeration, only in the exact study.

The dither reference is the analytic no-overload value
`sum(row_norm_squared)/12`. Raw rows include total, walk, and finish runtimes,
the sensitivity ratio, the rank/sensitivity certificate, the reached-face
size, and the walk invariant residual. The summary reports both pooled
residue-level statistics and the distribution of the ten per-block medians.

## Affine fixed-offset evidence

The affine study uses `K=256`, rank four, 10 blocks, 25 residue vectors, and
fixed-coordinate fractions `0`, `0.05`, and `0.10`. Fixed coordinates emulate
endpoint saturation: their outside-side signs are iid Rademacher and their
overshoots are iid `Uniform[0.25,1.25]` grid steps. It compares a walk starting
at the observed residues with a box-constrained least-squares QP start;
both use conditional-expectation finishing.

## Static evidence

The static protocol uses five deterministic stratified 60/20/20 Digits splits,
raw pixels divided by 16, and ridge `1e-3` with no intercept.  Within every
split it evaluates 20 stratified, without-replacement calibration subsets of
128 examples and one condition using the complete 359-example calibration
pool.  The methods are RTN, centered-metric one-flip descent with a fixed
output bias, the same centered endpoint with the optimal post-rounding
correction `c = -mean(X_cal) @ (B_hat - B)`, and uncentered-metric one-flip
descent.  All use the same per-column signed symmetric 3- and 4-bit grids and
RTN initialization.  Confidence intervals bootstrap the five split-level
medians; runtime, the affine analytic/direct product-MSE identity, and the
centered-metric/bias-correction identity are recorded and validated.

Generated evidence lives in:

- `experiments/results/discrepancy_{dynamic,affine,static,runtime}.csv`;
- `experiments/results/discrepancy_summary.json`;
- `experiments/results/discrepancy_manifest.json` (environment, dataset hash,
  and SHA-256 hashes for every discrepancy artifact);
- `figures/discrepancy/{dynamic_scaling,sensitivity_imbalance,affine_saturation,static_transfer}.svg`.
