# Reproducing the gauge-floor experiments

The experiment code is self-contained and uses only public synthetic data plus
the `scikit-learn` Digits dataset bundled with the package. It does not import
code or data from either companion repository.

Create an environment and run the complete suite from the repository root:

```sh
python3.13 -m venv .venv-experiments
.venv-experiments/bin/python -m pip install -r experiments/requirements-lock.txt
.venv-experiments/bin/python -m unittest discover -s experiments/tests -v
.venv-experiments/bin/python -m experiments.reproduce all
.venv-experiments/bin/python -m experiments.verify_artifacts
```

The reference environment is CPython 3.13.7 and the full CPU-only run takes
about 3--5 minutes on the recorded Apple-silicon system. Platform-specific
commands are in the [getting-started guide](../docs/getting-started.md).
The [reproduction contract](../docs/reproduction.md) explains which timing and
platform fields are expected to differ across machines.

The reproduction command runs every stated configuration; it has no hidden
quick mode. The positional selector may instead be `nuclear` or `discrepancy`. All random
streams derive from seed `20260723` and from stable textual stream names, so
reordering experiments does not alter their samples.

Outputs are:

- [`experiments/results/*.csv`](results/): one row per method and trial;
- [`experiments/results/*_summary.json`](results/): dimensions, estimands, uncertainty, and
  summary statistics;
- [`experiments/results/{nuclear,discrepancy}_manifest.json`](results/): family-specific
  environment versions, protocol metadata, dataset hashes, and artifact
  SHA-256 hashes;
- [`experiments/results/manifest.json`](results/manifest.json): the same provenance for the selector
  used by the most recent complete command (normally `all`);
- [`figures/{nuclear,discrepancy}/*.svg`](../figures/): accessible vector charts.

The public pipeline never writes or downloads manuscript source and has no
manuscript-output mode.

Static rounding uses raw Digits pixels divided by 16 across five deterministic
stratified 60/20/20 splits.  Each split includes twenty stratified 128-sample
calibration subsets and a full-calibration-pool condition; the centered-metric
method is evaluated both with a fixed output bias and with its optimal
post-rounding bias correction.  The reported held-out product MSE is the mean
over samples and output columns. Dynamic dither values are the analytic
no-overload expectations, not Monte Carlo draws.

## Nuclear-floor protocol

The main nonsingular experiment uses square, full-rank factors with
$m=n=K=64$, singular values `geomspace(1, 0.15, K)`, and 50 independent
trials per family. Gaussian QR draws all orthogonal matrices. The three
factorization families are:

- `spiky`: diagonal square-root factors with 64 independent
  $N(0,1.35^2)$ log-scales; eight uniformly selected coordinates receive an
  additional independent shift of $+2.25$ or $-2.25$. The log-scales are
  mean-centered and clipped to `[-log(25), log(25)]`.
- `dense_diagonal`: Haar-oriented square-root factors followed by a positive
  diagonal gauge whose 64 log-scales are independent $N(0,1.35^2)$.
- `general_gauge`: the same Haar-oriented factors followed by
  $Q\operatorname{diag}(\exp(\operatorname{linspace}(-2,2,K)))Q^\top$, with
  an independent Haar $Q$.

`diagonal_balance_transform` is the closed-form minimizer of the energy
surrogate over determinant-normalized positive diagonal gauges: before the
common geometric-mean normalization, its scales are
`(row_energy(B) / column_energy(A))**0.25`; it uses no iterative tolerance.
`balance_transform` evaluates the positive-definite Gram-matrix formula by
symmetric eigendecompositions, with eigenvalue floor
`2e-14 * max(trace(P)/K, 1)`. The singular experiment has
`rank(A)=rank(B)=32`. Its one-based latent supports are coordinates `1:32`
for $A$ and `17:48` for $B$, leaving a 16-coordinate overlap and
`rank(A @ B)=16`. Both factors use nonzero values
`sqrt(geomspace(1, 0.2, 32))`. The experiment uses ten regularizers from
$10^{-1}$ through $10^{-10}$ and reports every regularization point rather
than stopping on a numerical tolerance.

The `full_gauge_svd_hadamard` baseline computes the product SVD, aligns the
free row and column coordinates with its singular vectors, and applies a
normalized Hadamard matrix. Because $K=64$, its leading-objective ratio is
one up to floating-point error. It appears in both the family and flattening
outputs.

The dimension experiment uses
$K\in\{16,32,64,128,256,1024\}$, $m=3K/2$, $n=5K/4$, and eight
independent Gaussian-QR Stiefel pairs at each dimension. Factors are generated
directly in their balanced form with singular values
`geomspace(1, 0.15, K)`, then Haar and randomized-Hadamard inner rotations are
compared with the exact full-gauge baseline. That baseline uses the
generator's known SVD rather than forming and decomposing the same dense
product again.

Nuclear-specific generated artifacts additionally include:

- `experiments/results/nuclear_dimension_scaling.csv`;
- `figures/nuclear/dimension_scaling.svg`.
