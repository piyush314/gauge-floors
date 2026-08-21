# Generated chart descriptions and data alternatives

These descriptions make the computational charts usable without relying on
color or visual pattern recognition. Each chart is an SVG with text preserved
as text; its source measurements are also available as CSV.

## Nuclear-floor experiments

### Objective ratios

Short description: A log-scale grouped bar chart compares five transformations
on spiky, dense diagonal-gauge, and dense general-gauge factor families. Two
boxplots compare objective loss and finite-set flatness for four flattening
methods.

Long description: The full-gauge singular-value-decomposition plus Hadamard
construction is at the nuclear floor, ratio one, in all three families. The
untransformed spiky and dense-diagonal gauges can be orders of magnitude above
the floor. General inner balancing followed by Haar or deterministic Hadamard
flattening reduces the gap to a single-digit ratio. The lower panels preserve
the same method ordering. Hatching, not color alone, separates the bar series.

Data: [`nuclear_trials.csv`](../experiments/results/nuclear_trials.csv) and
[`nuclear_flattening.csv`](../experiments/results/nuclear_flattening.csv).

### Dimension scaling

Short description: A line chart follows the objective-to-floor ratio from
inner dimension 16 to 1024.

Long description: The full-gauge series remains at one across all dimensions.
The two inner-only series rise gradually and nearly overlap. Distinct marker
shapes and line styles carry every series distinction independently of color.

Data:
[`nuclear_dimension_scaling.csv`](../experiments/results/nuclear_dimension_scaling.csv).

### Singular frontier

Short description: A log-log chart follows energy-objective excess as a
regularized singular transform becomes more ill-conditioned.

Long description: Reducing the regularization increases the transform
condition number while the energy objective approaches its singular limiting
value. The plotted excess falls nearly monotonically.

Data: [`nuclear_singular.csv`](../experiments/results/nuclear_singular.csv).

### Flattening comparison

Short description: Two boxplots compare objective ratio and finite-set
coherence for four flattening constructions.

Long description: The full-gauge construction is lowest on objective ratio.
Haar, randomized-Hadamard, and deterministic-Hadamard inner transforms cluster
above it. The companion coherence panel shows how evenly each construction
spreads the finite input vectors.

Data:
[`nuclear_flattening.csv`](../experiments/results/nuclear_flattening.csv).

## Product-aware rounding experiments

### Dynamic scaling

Short description: Two log-log panels compare deterministic completion methods
and stochastic references as the dimension-to-rank ratio increases.

Long description: Conditional-expectation error falls below the dither
reference and continues downward as `K/r` grows. Round-to-nearest remains much
closer to one. Conditional expectation and the reached-face oracle form the
lowest constructive and oracle curves. Markers and line styles distinguish
methods and ranks independently of color.

Data:
[`discrepancy_dynamic.csv`](../experiments/results/discrepancy_dynamic.csv).

### Sensitivity imbalance

Short description: Lines compare measured error and a deterministic certificate
as row-norm imbalance increases for ranks one and four.

Long description: Measured errors remain below their certificates. Increasing
imbalance weakens the certificate and can worsen measured error. Marker shape
identifies rank; line style separates measured values from certificates.

Data:
[`discrepancy_dynamic.csv`](../experiments/results/discrepancy_dynamic.csv).

### Affine saturation

Short description: Lines compare two continuous starting points as the
percentage of clipped coordinates increases.

Long description: With no clipping, the residue and box-constrained starts are
similar. As the clipped fraction rises, residue-start error grows while the
box-constrained start remains near zero. Marker and line style identify the
starting rule independently of color.

Data:
[`discrepancy_affine.csv`](../experiments/results/discrepancy_affine.csv).

### Static transfer

Short description: Two scatter plots compare calibration and held-out
mean-squared error, normalized by split-matched round-to-nearest, for three- and
four-bit grids.

Long description: Reference lines at one separate improvements from
regressions. Centered rounding with fixed bias often exceeds one on held-out
data. Centered rounding with bias correction and uncentered rounding tend to
remain below one, especially with four bits. Marker shape identifies the
rounding method; filled markers use 128-sample calibration subsets and open
markers use the full calibration pool.

Data: [`discrepancy_static.csv`](../experiments/results/discrepancy_static.csv).
