# Reproduction and verification

## Canonical command

Use CPython 3.13.7 and the resolved lock file:

```sh
.venv-experiments/bin/python -m experiments.reproduce all
.venv-experiments/bin/python -m experiments.verify_artifacts
```

The selector may be `nuclear`, `discrepancy`, or `all`. The public default
writes only computational evidence: CSV, JSON, SVG, and provenance manifests.
It never writes or downloads a manuscript and has no manuscript-output mode.

## What the command changes

- `experiments/results/`
- `figures/nuclear/`
- `figures/discrepancy/`

Run in a clean clone or dedicated branch. Do not combine files from different
runs because each manifest describes one internally consistent inventory.

## Reproduction contract

| Artifact class | Expected behavior |
| --- | --- |
| Seeded scientific CSV and JSON fields | Deterministic under the pinned environment |
| Runtime columns | Machine- and load-dependent |
| Platform, package, and thread metadata | Records the actual run and may change |
| SVG charts | Stable under the pinned Matplotlib stack; always verify visually |
| SHA-256 manifests | Must match every file from the same complete run |

`experiments.verify_artifacts` checks integrity. It establishes that committed
files match their recorded hashes, not that an independently changed method is
scientifically equivalent. Unit tests and validation residuals supply that
second gate.

## Reference run

The committed artifacts use CPython 3.13.7, NumPy 2.3.5, SciPy 1.17.0,
Matplotlib 3.10.8, and scikit-learn 1.8.0. Full release runs took approximately
195--213 seconds on Apple-silicon macOS. The implementation is CPU-only.

## Release checks

```sh
make validate PYTHON=.venv-experiments/bin/python
python scripts/execute_notebooks.py --python .venv-experiments/bin/python
```

The release gate checks unit tests, committed artifact hashes, notebook
execution, and a strict public allowlist that rejects manuscript formats and
private-review paths.
