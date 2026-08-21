# Gauge Floors: Reproducibility and Tutorials

[![quality](https://github.com/piyush314/gauge-floors/actions/workflows/quality.yml/badge.svg)](https://github.com/piyush314/gauge-floors/actions/workflows/quality.yml)
[![notebooks](https://github.com/piyush314/gauge-floors/actions/workflows/notebooks.yml/badge.svg)](https://github.com/piyush314/gauge-floors/actions/workflows/notebooks.yml)
[![pages](https://github.com/piyush314/gauge-floors/actions/workflows/pages.yml/badge.svg)](https://github.com/piyush314/gauge-floors/actions/workflows/pages.yml)

Reproducible experiments, executable tutorials, and learning materials for two
studies of scalar quantization in matrix products:

- gauge transformations and nuclear-norm limits under independent noise; and
- product-aware deterministic rounding through low-rank cancellation.

> **Public-scope boundary:** this repository deliberately contains no
> manuscript source, LaTeX, bibliography, journal files, submission archives,
> or manuscript Git history. Those materials are maintained separately and
> privately. This public repository owns only the computational evidence and
> educational interfaces.

This is a research artifact, not a production quantization library.

## Start here

The reference environment is CPython 3.13.7 and runs on CPU only.

```sh
git clone https://github.com/piyush314/gauge-floors.git
cd gauge-floors
python3.13 -m venv .venv-experiments
.venv-experiments/bin/python -m pip install -r experiments/requirements-lock.txt
make validate PYTHON=.venv-experiments/bin/python
```

On Windows, replace `.venv-experiments/bin/python` with
`.venv-experiments\Scripts\python.exe`. The
[getting-started guide](docs/getting-started.md) gives complete commands and
troubleshooting notes.

## Four entry points

| Goal | Start with |
| --- | --- |
| Understand the ideas | [Guided tutorials](tutorials/README.md) |
| Work interactively | [Executable notebooks](notebooks/README.md) |
| Inspect committed evidence | [Data dictionary](docs/data-dictionary.md) |
| Regenerate everything | `python -m experiments.reproduce all` |

The [project site](https://piyush314.github.io/gauge-floors/) mirrors the
learning path and adds accessible explanations of every generated chart.

## Reproduce and verify

Run the complete, seeded CPU protocol:

```sh
.venv-experiments/bin/python -m experiments.reproduce all
.venv-experiments/bin/python -m experiments.verify_artifacts
```

The full reference run takes roughly 3--5 minutes on Apple silicon. It writes
CSV and JSON evidence, vector figures, and SHA-256 provenance manifests. The
pipeline has no manuscript-output mode and does not generate or download paper
source.

Selectors `nuclear` and `discrepancy` run either experiment family. Every
random stream derives from base seed `20260723` and a stable textual stream
name. Scientific fields are deterministic under the pinned environment;
runtime and platform metadata are intentionally observational.

## Repository map

| Path | Contents |
| --- | --- |
| `experiments/` | Implementations, protocols, tests, results, and manifests |
| `figures/` | Reproducible vector outputs with machine-readable data alternatives |
| `notebooks/tutorials/` | Short executable lessons |
| `notebooks/reproduction/` | Artifact and provenance walkthroughs |
| `tutorials/` | Text-first learning path and expected outputs |
| `docs/` | Quarto source for the GitHub Pages site |
| `scripts/` | Public-scope and notebook validation gates |

## Accessibility

Charts use markers, line styles, hatching, or fill state in addition to color.
The underlying CSV/JSON files, data dictionary, and long descriptions are
available from the [accessibility guide](docs/accessibility.md). Tutorials are
provided in both text-first and notebook forms. Please use the accessibility
issue form to report a barrier or request another format.

## Citation, contribution, and security

Use [`CITATION.cff`](CITATION.cff) and cite the study whose result you use.
Focused corrections and reproducibility reports are welcome; see
[`CONTRIBUTING.md`](CONTRIBUTING.md). Security-sensitive reports should follow
[`SECURITY.md`](SECURITY.md), not a public issue.

## License status

No open-source or open-data license has yet been granted for this repository.
Unless and until the copyright holders add an approved license, all rights are
reserved. Public visibility makes the materials inspectable; it does not grant
reuse rights beyond those provided by law.
