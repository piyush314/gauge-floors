# Getting started

## Requirements

- CPython 3.13.7 (the version recorded in `.python-version`)
- about 1 GB of free space for the virtual environment and generated artifacts
- a CPU; no GPU or private dataset is required
- Git and a SHA-256 implementation for provenance checks

The Digits dataset ships with scikit-learn. After installing the pinned Python
packages, the experiments make no network request.

## macOS or Linux

From the repository root:

```sh
python3.13 -m venv .venv-experiments
.venv-experiments/bin/python -m pip install --upgrade pip
.venv-experiments/bin/python -m pip install -r experiments/requirements-lock.txt
make validate PYTHON=.venv-experiments/bin/python
```

Use `experiments/requirements-lock.txt` for artifact reproduction. The shorter
`experiments/requirements.txt` lists only direct scientific dependencies.

## Windows PowerShell

```powershell
py -3.13 -m venv .venv-experiments
.venv-experiments\Scripts\python.exe -m pip install --upgrade pip
.venv-experiments\Scripts\python.exe -m pip install -r experiments\requirements-lock.txt
.venv-experiments\Scripts\python.exe -m unittest discover -s experiments\tests -v
.venv-experiments\Scripts\python.exe -m experiments.verify_artifacts
```

The Makefile is a convenience for macOS, Linux, and GitHub Actions. Every
validation target has an equivalent Python command.

## Reproduce the evidence

```sh
.venv-experiments/bin/python -m experiments.reproduce all
.venv-experiments/bin/python -m experiments.verify_artifacts
```

Use `nuclear` or `discrepancy` instead of `all` to run one family. The complete
run takes roughly 3--5 minutes on the reference Apple-silicon system.

## Common problems

`experiment environment does not match`: create a fresh environment from the
lock file. The canonical command deliberately refuses silently different
scientific dependency versions.

Artifact hash mismatch: start with `git status`. Regeneration rewrites tracked
artifacts. Use a clean clone if you intended only to verify the release.

Slow or inconsistent timings: BLAS/OpenMP thread counts default to one unless
you set them first. Timings describe the current machine; they are not
correctness criteria or portable benchmarks.

Notebook kernel not found: install `notebooks/requirements.txt` into the same
environment, then run `python scripts/execute_notebooks.py` from the repository
root.
