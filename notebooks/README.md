# Notebooks

These notebooks are executable interfaces, not screenshots of one successful
run.

- [`tutorials/`](tutorials/) contains two short concept lessons mirrored by the
  GitHub Pages site.
- [`reproduction/`](reproduction/) contains manifest and chart-provenance
  walkthroughs.

Install the pinned scientific environment and the lightweight notebook runner,
then execute every notebook from the repository root:

```sh
.venv-experiments/bin/python -m pip install -r notebooks/requirements.txt
.venv-experiments/bin/python scripts/execute_notebooks.py
```

CI runs clean copies in a temporary directory. Committed notebooks contain no
cell outputs, execution counts, widgets, embedded datasets, or manuscript
content.
