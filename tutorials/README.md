# Tutorials

The public learning path is available in two synchronized forms:

- text-first web pages for reading and assistive technology; and
- small output-free notebooks that CI executes from a clean kernel.

| Lesson | Web | Notebook | Main check |
| --- | --- | --- | --- |
| Gauge balancing | [Read](../docs/tutorials/nuclear-floor.qmd) | [Run](../notebooks/tutorials/01_gauge_balancing.ipynb) | Product is unchanged and energy reaches its floor |
| Coordinated rounding | [Read](../docs/tutorials/coordinated-rounding.qmd) | [Run](../notebooks/tutorials/02_coordinated_rounding.ipynb) | Four rank-one rounding errors cancel exactly |

The [reproduction notebooks](../notebooks/reproduction/) explain the committed
manifest and rebuild charts directly from CSV. They do not rerun the complete
3--5 minute protocol inside a notebook.

## Run locally

```sh
.venv-experiments/bin/python -m pip install -r notebooks/requirements.txt
.venv-experiments/bin/python scripts/execute_notebooks.py
```

Notebook execution writes clean copies to a temporary directory. It never
modifies or commits notebook outputs.
