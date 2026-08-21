# Contributing

Questions, reproducibility reports, documentation improvements, and focused
correctness fixes are welcome. Please open an issue before a large change so
that its computational scope and validation plan are clear.

## Local checks

Use CPython 3.13.7 and the resolved environment:

```sh
python3.13 -m venv .venv-experiments
.venv-experiments/bin/python -m pip install -r experiments/requirements-lock.txt
make validate PYTHON=.venv-experiments/bin/python
```

`make validate` enforces the public-scope boundary, runs unit tests, and verifies
the SHA-256 inventory of committed artifacts.

## Generated artifacts

Do not hand-edit files under `figures/` or `experiments/results/`. Change the
generator, add or update a test, then run:

```sh
.venv-experiments/bin/python -m experiments.reproduce all
.venv-experiments/bin/python -m experiments.verify_artifacts
```

Runtime measurements and platform metadata are expected to vary; scientific
fields must satisfy the exact identities and tolerances enforced by the test
suite. Explain any changed scientific result in the pull request.

## Documentation and accessibility

- Use descriptive link text and semantic Markdown headings.
- Do not communicate a chart distinction by color alone.
- Update `docs/figure-descriptions.md` and the data dictionary with new outputs.
- State limitations and assumptions near the claim they qualify.
- Execute every changed notebook from a clean kernel before opening a pull
  request.

## Public-scope boundary

Manuscript source does not belong in this repository. Do not add `.tex`, `.bib`,
`.cls`, or `.sty` files; paper PDFs; submission archives; private review notes;
or copied manuscript prose. `make scope-check` enforces the mechanical part of
this rule.

## Scope and rights

Contributors must have the right to submit their material. The repository does
not currently grant a public reuse license; an institution-approved license is
still pending.
