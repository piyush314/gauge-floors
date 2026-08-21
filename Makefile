PYTHON ?= python3
QUARTO ?= quarto
NOTEBOOK_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help install scope-check links test verify verify-artifacts notebooks validate ci reproduce docs site

help:
	@echo "Public reproducibility targets:"
	@echo "  install          install the pinned experiment environment"
	@echo "  scope-check      reject manuscript files, paths, and public history"
	@echo "  links            check every local Markdown and Quarto link target"
	@echo "  test             run the experiment unit tests"
	@echo "  verify-artifacts verify committed SHA-256 artifact hashes"
	@echo "  notebooks        execute clean notebook copies in fresh kernels"
	@echo "  validate         run scope, unit-test, and artifact-integrity gates"
	@echo "  ci               run validate plus notebook execution"
	@echo "  reproduce        regenerate all computational artifacts"
	@echo "  docs              render the Quarto Pages site"

install:
	$(PYTHON) -m pip install -r experiments/requirements-lock.txt

scope-check:
	$(PYTHON) scripts/check_public_scope.py

links:
	$(PYTHON) scripts/check_links.py

test:
	$(PYTHON) -m unittest discover -s experiments/tests -t . -p 'test_*.py' -v

verify-artifacts:
	$(PYTHON) -m experiments.verify_artifacts

verify: verify-artifacts

notebooks:
	$(PYTHON) scripts/execute_notebooks.py --python "$(PYTHON)" $(NOTEBOOK_ARGS)

validate: scope-check links test verify-artifacts

ci: validate notebooks

reproduce:
	$(PYTHON) -m experiments.reproduce all

docs site:
	$(QUARTO) render docs
