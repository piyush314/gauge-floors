# Documentation site

This directory contains the Quarto source for the public learning site. It
explains the computational models, mirrors the executable notebook tutorials,
documents every artifact, and states the public/private source boundary.

```sh
quarto preview docs
quarto render docs
```

GitHub Actions publishes `docs/_site` through the official Pages artifact
workflow. The workflow never publishes files outside Quarto's generated output.
