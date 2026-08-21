"""Reproducible experiments for the two arXiv manuscripts."""

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

BASE_SEED = 20260723

__all__ = ["BASE_SEED"]
